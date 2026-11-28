package engine

import (
	"context"
	"fmt"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"image"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/ai/vision"
	"github.com/shibukawa/hwnpu/ai/vision/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagedev"
	"github.com/shibukawa/hwnpu/internal/vision/segment"
)

func init() {
	r := bridge.Roles["segment"]
	r.Prepare = prepareSegment
	r.Load = loadSegment
}

// segmentSetup is what segment downloads for model m: its encoder's and decoder's ONNX files,
// which models.yaml's sha256 names.
func segmentSetup(m chat.Model, log io.Writer) (segment.Setup, error) {
	s := segment.Setup{Dir: chat.ModelDir(m), Repo: m.Source(), SHA256: m.SHA256, Log: log}
	for f := range m.SHA256 {
		switch {
		case strings.HasSuffix(f, "_encoder.onnx"):
			s.Encoder = f
		case strings.HasSuffix(f, "_decoder.onnx"):
			s.Decoder = f
		}
	}
	if s.Encoder == "" || s.Decoder == "" {
		return s, fmt.Errorf("%s: models.yaml names no *_encoder.onnx and *_decoder.onnx in its sha256", shown(m.Name))
	}
	return s, nil
}

func prepareSegment(e airole.PrepareEnv) (f airole.Files, err error) {
	m := e.Model
	s, err := segmentSetup(m, e.Log)
	if err != nil {
		return f, err
	}
	flag := imagedev.DeviceFlag(m)
	fetchEnc := func() (string, error) { enc, _, err := segment.Fetch(e.Ctx, s); return enc, err }
	have := func(dir string) error { _, err := os.Stat(filepath.Join(dir, "model.json")); return err }
	export := imagedev.AMDFirstRunExport(e.Ctx, e.Base, flag, false, logf(e.Log), e.Log, have, "iron_efficientsam.py", segment.ExportVersion, fetchEnc,
		func(python []string, model, dir string) error {
			return segment.ExportNPU(e.Ctx, python, e.Base.Name, model, dir, e.Log)
		})
	if export == "" && !imagedev.ProgramDevice(m, e.NPU.Vendor, flag) {
		// the CPU: the encoder's Core ML package with the decoder's part file made before
		// (hwnpu-dev models package MODEL/coreml), here or in the shared layers, on hwnpu's own
		// executor even where ONNX Runtime is, and no ONNX file downloaded
		if program, err := segment.PrepareCPU(llm.WithFetchOnly(e.Ctx), s, chat.ModelDir(e.Base)); err == nil {
			f.Set("cpu-program", program)
			return f, nil
		}
	}
	enc, dec, err := segment.Fetch(e.Ctx, s)
	if err != nil {
		return f, err
	}
	f.Set("encoder", enc)
	f.Set("decoder", dec)
	switch {
	case export != "":
		f.Set("export", export)
	case imagedev.ProgramDevice(m, e.NPU.Vendor, flag):
		// Intel's NPU runs the encoder converted from its ONNX file to OpenVINO IR, Apple's
		// Neural Engine the same as Core AI's program (M3: 0.4 s where hwnpu's evaluator
		// takes 33 s, the mask's IoU with the CPU's 0.9994; docs/design/vision-models.md),
		// converted into the model's directory at its first load
		dir := chat.ModelDir(e.Base)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return f, err
		}
		f.Set("programs", dir)
	}
	return f, nil
}

func loadSegment(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
	var (
		sm  *segment.Model
		dev *hwnpu.Device
		err error
	)
	device := "npu"
	switch {
	case f.Path("cpu-program") != "":
		device = "cpu"
		milcpu.Threads = threads(e.Def)
		sm, err = segment.OpenCPUProgram(ctx, f.Path("cpu-program"))
	case f.Path("export") != "":
		if dev, err = hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD)); err != nil {
			return nil, err
		}
		sm, err = segment.OpenNPU(ctx, dev, f.Path("export"), f.Path("decoder"))
	case f.Path("programs") != "":
		var backend string
		var opts []hwnpu.Option
		if e.NPU.Vendor == "apple" {
			opts = append(opts, hwnpu.WithPlacement(hwnpu.PlacementStrict))
		}
		if dev, backend, err = imagedev.Open(ctx, e.NPU.Device, e.Cache, opts...); err != nil {
			return nil, err
		}
		sm, err = segment.OpenProgram(ctx, dev, backend, f.Path("encoder"), f.Path("decoder"), f.Path("programs"), e.Log)
	default:
		device = "cpu"
		sm, err = segment.OpenCPU(f.Path("encoder"), f.Path("decoder"), threads(e.Def))
	}
	if err != nil {
		if dev != nil {
			dev.Close()
		}
		return nil, err
	}
	return bridge.Wrap["segment"](&segmenter{m: sm, dev: dev, name: e.Model, device: device}), nil
}

// points is prompt p as the decoder reads it: the points on the thing, those off it, then the
// box's corners.
func points(p vision.Prompt) []segment.Point {
	var out []segment.Point
	for _, q := range p.Points {
		out = append(out, segment.Point{X: q.X, Y: q.Y, Label: 1})
	}
	for _, q := range p.Negative {
		out = append(out, segment.Point{X: q.X, Y: q.Y, Label: 0})
	}
	if p.Box != nil {
		out = append(out, segment.Point{X: p.Box.Min.X, Y: p.Box.Min.Y, Label: 2}, segment.Point{X: p.Box.Max.X, Y: p.Box.Max.Y, Label: 3})
	}
	return out
}

// segmenter is a segmentation model of the engine (vision.Segmenter's).
type segmenter struct {
	mu     sync.Mutex
	m      *segment.Model
	dev    *hwnpu.Device
	name   string
	device string
}

func (s *segmenter) Model() string  { return s.name }
func (s *segmenter) Device() string { return s.device }
func (s *segmenter) Where() string  { return s.m.Where() }

// Encode is the picture's encoding, whose own is a *segment.Embedding.
func (s *segmenter) Encode(ctx context.Context, img image.Image) (*vision.PictureEncoding, any, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	t0 := time.Now()
	e, err := s.m.Embed(ctx, img)
	if err != nil {
		return nil, nil, err
	}
	return &vision.PictureEncoding{Took: time.Since(t0), Resize: s.m.Times[0], Encode: s.m.Times[1]}, e, nil
}

func (s *segmenter) Masks(encoding any, p vision.Prompt) (*vision.Masks, error) {
	e, _ := encoding.(*segment.Embedding)
	pts := points(p)
	s.mu.Lock()
	defer s.mu.Unlock()
	t0 := time.Now()
	ks, err := s.m.Segment(e, pts)
	if err != nil {
		return nil, err
	}
	out := &vision.Masks{Took: time.Since(t0)}
	for _, k := range ks {
		out.Masks = append(out.Masks, vision.Mask{W: k.W, H: k.H, In: k.In, Score: k.Score})
	}
	return out, nil
}

func (s *segmenter) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	err := s.m.Close()
	if s.dev != nil {
		s.dev.Close()
	}
	return err
}
