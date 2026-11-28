package engine

import (
	"context"
	"errors"
	"fmt"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"image"
	"io"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/ai/vision"
	"github.com/shibukawa/hwnpu/ai/vision/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagedev"
	"github.com/shibukawa/hwnpu/internal/vision/ocr"
)

func init() {
	r := bridge.Roles["ocr"]
	r.Prepare = prepareOCR
	r.Load = loadOCR
}

// ocrSetup is where the detector and the recognizer of model m come from: the recognizer's
// repository, and the detector's in models.yaml's with (detector), its files' sha256 under
// detector/.
func ocrSetup(m chat.Model, log io.Writer) (ocr.Setup, error) {
	det := m.With["detector"]
	if det == "" {
		return ocr.Setup{}, fmt.Errorf("%s: models.yaml gives no detector (with: {detector: hf:OWNER/NAME})", shown(m.Name))
	}
	detSHA, recSHA := map[string]string{}, map[string]string{}
	for f, d := range m.SHA256 {
		if rest, ok := strings.CutPrefix(f, "detector/"); ok {
			detSHA[rest] = d
		} else {
			recSHA[f] = d
		}
	}
	return ocr.Setup{Dir: chat.ModelsDir(), Det: ocr.Source{Repo: det, SHA256: detSHA}, Rec: ocr.Source{Repo: m.Source(), SHA256: recSHA}, Log: log}, nil
}

func prepareOCR(e airole.PrepareEnv) (f airole.Files, err error) {
	m := e.Model
	s, err := ocrSetup(m, e.Log)
	if err != nil {
		return f, err
	}
	ort, builtin := imagedev.CPURoute(e.Ctx, m, imagedev.DeviceFlag(m), interpreter(m), threads(m), quiet)
	if !ort && !interpreter(m) && !builtin && e.NPU.Format == "" {
		return f, errors.New("this machine's NPU runs neither Core AI programs nor OpenVINO IR (--device cpu runs PP-OCR on the CPU)")
	}
	if ort || builtin || interpreter(m) {
		// the CPU: the Core ML packages with their settings' part files made before (hwnpu-dev
		// models package MODEL/coreml), here or in the shared layers, on hwnpu's own executor
		// even where ONNX Runtime is, and no ONNX file downloaded
		p := s
		p.Backend, p.Parts = llm.BackendCoreML, true
		if r, err := ocr.Open(e.Ctx, nil, p); err == nil {
			r.Close()
			f.Set("models", s.Dir)
			f.SetValue("prebuilt", "1")
			return f, nil
		}
	}
	// the repositories' files (opening the reader downloads them; its programs are made later)
	s.CPU = true
	r, err := ocr.Open(e.Ctx, nil, s)
	if err != nil {
		return f, err
	}
	r.Close()
	// the programs are made in s.Dir at the first picture: it is there for the record of the
	// store, which counts a use whose files are missing as not prepared (a fresh cache has none)
	if err := os.MkdirAll(s.Dir, 0o755); err != nil {
		return f, err
	}
	f.Set("models", s.Dir)
	return f, nil
}

func loadOCR(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
	m := e.Def
	s, err := ocrSetup(m, e.Log)
	if err != nil {
		return nil, err
	}
	n, interp := threads(m), interpreter(m)
	ort, builtin := imagedev.CPURoute(ctx, named(m), imagedev.DeviceFlag(m), interp, n, logf(e.Log))
	if f.Values["prebuilt"] == "1" && ort {
		// prepared from the prebuilt Core ML packages, no ONNX file: hwnpu's own executor
		ort, builtin = false, true
		milcpu.Threads = max(1, n)
	}
	cpu := interp || builtin
	var (
		dev      *hwnpu.Device
		closeDev = func() {}
	)
	if !ort {
		if dev, s.Backend, closeDev, err = imagedev.OpenProgramDevice(ctx, cpu, e.Cache, e.NPU.Device, "PP-OCR"); err != nil {
			return nil, err
		}
	}
	s.Dir, s.CPU, s.Threads = f.Path("models"), ort, n
	r, err := ocr.Open(ctx, dev, s)
	if err != nil {
		closeDev()
		return nil, err
	}
	r.Logf = logf(e.Log)
	t := &textReader{r: r, closeDev: closeDev, name: e.Model, device: "cpu"}
	t.model, _ = chat.Lookup(e.Model)
	switch {
	case ort:
		t.where = fmt.Sprintf("on the CPU (ONNX Runtime, %d threads)", n)
	case interp:
		t.where = imagedev.Placement(true, false)
	case builtin:
		t.where = "on the CPU (hwnpu's own executor)"
	default:
		t.device, t.where = "npu", "on the NPU"
	}
	return bridge.Wrap["ocr"](t), nil
}

// textReader is an OCR model of the engine (vision.TextReader's).
type textReader struct {
	mu            sync.Mutex
	r             *ocr.Reader
	closeDev      func()
	name          string
	model         chat.Model // its programs are made at the first picture, from its recipe (chat.WithModel)
	device, where string
}

func (t *textReader) Model() string  { return t.name }
func (t *textReader) Device() string { return t.device }
func (t *textReader) Where() string  { return t.where }

func (t *textReader) Read(ctx context.Context, img image.Image, o vision.OCROptions) (*vision.Text, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if s := o.Size; s != [2]int{} && (s[0] <= 0 || s[1] <= 0 || s[0]%32 != 0 || s[1]%32 != 0) {
		return nil, fmt.Errorf("vision: the detector's size %dx%d is not of multiples of 32", s[0], s[1])
	}
	t.r.MinScore, t.r.Size = o.MinScore, o.Size
	t0 := time.Now()
	lines, err := t.r.Read(chat.WithModel(ctx, t.model), img)
	if err != nil {
		return nil, err
	}
	b := img.Bounds()
	W, H := t.r.Choose(b.Dx(), b.Dy())
	tm := t.r.Times
	out := &vision.Text{Lines: []vision.TextLine{}, Size: [2]int{W, H}, Took: time.Since(t0), Resize: tm.Resize, Detect: tm.Detect, Boxes: tm.Boxes,
		Crop: tm.Crop, Recognize: tm.Recognize, Decode: tm.Decode, Calls: t.r.Calls}
	x0, y0 := float64(b.Min.X), float64(b.Min.Y)
	for _, l := range lines {
		tl := vision.TextLine{Text: l.Text, Score: float64(l.Score)}
		for i, c := range l.Box {
			tl.Box[i] = vision.Point{X: float64(c[0]) + x0, Y: float64(c[1]) + y0}
		}
		out.Lines = append(out.Lines, tl)
	}
	return out, nil
}

func (t *textReader) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	err := t.r.Close()
	t.closeDev()
	return err
}
