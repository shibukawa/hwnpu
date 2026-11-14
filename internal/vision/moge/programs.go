package moge

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/imagegen/milpart"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/vision/depth"
	"github.com/shibukawa/hwnpu/internal/vision/moge/mil"
)

// ExportVersion is the layout of the programs Prepare and ConvertSize write. 2: the dense head's
// deconvolutions shuffled in bands (mil.deconv2, the M1 on Core AI).
const ExportVersion = 2

// Tokens is the ViT's patches an export of Prepare is made for (MoGe's num_tokens; iron_moge.py's
// default).
const Tokens = 2500

// Aspects are the picture aspects an export of Prepare holds (iron_moge.py's ASPECTS).
var Aspects = []float64{1, 4.0 / 3, 3.0 / 4, 3.0 / 2, 2.0 / 3, 16.0 / 10, 10.0 / 16, 16.0 / 9, 9.0 / 16}

// BaseSize is MoGe's patch grid of tokens at an aspect (columns, rows).
func BaseSize(tokens int, aspect float64) (int, int) {
	return int(math.RoundToEven(math.Sqrt(float64(tokens) * aspect))), int(math.RoundToEven(math.Sqrt(float64(tokens) / aspect)))
}

// progSized is one size's Core ML programs.
type progSized struct {
	vit, dense runner.Program
}

func (p *progSized) close() error {
	err := p.vit.Close()
	if err2 := p.dense.Close(); err == nil {
		err = err2
	}
	return err
}

// useProgram loads size w x h of an export of Prepare (Core ML packages or OpenVINO IR),
// converting it first when it is missing.
func (e *Estimator) useProgram(ctx context.Context, d string, w, h int) (*sized, error) {
	if !llm.FetchExport(ctx, d, d+".done") { // here, or from the shared layers
		if e.Logf != nil {
			e.Logf("moge: converting %dx%d (the first picture of this aspect)", w, h)
		}
		if err := ConvertSize(e.Checkpoint, d, w/14, h/14, e.Format); err != nil {
			return nil, err
		}
		llm.ShareExport(ctx, d)
	}
	b, err := os.ReadFile(filepath.Join(d, "model.json"))
	if err != nil {
		return nil, err
	}
	s := &sized{prog: &progSized{}}
	if err := json.Unmarshal(b, &s.sizeConfig); err != nil {
		return nil, fmt.Errorf("moge: %w", err)
	}
	for _, x := range []struct {
		p    *runner.Program
		name string
	}{{&s.prog.vit, "vit"}, {&s.prog.dense, "dense"}} {
		t0 := time.Now()
		p, err := runner.Open(ctx, e.dev, filepath.Join(d, x.name+depth.ProgramExt(e.Format)))
		// OpenVINO's CPU plugin (the Intel path checked without Intel's NPU) is not the NPU
		if err == nil && e.dev != nil && !p.OnNPU() && e.dev.Info().Backend != "openvino" {
			p.Close()
			err = fmt.Errorf("%s does not run on the NPU alone", x.name)
		}
		if err != nil {
			if s.prog.vit != nil {
				s.prog.vit.Close()
			}
			return nil, fmt.Errorf("moge: %s: %w", x.name, err)
		}
		*x.p = p
		if e.Logf != nil {
			e.Logf("moge: loaded %s (%dx%d) in %s", x.name, w, h, console.Duration(time.Since(t0)))
		}
	}
	e.cur = s
	return s, nil
}

// estimateProgram runs a size's programs on the picture's pixels px (the size's) and
// post-processes at ow x oh.
func (e *Estimator) estimateProgram(ctx context.Context, s *sized, px []uint8, ow, oh int, t0, tl time.Time) (*Result, error) {
	f, err := s.prog.vit.Run(ctx, map[string][]float32{"patches": depth.PatchRows(px, s.Width, s.Height, s.Patch, s.Tokens, s.Mean, s.Std)})
	if err != nil {
		return nil, fmt.Errorf("moge: vit: %w", err)
	}
	t1 := time.Now()
	out, err := s.prog.dense.Run(ctx, map[string][]float32{"f": f})
	if err != nil {
		return nil, fmt.Errorf("moge: dense: %w", err)
	}
	t2 := time.Now()
	H4, W4 := s.Config.BaseH<<4, s.Config.BaseW<<4
	if len(out) != 7*H4*W4 {
		return nil, fmt.Errorf("moge: dense: %d values for 7 maps of %dx%d", len(out), W4, H4)
	}
	raw := make([][]float32, 7)
	for i := range raw {
		raw[i] = out[i*H4*W4 : (i+1)*H4*W4]
	}
	r := Post(raw, W4, H4, ow, oh, f[0], e.FovX)
	if e.Logf != nil {
		e.Logf("moge: %dx%d, load %.2f s, vit %.2f s, neck and heads %.2f s, post %.2f s; scale %.2f, fov %.1f°",
			s.Width, s.Height, tl.Sub(t0).Seconds(), t1.Sub(tl).Seconds(), t2.Sub(t1).Seconds(), time.Since(t2).Seconds(), r.Scale, r.FovX)
	}
	return r, nil
}

// ConvertSize writes the programs of the MoGe-2 checkpoint (model.pt) for bw x bh patches into
// d in format ("coreai": .aimodel; "coreml": vit.mlpackage and dense.mlpackage; "intel": vit.xml and dense.xml,
// OpenVINO IR) and model.json; d.done, written last, says they are whole.
func ConvertSize(checkpoint, d string, bw, bh int, format string) error {
	wt, err := mil.OpenWeights(checkpoint)
	if err != nil {
		return err
	}
	defer wt.Close()
	c, err := wt.ReadConfig(bw, bh)
	if err != nil {
		return err
	}
	if err := os.RemoveAll(d); err != nil {
		return err
	}
	if err := os.MkdirAll(d, 0o755); err != nil {
		return err
	}
	for _, p := range []*milpart.Part{mil.Vit(c), mil.Dense(c)} {
		if _, err := llm.WritePart(filepath.Join(d, p.Name+depth.ProgramExt(format)), p, wt, llm.Options{Backend: depth.ProgramBackend(format), Format: "fp16"}, nil); err != nil {
			return fmt.Errorf("moge: %s: %w", p.Name, err)
		}
	}
	sc := sizeConfig{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}, Patch: 14, Tokens: c.Tokens(),
		Rows: c.Tokens(), Width: c.Width, Height: c.Height}
	sc.Config.BaseW, sc.Config.BaseH = bw, bh
	b, err := json.MarshalIndent(sc, "", " ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(d, "model.json"), append(b, '\n'), 0o644); err != nil {
		return err
	}
	return llm.FinishExport(d, d+".done")
}

// Setup is what Prepare makes ready.
type Setup struct {
	Dir  string // the checkpoints (Dir/NAME) and the exports
	Repo string // the Hugging Face repository (Ruicheng/moge-2-vits-normal)
	Name string
	// Backend is the program format (llm.Options.Backend): Core ML packages (the default) or
	// OpenVINO IR for Intel's NPU.
	Backend string
	Log     io.Writer
}

// Prepare makes a MoGe-2 ready for Apple's Neural Engine (or Intel's NPU, s.Backend) in s.Dir: it downloads model.pt when
// missing and writes the export's model.json; each size is converted when a picture first takes
// it.
func Prepare(ctx context.Context, s Setup) (string, error) {
	root := filepath.Join(s.Dir, s.Name)
	if err := os.MkdirAll(root, 0o755); err != nil {
		return "", err
	}
	pt := filepath.Join(root, "model.pt")
	if err := chat.Fetch(ctx, s.Repo, "model.pt", pt, s.Log); err != nil {
		return "", err
	}
	format := depth.ExportFormat(s.Backend)
	// the export beside the checkpoint, in the model's directory (as every model's); the one an
	// earlier hwnpu kept in the models' directory is removed (its sizes are converted again)
	old := filepath.Join(s.Dir, fmt.Sprintf("%s-v%d.%s", strings.ToLower(s.Name), ExportVersion, format))
	os.RemoveAll(old)
	os.Remove(old + ".done")
	out := filepath.Join(root, fmt.Sprintf("%s-v%d.%s", strings.ToLower(s.Name), ExportVersion, format))
	x := Export{Name: s.Name, Kind: "moge", Tokens: Tokens, Format: format, Checkpoint: pt}
	for _, a := range Aspects {
		bw, bh := BaseSize(Tokens, a)
		x.Sizes = append(x.Sizes, [2]int{14 * bw, 14 * bh})
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		return "", err
	}
	b, err := json.MarshalIndent(x, "", " ")
	if err != nil {
		return "", err
	}
	return out, os.WriteFile(filepath.Join(out, "model.json"), append(b, '\n'), 0o644)
}
