package depth

import (
	"context"
	"encoding/json"
	"fmt"
	"image"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/vision/depth/mil"
)

// ExportVersion is the layout of the programs Prepare and ConvertSize write; an export of
// another is written again.
const ExportVersion = 1

// Sizes are the sizes an export of Prepare holds (tools/amd/iron_da3.py's SIZES): the official
// processor's for common aspects, the longest side 504.
var Sizes = [][2]int{{504, 504}, {504, 378}, {378, 504}, {504, 336}, {336, 504}, {504, 322}, {322, 504}, {504, 280}, {280, 504}}

// progSized is one size's Core ML programs.
type progSized struct {
	vit, head runner.Program
}

func (p *progSized) close() error {
	err := p.vit.Close()
	if err2 := p.head.Close(); err == nil {
		err = err2
	}
	return err
}

// useProgram loads size w x h of an export of Prepare (Core ML packages or OpenVINO IR),
// converting it first when it is missing.
func (e *Estimator) useProgram(ctx context.Context, d string, w, h int) (*sized, error) {
	if !llm.FetchExport(ctx, d, d+".done") { // here, or from the shared layers
		if err := llm.MayConvert(ctx, d); err != nil {
			return nil, err
		}
		if e.Logf != nil {
			e.Logf("depth: converting %dx%d (the first picture of this aspect)", w, h)
		}
		if err := e.fetchCheckpoint(ctx, LogWriter(e.Logf)); err != nil {
			return nil, err
		}
		if err := ConvertSize(e.Checkpoint, d, w, h, e.Format); err != nil {
			return nil, err
		}
		llm.ShareExport(ctx, d)
	}
	b, err := os.ReadFile(filepath.Join(d, "model.json"))
	if err != nil {
		return nil, err
	}
	s := &sized{prog: &progSized{}}
	if err := json.Unmarshal(b, &s.Config); err != nil {
		return nil, fmt.Errorf("depth: %w", err)
	}
	for _, x := range []struct {
		p    *runner.Program
		name string
	}{{&s.prog.vit, "vit"}, {&s.prog.head, "head"}} {
		t0 := time.Now()
		p, err := runner.Open(ctx, e.dev, filepath.Join(d, x.name+ProgramExt(e.Format)))
		// OpenVINO's CPU plugin (the Intel path checked without Intel's NPU) is not the NPU
		if err == nil && e.dev != nil && !p.OnNPU() && e.dev.Info().Backend != "openvino" {
			p.Close()
			err = fmt.Errorf("%s does not run on the NPU alone", x.name)
		}
		if err != nil {
			if s.prog.vit != nil {
				s.prog.vit.Close()
			}
			return nil, fmt.Errorf("depth: %s: %w", x.name, err)
		}
		*x.p = p
		if e.Logf != nil {
			e.Logf("depth: loaded %s (%dx%d) in %s", x.name, w, h, console.Duration(time.Since(t0)))
		}
	}
	e.cur = s
	return s, nil
}

// estimateProgram runs a size's programs on img.
func (e *Estimator) estimateProgram(ctx context.Context, s *sized, img image.Image, t0, tl time.Time) (*Result, error) {
	feats, err := s.prog.vit.Run(ctx, map[string][]float32{"patches": s.Patches(img)})
	if err != nil {
		return nil, fmt.Errorf("depth: vit: %w", err)
	}
	t1 := time.Now()
	out, err := s.prog.head.Run(ctx, map[string][]float32{"feats": feats})
	if err != nil {
		return nil, fmt.Errorf("depth: head: %w", err)
	}
	t2 := time.Now()
	W, H := s.Model.Width, s.Model.Height
	if len(out) != 2*W*H {
		return nil, fmt.Errorf("depth: head: %d values for %dx%d", len(out), W, H)
	}
	r := e.result(W, H, out[:W*H], out[W*H:])
	if e.Logf != nil {
		e.Logf("depth: %dx%d, load %.2f s, vit %.2f s, head %.2f s", W, H, tl.Sub(t0).Seconds(), t1.Sub(tl).Seconds(), t2.Sub(t1).Seconds())
	}
	return r, nil
}

// ConvertSize writes the programs of the DA3 checkpoint root (config.json, model.safetensors)
// for w x h pixels into d in format ("coreai": vit.aimodel and head.aimodel; "coreml":
// vit.mlpackage and head.mlpackage; "intel": vit.xml and head.xml, OpenVINO IR) and model.json; d.done, written last, says they are whole.
func ConvertSize(root, d string, w, h int, format string) error {
	c, err := mil.ReadConfig(root, w, h)
	if err != nil {
		return err
	}
	wt, err := mil.OpenWeights(root)
	if err != nil {
		return err
	}
	defer wt.Close()
	if err := os.RemoveAll(d); err != nil {
		return err
	}
	if err := os.MkdirAll(d, 0o755); err != nil {
		return err
	}
	for _, p := range []*mil.Part{mil.Vit(c), mil.Head(c)} {
		if _, err := llm.WritePart(filepath.Join(d, p.Name+ProgramExt(format)), p, wt, llm.Options{Backend: ProgramBackend(format), Format: "fp16"}, nil); err != nil {
			return fmt.Errorf("depth: %s: %w", p.Name, err)
		}
	}
	cfg := Config{Name: filepath.Base(root), Kind: "depth", Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225},
		Patch: mil.Patch, Tokens: c.Tokens(), Rows: c.Tokens()}
	cfg.Model.Width, cfg.Model.Height, cfg.Model.Hidden = w, h, c.Hidden
	b, err := json.MarshalIndent(cfg, "", " ")
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
	Dir  string // the exports (Dir/NAME); the checkpoint is the repository's (chat.RepoDir)
	Repo string // the Hugging Face repository (depth-anything/DA3MONO-LARGE)
	Name string
	// Backend is the program format (llm.Options.Backend): Core AI programs, Core ML packages
	// (the default) or OpenVINO IR for Intel's NPU.
	Backend string
	Log     io.Writer
}

// Prepare makes a DA3 ready for Apple's Neural Engine (or Intel's NPU, s.Backend) in s.Dir, as hwnpu-chat makes its models
// ready: it downloads the checkpoint's config.json when missing and writes the export's
// model.json; each size is taken from the shared layers or converted when a picture first takes
// it (seconds; its first load compiles it), the checkpoint's weights downloaded then.
func Prepare(ctx context.Context, s Setup) (string, error) {
	root := filepath.Join(s.Dir, s.Name)
	if _, err := chat.Fetch(ctx, s.Repo, "config.json", s.Log); err != nil {
		return "", err
	}
	src := chat.RepoDir(s.Repo)
	c, err := mil.ReadConfig(src, Sizes[0][0], Sizes[0][1])
	if err != nil {
		return "", err
	}
	format := ExportFormat(s.Backend)
	// the export in the model's directory (as every model's); the one an earlier hwnpu kept in
	// the models' directory is removed (its sizes are converted again)
	old := filepath.Join(s.Dir, fmt.Sprintf("%s-v%d.%s", strings.ToLower(s.Name), ExportVersion, format))
	os.RemoveAll(old)
	os.Remove(old + ".done")
	out := filepath.Join(root, fmt.Sprintf("%s-v%d.%s", strings.ToLower(s.Name), ExportVersion, format))
	x := Export{Name: s.Name, Kind: "depth", Metric: strings.Contains(strings.ToLower(s.Name), "metric"), MetricScale: 300,
		Second: "sky", Format: format, Checkpoint: src, Repo: s.Repo, Version: ExportVersion}
	if c.Dual {
		x.Second = "confidence"
	}
	for _, sz := range Sizes {
		x.Sizes = append(x.Sizes, sz)
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

// ExportFormat is an export's Format for a program format (llm.Options.Backend): "intel" for
// OpenVINO IR, "coreai" for Core AI's programs, else "coreml".
func ExportFormat(backend string) string {
	switch backend {
	case llm.BackendIntel, llm.BackendCoreAI:
		return backend
	}
	return "coreml"
}

// ProgramExt is the file name extension of a program of an export's Format.
func ProgramExt(format string) string { return llm.ProgramExt(ProgramBackend(format)) }

// ProgramBackend is the program format (llm.Options.Backend) of an export's Format ("" for an
// export before Format: coreml).
func ProgramBackend(format string) string {
	switch format {
	case llm.BackendIntel, llm.BackendCoreAI:
		return format
	}
	return llm.BackendCoreML
}

// ConvertSizes converts each size of the export of Prepare in dir (Core AI or Core ML programs,
// OpenVINO IR) ahead, or takes it from the shared layers, as the first picture of each aspect
// would: an Estimator then needs no network nor conversion. log (nil: nowhere) is told of each
// size converted.
func ConvertSizes(ctx context.Context, dir string, log io.Writer) error {
	x, err := readExport(dir)
	if err != nil {
		return err
	}
	if x.Format != "coreml" && x.Format != "coreai" && x.Format != "intel" {
		return nil // AMD's graphs are whole in the export
	}
	for _, sz := range x.Sizes {
		d := filepath.Join(dir, fmt.Sprintf("%dx%d", sz[0], sz[1]))
		if llm.FetchExport(ctx, d, d+".done") {
			continue
		}
		if err := llm.MayConvert(ctx, d); err != nil {
			return err
		}
		if log != nil {
			fmt.Fprintf(log, "depth: converting %dx%d\n", sz[0], sz[1])
		}
		if err := x.fetchCheckpoint(ctx, log); err != nil {
			return err
		}
		if err := ConvertSize(x.Checkpoint, d, sz[0], sz[1], x.Format); err != nil {
			return err
		}
		llm.ShareExport(ctx, d)
		if err := ctx.Err(); err != nil {
			return err
		}
	}
	return nil
}

// Warm loads each size's programs once, converting what is missing: the first load of each
// compiles it, so that a picture of any aspect does not wait. AMD's graphs need none of it.
func (e *Estimator) Warm(ctx context.Context) error {
	if e.Format != "coreml" && e.Format != "coreai" && e.Format != "intel" {
		return nil
	}
	for _, sz := range e.Sizes {
		if _, err := e.use(ctx, sz[0], sz[1]); err != nil {
			return err
		}
	}
	return nil
}

// fetchCheckpoint downloads the checkpoint's weights (model.safetensors of Repo) for converting a
// size, unless they are there; an export without Repo (of an earlier hwnpu) had them downloaded
// by Prepare.
func (x *Export) fetchCheckpoint(ctx context.Context, log io.Writer) error {
	if x.Repo == "" {
		return nil
	}
	for _, f := range []string{"config.json", "model.safetensors"} {
		if _, err := chat.Fetch(ctx, x.Repo, f, log); err != nil {
			return err
		}
	}
	return nil
}

// LogWriter writes lines to logf (nil: nowhere).
func LogWriter(logf func(string, ...any)) io.Writer {
	if logf == nil {
		return nil
	}
	return lineWriter(logf)
}

type lineWriter func(string, ...any)

func (w lineWriter) Write(b []byte) (int, error) {
	w("%s", strings.TrimRight(string(b), "\n"))
	return len(b), nil
}
