package anima

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/ovcpu"
)

// ProgramKind is the kind of manifest.json of an export of hwnpu-image -convert.
const ProgramKind = "anima-programs"

// ProgramManifest is manifest.json of an export of hwnpu-image -convert: the programs (Core ML
// packages, or OpenVINO IR .xml) by part, relative to the export's directory.
type ProgramManifest struct {
	Kind    string `json:"kind"`
	Backend string `json:"backend"` // coreml or intel
	Width   int    `json:"width"`
	Height  int    `json:"height"`
	// Formats are the weight formats of the parts (text, dit), as hwnpu-convert's -format.
	Formats map[string]string `json:"formats"`
	Text    string            `json:"text"`
	Time    string            `json:"time"`
	// TimeRows is the rows the time part takes (mil.TimeRows); 0 in earlier exports, which
	// took 1.
	TimeRows int `json:"time_rows,omitempty"`
	// Patch is the patch embedding of exports before it moved into the first group of blocks
	// (which then takes the patches); empty in later exports.
	Patch  string   `json:"patch,omitempty"`
	Blocks []string `json:"blocks"` // the DiT's groups of blocks, in order
	Final  string   `json:"final"`
	Vae    []string `json:"vae"` // the VAE's stages, in order
}

// IsProgramExport reports whether dir holds an export of hwnpu-image -convert.
func IsProgramExport(dir string) bool {
	m, err := ReadProgramManifest(dir)
	return err == nil && m.Kind == ProgramKind
}

// ReadProgramManifest reads dir/manifest.json of an export of hwnpu-image -convert.
func ReadProgramManifest(dir string) (*ProgramManifest, error) {
	b, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return nil, err
	}
	var m ProgramManifest
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	if m.Kind != ProgramKind {
		return nil, fmt.Errorf("anima: %s is not an export of hwnpu-image -convert", dir)
	}
	return &m, nil
}

// WriteProgramManifest writes m as dir/manifest.json.
func WriteProgramManifest(dir string, m *ProgramManifest) error {
	m.Kind = ProgramKind
	b, err := json.MarshalIndent(m, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "manifest.json"), append(b, '\n'), 0o644)
}

// progEngine runs the programs of internal/imagegen/anima/mil, each a runner.Program (on the
// Neural Engine or Intel's NPU, or the CPU interpreter): they take and give float32.
type progEngine struct {
	dit                     DitConfig
	textP, time, patch, fin runner.Program
	blocks                  []runner.Program
	// vae is the VAE's stages; each is loaded only while it runs (at 1024 x 1024 their
	// working memory is a few GB, as AMD's stages, 16.72)
	vae      []string
	open     func(ctx context.Context, name string, show bool) (runner.Program, error)
	cond     []float32
	textRows int
	// timeRows is the rows the time part takes (mil.TimeRows; 1 in earlier exports).
	timeRows int
}

const (
	progRows   = 512
	progHidden = 1024
)

func openPrograms(ctx context.Context, dev *hwnpu.Device, dir string, log io.Writer) (*progEngine, DitConfig, TextConfig, error) {
	m, err := ReadProgramManifest(dir)
	var dc DitConfig
	tc := TextConfig{Rows: progRows, Hidden: progHidden, T5Vocab: 32128, QwenVocab: 151936}
	if err != nil {
		return nil, dc, tc, err
	}
	dc = DitConfig{Width: m.Width, Height: m.Height, Hidden: 2048, Layers: 28, TextDim: progHidden, TextTokens: progRows,
		LatentChannels: 16, Patch: 2, VAEScale: 8, Group: 28 / max(1, len(m.Blocks))}
	e := &progEngine{dit: dc, textRows: progRows}
	// the programs in the order they load, for "[3/19]"
	total := 3 + len(m.Blocks) + len(m.Vae)
	if m.Patch != "" {
		total++
	}
	where := "the CPU interpreter"
	if dev != nil {
		where = llm.DeviceName(dev)
	}
	if log != nil {
		fmt.Fprintf(log, "loading %d programs onto %s", total, where)
		if dev != nil {
			fmt.Fprint(log, " (the first time, each is compiled for it: minutes at 1024 x 1024)")
		}
		fmt.Fprintln(log)
	}
	loaded := 0
	open := func(ctx context.Context, name string, show bool) (runner.Program, error) {
		var bar *console.Bar
		if show && log != nil {
			loaded++
			bar = console.NewBar(log, fmt.Sprintf("  [%2d/%d] %s", loaded, total, name))
		}
		p, err := openProgram(ctx, dev, dir, m.Backend, name)
		if bar != nil {
			switch {
			case err != nil:
				bar.Done("failed")
			case p.OnNPU():
				bar.Done(console.Duration(bar.Elapsed()) + ", all on the NPU")
			default:
				bar.Done(console.Duration(bar.Elapsed()) + ", " + where)
			}
		}
		return p, err
	}
	fail := func(err error) (*progEngine, DitConfig, TextConfig, error) {
		e.close()
		return nil, dc, tc, err
	}
	for _, x := range []struct {
		p    *runner.Program
		name string
	}{{&e.textP, m.Text}, {&e.time, m.Time}, {&e.fin, m.Final}} {
		if *x.p, err = open(ctx, x.name, true); err != nil {
			return fail(err)
		}
	}
	if m.Patch != "" {
		if e.patch, err = open(ctx, m.Patch, true); err != nil {
			return fail(err)
		}
	}
	e.timeRows = max(1, m.TimeRows)
	for _, name := range m.Blocks {
		p, err := open(ctx, name, true)
		if err != nil {
			return fail(err)
		}
		e.blocks = append(e.blocks, p)
	}
	// the stages are checked (and compiled into the cache) now, so that a stage that does not
	// run on the NPU fails here and not after the denoising; Decode loads them again
	for _, name := range m.Vae {
		p, err := open(ctx, name, true)
		if err != nil {
			return fail(err)
		}
		p.Close()
	}
	e.vae, e.open = m.Vae, open
	return e, dc, tc, nil
}

// openProgram loads one program of an export onto dev (with the CPU interpreter when dev is nil;
// an Intel export then runs on OpenVINO's CPU plugin) and fails unless it is all on the NPU.
func openProgram(ctx context.Context, dev *hwnpu.Device, dir, backend, name string) (runner.Program, error) {
	if name == "" {
		return nil, errors.New("anima: the manifest misses a part")
	}
	if dev == nil && backend == "intel" {
		// for checking: OpenVINO's CPU plugin through tools/ov/run_ir.py
		py := ovcpu.Python()
		if py == "" {
			return nil, errors.New("anima: an Intel export runs without the NPU on OpenVINO's CPU, which needs HWNPU_OV_PYTHON")
		}
		return &ovProgram{py: py, xml: filepath.Join(dir, name), precision: os.Getenv("HWNPU_OV_PRECISION")}, nil
	}
	p, err := runner.Open(ctx, dev, filepath.Join(dir, name))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	if dev != nil && !p.OnNPU() {
		p.Close()
		return nil, fmt.Errorf("%s does not run on the NPU alone", name)
	}
	return p, nil
}

func (e *progEngine) close() error {
	var errs []error
	for _, p := range append([]runner.Program{e.textP, e.time, e.patch, e.fin}, e.blocks...) {
		if p != nil {
			errs = append(errs, p.Close())
		}
	}
	return errors.Join(errs...)
}

// bias is the additive attention mask of n valid rows of rows: 0, then -10000.
func bias(n, rows int) []float32 {
	b := make([]float32, rows)
	for i := n; i < rows; i++ {
		b[i] = -1e4
	}
	return b
}

func (e *progEngine) text(ctx context.Context, qx, tx []float32, nq, nt int) ([]float32, error) {
	return e.textP.Run(ctx, map[string][]float32{"qx": qx, "tx": tx, "qbias": bias(nq, e.textRows), "tbias": bias(nt, e.textRows)})
}

func (e *progEngine) setCond(cond []float32) error {
	e.cond = cond
	return nil
}

func (e *progEngine) velocity(ctx context.Context, patches, tproj []float32) ([]float32, error) {
	// the step's sinusoid in row 0, zeros in the rest; its embedding is row 0 of the output
	tx := make([]float32, e.timeRows*len(tproj))
	copy(tx, tproj)
	t, err := e.time.Run(ctx, map[string][]float32{"x": tx})
	if err != nil {
		return nil, fmt.Errorf("time: %w", err)
	}
	t = t[:len(t)/e.timeRows]
	x := patches // the first group embeds them
	if e.patch != nil {
		if x, err = e.patch.Run(ctx, map[string][]float32{"x": patches}); err != nil {
			return nil, fmt.Errorf("patch: %w", err)
		}
	}
	for i, b := range e.blocks {
		if x, err = b.Run(ctx, map[string][]float32{"x": x, "t": t, "ctx": e.cond}); err != nil {
			return nil, fmt.Errorf("blocks group %d: %w", i, err)
		}
	}
	return e.fin.Run(ctx, map[string][]float32{"x": x, "t": t})
}

func (e *progEngine) hasVAE() bool { return len(e.vae) > 0 }

// decode runs the VAE's stages, each loaded while it runs; the last gives the image (1, 3, h, w).
func (e *progEngine) decode(ctx context.Context, latents []float32, stage StageFunc) (*image.RGBA, error) {
	x := latents
	for i, name := range e.vae {
		t0 := time.Now()
		s, err := e.open(ctx, name, false)
		if err != nil {
			return nil, fmt.Errorf("anima: vae stage %d: %w", i, err)
		}
		x, err = s.Run(ctx, map[string][]float32{"x": x})
		s.Close()
		if err != nil {
			return nil, fmt.Errorf("anima: vae stage %d: %w", i, err)
		}
		if stage != nil {
			stage(i, len(e.vae), time.Since(t0))
		}
	}
	W, H := e.dit.Width, e.dit.Height
	if len(x) != 3*W*H {
		return nil, fmt.Errorf("anima: vae: %d values for a %dx%d image", len(x), W, H)
	}
	img := image.NewRGBA(image.Rect(0, 0, W, H))
	for y := 0; y < H; y++ {
		for xx := 0; xx < W; xx++ {
			i := y*W + xx
			img.SetRGBA(xx, y, color.RGBA{pixel(x[i]), pixel(x[W*H+i]), pixel(x[2*W*H+i]), 255})
		}
	}
	return img, nil
}

// ovProgram runs an IR on OpenVINO's CPU plugin (ovcpu), each call a Python process: for checking
// an Intel export on a machine without Intel's NPU. precision is the plugin's (f32, or f16).
type ovProgram struct {
	py, xml, precision string
}

func (p *ovProgram) Inputs() map[string][]int    { return nil }
func (p *ovProgram) Output() []int               { return nil }
func (p *ovProgram) Metadata() map[string]string { return nil }
func (p *ovProgram) OnNPU() bool                 { return false }
func (p *ovProgram) Close() error                { return nil }

func (p *ovProgram) Run(ctx context.Context, in map[string][]float32) ([]float32, error) {
	res, err := ovcpu.Run(p.py, p.xml, in, ovcpu.Options{Precision: p.precision})
	if err != nil {
		return nil, err
	}
	y, ok := res["y"]
	if !ok {
		return nil, fmt.Errorf("anima: %s gave no y", p.xml)
	}
	return y, nil
}
