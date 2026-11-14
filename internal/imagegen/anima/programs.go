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
	"github.com/shibukawa/hwnpu/internal/types"
)

// ProgramKind is the kind of manifest.json of an export of hwnpu-image convert.
const ProgramKind = "anima-programs"

// ProgramManifest is manifest.json of an export of hwnpu-image convert: the programs (Core ML
// packages, or OpenVINO IR .xml) by part, relative to the export's directory.
type ProgramManifest struct {
	Kind    string `json:"kind"`
	Backend string `json:"backend"` // coreml, coreai or intel
	Width   int    `json:"width"`
	Height  int    `json:"height"`
	// Formats are the weight formats of the parts (text, dit), as hwnpu-dev convert's -format.
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
	// Dit is the diffusion model's file the export was made from (ConvertOptions.Dit); empty
	// in earlier exports, which were all of Anima-Turbo.
	Dit string `json:"dit,omitempty"`
}

// IsProgramExport reports whether dir holds an export of hwnpu-image convert.
func IsProgramExport(dir string) bool {
	m, err := ReadProgramManifest(dir)
	return err == nil && m.Kind == ProgramKind
}

// ReadProgramManifest reads dir/manifest.json of an export of hwnpu-image convert.
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
		return nil, fmt.Errorf("anima: %s is not an export of hwnpu-image convert", dir)
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
//
// The programs stay loaded while they fit. When one does not load for want of the memory graphs
// are loaded into (Intel's NPU: ERROR_OUT_OF_HOST_MEMORY at about 4.6 GB of Anima's fp16 on a
// Core Ultra 5 135U), others are closed until it does (get): first the text encoder, which runs
// once a prompt, then the DiT's groups the most recently used first, so that of groups that run
// in the same order every step, those that fit stay and one place takes the others in turn.
type progEngine struct {
	dit                     DitConfig
	textP, time, patch, fin *slot
	blocks                  []*slot
	// vae is the VAE's stages; each is loaded only while it runs (at 1024 x 1024 their
	// working memory is a few GB, as AMD's stages, 16.72)
	vae      []string
	open     func(ctx context.Context, name string, show bool) (runner.Program, error)
	log      io.Writer
	textRows int
	// checkpoint is the diffusion model of the export (ProgramManifest.Dit).
	checkpoint string
	// timeRows is the rows the time part takes (mil.TimeRows; 1 in earlier exports).
	timeRows int

	tick     int64 // counts the uses, for slot.used
	unloaded int   // programs closed for memory, to tell once
}

// slot is a program that may be closed for memory and loaded again.
type slot struct {
	name  string
	kind  slotKind
	p     runner.Program
	used  int64 // e.tick at its last use
	shown bool  // its loading was shown
}

type slotKind int

const (
	slotSmall slotKind = iota // the time, the patch and the final layer: kept
	slotText                  // closed first: it runs once a prompt
	slotBlock                 // a group of the DiT's blocks
	slotStage                 // a stage of the VAE, closed after it runs
)

// get returns s's program, loading it when it is not loaded. A load that fails for memory closes
// another program (victim) and tries again.
func (e *progEngine) get(ctx context.Context, s *slot) (runner.Program, error) {
	e.tick++
	s.used = e.tick
	if s.p != nil {
		return s.p, nil
	}
	for retried := false; ; retried = true {
		p, err := e.open(ctx, s.name, !s.shown)
		s.shown = true
		if err == nil {
			if retried && e.log != nil {
				fmt.Fprintf(e.log, "  %s loaded after making room\n", s.name)
			}
			s.p = p
			return p, nil
		}
		if !types.IsOutOfMemory(err) {
			return nil, err
		}
		v := e.victim(s)
		if v == nil {
			return nil, fmt.Errorf("%w (no other program loaded to make room)", err)
		}
		if e.log != nil {
			fmt.Fprintf(e.log, "  %s unloaded for %s (loaded again when it runs)\n", v.name, s.name)
		}
		e.unloaded++
		v.p.Close()
		v.p = nil
	}
}

// victim is the loaded program to close to make room for s: the text encoder, else the DiT's
// group used last, else the small part used last; nil when none is loaded.
func (e *progEngine) victim(s *slot) *slot {
	var best *slot
	rank := func(k slotKind) int {
		switch k {
		case slotText:
			return 2
		case slotBlock:
			return 1
		}
		return 0
	}
	for _, c := range e.slots() {
		if c == s || c.p == nil || c.kind == slotStage {
			continue
		}
		if best == nil || rank(c.kind) > rank(best.kind) || (rank(c.kind) == rank(best.kind) && c.used > best.used) {
			best = c
		}
	}
	return best
}

func (e *progEngine) slots() []*slot {
	out := []*slot{e.textP, e.time, e.fin}
	if e.patch != nil {
		out = append(out, e.patch)
	}
	return append(out, e.blocks...)
}

// run gets s's program and runs it.
func (e *progEngine) run(ctx context.Context, s *slot, in map[string][]float32) ([]float32, error) {
	p, err := e.get(ctx, s)
	if err != nil {
		return nil, err
	}
	return p.Run(ctx, in)
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
	e := &progEngine{dit: dc, textRows: progRows, log: log, checkpoint: m.Dit}
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
			bar = console.NewLoadBar(log, fmt.Sprintf("  [%2d/%d] %s", loaded, total, name), llm.PartBytes(filepath.Join(dir, name)))
		}
		p, err := openProgram(ctx, dev, dir, m.Backend, name)
		if bar != nil {
			switch {
			case types.IsOutOfMemory(err):
				bar.Failed("does not fit in the NPU's memory beside the others: making room")
			case err != nil:
				bar.Failed("failed")
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
	e.open = open
	e.textP = &slot{name: m.Text, kind: slotText}
	e.time = &slot{name: m.Time, kind: slotSmall}
	e.fin = &slot{name: m.Final, kind: slotSmall}
	if m.Patch != "" {
		e.patch = &slot{name: m.Patch, kind: slotSmall}
	}
	e.timeRows = max(1, m.TimeRows)
	for _, name := range m.Blocks {
		e.blocks = append(e.blocks, &slot{name: name, kind: slotBlock})
	}
	// everything is loaded now (and compiled into the cache the first time), so that a program
	// that does not run on the NPU fails here and not after the denoising; one that does not fit
	// beside the others closes some of them (get)
	for _, s := range e.slots() {
		if _, err := e.get(ctx, s); err != nil {
			return fail(err)
		}
	}
	if e.unloaded > 0 && log != nil {
		fmt.Fprintf(log, "  not all the programs fit on the NPU at once: %d were unloaded and load again as they run (an int8 export, --dit-format int8 --text-format int8, needs half the memory)\n", e.unloaded)
	}
	// the stages too; decode loads them again
	for _, name := range m.Vae {
		s := &slot{name: name, kind: slotStage}
		p, err := e.get(ctx, s)
		if err != nil {
			return fail(err)
		}
		p.Close()
	}
	e.vae = m.Vae
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
	for _, s := range e.slots() {
		if s != nil && s.p != nil {
			errs = append(errs, s.p.Close())
			s.p = nil
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
	return e.run(ctx, e.textP, map[string][]float32{"qx": qx, "tx": tx, "qbias": bias(nq, e.textRows), "tbias": bias(nt, e.textRows)})
}

func (e *progEngine) velocity(ctx context.Context, patches, tproj []float32, conds [][]float32) ([][]float32, error) {
	// the step's sinusoid in row 0, zeros in the rest; its embedding is row 0 of the output
	tx := make([]float32, e.timeRows*len(tproj))
	copy(tx, tproj)
	t, err := e.run(ctx, e.time, map[string][]float32{"x": tx})
	if err != nil {
		return nil, fmt.Errorf("time: %w", err)
	}
	t = t[:len(t)/e.timeRows]
	x := patches // the first group embeds them
	if e.patch != nil {
		if x, err = e.run(ctx, e.patch, map[string][]float32{"x": patches}); err != nil {
			return nil, fmt.Errorf("patch: %w", err)
		}
	}
	// each group runs every conditioning before the next group: a group a program that is
	// loaded again when the NPU's memory is short (Intel) loads once a step
	xs := make([][]float32, len(conds))
	for k := range xs {
		xs[k] = x
	}
	for i, b := range e.blocks {
		for k, cond := range conds {
			if xs[k], err = e.run(ctx, b, map[string][]float32{"x": xs[k], "t": t, "ctx": cond}); err != nil {
				return nil, fmt.Errorf("blocks group %d: %w", i, err)
			}
		}
	}
	for k := range xs {
		if xs[k], err = e.run(ctx, e.fin, map[string][]float32{"x": xs[k], "t": t}); err != nil {
			return nil, err
		}
	}
	return xs, nil
}

func (e *progEngine) hasVAE() bool { return len(e.vae) > 0 }

// decode runs the VAE's stages, each loaded while it runs; the last gives the image (1, 3, h, w).
func (e *progEngine) decode(ctx context.Context, latents []float32, stage StageFunc) (*image.RGBA, error) {
	x := latents
	for i, name := range e.vae {
		t0 := time.Now()
		s, err := e.get(ctx, &slot{name: name, kind: slotStage, shown: true})
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
