package chroma

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
	"github.com/shibukawa/hwnpu/internal/imagegen/anima"
	"github.com/shibukawa/hwnpu/internal/imagegen/chroma/mil"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
)

// ProgramKind is the kind of manifest.json of an export of Convert.
const ProgramKind = "chroma-programs"

// Group is a program of the DiT's blocks: blocks First..First+Count-1.
type Group struct {
	Name  string `json:"name"`
	First int    `json:"first"`
	Count int    `json:"count"`
}

// ProgramManifest is manifest.json of an export of Convert: the shapes and the programs (Core ML
// packages) by part, relative to the export's directory.
type ProgramManifest struct {
	Kind    string `json:"kind"`
	Backend string `json:"backend"`
	Version int    `json:"version"` // ExportVersion
	// Formats are the weight formats of the parts (t5, dit, vae), as hwnpu-convert's -format.
	Formats map[string]string `json:"formats"`
	Dit     Config            `json:"dit"`
	T5      struct {
		Hidden int `json:"hidden"`
		Layers int `json:"layers"`
		Vocab  int `json:"vocab"`
	} `json:"t5"`
	// EmbedScale multiplies T5's embedding rows (text/embed.bin) before the first text program
	// (1 / mil.T5Scale).
	EmbedScale float32 `json:"embed_scale"`
	// Text are T5's groups of layers, in order; the last ends with the final norm and the DiT's
	// context_embedder.
	Text    []string `json:"text"`
	Mods    string   `json:"mods"`
	Doubles []Group  `json:"doubles"` // the first embeds the patches
	Singles []Group  `json:"singles"` // the last ends with norm_out and proj_out
	Vae     []string `json:"vae"`     // the VAE's stages, in order
}

// ExportFormat is the program format of the export dir (its manifest's backend), "" for none.
func ExportFormat(dir string) string {
	m, err := ReadProgramManifest(dir)
	if err != nil {
		return ""
	}
	return m.Backend
}

// IsProgramExport reports whether dir holds an export of Convert.
func IsProgramExport(dir string) bool {
	_, err := ReadProgramManifest(dir)
	return err == nil
}

// ReadProgramManifest reads dir/manifest.json of an export of Convert.
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
		return nil, fmt.Errorf("chroma: %s is not an export of hwnpu-image convert", dir)
	}
	return &m, nil
}

func writeProgramManifest(dir string, m *ProgramManifest) error {
	m.Kind = ProgramKind
	b, err := json.MarshalIndent(m, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "manifest.json"), append(b, '\n'), 0o644)
}

// progEngine runs the programs of internal/imagegen/chroma/mil, each a runner.Program on the
// Neural Engine (or the CPU interpreter). T5's groups, the approximator and each VAE stage are
// loaded only while they run; the DiT's groups from the first step to the last, or on Core AI each
// for its turn in a call (swapDiT).
type progEngine struct {
	m    *ProgramManifest
	dir  string
	dev  *hwnpu.Device
	log  io.Writer
	logf func(string, ...any)
	// mods are the approximator's rows of each sigma the steps run at (openDiT)
	mods map[float64][]float32
	// empty is the empty prompt's text rows (EmptyFile, 2 x hidden), divided by mil.DitScale;
	// nil in an export without it
	empty   []float32
	doubles []runner.Program
	singles []runner.Program
	// swapDiT loads each group of the DiT's blocks for each of its calls, runs it and closes it,
	// rather than holding the ten of them (8.3 GiB at lut8): Core AI loads a group from its cache
	// in a fraction of a second, and holding them all is no faster, since a group that follows
	// another pays for that either way (an M1 at 512 x 512: a picture in 1 m 25 to 30 s with 2.2
	// to 2.6 GiB on the Neural Engine, against 1 m 36 s with 8.3; design.md 16.210). checked says
	// openDiT loaded each once (the first load compiles it) and saw it on the NPU; ring gives the
	// groups of a call, the doubles then the singles, with ahead the next one loading while one
	// runs (without it a call takes 7.2 s in place of 5.4 to 5.8: a group is 0.7 to 0.9 GB).
	// ahead: T5's groups and the VAE's stages load one ahead as well (runner.Ahead).
	swapDiT, checked, ahead bool
	ring                    *runner.Ring
}

func openPrograms(dev *hwnpu.Device, dir string, log io.Writer, c *Config, model *Model, logf func(string, ...any)) (*progEngine, error) {
	m, err := ReadProgramManifest(dir)
	if err != nil {
		return nil, err
	}
	if m.Version != ExportVersion {
		return nil, fmt.Errorf("%s is of layout %d, this hwnpu reads %d: convert it again", dir, m.Version, ExportVersion)
	}
	*c = m.Dit
	b, err := os.ReadFile(filepath.Join(dir, "model.json"))
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(b, model); err != nil {
		return nil, err
	}
	e := &progEngine{m: m, dir: dir, dev: dev, log: log, logf: logf, swapDiT: runner.Swap(dev), ahead: runner.Ahead(dev)}
	if rows, err := readEmpty(dir, m.Dit.Hidden); err == nil {
		e.empty = make([]float32, m.Dit.TextRows*m.Dit.Hidden)
		for i, v := range rows {
			e.empty[i] = v / mil.DitScale
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	return e, nil
}

func (e *progEngine) textDir() string { return filepath.Join(e.dir, "text") }

// open loads one program, on the device unless it is nil, and fails unless it runs on the NPU
// alone. With a log it shows a line while it loads (the first load compiles it).
func (e *progEngine) open(ctx context.Context, name string) (runner.Program, error) {
	if name == "" {
		return nil, errors.New("the manifest misses a part")
	}
	var bar *console.Bar
	if e.log != nil {
		bar = console.NewLoadBar(e.log, "  loading "+name, llm.PartBytes(filepath.Join(e.dir, name)))
	}
	p, err := e.load(ctx, name)
	if bar != nil {
		switch {
		case err != nil:
			bar.Failed("failed")
		case e.dev != nil:
			bar.Done(console.Duration(bar.Elapsed()) + ", all on the NPU")
		default:
			bar.Done(console.Duration(bar.Elapsed()))
		}
	}
	if err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	return p, nil
}

// load is open without a line: the program on the device unless it is nil, on the NPU alone.
func (e *progEngine) load(ctx context.Context, name string) (runner.Program, error) {
	p, err := runner.Open(ctx, e.dev, filepath.Join(e.dir, name))
	if err == nil && e.dev != nil && !p.OnNPU() {
		p.Close()
		p, err = nil, fmt.Errorf("%s does not run on the NPU alone", name)
	}
	return p, err
}

// maskBias is the additive mask of n valid rows of rows: 0, then -10000.
func maskBias(n, rows int) []float32 {
	b := make([]float32, rows)
	for i := n; i < rows; i++ {
		b[i] = -1e4
	}
	return b
}

// encode runs T5's groups in turn, each loaded once for every prompt: at most one group's weights
// are held, on Core AI two (the next group loads while this one runs, runner.InTurn). The empty
// prompt (one token, </s>) takes the export's rows (EmptyFile) without T5.
func (e *progEngine) encode(ctx context.Context, xs [][]float32, ns []int) ([]Cond, error) {
	t0 := time.Now()
	R := e.m.Dit.TextRows
	masks := make([][]float32, len(xs))
	hs := append([][]float32(nil), xs...)
	var todo []int
	for k := range xs {
		masks[k] = maskBias(ns[k], R)
		if ns[k] == 1 && e.empty != nil {
			hs[k] = e.empty
		} else {
			todo = append(todo, k)
		}
	}
	if len(todo) > 0 {
		err := runner.InTurn(ctx, e.m.Text, e.ahead, e.open, func(_ int, p runner.Program) error {
			for _, k := range todo {
				y, err := p.Run(ctx, map[string][]float32{"x": hs[k], "mask": masks[k]})
				if err != nil {
					return err
				}
				hs[k] = y
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	conds := make([]Cond, len(xs))
	for k := range hs {
		conds[k] = Cond{rows: hs[k]}
	}
	if len(todo) > 0 {
		e.logf("T5-XXL %.2f s", time.Since(t0).Seconds())
	}
	return conds, nil
}

// openDiT runs the approximator once for each sigma the steps take (its rows depend on the
// timestep alone; Heun's steps share theirs with the next step's start, and CFG's two passes
// theirs), closes it and loads the DiT's groups: all of them held, or with swapDiT each in turn
// by the calls (group).
func (e *progEngine) openDiT(ctx context.Context, sigmas []float64) error {
	c := e.m.Dit
	if e.mods == nil {
		e.mods = map[float64][]float32{}
	}
	var todo []float64
	for _, s := range sigmas {
		if _, ok := e.mods[s]; !ok {
			e.mods[s] = nil
			todo = append(todo, s)
		}
	}
	if len(todo) > 0 {
		p, err := e.open(ctx, e.m.Mods)
		if err != nil {
			return err
		}
		for _, s := range todo {
			y, err := p.Run(ctx, map[string][]float32{"x": ModInput(1000*s, c.Mods, c.Mods, c.ApproxIn)})
			if err != nil {
				p.Close()
				return fmt.Errorf("%s: %w", e.m.Mods, err)
			}
			e.mods[s] = y
		}
		p.Close()
	}
	if e.swapDiT {
		if e.ring != nil {
			return nil
		}
		names := e.groups()
		if !e.checked { // each loaded once here (the first load compiles it), then for each call
			for _, name := range names {
				p, err := e.open(ctx, name)
				if err != nil {
					return err
				}
				p.Close()
			}
			e.checked = true
		}
		e.ring = runner.NewRing(names, e.ahead, e.load)
		return nil
	}
	if e.doubles != nil {
		return nil
	}
	for _, gs := range []struct {
		groups []Group
		to     *[]runner.Program
	}{{e.m.Doubles, &e.doubles}, {e.m.Singles, &e.singles}} {
		for _, g := range gs.groups {
			p, err := e.open(ctx, g.Name)
			if err != nil {
				e.closeDiT()
				return err
			}
			*gs.to = append(*gs.to, p)
		}
	}
	return nil
}

func (e *progEngine) closeDiT() error {
	var errs []error
	if e.ring != nil { // the group loading for a call that did not come
		errs = append(errs, e.ring.Close())
		e.ring = nil
	}
	for _, p := range append(append([]runner.Program(nil), e.doubles...), e.singles...) {
		errs = append(errs, p.Close())
	}
	e.doubles, e.singles = nil, nil
	return errors.Join(errs...)
}

// groups are the DiT's programs in the order a call runs them: the doubles, then the singles.
func (e *progEngine) groups() []string {
	var names []string
	for _, g := range e.m.Doubles {
		names = append(names, g.Name)
	}
	for _, g := range e.m.Singles {
		names = append(names, g.Name)
	}
	return names
}

// group gives the DiT's program i of groups for a call to run, and what to do with it after: with
// swapDiT it is loaded for this call (or was loading since the group before it ran) and release
// closes it; else it is one of those held.
func (e *progEngine) group(ctx context.Context, i int) (p runner.Program, release func(), err error) {
	if e.swapDiT {
		if e.ring == nil {
			return nil, nil, errors.New("the DiT is not open (openDiT)")
		}
		if p, err = e.ring.Take(ctx, i); err != nil {
			return nil, nil, err
		}
		return p, func() { p.Close() }, nil
	}
	if i < len(e.doubles) {
		return e.doubles[i], func() {}, nil
	}
	return e.singles[i-len(e.doubles)], func() {}, nil
}

func (e *progEngine) close() error { return e.closeDiT() }

func (e *progEngine) velocity(ctx context.Context, conds []Cond, x []float32, sigma float64) ([][]float32, error) {
	c := e.m.Dit
	E, N, R := c.Hidden, c.Tokens(), c.Rows()
	mods, ok := e.mods[sigma]
	if !ok || mods == nil {
		return nil, fmt.Errorf("chroma: no modulation rows of sigma %g (openDiT)", sigma)
	}
	rows := func(rs ...int) []float32 {
		out := make([]float32, 0, len(rs)*E)
		for _, r := range rs {
			out = append(out, mods[r*E:(r+1)*E]...)
		}
		return out
	}
	kb := make([][]float32, len(conds))
	hs := make([][]float32, len(conds))
	for k, cd := range conds {
		kb[k] = maskBias(R, R)
		for i := N + cd.Keys; i < R; i++ {
			kb[k][i] = -1e4
		}
	}
	// each group runs every conditioning before the next group (design.md 16.148, 16.160): a
	// group the Neural Engine has to load again loads once a call of the DiT
	for gi, g := range e.m.Doubles {
		var rs []int
		for b := g.First; b < g.First+g.Count; b++ {
			for _, text := range []int{0, 1} {
				r := 3*c.Singles + 6*b + text*6*c.Doubles
				rs = append(rs, r, r+1, r+2, r+3, r+4, r+5)
			}
		}
		m := rows(rs...)
		b, release, err := e.group(ctx, gi)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", g.Name, err)
		}
		for k, cd := range conds {
			in := map[string][]float32{"mods": m, "kbias": kb[k]}
			if gi == 0 {
				in["x"], in["ctx"] = x, cd.rows
			} else {
				in["x"] = hs[k]
			}
			if hs[k], err = b.Run(ctx, in); err != nil {
				release()
				return nil, fmt.Errorf("%s: %w", g.Name, err)
			}
		}
		release()
	}
	for gi, g := range e.m.Singles {
		var rs []int
		for b := g.First; b < g.First+g.Count; b++ {
			rs = append(rs, 3*b, 3*b+1, 3*b+2)
		}
		if gi == len(e.m.Singles)-1 {
			rs = append(rs, c.Mods-2, c.Mods-1)
		}
		m := rows(rs...)
		b, release, err := e.group(ctx, len(e.m.Doubles)+gi)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", g.Name, err)
		}
		for k := range conds {
			if hs[k], err = b.Run(ctx, map[string][]float32{"x": hs[k], "mods": m, "kbias": kb[k]}); err != nil {
				release()
				return nil, fmt.Errorf("%s: %w", g.Name, err)
			}
		}
		release()
	}
	return hs, nil
}

// decode runs the VAE's stages, each loaded while it runs (on Core AI while the stage before it
// runs); the last gives the picture (1, 3, h, w).
func (e *progEngine) decode(ctx context.Context, z []float32, stage anima.StageFunc) (*image.RGBA, error) {
	x := z
	t0 := time.Now()
	err := runner.InTurn(ctx, e.m.Vae, e.ahead, e.open, func(i int, p runner.Program) error {
		y, err := p.Run(ctx, map[string][]float32{"x": x})
		if err != nil {
			return err
		}
		x = y
		if stage != nil {
			stage(i, len(e.m.Vae), time.Since(t0))
		}
		t0 = time.Now()
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("vae: %w", err)
	}
	W, H := e.m.Dit.Width, e.m.Dit.Height
	if len(x) != 3*W*H {
		return nil, fmt.Errorf("vae: %d values for a %dx%d picture", len(x), W, H)
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

// pixel is a value in [-1, 1] as a byte.
func pixel(v float32) uint8 {
	f := (v + 1) * 127.5
	switch {
	case f <= 0:
		return 0
	case f >= 255:
		return 255
	}
	return uint8(f + 0.5)
}
