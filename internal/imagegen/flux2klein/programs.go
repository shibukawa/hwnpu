package flux2klein

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
	"runtime/debug"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/imagegen/anima"
	"github.com/shibukawa/hwnpu/internal/imagegen/flux2klein/mil"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// ProgramKind is the kind of manifest.json of an export of Convert.
const ProgramKind = "flux2klein-programs"

// Group is a program of the DiT's blocks: blocks First..First+Count-1.
type Group struct {
	Name  string `json:"name"`
	First int    `json:"first"`
	Count int    `json:"count"`
}

// TextProgram is a program of Qwen3's layers; Tap, when not -1, is the hidden state (of
// mil.TextTaps) its output is.
type TextProgram struct {
	Name string `json:"name"`
	Tap  int    `json:"tap"`
}

// ProgramManifest is manifest.json of an export of Convert: the shapes and the programs (Core AI
// programs) by part, relative to the export's directory.
type ProgramManifest struct {
	Kind    string `json:"kind"`
	Backend string `json:"backend"`
	Version int    `json:"version"` // ExportVersion
	// Formats are the weight formats of the parts (text, dit, vae), as hwnpu-convert's -format.
	Formats map[string]string `json:"formats"`
	Dit     Config            `json:"dit"`
	Text    TextConfig        `json:"text"`
	Vocab   int               `json:"vocab"`
	// EmbedScale multiplies Qwen3's embedding rows (EmbedFile) before the first text program
	// (1 / mil.TextScale).
	EmbedScale float32       `json:"embed_scale"`
	TextGroups []TextProgram `json:"text_groups"`
	Ctx        string        `json:"ctx"`
	Doubles    []Group       `json:"doubles"` // the first embeds the patches
	Singles    []Group       `json:"singles"` // the last ends with norm_out and proj_out
	Vae        []string      `json:"vae"`     // the VAE's stages, in order
	// BNMean and BNStd are the VAE's batch-norm statistics of the patched latents' channels.
	BNMean []float32 `json:"bn_mean"`
	BNStd  []float32 `json:"bn_std"`
	// Sizes, in an export of several sizes, are the parts of each that depend on the size
	// (functions of the same programs); Dit, Doubles, Singles and Vae are one of them.
	Sizes []SizeParts `json:"sizes,omitempty"`
}

// SizeParts are the parts of an export for pictures of Width x Height.
type SizeParts struct {
	Width   int      `json:"width"`
	Height  int      `json:"height"`
	Doubles []Group  `json:"doubles"`
	Singles []Group  `json:"singles"`
	Vae     []string `json:"vae"`
}

// SizeList is what the export holds: Sizes, else Dit's size.
func (m *ProgramManifest) SizeList() [][2]int {
	if len(m.Sizes) == 0 {
		return [][2]int{{m.Dit.Width, m.Dit.Height}}
	}
	var out [][2]int
	for _, s := range m.Sizes {
		out = append(out, [2]int{s.Width, s.Height})
	}
	return out
}

// At makes m the parts of its pictures of w x h (0 for the default), or fails when it holds none
// of that size.
func (m *ProgramManifest) At(w, h int) error {
	if w == 0 || w == m.Dit.Width && h == m.Dit.Height {
		return nil
	}
	for _, s := range m.Sizes {
		if s.Width == w && s.Height == h {
			m.Doubles, m.Singles, m.Vae = s.Doubles, s.Singles, s.Vae
			m.Dit.Width, m.Dit.Height = w, h
			return nil
		}
	}
	return fmt.Errorf("flux2klein: the export holds %v, not %dx%d", m.SizeList(), w, h)
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
		return nil, fmt.Errorf("flux2klein: %s is not an export of hwnpu-image convert", dir)
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

// progEngine runs the programs of internal/imagegen/flux2klein/mil, each a runner.Program on the
// Neural Engine. Qwen3's groups and each VAE stage are loaded only while they run (on Core AI the
// next one loading while one runs); the DiT's groups each for its turn in a step (a ring), as
// Chroma's on Core AI (docs/design/image-apple.md).
type progEngine struct {
	m    *ProgramManifest
	dir  string
	dev  *hwnpu.Device
	log  io.Writer
	logf func(string, ...any)
	// time is the time embedding's weights (TimeFile), read when a sigma's modulation rows are
	// missing (modulations) and dropped once they are made: 0.7 GB of float32 that the VAE's
	// stages need not share the memory with; mods are the rows of each sigma made so far
	time  *timeEmbed
	mods  map[float64][]float32
	ahead bool
	// checked says each of the DiT's groups of a step (by mil.RefsName of its references) was
	// loaded once (the first load compiles it) and seen on the NPU; ring loads the groups of the
	// step of ringRefs
	checked  map[string]bool
	ring     *runner.Ring
	ringRefs string
	// edit, when set, is the edit's programs at the pictures' size (UseEdit), editDir their export
	edit    *EditSize
	editDir string
	// loader, when set, loads the programs in place of runner.Open (tests)
	loader runner.OpenFunc
}

func openPrograms(dev *hwnpu.Device, dir string, w, h int, log io.Writer, logf func(string, ...any)) (*progEngine, error) {
	m, err := ReadProgramManifest(dir)
	if err != nil {
		return nil, err
	}
	if err := m.At(w, h); err != nil {
		return nil, err
	}
	if m.Version != ExportVersion {
		return nil, fmt.Errorf("%s is of layout %d, this hwnpu reads %d: convert it again", dir, m.Version, ExportVersion)
	}
	return &progEngine{m: m, dir: dir, dev: dev, log: log, logf: logf, ahead: runner.Ahead(dev)}, nil
}

// modulations makes the modulation rows of each of sigmas that are not made yet: the time
// embedding's weights are read for them (unless a test set them) and dropped after.
func (e *progEngine) modulations(sigmas []float64) error {
	if e.mods == nil {
		e.mods = map[float64][]float32{}
	}
	var todo []float64
	for _, s := range sigmas {
		if _, ok := e.mods[s]; !ok {
			todo = append(todo, s)
		}
	}
	if len(todo) == 0 {
		return nil
	}
	te := e.time
	if te == nil {
		var err error
		if te, err = readTimeFile(filepath.Join(e.dir, TimeFile), e.m.Dit.Hidden, 256); err != nil {
			return err
		}
	}
	for _, s := range todo {
		e.mods[s] = te.modulation(s)
	}
	return nil
}

// open loads one program, on the device unless it is nil, and fails unless it runs on the NPU
// alone. With a log it shows a line while it loads (the first load compiles it).
func (e *progEngine) open(ctx context.Context, name string) (runner.Program, error) {
	if name == "" {
		return nil, errors.New("the manifest misses a part")
	}
	var bar *console.Bar
	if e.log != nil {
		bar = console.NewLoadBar(e.log, "  loading "+filepath.Base(name), llm.PartBytes(e.path(name)))
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

// load is open without a line.
func (e *progEngine) load(ctx context.Context, name string) (runner.Program, error) {
	if e.loader != nil {
		return e.loader(ctx, name)
	}
	p, err := runner.Open(ctx, e.dev, e.path(name))
	if err == nil && e.dev != nil && !p.OnNPU() {
		p.Close()
		p, err = nil, fmt.Errorf("%s does not run on the NPU alone", filepath.Base(name))
	}
	return p, err
}

// path is the file of a program of the export (relative to it), or of the edit's (absolute).
func (e *progEngine) path(name string) string {
	if filepath.IsAbs(name) {
		return name
	}
	return filepath.Join(e.dir, name)
}

// encode runs Qwen3's groups in turn on every prompt's rows xs[k] (TextRows x hidden, divided by
// TextScale, the pad's rows past its ns[k] tokens), each group loaded once for all of them, then
// context_embedder of the three hidden states the DiT reads: each prompt's text rows (TextRows x
// hidden), divided by DitScale.
func (e *progEngine) encode(ctx context.Context, xs [][]float32, ns []int) ([][]float32, error) {
	t0 := time.Now()
	R, T := e.m.Text.Rows, e.m.Text.Hidden
	hs := append([][]float32(nil), xs...)
	kb := make([][]float32, len(xs))
	taps := make([][][]float32, len(xs))
	for k := range xs {
		kb[k] = make([]float32, R)
		for i := ns[k]; i < R; i++ {
			kb[k][i] = -1e4
		}
		taps[k] = make([][]float32, len(mil.TextTaps))
	}
	names := make([]string, len(e.m.TextGroups))
	for i, g := range e.m.TextGroups {
		names[i] = g.Name
	}
	err := runner.InTurn(ctx, names, e.ahead, e.open, func(i int, p runner.Program) error {
		for k := range hs {
			y, err := p.Run(ctx, map[string][]float32{"x": hs[k], "kbias": kb[k]})
			if err != nil {
				return err
			}
			hs[k] = y
			if t := e.m.TextGroups[i].Tap; t >= 0 {
				taps[k][t] = y
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	t1 := time.Now()
	p, err := e.open(ctx, e.m.Ctx)
	if err != nil {
		return nil, err
	}
	defer p.Close()
	out := make([][]float32, len(xs))
	n := len(mil.TextTaps)
	for k := range xs {
		x := make([]float32, R*n*T)
		for r := 0; r < R; r++ {
			for t := 0; t < n; t++ {
				copy(x[(r*n+t)*T:(r*n+t+1)*T], taps[k][t][r*T:(r+1)*T])
			}
		}
		if out[k], err = p.Run(ctx, map[string][]float32{"x": x}); err != nil {
			return nil, fmt.Errorf("%s: %w", e.m.Ctx, err)
		}
	}
	e.logf("the text in %.2f s: Qwen3-4B %.2f s, the projection %.2f s", time.Since(t0).Seconds(), t1.Sub(t0).Seconds(), time.Since(t1).Seconds())
	return out, nil
}

// step is the DiT's groups of a step with references of the picture's sides over divs: the
// export's for none, else the edit's (UseEdit), their names absolute.
func (e *progEngine) step(divs []int) (doubles, singles []Group, err error) {
	if len(divs) == 0 {
		return e.m.Doubles, e.m.Singles, nil
	}
	if e.edit == nil {
		return nil, nil, errNoEdit
	}
	st, ok := e.edit.Step(divs)
	if !ok {
		return nil, nil, fmt.Errorf("the edit's programs have no step of %s at %dx%d", mil.RefsName(divs), e.m.Dit.Width, e.m.Dit.Height)
	}
	abs := func(gs []Group) []Group {
		out := make([]Group, len(gs))
		for i, g := range gs {
			out[i] = Group{filepath.Join(e.editDir, g.Name), g.First, g.Count}
		}
		return out
	}
	return abs(st.Doubles), abs(st.Singles), nil
}

// errNoEdit is the error of references without the edit's programs.
var errNoEdit = errors.New("editing (references) needs the edit's programs (PrepareEdit, UseEdit)")

// groups are the DiT's programs of a step in the order it runs them: the doubles, then the
// singles.
func groups(doubles, singles []Group) []string {
	var names []string
	for _, g := range doubles {
		names = append(names, g.Name)
	}
	for _, g := range singles {
		names = append(names, g.Name)
	}
	return names
}

// openDiT loads each of the DiT's groups of a step with references divs once (the first load
// compiles it) and makes the ring that loads them for each step; another step's ring is closed.
func (e *progEngine) openDiT(ctx context.Context, divs []int) error {
	key := mil.RefsName(divs)
	if e.ring != nil && e.ringRefs == key {
		return nil
	}
	if err := e.closeDiT(); err != nil {
		return err
	}
	ds, ss, err := e.step(divs)
	if err != nil {
		return err
	}
	names := groups(ds, ss)
	if e.checked == nil {
		e.checked = map[string]bool{}
	}
	if !e.checked[key] {
		for _, name := range names {
			p, err := e.open(ctx, name)
			if err != nil {
				return err
			}
			p.Close()
		}
		e.checked[key] = true
	}
	e.ring, e.ringRefs = runner.NewRing(names, e.ahead, e.load), key
	return nil
}

func (e *progEngine) closeDiT() error {
	if e.ring == nil {
		return nil
	}
	err := e.ring.Close()
	e.ring = nil
	return err
}

// velocities runs the DiT at sigma on each picture's latents xs[k] (tokens x 128) under its text
// rows rows[k], with the references' latents refs (their rows one after another, the same for
// every picture) of the picture's sides over divs: each group of blocks runs every picture before
// the next group (a group that follows another pays for that in its first run alone,
// docs/design/image-models.md). The ring is the step's (openDiT of divs).
func (e *progEngine) velocities(ctx context.Context, rows, xs [][]float32, refs []float32, divs []int, sigma float64) ([][]float32, error) {
	E := e.m.Dit.Hidden
	if err := e.modulations([]float64{sigma}); err != nil {
		return nil, err
	}
	ds, ss, err := e.step(divs)
	if err != nil {
		return nil, err
	}
	mods := e.mods[sigma]
	m := func(lo, hi int) []float32 { return mods[lo*E : hi*E] }
	hs := make([][]float32, len(xs))
	for i, name := range groups(ds, ss) {
		p, err := e.ring.Take(ctx, i)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		for k := range xs {
			var in map[string][]float32
			switch {
			case i == 0:
				x := xs[k]
				if len(refs) > 0 { // the picture's patches, then the references'
					x = append(append(make([]float32, 0, len(x)+len(refs)), x...), refs...)
				}
				in = map[string][]float32{"x": x, "ctx": rows[k], "mods": m(mil.ModImage, mil.ModSingle)}
			case i < len(ds):
				in = map[string][]float32{"x": hs[k], "mods": m(mil.ModImage, mil.ModSingle)}
			case i == len(ds)+len(ss)-1:
				in = map[string][]float32{"x": hs[k], "mods": m(mil.ModSingle, mil.ModRows)}
			default:
				in = map[string][]float32{"x": hs[k], "mods": m(mil.ModSingle, mil.ModOut)}
			}
			if hs[k], err = p.Run(ctx, in); err != nil {
				p.Close()
				return nil, fmt.Errorf("%s: %w", filepath.Base(name), err)
			}
		}
		p.Close()
	}
	return hs, nil
}

// encodeReference runs the VAE's encoder of a reference of the picture's sides over div (the
// edit's programs, a stage at a time) on its pixels (3, h, w) in [-1, 1]: the latents' means
// (32, h/8, w/8).
func (e *progEngine) encodeReference(ctx context.Context, px []float32, div int) ([]float32, error) {
	if e.edit == nil {
		return nil, errNoEdit
	}
	stages := e.edit.Encoders[div]
	if len(stages) == 0 {
		return nil, fmt.Errorf("the edit's programs have no encoder of a reference of the sides over %d at %dx%d", div, e.m.Dit.Width, e.m.Dit.Height)
	}
	names := make([]string, len(stages))
	for i, s := range stages {
		names[i] = filepath.Join(e.editDir, s)
	}
	t0 := time.Now()
	zs, err := runner.Stages(ctx, names, e.ahead, e.open, [][]float32{px}, runner.StageBudget,
		func(_ int, p runner.Program, x []float32) ([]float32, error) {
			return p.Run(ctx, map[string][]float32{"x": x})
		}, nil)
	if err != nil {
		return nil, fmt.Errorf("the encoder: %w", err)
	}
	e.logf("the reference's latents in %.2f s", time.Since(t0).Seconds())
	return zs[0], nil
}

// decodeAll runs the VAE's stages on each picture's latents (32, H/8, W/8), stage by stage
// (runner.Stages): a stage is loaded once (on Core AI while the stage before it runs) and runs
// every picture before the next one; the last gives the pictures (1, 3, h, w).
func (e *progEngine) decodeAll(ctx context.Context, zs [][]float32, stage anima.StageFunc) ([]*image.RGBA, error) {
	xs, err := runner.Stages(ctx, e.m.Vae, e.ahead, e.open, zs, runner.StageBudget,
		func(_ int, p runner.Program, x []float32) ([]float32, error) {
			y, err := p.Run(ctx, map[string][]float32{"x": x})
			// the stage before's input and output (up to 0.5 GB each of float32 at 1024 x 1024)
			// back to the system: the peak 5.1 -> 4.1 GB, 0.4 s more of decoding
			debug.FreeOSMemory()
			return y, err
		},
		func(i, n int, d time.Duration) {
			if stage != nil {
				stage(i, n, d)
			}
		})
	if err != nil {
		return nil, fmt.Errorf("vae: %w", err)
	}
	W, H := e.m.Dit.Width, e.m.Dit.Height
	imgs := make([]*image.RGBA, len(xs))
	for k, x := range xs {
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
		imgs[k] = img
	}
	return imgs, nil
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

// openProgramPipeline reads an export of Convert in dir for pictures of w x h (0: its default).
func openProgramPipeline(dev *hwnpu.Device, dir string, w, h int, log io.Writer) (*Pipeline, error) {
	p := &Pipeline{dev: dev, dir: dir}
	fail := func(err error) (*Pipeline, error) {
		p.Close()
		return nil, fmt.Errorf("flux2klein: %w", err)
	}
	e, err := openPrograms(dev, dir, w, h, log, p.logf)
	if err != nil {
		return fail(err)
	}
	p.prog = e
	p.C, p.Text = e.m.Dit, e.m.Text
	p.Model = Model{Name: "FLUX.2-klein-4B", Steps: 4, Guidance: 1}
	if b, err := os.ReadFile(filepath.Join(dir, "model.json")); err == nil {
		if err := json.Unmarshal(b, &p.Model); err != nil {
			return fail(err)
		}
	}
	if p.tok, err = tokenizer.Load(filepath.Join(dir, TokenizerFile)); err != nil {
		return fail(err)
	}
	if p.embed, err = os.Open(filepath.Join(dir, EmbedFile)); err != nil {
		return fail(err)
	}
	if len(e.m.BNMean) != p.C.Latent || len(e.m.BNStd) != p.C.Latent {
		return fail(fmt.Errorf("the manifest has %d and %d batch-norm statistics for %d channels", len(e.m.BNMean), len(e.m.BNStd), p.C.Latent))
	}
	p.bnMean, p.bnStd = e.m.BNMean, e.m.BNStd
	return p, nil
}

// embedRows are Qwen3's embedding rows of ids, the pad's past them to the text's rows, times the
// export's EmbedScale.
func (p *Pipeline) embedRows(ids []int32) ([]float32, error) {
	R, E := p.Text.Rows, p.Text.Hidden
	s := p.prog.m.EmbedScale
	b := make([]byte, 2*E)
	x := make([]float32, R*E)
	for i := 0; i < R; i++ {
		id := int32(PadID)
		if i < len(ids) {
			id = ids[i]
		}
		if _, err := p.embed.ReadAt(b, int64(id)*int64(2*E)); err != nil {
			return nil, err
		}
		for j, v := range fromBF16s(b) {
			x[i*E+j] = v * s
		}
	}
	return x, nil
}

// encodePrograms is EncodeAll on Apple's Neural Engine.
func (p *Pipeline) encodePrograms(ctx context.Context, prompts []string) ([]Cond, error) {
	xs := make([][]float32, len(prompts))
	ns := make([]int, len(prompts))
	for k, pr := range prompts {
		ids := p.Tokens(pr)
		x, err := p.embedRows(ids)
		if err != nil {
			return nil, fmt.Errorf("flux2klein: %w", err)
		}
		xs[k], ns[k] = x, len(ids)
		p.logf("text: %d tokens", len(ids))
	}
	rows, err := p.prog.encode(ctx, xs, ns)
	if err != nil {
		return nil, fmt.Errorf("flux2klein: %w", err)
	}
	conds := make([]Cond, len(prompts))
	for k := range prompts {
		conds[k] = Cond{rows: rows[k], Tokens: ns[k]}
	}
	if p.Dump != "" && len(rows) > 0 {
		p.dump("ctx.bf16", toBF16(scaled(rows[0], mil.DitScale)))
	}
	return conds, nil
}

// scaled is xs times s.
func scaled(xs []float32, s float32) []float32 {
	out := make([]float32, len(xs))
	for i, x := range xs {
		out[i] = x * s
	}
	return out
}

// generatePrograms is GenerateAll on Apple's Neural Engine: the pictures' steps together, each
// group of the DiT's blocks running every picture before the next group loads. With references
// (their latents of EncodeImage) the step is the edit's programs' (UseEdit).
func (p *Pipeline) generatePrograms(ctx context.Context, conds []Cond, refs, noises [][]float32, sigmas []float64, step StepFunc) ([][]float32, error) {
	divs, err := p.refDivs(refs)
	if err != nil {
		return nil, err
	}
	var refRows []float32
	for _, r := range refs {
		refRows = append(refRows, r...)
	}
	rows := make([][]float32, len(conds))
	xs := make([][]float32, len(conds))
	for k, cd := range conds {
		if cd.rows == nil {
			return nil, errors.New("flux2klein: the DiT runs without a prompt (Encode)")
		}
		rows[k], xs[k] = cd.rows, append([]float32(nil), noises[k]...)
	}
	e := p.prog
	if err := e.modulations(sigmas[:len(sigmas)-1]); err != nil {
		return nil, fmt.Errorf("flux2klein: %w", err)
	}
	if err := e.openDiT(ctx, divs); err != nil {
		return nil, fmt.Errorf("flux2klein: %w", err)
	}
	defer e.closeDiT()
	n := len(sigmas) - 1
	for i := 0; i < n; i++ {
		t0 := time.Now()
		vs, err := e.velocities(ctx, rows, xs, refRows, divs, sigmas[i])
		if err != nil {
			return nil, fmt.Errorf("flux2klein: step %d: %w", i, err)
		}
		dt := float32(sigmas[i+1] - sigmas[i])
		for k, v := range vs {
			if len(v) != len(xs[k]) {
				return nil, fmt.Errorf("flux2klein: step %d: %d values of velocity for latents of %d", i, len(v), len(xs[k]))
			}
			if k == 0 {
				p.dump(fmt.Sprintf("v_%d.bf16", i), toBF16(v))
			}
			for j := range v {
				xs[k][j] += dt * v[j]
			}
		}
		if step != nil {
			step(i, n, time.Since(t0))
		}
	}
	return xs, nil
}

// DecodeAll is Decode of several pictures' latents: on Apple's Neural Engine each VAE stage over
// all of them; on AMD's NPU one picture after another (a stage hands its output to the next on
// the device, and opening a stage is 0.5 to 0.8 s of a picture's 2 to 6 s of decoding).
func (p *Pipeline) DecodeAll(ctx context.Context, xs ...[]float32) ([]*image.RGBA, error) {
	if p.prog == nil {
		out := make([]*image.RGBA, len(xs))
		for k, x := range xs {
			// the garbage of the steps of every picture, and of the picture decoded before (its
			// last stage's output, 128 MB at 1024 x 1024), back to the system before the VAE's
			// peak
			debug.FreeOSMemory()
			img, err := p.Decode(ctx, x)
			if err != nil {
				return nil, err
			}
			out[k] = img
		}
		return out, nil
	}
	zs := make([][]float32, len(xs))
	for k, x := range xs {
		zs[k] = p.Unpack(x)
	}
	// the steps' garbage (each group's output, 57 MB at 1024 x 1024) back to the system before
	// the VAE's stages, the largest of a picture's memory
	debug.FreeOSMemory()
	return p.prog.decodeAll(ctx, zs, p.Stage)
}

// UseEdit has an Apple pipeline draw from references too, with the edit's programs in dir (an
// export of ConvertEdit of the same model, PrepareEdit) at the pictures' size.
func (p *Pipeline) UseEdit(dir string) error {
	if p.prog == nil {
		return errors.New("flux2klein: the edit's programs are for Apple's Neural Engine (an IRON export edits by itself)")
	}
	m, err := ReadEditManifest(dir)
	if err != nil {
		return err
	}
	c := p.prog.m.Dit
	if m.Hidden != c.Hidden || m.Latent != c.Latent {
		return fmt.Errorf("flux2klein: %s is of a DiT of %d and patches of %d, the export's of %d and %d", dir, m.Hidden, m.Latent, c.Hidden, c.Latent)
	}
	s, ok := m.At(c.Width, c.Height)
	if !ok {
		var have []string
		for _, s := range m.Sizes {
			have = append(have, fmt.Sprintf("%dx%d", s.Width, s.Height))
		}
		return fmt.Errorf("flux2klein: %s edits at %v, not %dx%d", filepath.Base(dir), have, c.Width, c.Height)
	}
	if err := p.prog.closeDiT(); err != nil {
		return err
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		return err
	}
	p.prog.edit, p.prog.editDir = &s, abs
	return nil
}
