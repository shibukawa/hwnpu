package flux3klein

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/imagegen/flux3klein/mil"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// ProgramKind is the kind of manifest.json of an export of Convert.
const ProgramKind = "flux3klein-programs"

// Template is the chat template the pipeline puts the prompt in: Qwen3-VL's, one user message and
// the assistant's turn opened (apply_chat_template with add_generation_prompt).
const Template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

// PadID is the tokenizer's pad (<|endoftext|>): the text's rows past the prompt.
const PadID = 151643

// Steps, Shift: the distilled checkpoint's 16 flow-map steps, its sigmas linspace(1, 1/16) shifted
// by 6.93 (Flux3SetTimestepsStep).
const (
	Steps = 16
	Shift = 6.93
)

// Group is a program of the DiT's single blocks: blocks First..First+Count-1.
type Group struct {
	Name  string `json:"name"`
	First int    `json:"first"`
	Count int    `json:"count"`
}

// TextProgram is a program of Qwen3-VL's layers; Tap, when not -1, is the hidden state (of
// mil.TextTaps) its output is.
type TextProgram struct {
	Name string `json:"name"`
	Tap  int    `json:"tap"`
}

// Config is the DiT's shape an export holds.
type Config struct {
	Hidden      int `json:"hidden"`
	HeadDim     int `json:"head_dim"`
	Latent      int `json:"latent"`
	Context     int `json:"context"`
	TimeIn      int `json:"time_in"`
	Modes       int `json:"modes"`
	Singles     int `json:"singles"`
	SingleGroup int `json:"single_group"`
}

// TextConfig is Qwen3-VL's.
type TextConfig struct {
	Hidden  int     `json:"hidden"`
	HeadDim int     `json:"head_dim"`
	Theta   float64 `json:"theta"`
	Vocab   int     `json:"vocab"`
}

// Variant is the programs of pictures of Width x Height under text of TextRows rows (names relative
// to the export, NAME#FUNCTION for a function of a program of several).
type Variant struct {
	Width      int           `json:"width"`
	Height     int           `json:"height"`
	TextRows   int           `json:"text_rows"`
	TextGroups []TextProgram `json:"text_groups"`
	Ctx        string        `json:"ctx"`
	TextModes  string        `json:"text_modes"`
	ImageModes string        `json:"image_modes"` // embeds the patches
	Singles    []Group       `json:"singles"`     // the last ends with norm_out and proj_out
	Vae        []string      `json:"vae"`
}

// ProgramManifest is manifest.json of an export of Convert.
type ProgramManifest struct {
	Kind    string            `json:"kind"`
	Backend string            `json:"backend"`
	Version int               `json:"version"`
	Formats map[string]string `json:"formats"`
	// FittedTables says the DiT's lut8 tables were fitted to the activations (ConvertOptions.Stats).
	FittedTables bool       `json:"fitted_tables"`
	Dit          Config     `json:"dit"`
	Text         TextConfig `json:"text"`
	// EmbedScale multiplies Qwen3-VL's embedding rows before the first text program.
	EmbedScale float32   `json:"embed_scale"`
	Variants   []Variant `json:"variants"`
	BNMean     []float32 `json:"bn_mean"`
	BNStd      []float32 `json:"bn_std"`
}

// Sizes are the picture sizes m holds.
func (m *ProgramManifest) Sizes() [][2]int {
	var out [][2]int
	seen := map[[2]int]bool{}
	for _, v := range m.Variants {
		s := [2]int{v.Width, v.Height}
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	return out
}

// variant is m's programs for pictures of w x h under text of n tokens: the shortest text that
// holds them.
func (m *ProgramManifest) variant(w, h, n int) (*Variant, error) {
	var best *Variant
	most := 0
	for i := range m.Variants {
		v := &m.Variants[i]
		if v.Width != w || v.Height != h {
			continue
		}
		most = max(most, v.TextRows)
		if v.TextRows >= n && (best == nil || v.TextRows < best.TextRows) {
			best = v
		}
	}
	switch {
	case most == 0:
		return nil, fmt.Errorf("the export holds %v, not %dx%d", m.Sizes(), w, h)
	case best == nil:
		return nil, fmt.Errorf("the prompt is %d tokens in the chat template; the export takes up to %d (convert it with longer text)", n, most)
	}
	return best, nil
}

// IsExport reports whether dir holds an export of Convert or of tools/amd/iron_flux3klein.py.
func IsExport(dir string) bool {
	_, err := ReadProgramManifest(dir)
	return err == nil || IsIronExport(dir)
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
		return nil, fmt.Errorf("flux3klein: %s is not an export of FLUX.3 klein", dir)
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

// StepFunc reports step i of n done, in d.
type StepFunc func(i, n int, d time.Duration)

// Pipeline is an export for pictures of one size: of Convert on Apple's Neural Engine, or of
// tools/amd/iron_flux3klein.py on AMD's NPU.
type Pipeline struct {
	W, H int
	m    *ProgramManifest
	dir  string
	dev  *hwnpu.Device
	log  io.Writer
	tok  *tokenizer.Tokenizer
	emb  *os.File
	// Logf, when set, takes the pipeline's lines (the text's and the steps' times).
	Logf func(string, ...any)
	// Dump, when set, is a directory the text's rows for the DiT (ctx.f32, before context_embedder:
	// the eight hidden states side by side) and each step's velocity (v_I.f32) are written into,
	// to check against tools/ref/ref_flux3klein.py.
	Dump  string
	ahead bool
	// loader, when set, loads the programs in place of runner.Open (tests)
	loader runner.OpenFunc
	// checked are the programs loaded once (the first load compiles them) and seen on the NPU
	checked map[string]bool
	// iron runs an IRON export (AMD's NPU); nil for one of Convert
	iron *ironPipe
	// mods are the modulation rows of each step (sigma, next) made so far (modulations)
	modsMu sync.Mutex
	mods   map[[2]float64][]float32
}

// Open reads the export in dir for pictures of w x h, running its programs on dev: one of Convert
// (Apple's Neural Engine) or of tools/amd/iron_flux3klein.py (AMD's NPU).
func Open(ctx context.Context, dev *hwnpu.Device, dir string, w, h int, log io.Writer) (*Pipeline, error) {
	if IsIronExport(dir) {
		p := &Pipeline{W: w, H: h, dir: dir, dev: dev, log: log, checked: map[string]bool{}}
		if err := p.openIron(ctx, dev, dir, w, h); err != nil {
			p.Close()
			return nil, fmt.Errorf("flux3klein: %w", err)
		}
		return p, nil
	}
	m, err := ReadProgramManifest(dir)
	if err != nil {
		return nil, err
	}
	if m.Version != ExportVersion {
		return nil, fmt.Errorf("flux3klein: %s is of layout %d, this hwnpu reads %d: convert it again", dir, m.Version, ExportVersion)
	}
	if _, err := m.variant(w, h, 1); err != nil {
		return nil, fmt.Errorf("flux3klein: %w", err)
	}
	if len(m.BNMean) != m.Dit.Latent || len(m.BNStd) != m.Dit.Latent {
		return nil, fmt.Errorf("flux3klein: the manifest has %d and %d batch-norm statistics for %d channels", len(m.BNMean), len(m.BNStd), m.Dit.Latent)
	}
	p := &Pipeline{W: w, H: h, m: m, dir: dir, dev: dev, log: log, ahead: runner.Ahead(dev), checked: map[string]bool{}}
	if p.tok, err = tokenizer.Load(filepath.Join(dir, TokenizerFile)); err != nil {
		return nil, fmt.Errorf("flux3klein: %w", err)
	}
	if p.emb, err = os.Open(filepath.Join(dir, EmbedFile)); err != nil {
		return nil, fmt.Errorf("flux3klein: %w", err)
	}
	return p, nil
}

// Close releases the pipeline's files.
func (p *Pipeline) Close() error {
	if p.emb != nil {
		return p.emb.Close()
	}
	return nil
}

// Manifest is the export's manifest.
func (p *Pipeline) Manifest() *ProgramManifest { return p.m }

func (p *Pipeline) logf(format string, args ...any) {
	if p.Logf != nil {
		p.Logf(format, args...)
	}
}

func (p *Pipeline) dump(name string, xs []float32) {
	if p.Dump == "" {
		return
	}
	b := make([]byte, 4*len(xs))
	for i, x := range xs {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	if err := os.WriteFile(filepath.Join(p.Dump, name), b, 0o644); err != nil {
		p.logf("dump %s: %v", name, err)
	}
}

// Tokens are the prompt's tokens in the template.
func (p *Pipeline) Tokens(prompt string) []int32 {
	return p.tok.Encode(strings.Replace(Template, "{}", prompt, 1))
}

// Sigmas are the steps' noise levels: linspace(1, 1/16, 16) shifted by Shift, then 0
// (FlowMatchEulerDiscreteScheduler with a static shift).
func Sigmas() []float64 {
	s := make([]float64, Steps+1)
	for i := 0; i < Steps; i++ {
		t := 1 - float64(i)*(1-1.0/Steps)/(Steps-1)
		s[i] = float64(float32(Shift * t / (1 + (Shift-1)*t)))
	}
	return s
}

// Noise returns Gaussian latents for seed in the DiT's packed layout (tokens x 128: a 2 x 2 patch
// of the 32 channels, channel major).
func (p *Pipeline) Noise(seed uint64) []float32 {
	r := rand.New(rand.NewPCG(seed, 0x666c7833))
	out := make([]float32, (p.W/16)*(p.H/16)*p.m.Dit.Latent)
	for i := range out {
		out[i] = float32(r.NormFloat64())
	}
	return out
}

// open loads one program, failing unless it runs on the NPU alone; the first load of each shows a
// line (it compiles the program).
func (p *Pipeline) open(ctx context.Context, name string) (runner.Program, error) {
	if name == "" {
		return nil, errors.New("flux3klein: the manifest misses a part")
	}
	var bar *console.Bar
	if p.log != nil && !p.checked[name] {
		bar = console.NewLoadBar(p.log, "  loading "+name, llm.PartBytes(filepath.Join(p.dir, strings.SplitN(name, "#", 2)[0])))
	}
	pr, err := p.load(ctx, name)
	if bar != nil {
		switch {
		case err != nil:
			bar.Failed("failed")
		case p.dev != nil:
			bar.Done(console.Duration(bar.Elapsed()) + ", all on the NPU")
		default:
			bar.Done(console.Duration(bar.Elapsed()))
		}
	}
	if err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	p.checked[name] = true
	return pr, nil
}

func (p *Pipeline) load(ctx context.Context, name string) (runner.Program, error) {
	if p.loader != nil {
		return p.loader(ctx, name)
	}
	pr, err := runner.Open(ctx, p.dev, filepath.Join(p.dir, name))
	if err == nil && p.dev != nil && !pr.OnNPU() {
		pr.Close()
		pr, err = nil, fmt.Errorf("%s does not run on the NPU alone", name)
	}
	return pr, err
}

// Cond is a prompt's conditioning: the text's stream out of context_embedder (text rows x hidden,
// divided by mil.TextStreamScale) and the programs of its text's length.
type Cond struct {
	v      *Variant
	stream []float32
	// ironStream is the text slot's stream on AMD's NPU (bf16), its first rows the prompt's
	ironStream []byte
	rows       int
	Tokens     int
}

// embedRows are Qwen3-VL's embedding rows of ids, the pad's past them to rows, times EmbedScale.
func (p *Pipeline) embedRows(ids []int32, rows int) ([]float32, error) {
	E := p.m.Text.Hidden
	s := p.m.EmbedScale
	b := make([]byte, 2*E)
	x := make([]float32, rows*E)
	for i := 0; i < rows; i++ {
		id := int32(PadID)
		if i < len(ids) {
			id = ids[i]
		}
		if _, err := p.emb.ReadAt(b, int64(id)*int64(2*E)); err != nil {
			return nil, err
		}
		for j, v := range fromBF16s(b) {
			x[i*E+j] = v * s
		}
	}
	return x, nil
}

// Encode runs Qwen3-VL's groups on the prompt's rows (its tokens in the template, padded to the
// shortest text length the export holds for them), then context_embedder of the eight hidden
// states the DiT reads.
func (p *Pipeline) Encode(ctx context.Context, prompt string) (Cond, error) {
	if p.iron != nil {
		return p.encodeIron(ctx, prompt)
	}
	t0 := time.Now()
	ids := p.Tokens(prompt)
	v, err := p.m.variant(p.W, p.H, len(ids))
	if err != nil {
		return Cond{}, fmt.Errorf("flux3klein: %w", err)
	}
	R, T := v.TextRows, p.m.Text.Hidden
	p.logf("text: %d tokens, %d rows", len(ids), R)
	x, err := p.embedRows(ids, R)
	if err != nil {
		return Cond{}, fmt.Errorf("flux3klein: %w", err)
	}
	kb := make([]float32, R)
	for i := len(ids); i < R; i++ {
		kb[i] = -1e4
	}
	cos, sin := mil.TextRope(mil.Text{HeadDim: p.m.Text.HeadDim, Theta: p.m.Text.Theta}, mil.TextPositions(len(ids), R))
	taps := make([][]float32, len(mil.TextTaps))
	names := make([]string, len(v.TextGroups))
	for i, g := range v.TextGroups {
		names[i] = g.Name
	}
	err = runner.InTurn(ctx, names, p.ahead, p.open, func(i int, pr runner.Program) error {
		y, err := pr.Run(ctx, map[string][]float32{"x": x, "kbias": kb, "cos": cos, "sin": sin})
		if err != nil {
			return err
		}
		x = y
		if t := v.TextGroups[i].Tap; t >= 0 {
			taps[t] = y
		}
		return nil
	})
	if err != nil {
		return Cond{}, fmt.Errorf("flux3klein: %w", err)
	}
	t1 := time.Now()
	n := len(mil.TextTaps)
	half := n / 2
	side := func(lo int) []float32 {
		out := make([]float32, R*half*T)
		for r := 0; r < R; r++ {
			for t := 0; t < half; t++ {
				copy(out[(r*half+t)*T:(r*half+t+1)*T], taps[lo+t][r*T:(r+1)*T])
			}
		}
		return out
	}
	x0, x1 := side(0), side(half)
	if p.Dump != "" {
		all := make([]float32, R*n*T)
		for r := 0; r < R; r++ {
			for t := 0; t < n; t++ {
				for j, val := range taps[t][r*T : (r+1)*T] {
					all[(r*n+t)*T+j] = val * mil.TextScale
				}
			}
		}
		p.dump("ctx.f32", all)
	}
	pr, err := p.open(ctx, v.Ctx)
	if err != nil {
		return Cond{}, fmt.Errorf("flux3klein: %w", err)
	}
	defer pr.Close()
	stream, err := pr.Run(ctx, map[string][]float32{"x0": x0, "x1": x1})
	if err != nil {
		return Cond{}, fmt.Errorf("flux3klein: %s: %w", v.Ctx, err)
	}
	p.logf("the text in %.2f s: Qwen3-VL %.2f s, the projection %.2f s", time.Since(t0).Seconds(), t1.Sub(t0).Seconds(), time.Since(t1).Seconds())
	return Cond{v: v, stream: stream, Tokens: len(ids)}, nil
}

// modulations are the modulation rows of each step of sigmas (a step from sigmas[i] to
// sigmas[i+1]). Each step's are kept (a few hundred KB): the time file is opened only for a step
// not made before, as a picture of the same steps as the last takes none.
func (p *Pipeline) modulations(sigmas []float64) ([][]float32, error) {
	p.modsMu.Lock()
	defer p.modsMu.Unlock()
	if p.mods == nil {
		p.mods = map[[2]float64][]float32{}
	}
	var te *timeEmbed
	out := make([][]float32, len(sigmas)-1)
	for i := range out {
		k := [2]float64{sigmas[i], sigmas[i+1]}
		if out[i] = p.mods[k]; out[i] != nil {
			continue
		}
		if te == nil {
			var err error
			if te, err = readTimeFile(filepath.Join(p.dir, TimeFile), p.m.Dit.Hidden, p.m.Dit.TimeIn); err != nil {
				return nil, err
			}
			defer te.close()
		}
		out[i] = te.modulation(sigmas[i], sigmas[i+1])
		p.mods[k] = out[i]
	}
	return out, nil
}

// Generate runs the DiT's steps over sigmas from noise under cond: each step the text's mode blocks
// (their modulation follows the step's target), the image's, then the single blocks give the
// velocity, and an Euler step moves the latents (tokens x 128).
func (p *Pipeline) Generate(ctx context.Context, cond Cond, noise []float32, sigmas []float64, step StepFunc) ([]float32, error) {
	if p.iron != nil {
		return p.generateIron(ctx, cond, noise, sigmas, step)
	}
	v := cond.v
	if v == nil {
		return nil, errors.New("flux3klein: the DiT runs without a prompt (Encode)")
	}
	E := p.m.Dit.Hidden
	mods, err := p.modulations(sigmas)
	if err != nil {
		return nil, fmt.Errorf("flux3klein: %w", err)
	}
	names := []string{v.TextModes, v.ImageModes}
	for _, g := range v.Singles {
		names = append(names, g.Name)
	}
	for _, name := range names { // each loaded once first: the first load compiles it
		if !p.checked[name] {
			pr, err := p.open(ctx, name)
			if err != nil {
				return nil, fmt.Errorf("flux3klein: %w", err)
			}
			pr.Close()
		}
	}
	ring := runner.NewRing(names, p.ahead, p.load)
	defer ring.Close()
	x := append([]float32(nil), noise...)
	L, N := v.TextRows, len(noise)/p.m.Dit.Latent
	n := len(sigmas) - 1
	for i := 0; i < n; i++ {
		t0 := time.Now()
		m := func(lo, hi int) []float32 { return mods[i][lo*E : hi*E] }
		run := func(k int, in map[string][]float32) ([]float32, error) {
			pr, err := ring.Take(ctx, k)
			if err != nil {
				return nil, fmt.Errorf("%s: %w", names[k], err)
			}
			defer pr.Close()
			y, err := pr.Run(ctx, in)
			if err != nil {
				return nil, fmt.Errorf("%s: %w", names[k], err)
			}
			return y, nil
		}
		txt, err := run(0, map[string][]float32{"x": cond.stream, "mods": m(mil.ModTextEarly, mil.ModTextEarly+3)})
		if err != nil {
			return nil, fmt.Errorf("flux3klein: step %d: %w", i, err)
		}
		img, err := run(1, map[string][]float32{"x": x, "mods": m(mil.ModImageEarly, mil.ModImageEarly+3)})
		if err != nil {
			return nil, fmt.Errorf("flux3klein: step %d: %w", i, err)
		}
		if len(txt) != L*E || len(img) != N*E {
			return nil, fmt.Errorf("flux3klein: step %d: the mode blocks give %d and %d values for %d and %d rows", i, len(txt), len(img), L, N)
		}
		h := append(txt, img...)
		for k := range v.Singles {
			hi := mil.ModOut
			if k == len(v.Singles)-1 {
				hi = mil.ModRows
			}
			if h, err = run(2+k, map[string][]float32{"x": h, "mods": m(mil.ModTextSingle, hi)}); err != nil {
				return nil, fmt.Errorf("flux3klein: step %d: %w", i, err)
			}
		}
		if len(h) != len(x) {
			return nil, fmt.Errorf("flux3klein: step %d: %d values of velocity for latents of %d", i, len(h), len(x))
		}
		p.dump(fmt.Sprintf("v_%d.f32", i), h)
		dt := float32(sigmas[i+1] - sigmas[i])
		for j := range x {
			x[j] += dt * h[j]
		}
		if step != nil {
			step(i, n, time.Since(t0))
		}
	}
	return x, nil
}

// Unpack is the DiT's latents (tokens x 128) as the VAE's input (32 x H/8 x W/8): each channel
// un-normalized by the VAE's batch-norm statistics, the 2 x 2 patches laid out.
func (p *Pipeline) Unpack(x []float32) []float32 {
	gh, gw, L := p.H/16, p.W/16, p.m.Dit.Latent
	H, W := 2*gh, 2*gw
	z := make([]float32, L/4*H*W)
	for py := 0; py < gh; py++ {
		for px := 0; px < gw; px++ {
			t := (py*gw + px) * L
			for ch := 0; ch < L/4; ch++ {
				for dy := 0; dy < 2; dy++ {
					for dx := 0; dx < 2; dx++ {
						k := ch*4 + dy*2 + dx
						z[(ch*H+2*py+dy)*W+2*px+dx] = x[t+k]*p.m.BNStd[k] + p.m.BNMean[k]
					}
				}
			}
		}
	}
	return z
}

// Decode runs the VAE's stages on the latents (Generate's) and returns the picture.
func (p *Pipeline) Decode(ctx context.Context, cond Cond, x []float32, stage func(i, n int, d time.Duration)) (*image.RGBA, error) {
	if p.iron != nil {
		return p.decodeIron(ctx, x, stage)
	}
	z := p.Unpack(x)
	debug.FreeOSMemory()
	xs, err := runner.Stages(ctx, cond.v.Vae, p.ahead, p.open, [][]float32{z}, runner.StageBudget,
		func(_ int, pr runner.Program, in []float32) ([]float32, error) {
			y, err := pr.Run(ctx, map[string][]float32{"x": in})
			debug.FreeOSMemory()
			return y, err
		},
		func(i, n int, d time.Duration) {
			if stage != nil {
				stage(i, n, d)
			}
		})
	if err != nil {
		return nil, fmt.Errorf("flux3klein: vae: %w", err)
	}
	W, H := p.W, p.H
	out := xs[0]
	if len(out) != 3*W*H {
		return nil, fmt.Errorf("flux3klein: vae: %d values for a %dx%d picture", len(out), W, H)
	}
	img := image.NewRGBA(image.Rect(0, 0, W, H))
	for y := 0; y < H; y++ {
		for xx := 0; xx < W; xx++ {
			i := y*W + xx
			img.SetRGBA(xx, y, color.RGBA{pixel(out[i]), pixel(out[W*H+i]), pixel(out[2*W*H+i]), 255})
		}
	}
	return img, nil
}

// pixel is a value in [-1, 1] as a byte (the pipeline's (x / 2 + 0.5).clamp(0, 1), rounded).
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
