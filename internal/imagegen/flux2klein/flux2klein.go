// Package flux2klein runs FLUX.2 [klein] 4B (black-forest-labs/FLUX.2-klein-4B, Apache-2.0), an
// image model that draws from a prompt and edits reference images, on an NPU: on AMD's, the graphs
// tools/amd/iron_flux2klein.py exports; on Apple's Neural Engine, the Core AI programs of
// internal/imagegen/flux2klein/mil that Convert writes (progEngine: text to image for now).
//
// The DiT is FLUX.1's shape with 5 double-stream and 20 single-stream blocks, 3072 wide; Qwen3-4B
// reads the prompt (512 rows: the hidden states after layers 9, 18 and 27 side by side; the DiT
// attends to the pads too); the latents are 2 x 2 patches of the FLUX.2 VAE's 32 channels. A
// picture is:
//
//  1. the prompt in Qwen3's chat template (no thinking), padded to 512 tokens, their embedding
//     rows, the text graph a group of layers at a time (the prompt's rows causal, the pads
//     attending to the prompt's keys only), and context_embedder: the DiT's text rows;
//  2. from Gaussian noise, 4 flow-matching Euler steps (the checkpoint is distilled: no
//     guidance) on a linear grid shifted by the pipeline's empirical mu: each step the
//     timestep's modulation rows (mods), the latents embedded, the double and single blocks over
//     [image rows, text rows], a group at a time, and the final norm and projection;
//  3. the latents un-normalized by the VAE's batch-norm statistics and un-patched, and the
//     VAE's decoder.
//
// Editing: each reference's latents (the VAE's encoder, patched and normalized) are rows of the
// image stream after the picture's, told apart by the first RoPE axis (10, 20, ...). They run
// through every block with the picture's rows; only the picture's rows are read at the end.
//
// The export is for one image size, and for the references it was compiled for: of the picture's
// size, or of half its sides (a quarter of the rows: EncodeImage's div).
package flux2klein

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/imagegen/anima"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3vl"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// Config is the DiT's shape (iron_flux2klein.KleinConfig).
type Config struct {
	Width       int     `json:"width"`
	Height      int     `json:"height"`
	TextRows    int     `json:"text_rows"`
	Hidden      int     `json:"hidden"`
	HeadDim     int     `json:"head_dim"`
	Latent      int     `json:"latent"`
	Text        int     `json:"text"`
	Axes        []int   `json:"axes"`
	Theta       float64 `json:"theta"`
	Doubles     int     `json:"doubles"`
	Singles     int     `json:"singles"`
	DoubleGroup int     `json:"double_group"`
	SingleGroup int     `json:"single_group"`
}

// GH is the patches' rows (2 x 2 latent pixels each).
func (c Config) GH() int { return c.Height / 16 }

// GW is the patches' columns.
func (c Config) GW() int { return c.Width / 16 }

// Tokens is the picture's rows of the DiT.
func (c Config) Tokens() int { return c.GH() * c.GW() }

// TextConfig is the text encoder's shape (iron_flux2klein.TextConfig).
type TextConfig struct {
	Rows    int     `json:"rows"`
	Hidden  int     `json:"hidden"`
	Heads   int     `json:"heads"`
	HeadDim int     `json:"head_dim"`
	Theta   float64 `json:"theta"`
}

// Model is model.json of an export: the checkpoint, its scheduler and its defaults.
type Model struct {
	Name      string `json:"name"`
	Scheduler struct {
		BaseShift float64 `json:"base_shift"`
	} `json:"scheduler"`
	Steps    int     `json:"steps"`
	Guidance float64 `json:"guidance"`
}

// Template is the chat template the pipeline puts the prompt in: Qwen3's, a user message and
// the assistant's turn opened without thinking.
const Template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

// PadID is Qwen's <|endoftext|>, the tokenizer's pad.
const PadID = 151643

// StepFunc is told that step i (of n) took d.
type StepFunc func(i, n int, d time.Duration)

// IsExport reports whether dir is an export of FLUX.2 klein: of tools/amd/iron_flux2klein.py, or
// of Convert.
func IsExport(dir string) bool { return isIronExport(dir) || IsProgramExport(dir) }

// isIronExport reports whether dir is an export of tools/amd/iron_flux2klein.py.
func isIronExport(dir string) bool {
	for _, p := range []string{"doubles/manifest.json", "singles/manifest.json", "mods/manifest.json", "ctx/manifest.json", "text/manifest.json"} {
		if _, err := os.Stat(filepath.Join(dir, p)); err != nil {
			return false
		}
	}
	return true
}

// Pipeline is a loaded export.
type Pipeline struct {
	C     Config
	Text  TextConfig
	Model Model

	dev   *hwnpu.Device
	dir   string
	taps  []int // the text graph's groups whose outputs the DiT reads
	tok   *tokenizer.Tokenizer
	embed *os.File // bf16 rows of Text.Hidden
	vae   *anima.VAE
	// the VAE's batch-norm mean and standard deviation of the patched latents' channels
	bnMean, bnStd []float32

	// Dump, when set, is a directory the text's hidden states (text_h0..2.bf16), the DiT's text
	// rows (ctx.bf16) and each step's velocity (v_N.bf16) are written to, raw bf16, to check
	// against tools/ref/ref_flux2klein.py.
	Dump string
	// Logf, when set, is told what is done.
	Logf func(format string, args ...any)
	// Stage, when set, is told of each stage of the VAE as it ends.
	Stage anima.StageFunc

	// prog runs an export of Convert (Apple's Neural Engine); nil for an IRON export
	prog *progEngine
}

// Open reads the export in dir: of tools/amd/iron_flux2klein.py on AMD's NPU, or of Convert on
// Apple's Neural Engine (log, when not nil, shows the programs loading).
func Open(ctx context.Context, dev *hwnpu.Device, dir string, log io.Writer) (*Pipeline, error) {
	return OpenAt(ctx, dev, dir, 0, 0, log)
}

// OpenAt is Open for pictures of w x h, one of the sizes an export of Convert holds
// (ConvertOptions.Sizes); 0 x 0 is the export's default (an IRON export's one size).
func OpenAt(ctx context.Context, dev *hwnpu.Device, dir string, w, h int, log io.Writer) (*Pipeline, error) {
	if IsProgramExport(dir) {
		return openProgramPipeline(dev, dir, w, h, log)
	}
	p := &Pipeline{dev: dev, dir: dir}
	fail := func(err error) (*Pipeline, error) {
		p.Close()
		return nil, fmt.Errorf("flux2klein: %w", err)
	}
	if dev == nil {
		return fail(errors.New("an IRON export runs on AMD's NPU only"))
	}
	m, err := iron.ReadGraphManifest(filepath.Join(dir, "doubles"))
	if err != nil {
		return fail(err)
	}
	var x struct {
		Config Config `json:"config"`
	}
	if err := json.Unmarshal(m.Meta, &x); err != nil {
		return fail(err)
	}
	p.C = x.Config
	tm, err := iron.ReadGraphManifest(filepath.Join(dir, "text"))
	if err != nil {
		return fail(err)
	}
	var t struct {
		Config TextConfig `json:"config"`
		Taps   []int      `json:"taps"`
	}
	if err := json.Unmarshal(tm.Meta, &t); err != nil {
		return fail(err)
	}
	if t.Config.Rows != p.C.TextRows || t.Config.Hidden != p.C.Text || len(t.Taps) != 3 {
		return fail(fmt.Errorf("text: %d rows of %d and %d hidden states for the DiT's %d of %d and 3", t.Config.Rows, t.Config.Hidden, len(t.Taps), p.C.TextRows, p.C.Text))
	}
	p.Text, p.taps = t.Config, t.Taps
	p.Model = Model{Name: "FLUX.2-klein-4B", Steps: 4, Guidance: 1}
	if b, err := os.ReadFile(filepath.Join(dir, "model.json")); err == nil {
		if err := json.Unmarshal(b, &p.Model); err != nil {
			return fail(err)
		}
	}
	td := filepath.Join(dir, "text")
	if p.tok, err = tokenizer.Load(filepath.Join(td, "tokenizer.json")); err != nil {
		return fail(err)
	}
	if p.embed, err = os.Open(filepath.Join(td, "embed.bin")); err != nil {
		return fail(err)
	}
	if p.vae, err = anima.OpenVAE(ctx, dev, filepath.Join(dir, "vae")); err != nil {
		return fail(err)
	}
	b, err := os.ReadFile(filepath.Join(dir, "vae", "manifest.json"))
	if err != nil {
		return fail(err)
	}
	var vm struct {
		Mean []float32 `json:"bn_mean"`
		Std  []float32 `json:"bn_std"`
	}
	if err := json.Unmarshal(b, &vm); err != nil || len(vm.Mean) != p.C.Latent || len(vm.Std) != p.C.Latent {
		return fail(fmt.Errorf("vae: %d and %d batch-norm statistics for %d channels (%v)", len(vm.Mean), len(vm.Std), p.C.Latent, err))
	}
	p.bnMean, p.bnStd = vm.Mean, vm.Std
	return p, nil
}

// Close releases what Open holds.
func (p *Pipeline) Close() error {
	if p.embed != nil {
		p.embed.Close()
	}
	if p.prog != nil {
		return p.prog.closeDiT()
	}
	return nil
}

func (p *Pipeline) logf(format string, args ...any) {
	if p.Logf != nil {
		p.Logf(format, args...)
	}
}

func (p *Pipeline) dump(name string, b []byte) {
	if p.Dump == "" {
		return
	}
	if err := os.WriteFile(filepath.Join(p.Dump, name), b, 0o644); err != nil {
		p.logf("dump: %v", err)
	}
}

// MaxReferences is the model's limit of reference images.
const MaxReferences = 4

// References are the references the export runs, as hwnpu-image names them: "0" (text to image),
// "2" (two of the picture's size), "2@half" (two of half its sides), "1+2@half" (a mix): every
// mix of up to MaxReferences whose step's rows the export's blocks were compiled for
// (tools/amd/iron_flux2klein.py --refs) and whose sizes it has an encoder for.
func (p *Pipeline) References() []string {
	out := []string{"0"}
	if p.prog != nil {
		return out
	}
	for full := 0; full <= MaxReferences; full++ {
		for half := 0; full+half <= MaxReferences; half++ {
			if full+half == 0 {
				continue
			}
			divs := make([]int, 0, full+half)
			for i := 0; i < full; i++ {
				divs = append(divs, 1)
			}
			for i := 0; i < half; i++ {
				divs = append(divs, 2)
			}
			if !p.HasReferences(divs) {
				continue
			}
			var name []string
			if full > 0 {
				name = append(name, fmt.Sprint(full))
			}
			if half > 0 {
				name = append(name, fmt.Sprintf("%d@half", half))
			}
			out = append(out, strings.Join(name, "+"))
		}
	}
	return out
}

// HasReferences reports whether the export holds the blocks for a step with references of the
// picture's sides over each of divs (in any order: the blocks depend on the rows alone) and the
// encoders of those sizes.
func (p *Pipeline) HasReferences(divs []int) bool {
	if p.prog != nil {
		return len(divs) == 0
	}
	rows, ok := p.rows(divs)
	if !ok {
		return false
	}
	need := []string{"doubles" + p.suffix(rows), "singles" + p.suffix(rows), "dit_in" + p.suffix(rows)}
	for _, d := range divs {
		if d > 1 {
			need = append(need, fmt.Sprintf("vae_enc_d%d", d))
		} else {
			need = append(need, "vae_enc")
		}
	}
	for _, f := range need {
		if _, err := os.Stat(filepath.Join(p.dir, f, "manifest.json")); err != nil {
			return false
		}
	}
	return true
}

// rows is a step's rows with references of the picture's sides over each of divs: the picture's,
// the references' and the text's.
func (p *Pipeline) rows(divs []int) (int, bool) {
	c := p.C
	n := c.Tokens() + c.TextRows
	for _, d := range divs {
		if d < 1 || c.GH()%d != 0 || c.GW()%d != 0 {
			return 0, false
		}
		n += (c.GH() / d) * (c.GW() / d)
	}
	return n, true
}

// suffix is the name of the graphs of a step of that many rows: none for the picture and the
// text alone, else _nROWS (iron_flux2klein.suffix).
func (p *Pipeline) suffix(rows int) string {
	if rows == p.C.Tokens()+p.C.TextRows {
		return ""
	}
	return fmt.Sprintf("_n%d", rows)
}

// Tokens are the prompt's tokens in the template, cut to the text's rows (the pipeline
// truncates on the right).
func (p *Pipeline) Tokens(prompt string) []int32 {
	ids := p.tok.Encode(strings.Replace(Template, "{}", prompt, 1))
	if len(ids) > p.Text.Rows {
		ids = ids[:p.Text.Rows]
	}
	return ids
}

// Cond is a prompt's conditioning: the DiT's text rows (TextRows x hidden, bf16; on Apple's Neural
// Engine float32, divided by mil.DitScale).
type Cond struct {
	ctx  []byte
	rows []float32
	// Tokens is the prompt's tokens in the template (the rest of the rows are pads).
	Tokens int
}

func (p *Pipeline) open(ctx context.Context, name string, swap, prefetch bool) (*iron.Graph, error) {
	return iron.OpenGraphWith(ctx, p.dev, filepath.Join(p.dir, name), iron.GraphOptions{SwapVariants: swap, Prefetch: prefetch, Barrier: true})
}

func run(ctx context.Context, g *iron.Graph, in map[string][]byte, values map[string]int64) error {
	for k, v := range in {
		if err := g.SetInput(k, v); err != nil {
			return err
		}
	}
	return g.Run(ctx, values)
}

func output(g *iron.Graph, i int) ([]byte, error) {
	if i < 0 {
		i = len(g.Outputs()) - 1
	}
	b, err := g.Output(g.Outputs()[i].Name)
	return append([]byte(nil), b...), err
}

// Encode runs the text encoder and context_embedder on the prompt.
func (p *Pipeline) Encode(ctx context.Context, prompt string) (Cond, error) {
	c, err := p.EncodeAll(ctx, prompt)
	if err != nil {
		return Cond{}, err
	}
	return c[0], nil
}

// EncodeAll is Encode of each prompt with each group of the text encoder's layers read once.
//
// The text graph holds a group of layers; its bank images are read into one bank in turn. Each
// group runs every prompt's 512 rows: the prompt's rows attend causally, the pads to the
// prompt's keys only (sel picks each row's attention, nk is the prompt's tokens). The outputs of
// the groups ending at layers 9, 18 and 27 go to ctx, the DiT's context_embedder.
func (p *Pipeline) EncodeAll(ctx context.Context, prompts ...string) ([]Cond, error) {
	if p.prog != nil {
		return p.encodePrograms(ctx, prompts)
	}
	t0 := time.Now()
	T := p.Text
	R, E := T.Rows, T.Hidden
	pad := make([]byte, 2*E)
	if _, err := p.embed.ReadAt(pad, int64(PadID)*int64(2*E)); err != nil {
		return nil, fmt.Errorf("flux2klein: %w", err)
	}
	xs := make([][]byte, len(prompts))
	sels := make([][]byte, len(prompts))
	ns := make([]int, len(prompts))
	one := toBF16([]float32{1})
	W := T.Heads * T.HeadDim
	for k, pr := range prompts {
		ids := p.Tokens(pr)
		x := make([]byte, 2*R*E)
		for i := 0; i < R; i++ {
			if i >= len(ids) {
				copy(x[2*i*E:], pad)
				continue
			}
			if _, err := p.embed.ReadAt(x[2*i*E:2*(i+1)*E], int64(ids[i])*int64(2*E)); err != nil {
				return nil, fmt.Errorf("flux2klein: %w", err)
			}
		}
		sel := make([]byte, 2*R*W)
		for i := 0; i < len(ids)*W; i++ {
			copy(sel[2*i:], one)
		}
		xs[k], sels[k], ns[k] = x, sel, len(ids)
		p.logf("text: %d tokens", len(ids))
	}
	angles := toBF16(halfRope(R, T.HeadDim, T.Theta))
	g, err := p.open(ctx, "text", true, anima.PrefetchFits(p.vae, p.dir, []string{"text"}, "text"))
	if err != nil {
		return nil, fmt.Errorf("flux2klein: %w", err)
	}
	taps := make([][3][]byte, len(prompts))
	var tRead, tRun time.Duration
	tOpen := time.Since(t0)
	for gi := 0; gi < max(g.Variants(), 1); gi++ {
		if g.Variants() > 0 {
			t := time.Now()
			if err := g.Use(gi); err != nil {
				g.Close()
				return nil, fmt.Errorf("flux2klein: %w", err)
			}
			tRead += time.Since(t)
		}
		for k := range prompts {
			t := time.Now()
			if err := run(ctx, g, map[string][]byte{"x": xs[k], "angles": angles, "sel": sels[k]}, map[string]int64{"nk": int64(ns[k])}); err != nil {
				g.Close()
				return nil, fmt.Errorf("flux2klein: text group %d: %w", gi, err)
			}
			tRun += time.Since(t)
			if xs[k], err = output(g, 0); err != nil {
				g.Close()
				return nil, fmt.Errorf("flux2klein: %w", err)
			}
			for i, tap := range p.taps {
				if tap == gi {
					taps[k][i] = xs[k]
				}
			}
			if k == 0 {
				p.dump(fmt.Sprintf("text_g%d.bf16", gi), xs[k])
			}
		}
	}
	g.Close()
	t1 := time.Now()
	c, err := p.open(ctx, "ctx", false, false)
	if err != nil {
		return nil, fmt.Errorf("flux2klein: %w", err)
	}
	defer c.Close()
	conds := make([]Cond, len(prompts))
	for k := range prompts {
		if err := run(ctx, c, map[string][]byte{"h0": taps[k][0], "h1": taps[k][1], "h2": taps[k][2]}, nil); err != nil {
			return nil, fmt.Errorf("flux2klein: ctx: %w", err)
		}
		out, err := output(c, 0)
		if err != nil {
			return nil, fmt.Errorf("flux2klein: %w", err)
		}
		conds[k] = Cond{ctx: out, Tokens: ns[k]}
		if k == 0 {
			for i := range taps[k] {
				p.dump(fmt.Sprintf("text_h%d.bf16", i), taps[k][i])
			}
			p.dump("ctx.bf16", out)
		}
	}
	p.logf("the text in %.2f s: Qwen3-4B %.2f s (opening %.2f, the groups' weights %.2f, the layers %.2f), the projection %.2f s",
		time.Since(t0).Seconds(), t1.Sub(t0).Seconds(), tOpen.Seconds(), tRead.Seconds(), tRun.Seconds(), time.Since(t1).Seconds())
	return conds, nil
}

// EmpiricalMu is the pipeline's shift of the sigmas for a picture of rows rows drawn in steps
// steps (compute_empirical_mu).
func EmpiricalMu(rows, steps int) float64 {
	const a1, b1 = 8.73809524e-05, 1.89833333
	const a2, b2 = 0.00016927, 0.45666666
	n := float64(rows)
	if rows > 4300 {
		return a2*n + b2
	}
	m200, m10 := a2*n+b2, a1*n+b1
	a := (m200 - m10) / 190
	return a*float64(steps) + m200 - 200*a
}

// Sigmas are the steps' sigmas and the terminal 0: a linear grid from 1 to 1/steps through the
// scheduler's exponential time shift by EmpiricalMu (FlowMatchEulerDiscreteScheduler with
// dynamic shifting). steps 0 is the export's default.
func (p *Pipeline) Sigmas(steps int) []float64 {
	if steps <= 0 {
		steps = p.Model.Steps
	}
	mu := math.Exp(EmpiricalMu(p.C.Tokens(), steps))
	s := make([]float64, steps+1)
	for i := 0; i < steps; i++ {
		t := 1 + (1/float64(steps)-1)*float64(i)/float64(max(1, steps-1))
		s[i] = mu / (mu + (1/t - 1))
	}
	return s
}

// Noise returns Gaussian latents for seed in the DiT's packed layout (tokens x 128: a 2 x 2
// patch of the 32 channels, channel major).
func (p *Pipeline) Noise(seed uint64) []float32 {
	r := rand.New(rand.NewPCG(seed, 0x6b6c6569))
	out := make([]float32, p.C.Tokens()*p.C.Latent)
	for i := range out {
		out[i] = float32(r.NormFloat64())
	}
	return out
}

// positions are the DiT's rows' position ids (reference index, row, column, text position): the
// picture's (0, y, x, 0), each reference's (10 (i + 1), y, x, 0) over its own grid (the picture's
// sides over its div), then the text's (0, 0, 0, l).
func (p *Pipeline) positions(divs []int) [][4]float64 {
	c := p.C
	var out [][4]float64
	for i := 0; i <= len(divs); i++ {
		gh, gw := c.GH(), c.GW()
		if i > 0 {
			gh, gw = gh/divs[i-1], gw/divs[i-1]
		}
		for y := 0; y < gh; y++ {
			for x := 0; x < gw; x++ {
				out = append(out, [4]float64{float64(10 * i), float64(y), float64(x), 0})
			}
		}
	}
	for l := 0; l < c.TextRows; l++ {
		out = append(out, [4]float64{0, 0, 0, float64(l)})
	}
	return out
}

// ropeAngles is IRON's RoPE table of positions pos (rows x HeadDim, cos and sin interleaved),
// each axis's frequencies theta^(-2k / d) (iron_qwenimage.rope_angles).
func (p *Pipeline) ropeAngles(pos [][4]float64) []float32 {
	c := p.C
	var freqs []float64
	var axis []int
	for a, d := range c.Axes {
		for k := 0; k < d; k += 2 {
			freqs = append(freqs, 1/math.Pow(c.Theta, float64(k)/float64(d)))
			axis = append(axis, a)
		}
	}
	out := make([]float32, len(pos)*c.HeadDim)
	for r, ps := range pos {
		for i, f := range freqs {
			a := ps[axis[i]] * f
			out[r*c.HeadDim+2*i] = float32(math.Cos(a))
			out[r*c.HeadDim+2*i+1] = float32(math.Sin(a))
		}
	}
	return out
}

// Generate runs Euler's steps over sigmas (Sigmas) from the noise (Noise's layout) under cond.
// refs are the references' latents, each in Noise's layout (EncodeImage): the export must hold
// the blocks compiled for that many (References).
func (p *Pipeline) Generate(ctx context.Context, cond Cond, refs [][]float32, noise []float32, sigmas []float64, step StepFunc) ([]float32, error) {
	c := p.C
	E, N := c.Hidden, c.Tokens()
	if len(noise) != N*c.Latent {
		return nil, fmt.Errorf("flux2klein: %d noise values for latents of %d", len(noise), N*c.Latent)
	}
	if p.prog != nil {
		if len(refs) > 0 {
			return nil, errors.New("flux2klein: editing (--image) runs on AMD's NPU only for now")
		}
		return p.generatePrograms(ctx, cond, noise, sigmas, step)
	}
	if cond.ctx == nil {
		return nil, errors.New("flux2klein: the DiT runs without a prompt (Encode)")
	}
	// the references' rows: each of the picture's size, or of half or a quarter of its sides
	divs := make([]int, len(refs))
	for i, r := range refs {
		for _, k := range []int{1, 2, 4} {
			if len(r)*k*k == N*c.Latent && c.GH()%k == 0 && c.GW()%k == 0 {
				divs[i] = k
			}
		}
		if divs[i] == 0 {
			return nil, fmt.Errorf("flux2klein: reference %d: %d values for latents of %d (a reference is of the picture's size, or of half its sides)", i+1, len(r), N*c.Latent)
		}
	}
	if !p.HasReferences(divs) {
		return nil, fmt.Errorf("flux2klein: the export has no blocks for these %d references (it runs %s)", len(refs), strings.Join(p.References(), ", "))
	}
	rows, _ := p.rows(divs)
	sfx := p.suffix(rows)
	n := len(sigmas) - 1
	// every step's modulation rows first: the mods graph (0.3 GB of bf16) is not held during the blocks
	mg, err := p.open(ctx, "mods", false, false)
	if err != nil {
		return nil, fmt.Errorf("flux2klein: %w", err)
	}
	mods := make([][]byte, n)
	for i := 0; i < n; i++ {
		if err := run(ctx, mg, map[string][]byte{"tproj": toBF16(sinusoid(1000*sigmas[i], 256))}, nil); err != nil {
			mg.Close()
			return nil, fmt.Errorf("flux2klein: mods: %w", err)
		}
		if mods[i], err = output(mg, -1); err != nil {
			mg.Close()
			return nil, fmt.Errorf("flux2klein: %w", err)
		}
	}
	mg.Close()
	var in, doubles, singles, out *iron.Graph
	closeAll := func() {
		for _, g := range []*iron.Graph{in, doubles, singles, out} {
			if g != nil {
				g.Close()
			}
		}
	}
	defer closeAll()
	names := []string{"dit_in" + sfx, "doubles" + sfx, "singles" + sfx, "dit_out"}
	prefetch := anima.PrefetchFits(p.vae, p.dir, names, "doubles"+sfx, "singles"+sfx)
	for _, x := range []struct {
		g    **iron.Graph
		name string
		swap bool
	}{{&in, names[0], false}, {&doubles, names[1], true}, {&singles, names[2], true}, {&out, names[3], false}} {
		if *x.g, err = p.open(ctx, x.name, x.swap, prefetch && x.swap); err != nil {
			return nil, fmt.Errorf("flux2klein: %w", err)
		}
	}
	angles := toBF16(p.ropeAngles(p.positions(divs)))
	var refRows []byte // the references' latents: the same rows every step
	for _, r := range refs {
		refRows = append(refRows, toBF16(r)...)
	}
	x := append([]float32(nil), noise...)
	var tRead, tRun time.Duration
	tLoop := time.Now()
	blocks := func(g *iron.Graph, h, m []byte, o int, what string) ([]byte, error) {
		for gi := 0; gi < max(g.Variants(), 1); gi++ {
			if g.Variants() > 0 {
				t := time.Now()
				if err := g.Use(gi); err != nil {
					return nil, err
				}
				tRead += time.Since(t)
			}
			t := time.Now()
			if err := run(ctx, g, map[string][]byte{"x": h, "mods": m, "angles": angles}, map[string]int64{"nk": int64(rows)}); err != nil {
				return nil, fmt.Errorf("%s group %d: %w", what, gi, err)
			}
			tRun += time.Since(t)
			var err error
			if h, err = output(g, o); err != nil {
				return nil, err
			}
		}
		return h, nil
	}
	for i := 0; i < n; i++ {
		t0 := time.Now()
		m := mods[i]
		row := func(a, b int) []byte { return m[2*a*E : 2*b*E] }
		if err := run(ctx, in, map[string][]byte{"x": append(toBF16(x), refRows...)}, nil); err != nil {
			return nil, fmt.Errorf("flux2klein: step %d: %w", i, err)
		}
		h, err := output(in, 0)
		if err != nil {
			return nil, fmt.Errorf("flux2klein: step %d: %w", i, err)
		}
		h = append(h, cond.ctx...) // [image rows, text rows]
		if h, err = blocks(doubles, h, row(0, 12), -1, "double blocks"); err == nil {
			h, err = blocks(singles, h, row(12, 15), 0, "single blocks")
		}
		if err != nil {
			return nil, fmt.Errorf("flux2klein: step %d: %w", i, err)
		}
		if err := run(ctx, out, map[string][]byte{"x": h[:2*N*E], "scale": row(15, 16), "shift": row(16, 17)}, nil); err != nil {
			return nil, fmt.Errorf("flux2klein: step %d: %w", i, err)
		}
		vb, err := output(out, 0)
		if err != nil {
			return nil, fmt.Errorf("flux2klein: step %d: %w", i, err)
		}
		p.dump(fmt.Sprintf("v_%d.bf16", i), vb)
		v := fromBF16s(vb)
		dt := float32(sigmas[i+1] - sigmas[i])
		for j := range x {
			x[j] += dt * v[j]
		}
		if step != nil {
			step(i, n, time.Since(t0))
		}
	}
	all := time.Since(tLoop)
	p.logf("the steps' %.2f s: the blocks %.2f s, the groups' weights %.2f s, the rest (the embedding, the output, the host) %.2f s",
		all.Seconds(), tRun.Seconds(), tRead.Seconds(), (all - tRun - tRead).Seconds())
	return x, nil
}

// Unpack is the packed latents (Noise's layout) as the VAE takes them, (32, H/8, W/8): each
// channel un-normalized by the VAE's batch-norm statistics, the 2 x 2 patches laid out (the
// pipeline's _unpatchify_latents).
func (p *Pipeline) Unpack(x []float32) []float32 {
	c := p.C
	gh, gw, L := c.GH(), c.GW(), c.Latent
	H, W := 2*gh, 2*gw
	z := make([]float32, L/4*H*W)
	for py := 0; py < gh; py++ {
		for px := 0; px < gw; px++ {
			t := (py*gw + px) * L
			for ch := 0; ch < L/4; ch++ {
				for dy := 0; dy < 2; dy++ {
					for dx := 0; dx < 2; dx++ {
						k := ch*4 + dy*2 + dx
						z[(ch*H+2*py+dy)*W+2*px+dx] = x[t+k]*p.bnStd[k] + p.bnMean[k]
					}
				}
			}
		}
	}
	return z
}

// EncodeImage is a reference image's latents in Noise's layout, for Generate's refs: the image's
// centre cropped to the picture's aspect ratio and scaled to its sides over div (bicubic), the
// VAE's encoder of that size (the latents' means), 2 x 2 patches normalized by the VAE's batch-norm
// statistics. div 1 reads the reference at the picture's size; 2 at half its sides, a quarter of
// the rows a step carries (HasReferences).
func (p *Pipeline) EncodeImage(ctx context.Context, img image.Image, div int) ([]float32, error) {
	c := p.C
	if p.prog != nil {
		return nil, errors.New("flux2klein: editing (--image) runs on AMD's NPU only for now")
	}
	if div < 1 || c.Width%(16*div) != 0 || c.Height%(16*div) != 0 {
		return nil, fmt.Errorf("flux2klein: a reference of the picture's sides over %d", div)
	}
	W, H := c.Width/div, c.Height/div
	dir := "vae_enc"
	if div > 1 {
		dir = fmt.Sprintf("vae_enc_d%d", div)
	}
	b := img.Bounds()
	sw, sh := b.Dx(), b.Dy()
	cw, ch := sw, sh
	if sw*H > sh*W {
		cw = sh * W / H
	} else {
		ch = sw * H / W
	}
	if cw == 0 || ch == 0 {
		return nil, fmt.Errorf("flux2klein: a reference of %dx%d", sw, sh)
	}
	src := lfm3vl.RGB(img)
	x0, y0 := (sw-cw)/2, (sh-ch)/2
	crop := make([]uint8, cw*ch*3)
	for y := 0; y < ch; y++ {
		copy(crop[y*cw*3:(y+1)*cw*3], src[((y0+y)*sw+x0)*3:])
	}
	rgb := lfm3vl.ResizeRGB(crop, cw, ch, W, H, "bicubic")
	px := make([]float32, 3*H*W)
	for i := 0; i < H*W; i++ {
		for k := 0; k < 3; k++ {
			px[k*H*W+i] = float32(rgb[3*i+k])/127.5 - 1
		}
	}
	enc, err := anima.OpenVAE(ctx, p.dev, filepath.Join(p.dir, dir))
	if err != nil {
		return nil, fmt.Errorf("flux2klein: the VAE's encoder: %w", err)
	}
	enc.Stage = p.Stage
	z, err := enc.Encode(ctx, px)
	if err != nil {
		return nil, fmt.Errorf("flux2klein: %w", err)
	}
	gh, gw, L := c.GH()/div, c.GW()/div, c.Latent
	h, w := 2*gh, 2*gw
	if len(z) != L/4*h*w {
		return nil, fmt.Errorf("flux2klein: the encoder gave %d values for %dx%dx%d", len(z), L/4, h, w)
	}
	out := make([]float32, gh*gw*L)
	for py := 0; py < gh; py++ {
		for px := 0; px < gw; px++ {
			t := (py*gw + px) * L
			for ch := 0; ch < L/4; ch++ {
				for dy := 0; dy < 2; dy++ {
					for dx := 0; dx < 2; dx++ {
						k := ch*4 + dy*2 + dx
						out[t+k] = (z[(ch*h+2*py+dy)*w+2*px+dx] - p.bnMean[k]) / p.bnStd[k]
					}
				}
			}
		}
	}
	return out, nil
}

// Decode turns the latents (Noise's layout) into the picture.
func (p *Pipeline) Decode(ctx context.Context, x []float32) (*image.RGBA, error) {
	if p.prog != nil {
		imgs, err := p.DecodeAll(ctx, x)
		if err != nil {
			return nil, err
		}
		return imgs[0], nil
	}
	p.vae.Stage = p.Stage
	return p.vae.Decode(ctx, p.Unpack(x))
}

// halfRope is Hugging Face's RoPE (two halves) at positions 0..rows-1 in IRON's table, cos and
// sin interleaved (iron_anima.half_rope_table).
func halfRope(rows, dim int, theta float64) []float32 {
	out := make([]float32, rows*dim)
	for r := 0; r < rows; r++ {
		for k := 0; k < dim/2; k++ {
			a := float64(r) / math.Pow(theta, float64(2*k)/float64(dim))
			out[r*dim+2*k] = float32(math.Cos(a))
			out[r*dim+2*k+1] = float32(math.Sin(a))
		}
	}
	return out
}

// sinusoid is diffusers' get_timestep_embedding(t, dim, flip_sin_to_cos=True,
// downscale_freq_shift=0): the cosines, then the sines.
func sinusoid(t float64, dim int) []float32 {
	half := dim / 2
	out := make([]float32, dim)
	for i := 0; i < half; i++ {
		a := t * math.Exp(-math.Log(10000)*float64(i)/float64(half))
		out[i] = float32(math.Cos(a))
		out[half+i] = float32(math.Sin(a))
	}
	return out
}

func toBF16(xs []float32) []byte {
	b := make([]byte, 2*len(xs))
	for i, x := range xs {
		u := math.Float32bits(x)
		binary.LittleEndian.PutUint16(b[2*i:], uint16((u+0x7fff+(u>>16)&1)>>16))
	}
	return b
}

func fromBF16s(b []byte) []float32 {
	out := make([]float32, len(b)/2)
	for i := range out {
		out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[2*i:])) << 16)
	}
	return out
}
