// Package qwenimage runs Qwen-Image-2.1 (Qwen/Qwen-Image-2.1-Turbo), an image model, on an NPU:
// on AMD's, the graphs tools/amd/iron_qwenimage.py exports; on Apple's Neural Engine, the Core ML
// packages of internal/imagegen/qwenimage/mil (Convert, Prepare). The host does what is not a
// tensor computation.
//
// Qwen-Image-2.1 is a 7B single-stream DiT over one sequence, the prompt's rows and then the
// image's, with Qwen3-VL-8B reading the prompt and a VAE of 64 latent channels at a 16th of the
// sides that decodes RGBA. A picture is:
//
//  1. the prompt in the pipeline's template (a system message, the prompt as the user's), its
//     Qwen tokens and their embedding rows; the text graph (Qwen3-VL-8B's language model, its
//     last layer before the final norm, then the DiT's txt_in) in chunks of 64 positions; the
//     rows after the system message's are the DiT's text;
//  2. the DiT's text pass once a prompt (dit_text): the text is modulated at t = 0, so its keys
//     and values are the same at every step (diffusers' prefix KV cache);
//  3. from Gaussian noise, flow-matching Euler steps on the checkpoint's sigmas (Turbo's eight):
//     each step the DiT's image rows only (dit_in, dit_blocks, dit_out), attending to their own
//     keys and the text's;
//  4. the VAE's decoder on the latents.
//
// The export is for one image size (1024 x 1024 by default). Classifier-free guidance (Qwen-Image
// 2.1's base checkpoint with a negative prompt) is not supported yet.
package qwenimage

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
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// DitConfig is the DiT's shape (iron_qwenimage.DitConfig).
type DitConfig struct {
	Width     int     `json:"width"`
	Height    int     `json:"height"`
	Hidden    int     `json:"hidden"`
	Heads     int     `json:"heads"`
	HeadDim   int     `json:"head_dim"`
	Layers    int     `json:"layers"`
	Latent    int     `json:"latent"`
	Axes      []int   `json:"axes"`
	Theta     float64 `json:"theta"`
	Group     int     `json:"group"`
	TextRows  int     `json:"text_rows"`
	TextChunk int     `json:"text_chunk"`
	VAEScale  int     `json:"vae_scale"`
}

// GH and GW are the latents' rows and columns.
func (c DitConfig) GH() int { return c.Height / c.VAEScale }

// GW is the latents' columns.
func (c DitConfig) GW() int { return c.Width / c.VAEScale }

// Tokens is the image's rows of the DiT.
func (c DitConfig) Tokens() int { return c.GH() * c.GW() }

// TextConfig is the text graph's shape (iron_qwenimage.TextConfig).
type TextConfig struct {
	Rows    int     `json:"rows"`
	Cache   int     `json:"cache"`
	Hidden  int     `json:"hidden"`
	Vocab   int     `json:"vocab"`
	HeadDim int     `json:"head_dim"`
	Theta   float64 `json:"theta"`
}

// Scheduler is the export's FlowMatchEulerDiscreteScheduler settings (scheduler_config.json).
type Scheduler struct {
	Shift         float64  `json:"shift"`
	Dynamic       bool     `json:"use_dynamic_shifting"`
	BaseSeq       int      `json:"base_image_seq_len"`
	MaxSeq        int      `json:"max_image_seq_len"`
	BaseShift     float64  `json:"base_shift"`
	MaxShift      float64  `json:"max_shift"`
	ShiftTerminal *float64 `json:"shift_terminal"`
}

// Model is model.json of an export.
type Model struct {
	Name string `json:"name"`
	// SampleSigmas are the checkpoint's sigmas (model_index.json's, Turbo's eight), nil for a
	// linear grid.
	SampleSigmas []float64 `json:"sample_sigmas"`
	Scheduler    Scheduler `json:"scheduler"`
}

// StageFunc is told that stage i of n took d.
type StageFunc = anima.StageFunc

// StepFunc is told that step i (of n) took d.
type StepFunc func(i, n int, d time.Duration)

// IsExport reports whether dir is an export of tools/amd/iron_qwenimage.py or of Convert.
func IsExport(dir string) bool {
	if IsProgramExport(dir) {
		return true
	}
	for _, p := range []string{"model.json", "dit_text/manifest.json", "dit_blocks/manifest.json"} {
		if _, err := os.Stat(filepath.Join(dir, p)); err != nil {
			return false
		}
	}
	return true
}

// Pipeline is a loaded export.
type Pipeline struct {
	Dit   DitConfig
	Text  TextConfig
	Model Model

	dev      *hwnpu.Device
	dir      string
	tok      *tokenizer.Tokenizer
	embed    *os.File // bf16 rows of Text.Hidden
	template string   // with {} for the prompt
	drop     int      // the template's system message's tokens, not the DiT's
	vae      *anima.VAE
	prog     *progEngine // a Core ML export's

	// Logf, when set, is told what is done.
	Logf func(format string, args ...any)
	// Stage, when set, is told of each stage of the VAE as it ends.
	Stage StageFunc
	// ResidentDiT keeps every group's weights on the NPU instead of reading each group's into
	// one bank before it runs.
	ResidentDiT bool
}

func meta[T any](dir string, field string) (T, *iron.GraphManifest, error) {
	var x T
	m, err := iron.ReadGraphManifest(dir)
	if err != nil {
		return x, nil, err
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(m.Meta, &raw); err != nil {
		return x, m, err
	}
	err = json.Unmarshal(raw[field], &x)
	return x, m, err
}

// Open reads the export in dir; its graphs are loaded onto dev as they run.
func Open(ctx context.Context, dev *hwnpu.Device, dir string) (*Pipeline, error) {
	return OpenWith(ctx, dev, dir, nil)
}

// OpenWith is Open; an export of Convert (Core ML) runs its programs on dev's Neural Engine, or
// with the CPU interpreter when dev is nil, and shows each program on log (when not nil) as it
// loads (the first load compiles it).
func OpenWith(ctx context.Context, dev *hwnpu.Device, dir string, log io.Writer) (*Pipeline, error) {
	p := &Pipeline{dev: dev, dir: dir}
	fail := func(err error) (*Pipeline, error) {
		p.Close()
		return nil, fmt.Errorf("qwenimage: %w", err)
	}
	b, err := os.ReadFile(filepath.Join(dir, "model.json"))
	if err != nil {
		return fail(err)
	}
	if err := json.Unmarshal(b, &p.Model); err != nil {
		return fail(err)
	}
	if IsProgramExport(dir) {
		if p.prog, err = openPrograms(dev, dir, log); err != nil {
			return fail(err)
		}
		m := p.prog.m
		p.Dit, p.Text, p.template, p.drop = m.Dit, m.Text, m.Template, m.Drop
		td := filepath.Join(dir, "text")
		if p.tok, err = tokenizer.Load(filepath.Join(td, "tokenizer.json")); err != nil {
			return fail(err)
		}
		if p.embed, err = os.Open(filepath.Join(td, "embed.bin")); err != nil {
			return fail(err)
		}
		return p, nil
	}
	if p.Dit, _, err = meta[DitConfig](filepath.Join(dir, "dit_blocks"), "config"); err != nil {
		return fail(err)
	}
	td := filepath.Join(dir, "text")
	var tm *iron.GraphManifest
	if p.Text, tm, err = meta[TextConfig](td, "config"); err != nil {
		return fail(err)
	}
	var x struct {
		Template string `json:"template"`
		Drop     int    `json:"drop"`
	}
	if err := json.Unmarshal(tm.Meta, &x); err != nil || !strings.Contains(x.Template, "{}") {
		return fail(fmt.Errorf("text: no template in its manifest (%v)", err))
	}
	p.template, p.drop = x.Template, x.Drop
	if p.tok, err = tokenizer.Load(filepath.Join(td, "tokenizer.json")); err != nil {
		return fail(err)
	}
	if p.embed, err = os.Open(filepath.Join(td, "embed.bin")); err != nil {
		return fail(err)
	}
	if p.vae, err = anima.OpenVAE(ctx, dev, filepath.Join(dir, "vae")); err != nil {
		return fail(err)
	}
	// a stage loads beside the one before only when they hold 3 GiB or less together: the 1024 x
	// 1024 VAE's last stages (2 to 3 GiB each) would otherwise set the run's peak
	p.vae.PrefetchLimit = 3 << 30
	return p, nil
}

// Close releases what Open holds (no graph stays on the device between calls; a Core ML export's
// DiT is released).
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

// Tokens are the prompt's tokens in the template (an empty prompt is a space: Qwen has no
// <bos> for the encoder to read).
func (p *Pipeline) Tokens(prompt string) []int32 {
	if prompt == "" {
		prompt = " "
	}
	return p.tok.Encode(strings.Replace(p.template, "{}", prompt, 1))
}

// Cond is a prompt's conditioning: the text's keys and values of each group of the DiT's
// blocks (bf16, group x TextRows rows of hidden each; a Core ML export's as float32 in ptk and ptv),
// and how many rows count.
type Cond struct {
	tk, tv   [][]byte
	ptk, ptv [][]float32
	Rows     int
}

// Encode runs the text graph and the DiT's text pass on the prompt.
func (p *Pipeline) Encode(ctx context.Context, prompt string) (Cond, error) {
	c, err := p.EncodeAll(ctx, prompt)
	if err != nil {
		return Cond{}, err
	}
	return c[0], nil
}

// EncodeAll is Encode of each prompt with each graph loaded once (a prompt and its negative
// prompt: the text graph's 5.6 GB load is paid once).
func (p *Pipeline) EncodeAll(ctx context.Context, prompts ...string) ([]Cond, error) {
	t0 := time.Now()
	ids := make([][]int32, len(prompts))
	for k, pr := range prompts {
		ids[k] = p.Tokens(pr)
		T := len(ids[k]) - p.drop
		switch {
		case len(ids[k]) > p.Text.Cache:
			return nil, fmt.Errorf("qwenimage: %d tokens, the text graph holds %d", len(ids[k]), p.Text.Cache)
		case T > p.Dit.TextRows:
			return nil, fmt.Errorf("qwenimage: %d text rows, the DiT takes %d", T, p.Dit.TextRows)
		case T <= 0:
			return nil, errors.New("qwenimage: no text rows")
		}
	}
	if p.prog != nil {
		conds, err := p.prog.encode(ctx, p.embed, ids)
		if err != nil {
			return nil, fmt.Errorf("qwenimage: %w", err)
		}
		for k := range prompts {
			p.logf("text: %d tokens (%d rows for the DiT)", len(ids[k]), conds[k].Rows)
		}
		p.logf("the text in %.2f s", time.Since(t0).Seconds())
		return conds, nil
	}
	tLoad := time.Now()
	rows, err := p.text(ctx, ids)
	if err != nil {
		return nil, err
	}
	t1 := time.Now()
	conds, err := p.ditText(ctx, rows, ids)
	if err != nil {
		return nil, err
	}
	for k := range prompts {
		p.logf("text: %d tokens (%d rows for the DiT)", len(ids[k]), conds[k].Rows)
	}
	p.logf("Qwen3-VL-8B %.2f s (loading %.2f s), the DiT's text pass %.2f s", t1.Sub(t0).Seconds(), tLoad.Sub(t0).Seconds(), time.Since(t1).Seconds())
	return conds, nil
}

// text runs the text graphs on each prompt's ids, a chunk of Text.Rows at a time: Qwen3-VL-8B's
// layers a group at a time, read into one bank in turn (every prompt and chunk runs a group
// before the next is read), then txt_in (text_out). Each prompt's rows (bf16 through float32),
// len(ids) of Text.Hidden.
func (p *Pipeline) text(ctx context.Context, ids [][]int32) ([][]float32, error) {
	g, err := iron.OpenGraphWith(ctx, p.dev, filepath.Join(p.dir, "text"), iron.GraphOptions{SwapVariants: true, Barrier: true})
	if err != nil {
		return nil, err
	}
	defer g.Close()
	E, R := p.Text.Hidden, p.Text.Rows
	type chunk struct {
		x      []byte
		start  int
		angles []byte
	}
	chunks := make([][]chunk, len(ids))
	for k, ii := range ids {
		for at := 0; at < len(ii); at += R {
			x := make([]byte, 2*R*E)
			for i, id := range ii[at:min(at+R, len(ii))] {
				if id < 0 || int(id) >= p.Text.Vocab {
					return nil, fmt.Errorf("qwenimage: token %d past the vocabulary", id)
				}
				if _, err := p.embed.ReadAt(x[2*i*E:2*(i+1)*E], int64(id)*int64(2*E)); err != nil {
					return nil, err
				}
			}
			chunks[k] = append(chunks[k], chunk{x, at, toBF16(halfRope(at, R, p.Text.HeadDim, p.Text.Theta))})
		}
	}
	for gi := 0; gi < g.Variants(); gi++ {
		if err := g.Use(gi); err != nil {
			return nil, err
		}
		for k := range chunks {
			for _, ch := range chunks[k] {
				if err := g.SetInput("x", ch.x); err != nil {
					return nil, err
				}
				if err := g.SetInput("angles", ch.angles); err != nil {
					return nil, err
				}
				if err := g.Run(ctx, map[string]int64{"start": int64(ch.start)}); err != nil {
					return nil, fmt.Errorf("text layers group %d: %w", gi, err)
				}
				o, err := g.Output(g.Outputs()[0].Name)
				if err != nil {
					return nil, err
				}
				copy(ch.x, o)
			}
		}
	}
	g.Close()
	t, err := iron.OpenGraphWith(ctx, p.dev, filepath.Join(p.dir, "text_out"), iron.GraphOptions{Barrier: true})
	if err != nil {
		return nil, err
	}
	defer t.Close()
	out := make([][]float32, len(ids))
	for k := range chunks {
		for _, ch := range chunks[k] {
			if err := t.SetInput("x", ch.x); err != nil {
				return nil, err
			}
			if err := t.Run(ctx, nil); err != nil {
				return nil, err
			}
			o, err := t.Output(t.Outputs()[0].Name)
			if err != nil {
				return nil, err
			}
			n := min(R, len(ids[k])-ch.start)
			out[k] = append(out[k], fromBF16s(o[:2*n*E])...)
		}
	}
	return out, nil
}

// halfRope is Hugging Face's RoPE (two halves) at positions start..start+rows-1 in IRON's table,
// cos and sin interleaved (iron_anima.half_rope_table).
func halfRope(start, rows, dim int, theta float64) []float32 {
	out := make([]float32, rows*dim)
	for r := 0; r < rows; r++ {
		for k := 0; k < dim/2; k++ {
			a := float64(start+r) / math.Pow(theta, float64(2*k)/float64(dim))
			out[r*dim+2*k] = float32(math.Cos(a))
			out[r*dim+2*k+1] = float32(math.Sin(a))
		}
	}
	return out
}

// modulation runs dit_in on the timestep sigma alone (zero latents): the shared modulation's rows
// and norm_out's scale.
func (p *Pipeline) modulation(ctx context.Context, g *iron.Graph, patches []byte, sigma float64) (h, mod, scale []byte, err error) {
	if err = g.SetInput("patches", patches); err != nil {
		return
	}
	if err = g.SetInput("tproj", toBF16(timestepSinusoid(1000*sigma, 256))); err != nil {
		return
	}
	if err = g.Run(ctx, nil); err != nil {
		return
	}
	outs := g.Outputs()
	var o [3][]byte
	for i := range o {
		b, e := g.Output(outs[i].Name)
		if e != nil {
			return nil, nil, nil, e
		}
		o[i] = append([]byte(nil), b...)
	}
	return o[0], o[1], o[2], nil
}

// ditText runs the DiT's blocks on each prompt's text rows (txt_in's, after the system message's)
// at t = 0, a group at a time, in chunks of TextChunk rows: each block's keys and values of the
// rows. Every prompt runs a group before the next group's weights are read.
func (p *Pipeline) ditText(ctx context.Context, rows [][]float32, ids [][]int32) ([]Cond, error) {
	c := p.Dit
	E, R, TR := c.Hidden, c.TextChunk, c.TextRows
	in, err := iron.OpenGraphWith(ctx, p.dev, filepath.Join(p.dir, "dit_in"), iron.GraphOptions{Barrier: true})
	if err != nil {
		return nil, err
	}
	_, mod0, _, err := p.modulation(ctx, in, make([]byte, 2*c.Tokens()*c.Latent), 0)
	in.Close()
	if err != nil {
		return nil, err
	}
	g, err := iron.OpenGraphWith(ctx, p.dev, filepath.Join(p.dir, "dit_text"), iron.GraphOptions{SwapVariants: !p.ResidentDiT, Barrier: true})
	if err != nil {
		return nil, err
	}
	defer g.Close()
	groups := g.Variants()
	if groups*c.Group != c.Layers {
		return nil, fmt.Errorf("qwenimage: dit_text holds %d groups of %d blocks, the DiT %d", groups, c.Group, c.Layers)
	}
	outs := g.Outputs() // x, then k and v of each block
	if len(outs) != 1+2*c.Group {
		return nil, fmt.Errorf("qwenimage: dit_text gives %d outputs, not %d", len(outs), 1+2*c.Group)
	}
	if err := g.SetInput("mod", mod0); err != nil {
		return nil, err
	}
	// each prompt's chunks of rows
	xs := make([][][]byte, len(rows))
	conds := make([]Cond, len(rows))
	for k := range rows {
		T := len(ids[k]) - p.drop
		conds[k].Rows = T
		r := rows[k][p.drop*E:]
		for at := 0; at < T; at += R {
			x := make([]byte, 2*R*E)
			copy(x, toBF16(r[at*E:min(at+R, T)*E]))
			xs[k] = append(xs[k], x)
		}
	}
	// each chunk's RoPE: the text's positions p on all three axes
	angle := func(k int) []byte {
		pos := make([][3]float64, R)
		for i := range pos {
			q := float64(k*R + i)
			pos[i] = [3]float64{q, q, q}
		}
		return toBF16(ropeAngles(c, pos))
	}
	for gi := 0; gi < groups; gi++ {
		if err := g.Use(gi); err != nil {
			return nil, err
		}
		for k := range xs {
			tk, tv := make([]byte, 2*c.Group*TR*E), make([]byte, 2*c.Group*TR*E)
			for ch := range xs[k] {
				if err := g.SetInput("x", xs[k][ch]); err != nil {
					return nil, err
				}
				if err := g.SetInput("angles", angle(ch)); err != nil {
					return nil, err
				}
				if err := g.Run(ctx, map[string]int64{"start": int64(ch * R)}); err != nil {
					return nil, fmt.Errorf("dit_text blocks %d..%d: %w", gi*c.Group, (gi+1)*c.Group-1, err)
				}
				o, err := g.Output(outs[0].Name)
				if err != nil {
					return nil, err
				}
				copy(xs[k][ch], o)
				for j := 0; j < c.Group; j++ {
					at := 2 * (j*TR + ch*R) * E
					for s, dst := range [][]byte{tk, tv} {
						o, err := g.Output(outs[1+2*j+s].Name)
						if err != nil {
							return nil, err
						}
						copy(dst[at:at+2*R*E], o)
					}
				}
			}
			conds[k].tk, conds[k].tv = append(conds[k].tk, tk), append(conds[k].tv, tv)
		}
	}
	return conds, nil
}

// Sigmas are the steps' sigmas and the terminal 0: the checkpoint's (steps 0) or a linear grid
// of steps, through the scheduler's shift (FlowMatchEulerDiscreteScheduler.set_timesteps).
func (p *Pipeline) Sigmas(steps int) []float64 {
	s := p.Model.Scheduler
	sig := p.Model.SampleSigmas
	if steps > 0 || sig == nil {
		if steps <= 0 {
			steps = 40
		}
		sig = make([]float64, steps)
		for i := range sig {
			sig[i] = 1 + (1/float64(steps)-1)*float64(i)/float64(max(1, steps-1))
		}
	} else {
		sig = append([]float64(nil), sig...)
	}
	if s.Dynamic {
		m := (s.MaxShift - s.BaseShift) / float64(s.MaxSeq-s.BaseSeq)
		mu := float64(p.Dit.Tokens())*m + s.BaseShift - m*float64(s.BaseSeq)
		for i, t := range sig {
			sig[i] = math.Exp(mu) / (math.Exp(mu) + (1/t - 1))
		}
	} else if s.Shift != 0 && s.Shift != 1 {
		for i, t := range sig {
			sig[i] = s.Shift * t / (1 + (s.Shift-1)*t)
		}
	}
	if s.ShiftTerminal != nil && len(sig) > 0 {
		scale := (1 - sig[len(sig)-1]) / (1 - *s.ShiftTerminal)
		for i, t := range sig {
			sig[i] = 1 - (1-t)/scale
		}
	}
	return append(sig, 0)
}

// Noise returns Gaussian latents for seed, (latent, rows, columns) channel major.
func (p *Pipeline) Noise(seed uint64) []float32 {
	r := rand.New(rand.NewPCG(seed, 0x7177656e))
	out := make([]float32, p.Dit.Latent*p.Dit.Tokens())
	for i := range out {
		out[i] = float32(r.NormFloat64())
	}
	return out
}

// Sampling is how Generate steps: the solver, "euler" ("" too) or "dpm2m" (DPM-Solver++ 2M,
// anima.DPM2M), and the classifier-free guidance: its scale (1: none) and the sigmas it runs at
// (CFGMax 0: every step).
type Sampling struct {
	Solver         string
	Guidance       float64
	CFGMin, CFGMax float64
}

func (s Sampling) guided(sigma float64) bool {
	return s.Guidance != 1 && (s.CFGMax <= 0 || sigma >= s.CFGMin && sigma <= s.CFGMax)
}

// Generate runs the steps over sigmas (Sigmas) from the noise (latent, rows, columns) under cond,
// guided by neg (the negative prompt's, Encode) at the steps smp guides: the latents, the same
// layout. A guided step runs the prompt and the negative prompt back to back in each group of
// blocks, so the weights are read once a step.
func (p *Pipeline) Generate(ctx context.Context, cond Cond, neg *Cond, noise []float32, sigmas []float64, smp Sampling, step StepFunc) ([]float32, error) {
	c := p.Dit
	N, L := c.Tokens(), c.Latent
	if len(noise) != L*N {
		return nil, fmt.Errorf("qwenimage: %d noise values for latents of %d", len(noise), L*N)
	}
	if len(cond.tk) == 0 && len(cond.ptk) == 0 {
		return nil, errors.New("qwenimage: the DiT runs without a prompt (Encode)")
	}
	switch smp.Solver {
	case "", "euler", "dpm2m":
	default:
		return nil, fmt.Errorf("qwenimage: no solver %q (euler, dpm2m)", smp.Solver)
	}
	if smp.Guidance == 0 {
		smp.Guidance = 1
	}
	if smp.Guidance != 1 && neg == nil {
		return nil, errors.New("qwenimage: guidance without a negative prompt's conditioning")
	}
	if p.prog != nil {
		return p.generatePrograms(ctx, cond, neg, noise, sigmas, smp, step)
	}
	open := func(name string, swap bool) (*iron.Graph, error) {
		return iron.OpenGraphWith(ctx, p.dev, filepath.Join(p.dir, name), iron.GraphOptions{SwapVariants: swap, Barrier: true})
	}
	in, err := open("dit_in", false)
	if err != nil {
		return nil, err
	}
	defer in.Close()
	blocks, err := open("dit_blocks", !p.ResidentDiT)
	if err != nil {
		return nil, err
	}
	defer blocks.Close()
	out, err := open("dit_out", false)
	if err != nil {
		return nil, err
	}
	defer out.Close()
	if blocks.Variants() != len(cond.tk) {
		return nil, fmt.Errorf("qwenimage: dit_blocks holds %d groups, the text %d", blocks.Variants(), len(cond.tk))
	}
	conds := []Cond{cond}
	if neg != nil && smp.Guidance != 1 {
		conds = append(conds, *neg)
	}
	// each conditioning's image RoPE (the frame follows its text's length) and key count
	angles := make([][]byte, len(conds))
	for k, cd := range conds {
		angles[k] = toBF16(ropeAngles(c, imagePositions(c, cd.Rows)))
	}
	// x channel last (tokens x latent), as the DiT reads it
	x := make([]float32, N*L)
	for ch := 0; ch < L; ch++ {
		for t := 0; t < N; t++ {
			x[t*L+ch] = noise[ch*N+t]
		}
	}
	steps := len(sigmas) - 1
	var d0, d1 []float32
	var hLast float64
	for i := 0; i < steps; i++ {
		t0 := time.Now()
		h, mod, scale, err := p.modulation(ctx, in, toBF16(x), sigmas[i])
		if err != nil {
			return nil, err
		}
		if err := blocks.SetInput("mod", mod); err != nil {
			return nil, err
		}
		run := conds[:1]
		if smp.guided(sigmas[i]) {
			run = conds
		}
		hs := make([][]byte, len(run))
		for k := range hs {
			hs[k] = append([]byte(nil), h...)
		}
		for gi := 0; gi < blocks.Variants(); gi++ {
			if err := blocks.Use(gi); err != nil {
				return nil, err
			}
			for k, cd := range run {
				for _, s := range []struct {
					name string
					data []byte
				}{{"x", hs[k]}, {"tk", cd.tk[gi]}, {"tv", cd.tv[gi]}, {"angles", angles[k]}} {
					if err := blocks.SetInput(s.name, s.data); err != nil {
						return nil, err
					}
				}
				if err := blocks.Run(ctx, map[string]int64{"nk": int64(N + cd.Rows)}); err != nil {
					return nil, fmt.Errorf("qwenimage: step %d, blocks %d..%d: %w", i, gi*c.Group, (gi+1)*c.Group-1, err)
				}
				o, err := blocks.Output(blocks.Outputs()[0].Name)
				if err != nil {
					return nil, err
				}
				hs[k] = append(hs[k][:0], o...)
			}
		}
		vs := make([][]float32, len(run))
		for k := range run {
			if err := out.SetInput("x", hs[k]); err != nil {
				return nil, err
			}
			if err := out.SetInput("scale", scale); err != nil {
				return nil, err
			}
			if err := out.Run(ctx, nil); err != nil {
				return nil, err
			}
			o, err := out.Output(out.Outputs()[0].Name)
			if err != nil {
				return nil, err
			}
			vs[k] = fromBF16s(o)
		}
		v := vs[0]
		if len(vs) == 2 { // the pipeline's true_cfg_scale: neg + g (cond - neg)
			g := float32(smp.Guidance)
			for j := range v {
				v[j] = vs[1][j] + g*(v[j]-vs[1][j])
			}
		}
		if smp.Solver == "dpm2m" {
			hLast = anima.DPM2M(x, v, &d0, &d1, sigmas[i], sigmas[i+1], hLast)
		} else {
			d := float32(sigmas[i+1] - sigmas[i])
			for j := range x {
				x[j] += d * v[j]
			}
		}
		if step != nil {
			step(i, steps, time.Since(t0))
		}
	}
	res := make([]float32, N*L)
	for ch := 0; ch < L; ch++ {
		for t := 0; t < N; t++ {
			res[ch*N+t] = x[t*L+ch]
		}
	}
	return res, nil
}

// Decode turns the latents (latent, rows, columns) into the picture (RGBA). A Core ML export's DiT
// is released first.
func (p *Pipeline) Decode(ctx context.Context, latents []float32) (*image.RGBA, error) {
	imgs, err := p.DecodeAll(ctx, latents)
	if err != nil {
		return nil, err
	}
	return imgs[0], nil
}

// DecodeAll is Decode of several pictures' latents: a Core ML export runs each VAE stage on all of
// them, so that each stage is loaded once.
func (p *Pipeline) DecodeAll(ctx context.Context, latents ...[]float32) ([]*image.RGBA, error) {
	if p.prog != nil {
		if err := p.prog.closeDiT(); err != nil {
			return nil, err
		}
		imgs, err := p.prog.decode(ctx, latents, p.Stage)
		if err != nil {
			return nil, fmt.Errorf("qwenimage: %w", err)
		}
		return imgs, nil
	}
	p.vae.Stage = p.Stage
	var imgs []*image.RGBA
	for _, l := range latents {
		img, err := p.vae.Decode(ctx, l)
		if err != nil {
			return nil, err
		}
		imgs = append(imgs, img)
	}
	return imgs, nil
}

// Batched reports whether the pipeline is best run with every prompt's conditioning first, every
// picture's steps next and the decoding of all last (EncodeAll, Generate, DecodeAll): a Core ML
// export loads its programs for each phase, which takes longer than running them.
func (p *Pipeline) Batched() bool { return p.prog != nil }

// imagePositions are the image's rows' RoPE positions after a text of textLen rows (frame,
// height, width): the frame at textLen, the height and width centred on zero (QwenImage21Rope).
func imagePositions(c DitConfig, textLen int) [][3]float64 {
	h, w := c.GH(), c.GW()
	out := make([][3]float64, 0, h*w)
	for y := -(h - h/2); y < h/2; y++ {
		for x := -(w - w/2); x < w/2; x++ {
			out = append(out, [3]float64{float64(textLen), float64(y), float64(x)})
		}
	}
	return out
}

// ropeAngles is IRON's RoPE table of positions pos (rows x HeadDim, cos and sin interleaved),
// each axis's frequencies theta^(-2k / d) (iron_qwenimage.rope_angles).
func ropeAngles(c DitConfig, pos [][3]float64) []float32 {
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

// timestepSinusoid is QwenImage21TemporalTimesteps of t (1000 sigma): the cosines, then the sines.
func timestepSinusoid(t float64, dim int) []float32 {
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
		binary.LittleEndian.PutUint16(b[2*i:], bf16(x))
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

// bf16 rounds x to bfloat16 (to nearest, ties to even).
func bf16(x float32) uint16 {
	b := math.Float32bits(x)
	if b&0x7fffffff > 0x7f800000 {
		return uint16(b>>16) | 0x40
	}
	return uint16((b + 0x7fff + (b>>16)&1) >> 16)
}
