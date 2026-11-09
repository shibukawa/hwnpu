// Package sana runs Sana and Sana-Sprint (NVlabs, Efficient-Large-Model/*_diffusers), image
// models, on AMD's NPU: the graphs tools/amd/iron_sana.py exports. The host does what is not a
// tensor computation.
//
// Sana is a linear DiT (0.6B or 1.6B) over DC-AE's latents, 32 channels at a 32nd of the
// image's sides, with Gemma 2 2B reading the prompt. A picture is:
//
//  1. the prompt, lowered, after diffusers' instruction ("Given a user prompt, generate an
//     'Enhanced prompt'..."), its Gemma tokens and their embedding rows;
//  2. the text graph (Gemma 2): the last hidden state, of which Sana keeps the first row and
//     the last 299 of the padded prompt (the pad rows masked), and the caption graph (the DiT's
//     caption projection) on those 300 rows;
//  3. from Gaussian noise, the steps: Sana-Sprint's SCM steps (1 to 4, the guidance scale
//     embedded, a new noise each step) or Sana's flow DPM-Solver++ with classifier-free
//     guidance (the DiT on the prompt and on the empty prompt: its one row, Gemma's first
//     row, is the prompt's own, so the second call is the first row alone);
//  4. DC-AE's decoder on the latents.
//
// The export is for one image size (1024 x 1024 by default).
package sana

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// Instruction is diffusers' complex_human_instruction for Sana's pipelines, joined by new lines;
// the prompt follows it.
var Instruction = strings.Join([]string{
	"Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
	"- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
	"- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
	"Here are examples of how to transform or refine prompts:",
	"- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
	"- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
	"Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
	"User Prompt: ",
}, "\n")

// MaxSequence is the rows of the prompt Sana keeps (the pipelines' max_sequence_length).
const MaxSequence = 300

// DitConfig is the DiT's shape (iron_sana.DitConfig).
type DitConfig struct {
	Size          int     `json:"size"`
	Hidden        int     `json:"hidden"`
	Layers        int     `json:"layers"`
	Latent        int     `json:"latent"`
	Guidance      bool    `json:"guidance"`
	GuidanceScale float64 `json:"guidance_scale"`
	TextRows      int     `json:"text_rows"`
	Group         int     `json:"group"`
	VAEScale      int     `json:"vae_scale"`
	Caption       int     `json:"caption"`
}

// Grid is the tokens down (and across).
func (c DitConfig) Grid() int { return c.Size / c.VAEScale }

// TextConfig is Gemma 2's shape (iron_sana.TextConfig).
type TextConfig struct {
	Rows   int `json:"rows"`
	Hidden int `json:"hidden"`
	Vocab  int `json:"vocab"`
}

// Model is model.json of an export: which Sana it is and its scheduler's settings.
type Model struct {
	Name     string  `json:"name"`
	Sprint   bool    `json:"sprint"`
	Steps    int     `json:"steps"`    // the default
	Guidance float64 `json:"guidance"` // the default
	// Sana-Sprint (SCMScheduler)
	SigmaData float64 `json:"sigma_data"`
	MaxT      float64 `json:"max_timesteps"`
	// Sana (DPMSolverMultistepScheduler)
	FlowShift  float64 `json:"flow_shift"`
	TrainSteps int     `json:"num_train_timesteps"`
}

// StageFunc is told that stage i of n took d.
type StageFunc func(i, n int, d time.Duration)

// Pipeline is a loaded Sana export.
type Pipeline struct {
	Dit   DitConfig
	Text  TextConfig
	Model Model

	dev   *hwnpu.Device
	dir   string
	tok   *tokenizer.Tokenizer
	embed *os.File // bf16 rows of Text.Hidden
	bos   int32
	vae   *VAE

	ditIn, ditBlocks, ditOut *iron.Graph // open while the steps run
	cond                     []byte      // the caption, bf16 (TextRows, Hidden)

	// ResidentDiT keeps every group's weights on the NPU instead of reading each group's
	// into one bank before it runs.
	ResidentDiT bool
	// Logf, when set, is told what is done.
	Logf func(format string, args ...any)
	// Stage, when set, is told of each stage of the VAE as it ends.
	Stage StageFunc
	// Dump, when set, is a directory each step's DiT outputs are written to (raw float32,
	// out<i>_<k>.bin for the call k of step i: Sana's 0 the empty prompt, 1 the prompt), to
	// check against tools/ref/ref_sana.py.
	Dump string
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

// Cond is a prompt's conditioning: the caption rows (in the pipeline) and how many count.
type Cond struct {
	Rows int // the rows of the caption that count (the rest masked)
}

func meta[T any](m *iron.GraphManifest) (T, error) {
	var x struct {
		Config T `json:"config"`
	}
	err := json.Unmarshal(m.Meta, &x)
	return x.Config, err
}

// IsExport reports whether dir is an export of tools/amd/iron_sana.py.
func IsExport(dir string) bool {
	_, err := os.Stat(filepath.Join(dir, "model.json"))
	if err != nil {
		return false
	}
	_, err = os.Stat(filepath.Join(dir, "caption", "manifest.json"))
	return err == nil
}

// Open reads the export in dir; its graphs are loaded onto dev as they run.
func Open(ctx context.Context, dev *hwnpu.Device, dir string) (*Pipeline, error) {
	p := &Pipeline{dev: dev, dir: dir}
	fail := func(err error) (*Pipeline, error) {
		p.Close()
		return nil, fmt.Errorf("sana: %w", err)
	}
	b, err := os.ReadFile(filepath.Join(dir, "model.json"))
	if err != nil {
		return fail(err)
	}
	if err := json.Unmarshal(b, &p.Model); err != nil {
		return fail(err)
	}
	mb, err := iron.ReadGraphManifest(filepath.Join(dir, "dit_blocks"))
	if err != nil {
		return fail(err)
	}
	if p.Dit, err = meta[DitConfig](mb); err != nil {
		return fail(err)
	}
	if n, v := p.Dit.Layers/p.Dit.Group, max(1, len(mb.Variants)); v != n {
		return fail(fmt.Errorf("dit_blocks holds %d groups, the DiT %d", v, n))
	}
	td := filepath.Join(dir, "text")
	mt, err := iron.ReadGraphManifest(td)
	if err != nil {
		return fail(err)
	}
	if p.Text, err = meta[TextConfig](mt); err != nil {
		return fail(err)
	}
	if p.tok, err = tokenizer.Load(filepath.Join(td, "tokenizer.json")); err != nil {
		return fail(err)
	}
	bos, ok := p.tok.ID("<bos>")
	if !ok {
		return fail(errors.New("the tokenizer has no <bos>"))
	}
	p.bos = bos
	if p.embed, err = os.Open(filepath.Join(td, "embed.bin")); err != nil {
		return fail(err)
	}
	if p.vae, err = OpenVAE(dev, filepath.Join(dir, "vae")); err != nil {
		return fail(err)
	}
	return p, nil
}

// Close releases the device memory.
func (p *Pipeline) Close() error {
	err := p.closeDiT()
	if p.embed != nil {
		p.embed.Close()
	}
	return err
}

func (p *Pipeline) logf(format string, args ...any) {
	if p.Logf != nil {
		p.Logf(format, args...)
	}
}

// Tokens returns the Gemma tokens of the prompt after the instruction (<bos> first), cut as
// the pipeline's tokenizer cuts them, and the length it pads them to.
func (p *Pipeline) Tokens(prompt string) (ids []int32, total int) {
	prompt = strings.ToLower(strings.TrimSpace(prompt))
	chi := 1 + len(p.tok.Encode(Instruction))
	total = chi + MaxSequence - 2
	ids = append([]int32{p.bos}, p.tok.Encode(Instruction+prompt)...)
	if len(ids) > total {
		ids = ids[:total]
	}
	return ids, total
}

// Encode runs Gemma 2 and the caption projection on the prompt.
func (p *Pipeline) Encode(ctx context.Context, prompt string) (Cond, error) {
	t0 := time.Now()
	ids, total := p.Tokens(prompt)
	E, R := p.Text.Hidden, p.Text.Rows
	if len(ids) > R {
		return Cond{}, fmt.Errorf("sana: %d tokens, the text graph takes %d", len(ids), R)
	}
	// the embedding rows times sqrt(hidden) (Gemma's normalizer, 48 in bf16)
	x := make([]byte, 2*R*E)
	row := make([]byte, 2*E)
	norm := float32(math.Sqrt(float64(E)))
	for i, id := range ids {
		if _, err := p.embed.ReadAt(row, int64(id)*int64(2*E)); err != nil {
			return Cond{}, err
		}
		for j := 0; j < E; j++ {
			v := fromBF16(binary.LittleEndian.Uint16(row[2*j:])) * norm
			binary.LittleEndian.PutUint16(x[2*(i*E+j):], bf16(v))
		}
	}
	g, err := iron.OpenGraph(ctx, p.dev, filepath.Join(p.dir, "text"))
	if err != nil {
		return Cond{}, err
	}
	if err := g.SetInput("x", x); err != nil {
		g.Close()
		return Cond{}, err
	}
	if err := g.Run(ctx, nil); err != nil {
		g.Close()
		return Cond{}, err
	}
	h, err := g.Output(g.Outputs()[0].Name)
	if err != nil {
		g.Close()
		return Cond{}, err
	}
	// the rows Sana keeps: the first and the last 299 of the padded length, of which those
	// before len(ids) count (a prefix: the first, then a run)
	first := total - (MaxSequence - 1)
	keep := 1 + max(0, len(ids)-first)
	C := p.Dit.Caption
	y := make([]byte, 2*p.Dit.TextRows*C)
	copy(y[:2*C], h[:2*E])
	for j := 1; j < keep; j++ {
		copy(y[2*j*C:2*(j+1)*C], h[2*(first+j-1)*E:2*(first+j)*E])
	}
	g.Close()
	t1 := time.Now()
	cg, err := iron.OpenGraph(ctx, p.dev, filepath.Join(p.dir, "caption"))
	if err != nil {
		return Cond{}, err
	}
	defer cg.Close()
	if err := cg.SetInput("y", y); err != nil {
		return Cond{}, err
	}
	if err := cg.Run(ctx, nil); err != nil {
		return Cond{}, err
	}
	out, err := cg.Output(cg.Outputs()[0].Name)
	if err != nil {
		return Cond{}, err
	}
	p.cond = append(p.cond[:0], out...)
	p.dump("cond.bin", fromBF16s(out))
	p.logf("text: %d tokens (%d kept) in %.2f s (Gemma 2 %.2f s)", len(ids), keep, time.Since(t0).Seconds(), t1.Sub(t0).Seconds())
	return Cond{Rows: keep}, nil
}

// Noise returns n Gaussian latents for seed, each (latent, grid, grid) channel major.
func (p *Pipeline) Noise(seed uint64, n int) [][]float32 {
	g := p.Dit.Grid()
	r := rand.New(rand.NewPCG(seed, 0x73616e61))
	out := make([][]float32, n)
	for k := range out {
		out[k] = make([]float32, p.Dit.Latent*g*g)
		for i := range out[k] {
			out[k][i] = float32(r.NormFloat64())
		}
	}
	return out
}

// NoiseCount is how many noises Generate takes for steps: Sana-Sprint draws a new one each step.
func (p *Pipeline) NoiseCount(steps int) int {
	if p.Model.Sprint {
		return steps + 1
	}
	return 1
}

// StepFunc is told that step i (of n) took d.
type StepFunc func(i, n int, d time.Duration)

// Generate runs the steps from the noises (NoiseCount of them) under cond: the latents.
// guidance is the guidance scale (Sana-Sprint: embedded; Sana: classifier-free, the DiT twice
// a step).
func (p *Pipeline) Generate(ctx context.Context, cond Cond, noises [][]float32, steps int, guidance float64, step StepFunc) ([]float32, error) {
	if len(noises) < p.NoiseCount(steps) {
		return nil, fmt.Errorf("sana: %d noises for %d steps", len(noises), steps)
	}
	if err := p.openDiT(ctx); err != nil {
		return nil, err
	}
	if p.Model.Sprint {
		return p.sprint(ctx, cond, noises, steps, guidance, step)
	}
	return p.dpm(ctx, cond, noises[0], steps, guidance, step)
}

func (p *Pipeline) sprint(ctx context.Context, cond Cond, noises [][]float32, steps int, guidance float64, step StepFunc) ([]float32, error) {
	m := p.Model
	sd := float32(m.SigmaData)
	T := SCMTimesteps(steps, m.MaxT)
	x := make([]float32, len(noises[0]))
	for i := range x {
		x[i] = noises[0][i] * sd
	}
	gproj := TimestepSinusoid(guidance*p.Dit.GuidanceScale, 256)
	in := make([]float32, len(x))
	var x0 []float32
	for i := 0; i < steps; i++ {
		t0 := time.Now()
		s, t := T[i], T[i+1]
		scale, st := SCMInput(s)
		for j := range x {
			in[j] = x[j] / sd * float32(scale)
		}
		outs, err := p.velocity(ctx, in, TimestepSinusoid(st, 256), gproj, []int{cond.Rows})
		if err != nil {
			return nil, fmt.Errorf("sana: step %d: %w", i, err)
		}
		out := outs[0]
		p.dump(fmt.Sprintf("out%d_0.bin", i), out)
		a, b := float32(1-2*st), float32(1-2*st+2*st*st)
		cs, ss := float32(math.Cos(s)), float32(math.Sin(s))
		ct, stt := float32(math.Cos(t)), float32(math.Sin(t))
		x0 = make([]float32, len(x))
		for j := range x {
			pred := (a*in[j] + b*out[j]) / float32(scale) * sd
			x0[j] = cs*x[j] - ss*pred
			x[j] = ct*x0[j] + stt*noises[i+1][j]*sd
		}
		if step != nil {
			step(i, steps, time.Since(t0))
		}
	}
	for j := range x0 {
		x0[j] /= sd
	}
	return x0, nil
}

func (p *Pipeline) dpm(ctx context.Context, cond Cond, noise []float32, steps int, guidance float64, step StepFunc) ([]float32, error) {
	m := p.Model
	d := NewDPM(steps, m.FlowShift, m.TrainSteps)
	x := append([]float32(nil), noise...)
	for i := 0; i < steps; i++ {
		t0 := time.Now()
		// the empty negative prompt is Gemma's first row alone, the prompt's own first row
		outs, err := p.velocity(ctx, x, TimestepSinusoid(d.Timesteps[i], 256), nil, []int{1, cond.Rows})
		if err != nil {
			return nil, fmt.Errorf("sana: step %d: %w", i, err)
		}
		u, c := outs[0], outs[1]
		p.dump(fmt.Sprintf("out%d_0.bin", i), u)
		p.dump(fmt.Sprintf("out%d_1.bin", i), c)
		v := make([]float32, len(x))
		for j := range v {
			v[j] = u[j] + float32(guidance)*(c[j]-u[j])
		}
		x = d.Step(i, v, x)
		if step != nil {
			step(i, steps, time.Since(t0))
		}
	}
	return x, nil
}

// TimestepSinusoid is diffusers' Timesteps(dim, flip_sin_to_cos=True, downscale_freq_shift=0) of
// t: the cosines, then the sines.
func TimestepSinusoid(t float64, dim int) []float32 {
	half := dim / 2
	out := make([]float32, dim)
	for i := 0; i < half; i++ {
		a := t * math.Exp(-math.Log(10000)*float64(i)/float64(half))
		out[i] = float32(math.Cos(a))
		out[half+i] = float32(math.Sin(a))
	}
	return out
}

func (p *Pipeline) closeDiT() error {
	var errs []error
	for _, g := range []**iron.Graph{&p.ditIn, &p.ditBlocks, &p.ditOut} {
		if *g != nil {
			errs = append(errs, (*g).Close())
			*g = nil
		}
	}
	return errors.Join(errs...)
}

func (p *Pipeline) openDiT(ctx context.Context) error {
	if p.ditBlocks != nil {
		return nil
	}
	if p.cond == nil {
		return errors.New("sana: the DiT runs without a prompt (Encode)")
	}
	for _, x := range []struct {
		g    **iron.Graph
		name string
	}{{&p.ditIn, "dit_in"}, {&p.ditBlocks, "dit_blocks"}, {&p.ditOut, "dit_out"}} {
		g, err := iron.OpenGraphWith(ctx, p.dev, filepath.Join(p.dir, x.name), iron.GraphOptions{SwapVariants: !p.ResidentDiT})
		if err != nil {
			p.closeDiT()
			return err
		}
		*x.g = g
	}
	if err := p.ditBlocks.SetInput("ctx", p.cond); err != nil {
		p.closeDiT()
		return err
	}
	return nil
}

// velocity runs the DiT on the latents x (latent, grid, grid) at the timestep's sinusoid tproj
// (and the guidance's, gproj), once for each count of the caption's rows in rows: the outputs,
// the latents' layout. Each group of blocks runs every call before the next group's weights
// are read.
func (p *Pipeline) velocity(ctx context.Context, x []float32, tproj, gproj []float32, rows []int) ([][]float32, error) {
	c := p.Dit
	n := c.Grid() * c.Grid()
	L := c.Latent
	patches := make([]float32, n*L)
	for ch := 0; ch < L; ch++ {
		for t := 0; t < n; t++ {
			patches[t*L+ch] = x[ch*n+t]
		}
	}
	g := p.ditIn
	if err := g.SetInput("patches", toBF16(patches)); err != nil {
		return nil, err
	}
	if err := g.SetInput("tproj", toBF16(tproj)); err != nil {
		return nil, err
	}
	if c.Guidance {
		if err := g.SetInput("gproj", toBF16(gproj)); err != nil {
			return nil, err
		}
	}
	if err := g.Run(ctx, nil); err != nil {
		return nil, err
	}
	outs := g.Outputs() // x, t6, emb
	h, err := g.Output(outs[0].Name)
	if err != nil {
		return nil, err
	}
	t6, err := g.Output(outs[1].Name)
	if err != nil {
		return nil, err
	}
	emb, err := g.Output(outs[2].Name)
	if err != nil {
		return nil, err
	}
	hs := make([][]byte, len(rows))
	for k := range hs {
		hs[k] = append([]byte(nil), h...)
	}
	b := p.ditBlocks
	if err := b.SetInput("t6", t6); err != nil {
		return nil, err
	}
	if err := p.ditOut.SetInput("emb", emb); err != nil {
		return nil, err
	}
	for gi := 0; gi < b.Variants(); gi++ {
		if err := b.Use(gi); err != nil {
			return nil, err
		}
		for k, nk := range rows {
			if err := b.SetInput("x", hs[k]); err != nil {
				return nil, err
			}
			if err := b.Run(ctx, map[string]int64{"nk": int64(nk)}); err != nil {
				return nil, fmt.Errorf("blocks %d..%d: %w", gi*c.Group, (gi+1)*c.Group-1, err)
			}
			o, err := b.Output(b.Outputs()[0].Name)
			if err != nil {
				return nil, err
			}
			hs[k] = append(hs[k][:0], o...)
		}
	}
	res := make([][]float32, len(rows))
	for k := range rows {
		if err := p.ditOut.SetInput("x", hs[k]); err != nil {
			return nil, err
		}
		if err := p.ditOut.Run(ctx, nil); err != nil {
			return nil, err
		}
		v, err := p.ditOut.Output(p.ditOut.Outputs()[0].Name)
		if err != nil {
			return nil, err
		}
		vs := fromBF16s(v)
		out := make([]float32, n*L)
		for ch := 0; ch < L; ch++ {
			for t := 0; t < n; t++ {
				out[ch*n+t] = vs[t*L+ch]
			}
		}
		res[k] = out
	}
	return res, nil
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
		out[i] = fromBF16(binary.LittleEndian.Uint16(b[2*i:]))
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

func fromBF16(h uint16) float32 { return math.Float32frombits(uint32(h) << 16) }
