// Package sana runs Sana and Sana-Sprint (NVlabs, Efficient-Large-Model/*_diffusers), image
// models, on an NPU: on AMD's, the graphs tools/amd/iron_sana.py exports; on Apple's Neural
// Engine, the Core ML packages of internal/imagegen/sana/mil (Convert, Prepare). The host does
// what is not a tensor computation.
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
	"image"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
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
	Size          int     `json:"size"` // exports before width and height
	Width         int     `json:"width"`
	Height        int     `json:"height"`
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
// W and H are the picture's width and height.
func (c DitConfig) W() int {
	if c.Width == 0 {
		return c.Size
	}
	return c.Width
}

// H is the picture's height.
func (c DitConfig) H() int {
	if c.Height == 0 {
		return c.Size
	}
	return c.Height
}

// GW and GH are the latents' columns and rows.
func (c DitConfig) GW() int { return c.W() / c.VAEScale }

// GH is the latents' rows.
func (c DitConfig) GH() int { return c.H() / c.VAEScale }

// Tokens is the latent pixels; NTok the rows the graphs hold them in (whole 256, zeros past).
func (c DitConfig) Tokens() int { return c.GW() * c.GH() }

// NTok is Tokens rounded up to 256.
func (c DitConfig) NTok() int { return (c.Tokens() + 255) / 256 * 256 }

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

	eng   engine
	tok   *tokenizer.Tokenizer
	embed *os.File // bf16 rows of Text.Hidden
	bos   int32

	// ResidentDiT keeps every group's weights on the NPU instead of reading each group's
	// into one bank before it runs.
	ResidentDiT bool
	// KeepDiT keeps the DiT on the device through Decode, for the next picture's steps (it is
	// otherwise released before the VAE runs).
	KeepDiT bool
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

// engine runs the models of an export: the IRON graphs on AMD's NPU (ironEngine), or the
// Core ML programs Convert writes (progEngine).
type engine interface {
	// text runs Gemma 2 on x (Text.Rows x Text.Hidden: the embedding rows times sqrt(hidden),
	// zeros past them): its last hidden state, the same shape.
	text(ctx context.Context, x []float32) ([]float32, error)
	// caption runs the caption projection on y (Dit.TextRows x Dit.Caption): the DiT's ctx
	// (Dit.TextRows x Dit.Hidden).
	caption(ctx context.Context, y []float32) ([]float32, error)
	// openDiT loads the DiT for velocity; resident is ResidentDiT.
	openDiT(ctx context.Context, resident bool) error
	// velocity runs the DiT on the latents channel last (tokens x latent) at the timestep's
	// sinusoid tproj (and the guidance's, gproj), once for each conditioning: the outputs,
	// channel last.
	velocity(ctx context.Context, patches, tproj, gproj []float32, conds []Cond) ([][]float32, error)
	// closeDiT releases the DiT.
	closeDiT() error
	// decode runs the VAE on the latents (latent, rows, columns), telling stage (when not
	// nil) of each stage as it ends.
	decode(ctx context.Context, latents []float32, stage StageFunc) (*image.RGBA, error)
	close() error
}

// Cond is a prompt's conditioning: the caption (TextRows x hidden) and how many of its rows
// count.
type Cond struct {
	ctx  []float32
	Rows int // the rows of the caption that count (the rest masked)
}

// IsExport reports whether dir is an export of Sana: of tools/amd/iron_sana.py, or of Convert.
func IsExport(dir string) bool { return isIronExport(dir) || IsProgramExport(dir) }

// Open reads the export in dir; its graphs or programs are loaded onto dev as they run (Core ML
// programs with the CPU interpreter when dev is nil). log, when not nil, is shown each program of
// a Core ML export as it loads (the first load compiles it for the NPU).
func Open(ctx context.Context, dev *hwnpu.Device, dir string, log io.Writer) (*Pipeline, error) {
	p := &Pipeline{}
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
	if IsProgramExport(dir) {
		var e *progEngine
		e, p.Dit, p.Text, err = openPrograms(dev, dir, log)
		if e != nil {
			p.eng = e
		}
	} else {
		if dev == nil {
			return fail(errors.New("an IRON export runs on AMD's NPU only"))
		}
		var e *ironEngine
		e, p.Dit, p.Text, err = openIron(dev, dir)
		if e != nil {
			p.eng = e
		}
	}
	if err != nil {
		return fail(err)
	}
	td := filepath.Join(dir, "text")
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
	return p, nil
}

// Close releases the device memory.
func (p *Pipeline) Close() error {
	var err error
	if p.eng != nil {
		err = p.eng.close()
	}
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
	return p.tokens(prompt, true)
}

// tokens is Tokens, or without the instruction (the negative prompt's: padded to 300).
func (p *Pipeline) tokens(prompt string, instruction bool) (ids []int32, total int) {
	prompt = strings.ToLower(strings.TrimSpace(prompt))
	if !instruction {
		ids = append([]int32{p.bos}, p.tok.Encode(prompt)...)
		return ids[:min(len(ids), MaxSequence)], MaxSequence
	}
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
	return p.encode(ctx, prompt, true)
}

// EncodeNegative is Sana's negative prompt's conditioning (Gemma 2 without the instruction).
// The empty prompt needs no run: its one row, Gemma's first (<bos>), is any prompt's first row,
// so it is cond's caption with one row counting.
func (p *Pipeline) EncodeNegative(ctx context.Context, prompt string, cond Cond) (Cond, error) {
	if strings.TrimSpace(prompt) == "" {
		return Cond{ctx: cond.ctx, Rows: 1}, nil
	}
	return p.encode(ctx, prompt, false)
}

func (p *Pipeline) encode(ctx context.Context, prompt string, instruction bool) (Cond, error) {
	t0 := time.Now()
	ids, total := p.tokens(prompt, instruction)
	E, R := p.Text.Hidden, p.Text.Rows
	if len(ids) > R {
		return Cond{}, fmt.Errorf("sana: %d tokens, the text graph takes %d", len(ids), R)
	}
	// the embedding rows times sqrt(hidden) (Gemma's normalizer, 48 in bf16)
	x := make([]float32, R*E)
	row := make([]byte, 2*E)
	norm := float32(math.Sqrt(float64(E)))
	for i, id := range ids {
		if _, err := p.embed.ReadAt(row, int64(id)*int64(2*E)); err != nil {
			return Cond{}, err
		}
		for j := 0; j < E; j++ {
			// rounded to bf16 as the reference's embedding in bf16
			x[i*E+j] = fromBF16(bf16(fromBF16(binary.LittleEndian.Uint16(row[2*j:])) * norm))
		}
	}
	h, err := p.eng.text(ctx, x)
	if err != nil {
		return Cond{}, err
	}
	// the rows Sana keeps: the first and the last 299 of the padded length, of which those
	// before len(ids) count (a prefix: the first, then a run)
	first := total - (MaxSequence - 1)
	keep := 1 + max(0, len(ids)-first)
	C := p.Dit.Caption
	y := make([]float32, p.Dit.TextRows*C)
	copy(y[:C], h[:E])
	for j := 1; j < keep; j++ {
		copy(y[j*C:(j+1)*C], h[(first+j-1)*E:(first+j)*E])
	}
	t1 := time.Now()
	cond, err := p.eng.caption(ctx, y)
	if err != nil {
		return Cond{}, err
	}
	if instruction {
		p.dump("cond.bin", cond)
	}
	p.logf("text: %d tokens (%d kept) in %.2f s (Gemma 2 %.2f s)", len(ids), keep, time.Since(t0).Seconds(), t1.Sub(t0).Seconds())
	return Cond{ctx: cond, Rows: keep}, nil
}

// Noise returns n Gaussian latents for seed, each (latent, rows, columns) channel major.
func (p *Pipeline) Noise(seed uint64, n int) [][]float32 {
	r := rand.New(rand.NewPCG(seed, 0x73616e61))
	out := make([][]float32, n)
	for k := range out {
		out[k] = make([]float32, p.Dit.Latent*p.Dit.Tokens())
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
// a step, on neg (EncodeNegative) and on cond).
func (p *Pipeline) Generate(ctx context.Context, cond, neg Cond, noises [][]float32, steps int, guidance float64, step StepFunc) ([]float32, error) {
	if len(noises) < p.NoiseCount(steps) {
		return nil, fmt.Errorf("sana: %d noises for %d steps", len(noises), steps)
	}
	if cond.ctx == nil {
		return nil, errors.New("sana: the DiT runs without a prompt (Encode)")
	}
	if err := p.eng.openDiT(ctx, p.ResidentDiT); err != nil {
		return nil, err
	}
	if p.Model.Sprint {
		return p.sprint(ctx, cond, noises, steps, guidance, step)
	}
	if neg.ctx == nil {
		neg = Cond{ctx: cond.ctx, Rows: 1}
	}
	return p.dpm(ctx, cond, neg, noises[0], steps, guidance, step)
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
		outs, err := p.velocity(ctx, in, TimestepSinusoid(st, 256), gproj, []Cond{cond})
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

func (p *Pipeline) dpm(ctx context.Context, cond, neg Cond, noise []float32, steps int, guidance float64, step StepFunc) ([]float32, error) {
	m := p.Model
	d := NewDPM(steps, m.FlowShift, m.TrainSteps)
	x := append([]float32(nil), noise...)
	for i := 0; i < steps; i++ {
		t0 := time.Now()
		outs, err := p.velocity(ctx, x, TimestepSinusoid(d.Timesteps[i], 256), nil, []Cond{neg, cond})
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

// velocity runs the DiT on the latents x (latent, rows, columns) at the timestep's sinusoid
// tproj (and the guidance's, gproj), once for each conditioning in conds: the outputs, the
// latents' layout.
func (p *Pipeline) velocity(ctx context.Context, x []float32, tproj, gproj []float32, conds []Cond) ([][]float32, error) {
	n, L := p.Dit.Tokens(), p.Dit.Latent
	patches := make([]float32, n*L)
	for ch := 0; ch < L; ch++ {
		for t := 0; t < n; t++ {
			patches[t*L+ch] = x[ch*n+t]
		}
	}
	vs, err := p.eng.velocity(ctx, patches, tproj, gproj, conds)
	if err != nil {
		return nil, err
	}
	res := make([][]float32, len(vs))
	for k, v := range vs {
		out := make([]float32, n*L)
		for ch := 0; ch < L; ch++ {
			for t := 0; t < n; t++ {
				out[ch*n+t] = v[t*L+ch]
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
