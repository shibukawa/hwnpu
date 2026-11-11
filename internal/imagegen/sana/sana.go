// Package sana runs Sana (NVlabs, Efficient-Large-Model/*_diffusers: Sana 600M, SANA 1.5 1.6B),
// image models, on an NPU: on AMD's, the graphs tools/amd/iron_sana.py exports; on Apple's Neural
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
//  3. from Gaussian noise, the steps: flow DPM-Solver++ with classifier-free guidance (the DiT
//     on the prompt and on the empty prompt: its one row, Gemma's first row, is the prompt's
//     own, so the second call is the first row alone);
//  4. DC-AE's decoder on the latents.
//
// The export is for one image size (1024 x 1024 by default). Sana-Sprint (the guidance
// embedded, SCM steps) is not supported: its pictures were far below Sana's (design.md 16.148).
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
	"github.com/shibukawa/hwnpu/internal/imagegen/sana/mil"
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
	Size     int `json:"size"` // exports before width and height
	Width    int `json:"width"`
	Height   int `json:"height"`
	Hidden   int `json:"hidden"`
	Layers   int `json:"layers"`
	Latent   int `json:"latent"`
	TextRows int `json:"text_rows"`
	Group    int `json:"group"`
	VAEScale int `json:"vae_scale"`
	Caption  int `json:"caption"`
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
	// Prefix is the first rows of the instruction the graph holds as keys and values
	// (iron_sana's Prefix, whole chunks), 0 for an export that computes every row in one call;
	// with it the graph is a chunk of Rows positions from a call's start, and Cache the
	// positions its keys and values hold.
	Prefix int `json:"prefix"`
	Cache  int `json:"cache"`
}

// Model is model.json of an export: which Sana it is and its scheduler's settings.
type Model struct {
	Name string `json:"name"`
	// Sprint marks an export of Sana-Sprint, which Open refuses (mil.ErrSprint).
	Sprint   bool    `json:"sprint"`
	Steps    int     `json:"steps"`    // the default
	Guidance float64 `json:"guidance"` // the default
	// DPMSolverMultistepScheduler's
	FlowShift  float64 `json:"flow_shift"`
	TrainSteps int     `json:"num_train_timesteps"`
}

// StageFunc is told that stage i of n took d.
type StageFunc func(i, n int, d time.Duration)

// Pipeline is a loaded Sana export.
type Pipeline struct {
	Dit  DitConfig
	Text TextConfig
	// prefixIDs are the tokens of the instruction the text graph holds, prefixOut their last
	// hidden states (bf16, rows of Text.Hidden).
	prefixIDs []int32
	prefixOut []byte
	Model     Model

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
	// CFGMin and CFGMax, when CFGMax > 0, bound the sigmas at which a step runs the guidance;
	// the other steps run the DiT once, on the prompt (guidance in a limited interval,
	// Kynkäänniemi et al. 2024; Anima's anima.Sampling). Open sets DefaultCFGSigmas; CFGMax 0
	// guides every step.
	CFGMin, CFGMax float64
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
	// openText loads Gemma 2: its run takes x (Text.Rows x Text.Hidden: the embedding rows
	// times sqrt(hidden), zeros past them) at position start (0 without Text.Prefix) and gives
	// the last hidden state, the same shape.
	openText(ctx context.Context) (part, error)
	// openCaption loads the caption projection: its run takes y (Dit.TextRows x Dit.Caption)
	// and gives the DiT's ctx (Dit.TextRows x Dit.Hidden).
	openCaption(ctx context.Context) (part, error)
	// openDiT loads the DiT for velocity; resident is ResidentDiT.
	openDiT(ctx context.Context, resident bool) error
	// velocity runs the DiT on the latents channel last (tokens x latent) at the timestep's
	// sinusoid tproj, once for each conditioning: the outputs, channel last.
	velocity(ctx context.Context, patches, tproj []float32, conds []Cond) ([][]float32, error)
	// closeDiT releases the DiT.
	closeDiT() error
	// decode runs the VAE on the latents (latent, rows, columns), telling stage (when not
	// nil) of each stage as it ends.
	decode(ctx context.Context, latents []float32, stage StageFunc) (*image.RGBA, error)
	close() error
}

// part is a loaded graph or program the pipeline runs more than once.
type part interface {
	run(ctx context.Context, in []float32, start int) ([]float32, error)
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
	p := &Pipeline{CFGMin: DefaultCFGSigmas[0], CFGMax: DefaultCFGSigmas[1]}
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
	if p.Model.Sprint {
		return nil, mil.ErrSprint
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
	if p.Text.Prefix > 0 {
		// an IRON export whose text graph holds the instruction's first rows (iron_sana's Prefix)
		mt, err := iron.ReadGraphManifest(td)
		if err != nil {
			return fail(err)
		}
		var x struct {
			Prefix []int32 `json:"prefix"`
		}
		if err := json.Unmarshal(mt.Meta, &x); err != nil || len(x.Prefix) < p.Text.Prefix {
			return fail(fmt.Errorf("text: the prefix's %d tokens are not in its manifest (%v)", p.Text.Prefix, err))
		}
		p.prefixIDs = x.Prefix
		if p.prefixOut, err = os.ReadFile(filepath.Join(td, "prefix_out.bin")); err != nil {
			return fail(err)
		}
		if len(p.prefixOut) != 2*len(p.prefixIDs)*p.Text.Hidden {
			return fail(fmt.Errorf("text: prefix_out.bin is %d bytes, not %d rows", len(p.prefixOut), len(p.prefixIDs)))
		}
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
	c, err := p.encode(ctx, []textPrompt{{prompt, true}})
	if err != nil {
		return Cond{}, err
	}
	return c[0], nil
}

// EncodeNegative is Sana's negative prompt's conditioning (Gemma 2 without the instruction).
// The empty prompt needs no run: its one row, Gemma's first (<bos>), is any prompt's first row,
// so it is cond's caption with one row counting.
func (p *Pipeline) EncodeNegative(ctx context.Context, prompt string, cond Cond) (Cond, error) {
	if strings.TrimSpace(prompt) == "" {
		return Cond{ctx: cond.ctx, Rows: 1}, nil
	}
	c, err := p.encode(ctx, []textPrompt{{prompt, false}})
	if err != nil {
		return Cond{}, err
	}
	return c[0], nil
}

// EncodeBoth is Encode's and EncodeNegative's conditionings with each graph loaded once (the
// text graph's 1.7 GB is most of the time of a short prompt).
func (p *Pipeline) EncodeBoth(ctx context.Context, prompt, negative string) (cond, neg Cond, err error) {
	if strings.TrimSpace(negative) == "" {
		if cond, err = p.Encode(ctx, prompt); err != nil {
			return
		}
		neg, err = p.EncodeNegative(ctx, negative, cond)
		return
	}
	c, err := p.encode(ctx, []textPrompt{{prompt, true}, {negative, false}})
	if err != nil {
		return Cond{}, Cond{}, err
	}
	return c[0], c[1], nil
}

// textPrompt is a prompt for the text graph, after the instruction (Sana's prompt) or alone
// (its negative prompt).
type textPrompt struct {
	text        string
	instruction bool
}

// encode runs Gemma 2 on each prompt, then the caption projection on each, loading each part
// once.
func (p *Pipeline) encode(ctx context.Context, prompts []textPrompt) ([]Cond, error) {
	t0 := time.Now()
	E, R, C := p.Text.Hidden, p.Text.Rows, p.Dit.Caption
	type run struct {
		ids          []int32
		total, start int
		keep         int
		y            []float32 // the rows Sana keeps, for the caption
	}
	runs := make([]run, len(prompts))
	for k, tp := range prompts {
		ids, total := p.tokens(tp.text, tp.instruction)
		start := textStart(ids, p.prefixIDs, p.Text.Prefix, R)
		if p.Text.Prefix > 0 {
			if len(ids) > p.Text.Cache {
				return nil, fmt.Errorf("sana: %d tokens, the text graph holds %d", len(ids), p.Text.Cache)
			}
		} else if len(ids) > R {
			return nil, fmt.Errorf("sana: %d tokens, the text graph takes %d", len(ids), R)
		}
		runs[k] = run{ids: ids, total: total, start: start}
	}
	text, err := p.eng.openText(ctx)
	if err != nil {
		return nil, err
	}
	closed := false
	closeText := func() {
		if !closed {
			text.close() // before the caption loads
			closed = true
		}
	}
	defer closeText()
	tLoad := time.Now()
	x := make([]float32, R*E)
	row := make([]byte, 2*E)
	norm := float32(math.Sqrt(float64(E)))
	for k := range runs {
		r := &runs[k]
		ids, start := r.ids, r.start
		// the embedding rows times sqrt(hidden) (Gemma's normalizer, 48 in bf16, rounded to
		// bf16 as the reference's embedding), a chunk at a time from start (one call of every
		// row without a prefix)
		var h []float32
		for at := start; at < len(ids); at += R {
			clear(x)
			for i, id := range ids[at:min(at+R, len(ids))] {
				if _, err := p.embed.ReadAt(row, int64(id)*int64(2*E)); err != nil {
					return nil, err
				}
				for j := 0; j < E; j++ {
					x[i*E+j] = fromBF16(bf16(fromBF16(binary.LittleEndian.Uint16(row[2*j:])) * norm))
				}
			}
			out, err := text.run(ctx, x, at)
			if err != nil {
				return nil, err
			}
			h = append(h, out...)
		}
		// hidden is row i's last hidden state: the prefix's below start, the graph's from it
		hidden := func(i int) []float32 {
			if i < start {
				return fromBF16s(p.prefixOut[2*i*E : 2*(i+1)*E])
			}
			return h[(i-start)*E : (i-start+1)*E]
		}
		// the rows Sana keeps: the first and the last 299 of the padded length, of which those
		// before len(ids) count (a prefix: the first, then a run)
		first := r.total - (MaxSequence - 1)
		r.keep = 1 + max(0, len(ids)-first)
		r.y = make([]float32, p.Dit.TextRows*C)
		copy(r.y[:C], hidden(0))
		for j := 1; j < r.keep; j++ {
			copy(r.y[j*C:(j+1)*C], hidden(first+j-1))
		}
	}
	closeText()
	t1 := time.Now()
	cap, err := p.eng.openCaption(ctx)
	if err != nil {
		return nil, err
	}
	defer cap.close()
	conds := make([]Cond, len(runs))
	for k, r := range runs {
		out, err := cap.run(ctx, r.y, 0)
		if err != nil {
			return nil, err
		}
		conds[k] = Cond{ctx: out, Rows: r.keep}
		if prompts[k].instruction {
			p.dump("gemma.bin", r.y)
			p.dump("cond.bin", out)
		} else {
			p.dump("gemma_neg.bin", r.y)
		}
	}
	for k, r := range runs {
		what := "text"
		if !prompts[k].instruction {
			what = "negative"
		}
		p.logf("%s: %d tokens (%d kept, %d computed)", what, len(r.ids), r.keep, len(r.ids)-r.start)
	}
	p.logf("Gemma 2 %.2f s (loading %.2f s, running %.2f s), the caption %.2f s", t1.Sub(t0).Seconds(), tLoad.Sub(t0).Seconds(),
		t1.Sub(tLoad).Seconds(), time.Since(t1).Seconds())
	return conds, nil
}

// Noise returns Gaussian latents for seed, (latent, rows, columns) channel major.
func (p *Pipeline) Noise(seed uint64) []float32 {
	r := rand.New(rand.NewPCG(seed, 0x73616e61))
	out := make([]float32, p.Dit.Latent*p.Dit.Tokens())
	for i := range out {
		out[i] = float32(r.NormFloat64())
	}
	return out
}

// StepFunc is told that step i (of n) took d.
type StepFunc func(i, n int, d time.Duration)

// Generate runs the steps from the noise under cond: the latents. guidance is the
// classifier-free guidance's scale: the DiT twice a step, on neg (EncodeNegative; the empty
// prompt's when it is the zero Cond) and on cond.
func (p *Pipeline) Generate(ctx context.Context, cond, neg Cond, noise []float32, steps int, guidance float64, step StepFunc) ([]float32, error) {
	if len(noise) != p.Dit.Latent*p.Dit.Tokens() {
		return nil, fmt.Errorf("sana: %d noise values for latents of %d", len(noise), p.Dit.Latent*p.Dit.Tokens())
	}
	if cond.ctx == nil {
		return nil, errors.New("sana: the DiT runs without a prompt (Encode)")
	}
	if err := p.eng.openDiT(ctx, p.ResidentDiT); err != nil {
		return nil, err
	}
	if neg.ctx == nil {
		neg = Cond{ctx: cond.ctx, Rows: 1}
	}
	return p.dpm(ctx, cond, neg, noise, steps, guidance, step)
}

// DefaultCFGSigmas are the sigmas whose steps run the guidance by default: 14 of the 20 steps.
// Against every step guided, 3 prompts x 2 seeds at 1024 x 1024 on the M1 drew the same pictures
// (Sana 600M's latents 0.996-0.998); from 0.8 a small object went missing (design.md 16.157).
var DefaultCFGSigmas = [2]float64{0.6, 1}

// Guided reports whether the step at sigma runs the guidance (CFGMin, CFGMax).
func (p *Pipeline) Guided(sigma float64) bool {
	return p.CFGMax <= 0 || (sigma >= p.CFGMin && sigma <= p.CFGMax)
}

// Sigmas are the scheduler's sigmas of steps steps (steps+1, the last 0).
func (p *Pipeline) Sigmas(steps int) []float64 {
	return NewDPM(steps, p.Model.FlowShift, p.Model.TrainSteps).Sigmas
}

func (p *Pipeline) dpm(ctx context.Context, cond, neg Cond, noise []float32, steps int, guidance float64, step StepFunc) ([]float32, error) {
	m := p.Model
	d := NewDPM(steps, m.FlowShift, m.TrainSteps)
	x := append([]float32(nil), noise...)
	for i := 0; i < steps; i++ {
		t0 := time.Now()
		tproj := TimestepSinusoid(d.Timesteps[i], 256)
		var v []float32
		if guidance == 1 || !p.Guided(d.Sigmas[i]) {
			// the DiT once, on the prompt: u + 1 (c - u) is c
			outs, err := p.velocity(ctx, x, tproj, []Cond{cond})
			if err != nil {
				return nil, fmt.Errorf("sana: step %d: %w", i, err)
			}
			v = outs[0]
			p.dump(fmt.Sprintf("out%d_1.bin", i), v)
		} else {
			outs, err := p.velocity(ctx, x, tproj, []Cond{neg, cond})
			if err != nil {
				return nil, fmt.Errorf("sana: step %d: %w", i, err)
			}
			u, c := outs[0], outs[1]
			p.dump(fmt.Sprintf("out%d_0.bin", i), u)
			p.dump(fmt.Sprintf("out%d_1.bin", i), c)
			v = make([]float32, len(x))
			for j := range v {
				v[j] = u[j] + float32(guidance)*(c[j]-u[j])
			}
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
// tproj, once for each conditioning in conds: the outputs, the latents' layout.
func (p *Pipeline) velocity(ctx context.Context, x []float32, tproj []float32, conds []Cond) ([][]float32, error) {
	n, L := p.Dit.Tokens(), p.Dit.Latent
	patches := make([]float32, n*L)
	for ch := 0; ch < L; ch++ {
		for t := 0; t < n; t++ {
			patches[t*L+ch] = x[ch*n+t]
		}
	}
	vs, err := p.eng.velocity(ctx, patches, tproj, conds)
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

// textStart is the first position the text graph computes for the tokens ids: of the tokens they
// share with the instruction's (prefix) from the first, the whole chunks of chunk positions within
// the held ones the graph holds (the instruction's keys and values), less one token so that a call
// computes a row at least. A negative prompt shares <bos> alone and starts at 0; so does any
// prompt of an export without held rows.
func textStart(ids, prefix []int32, held, chunk int) int {
	if held <= 0 {
		return 0
	}
	same := 0
	for same < len(prefix) && same < len(ids)-1 && ids[same] == prefix[same] {
		same++
	}
	return min(same, held) / chunk * chunk
}
