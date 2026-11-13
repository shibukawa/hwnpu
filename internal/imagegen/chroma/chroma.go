// Package chroma runs Chroma1 (lodestones/Chroma1-Flash, Chroma1-HD), an image model, on an NPU:
// on AMD's, the graphs tools/amd/iron_chroma.py (the DiT) and iron_chroma_parts.py (T5-XXL and the
// VAE) export (ironEngine); on Apple's Neural Engine, the Core ML packages of
// internal/imagegen/chroma/mil that Convert writes (progEngine).
//
// Chroma is FLUX.1 schnell's DiT (19 double-stream and 38 single-stream blocks) whose blocks'
// modulations come from one FFN of the timestep (the approximator: 344 rows), with T5-XXL reading
// the prompt (512 rows, the pads masked past the prompt and one pad) and FLUX's VAE. A picture is:
//
//  1. the prompt's T5 tokens (</s> last), their embedding rows, T5 and context_embedder: the DiT's
//     text rows;
//  2. from Gaussian noise, flow-matching steps (Heun by default: Chroma1-Flash's 8 steps, the DiT
//     twice a step but the last): each DiT call the approximator's rows of its timestep (mods),
//     the latents' patches embedded, the double and single blocks over [image rows, text rows],
//     a group at a time, and the final norm and projection; with classifier-free guidance
//     (Chroma1-HD) the DiT runs the negative prompt's rows too, in each group after the prompt's;
//  3. the VAE's decoder on the latents.
//
// The export is for one image size. Chroma1-HD (the undistilled full model; Flash bakes its guidance
// in) runs classifier-free guidance with Generate's Sampling.
package chroma

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"strconv"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/imagegen/anima"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// Config is the DiT's shape (iron_chroma.ChromaConfig).
type Config struct {
	Width       int `json:"width"`
	Height      int `json:"height"`
	TextRows    int `json:"text_rows"`
	Hidden      int `json:"hidden"`
	Latent      int `json:"latent"`
	T5          int `json:"t5"`
	Doubles     int `json:"doubles"`
	Singles     int `json:"singles"`
	DoubleGroup int `json:"double_group"`
	SingleGroup int `json:"single_group"`
	Mods        int `json:"mods"`
	ModRows     int `json:"mod_rows"`
	ApproxIn    int `json:"approx_in"`
}

// GH and GW are the patches' rows and columns (2 x 2 latent pixels each).
func (c Config) GH() int { return c.Height / 16 }

// GW is the patches' columns.
func (c Config) GW() int { return c.Width / 16 }

// Tokens is the image's rows of the DiT.
func (c Config) Tokens() int { return c.GH() * c.GW() }

// Rows is the DiT's rows: the image's, then the text's.
func (c Config) Rows() int { return c.Tokens() + c.TextRows }

// StepFunc is told that step i (of n) took d.
type StepFunc func(i, n int, d time.Duration)

// IsExport reports whether dir is an export of Chroma: of tools/amd/iron_chroma.py, or of
// Convert.
func IsExport(dir string) bool { return isIronExport(dir) || IsProgramExport(dir) }

// Model is model.json of an export: the checkpoint, its scheduler's shift and its defaults.
type Model struct {
	Name      string `json:"name"`
	Scheduler struct {
		Shift float64 `json:"shift"`
	} `json:"scheduler"`
	Steps    int     `json:"steps"`
	Guidance float64 `json:"guidance"`
	Sampler  string  `json:"sampler"`
	// CFGSigmas, [lo, hi], are the sigmas the guidance runs at by default (none: every step)
	CFGSigmas []float64 `json:"cfg_sigmas"`
}

// engine runs the parts of an export.
type engine interface {
	// encode gives each prompt's text rows from its T5 tokens' embedding rows x (TextRows x the
	// T5's width, the pad's row past the tokens) and the tokens' count n (</s> included).
	encode(ctx context.Context, xs [][]float32, ns []int) ([]Cond, error)
	// openDiT loads the DiT for the steps, which run at sigmas.
	openDiT(ctx context.Context, sigmas []float64) error
	// velocity runs the DiT on the latents x (tokens x 64) at sigma for each conditioning, back
	// to back in each group of blocks: the velocities.
	velocity(ctx context.Context, conds []Cond, x []float32, sigma float64) ([][]float32, error)
	closeDiT() error
	// decode runs the VAE on the latents (16, H/8, W/8).
	decode(ctx context.Context, z []float32, stage anima.StageFunc) (*image.RGBA, error)
	close() error
}

// Pipeline is a loaded export.
type Pipeline struct {
	C     Config
	Model Model
	eng   engine
	tok   *tokenizer.Unigram
	embed *os.File // T5's bf16 rows
	t5dim int
	// embedScale multiplies the embedding rows (the Core ML programs take them divided by
	// mil.T5Scale)
	embedScale float32

	// KeepDiT keeps the DiT loaded after Generate, for the next picture's steps (Close
	// releases it).
	KeepDiT bool
	// Logf, when set, is told what is done.
	Logf func(format string, args ...any)
	// Stage, when set, is told of each stage of the VAE as it ends.
	Stage anima.StageFunc
}

// Open reads the export in dir: on AMD's NPU an export of tools/amd/iron_chroma.py, on Apple's
// Neural Engine (or, with dev nil, the CPU interpreter) one of Convert. log, when not nil, is
// shown each Core ML program as it loads (the first load compiles it for the NPU).
func Open(ctx context.Context, dev *hwnpu.Device, dir string, log io.Writer) (*Pipeline, error) {
	p := &Pipeline{embedScale: 1}
	fail := func(err error) (*Pipeline, error) {
		p.Close()
		return nil, fmt.Errorf("chroma: %w", err)
	}
	var err error
	var td string
	if IsProgramExport(dir) {
		var e *progEngine
		e, err = openPrograms(dev, dir, log, &p.C, &p.Model, p.logf)
		if e != nil {
			p.eng = e
			p.t5dim = e.m.T5.Hidden
			p.embedScale = e.m.EmbedScale
			td = e.textDir()
		}
	} else {
		if dev == nil {
			return fail(errors.New("an IRON export runs on AMD's NPU only"))
		}
		var e *ironEngine
		e, err = openIron(ctx, dev, dir, &p.C, &p.Model, p.logf)
		if e != nil {
			p.eng = e
			p.t5dim = e.t5dim
			td = e.textDir()
		}
	}
	if err != nil {
		return fail(err)
	}
	if p.tok, err = tokenizer.LoadUnigram(td + "/tokenizer.json"); err != nil {
		return fail(err)
	}
	if p.embed, err = os.Open(td + "/embed.bin"); err != nil {
		return fail(err)
	}
	return p, nil
}

// Close releases what Open holds.
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

// Cond is a prompt's conditioning: the DiT's text rows (TextRows x hidden; IRON's bf16, the
// programs' float32) and the keys they give (the prompt's tokens and one pad).
type Cond struct {
	ctx  []byte
	rows []float32
	Keys int
}

// Encode runs T5 and context_embedder on the prompt.
func (p *Pipeline) Encode(ctx context.Context, prompt string) (Cond, error) {
	c, err := p.EncodeAll(ctx, prompt)
	if err != nil {
		return Cond{}, err
	}
	return c[0], nil
}

// EncodeAll is Encode of each prompt with each part loaded once (a prompt and its negative
// prompt: T5's load is paid once).
func (p *Pipeline) EncodeAll(ctx context.Context, prompts ...string) ([]Cond, error) {
	R, E := p.C.TextRows, p.t5dim
	row := func(id int) ([]float32, error) {
		b := make([]byte, 2*E)
		if _, err := p.embed.ReadAt(b, int64(id)*int64(2*E)); err != nil {
			return nil, err
		}
		v := fromBF16s(b)
		for i := range v {
			v[i] *= p.embedScale
		}
		return v, nil
	}
	pad, err := row(0) // T5's <pad>
	if err != nil {
		return nil, err
	}
	xs := make([][]float32, len(prompts))
	ns := make([]int, len(prompts))
	for k, pr := range prompts {
		ids := p.tok.Encode(pr)
		if len(ids) > R { // transformers truncates the text and keeps </s> last
			ids = append(ids[:R-1], ids[len(ids)-1])
		}
		x := make([]float32, R*E)
		for i := 0; i < R; i++ {
			if i < len(ids) {
				r, err := row(int(ids[i]))
				if err != nil {
					return nil, err
				}
				copy(x[i*E:], r)
			} else {
				copy(x[i*E:], pad)
			}
		}
		xs[k], ns[k] = x, len(ids)
		p.logf("text: %d tokens", len(ids))
	}
	conds, err := p.eng.encode(ctx, xs, ns)
	if err != nil {
		return nil, err
	}
	for k := range conds {
		// the prompt's tokens and one pad (the pipeline's mask)
		conds[k].Keys = min(ns[k]+1, R)
		if n, err := strconv.Atoi(os.Getenv("HWNPU_CHROMA_KEYS")); err == nil && n > 0 { // a test of the text's mask
			conds[k].Keys = min(n, R)
		}
	}
	return conds, nil
}

// Sigmas are the steps' sigmas and the terminal 0: the pipeline's linear grid (Chroma1-Flash's
// scheduler has no shift).
func Sigmas(steps int) []float64 { return ShiftedSigmas(steps, 1) }

// ShiftedSigmas are Sigmas through the scheduler's shift (Chroma1-HD's 3): shift s / (1 + (shift -
// 1) s), as FlowMatchEulerDiscreteScheduler without dynamic shifting.
func ShiftedSigmas(steps int, shift float64) []float64 {
	s := make([]float64, steps+1)
	for i := 0; i < steps; i++ {
		t := 1 + (1/float64(steps)-1)*float64(i)/float64(max(1, steps-1))
		if shift != 0 && shift != 1 {
			t = shift * t / (1 + (shift-1)*t)
		}
		s[i] = t
	}
	return s
}

// Sigmas are the export's sigmas for steps (0: its default): its scheduler's shift.
func (p *Pipeline) Sigmas(steps int) []float64 {
	if steps <= 0 {
		steps = p.Model.Steps
	}
	return ShiftedSigmas(steps, p.Model.Scheduler.Shift)
}

// Noise returns Gaussian latents for seed, the DiT's packed layout (tokens x 64: a 2 x 2 patch
// of the 16 channels, channel major).
func (p *Pipeline) Noise(seed uint64) []float32 {
	r := rand.New(rand.NewPCG(seed, 0x6368726f))
	out := make([]float32, p.C.Tokens()*p.C.Latent)
	for i := range out {
		out[i] = float32(r.NormFloat64())
	}
	return out
}

// ModInput is the approximator's input of timestep t (1000 sigma): rows rows of in values (64),
// each of the first mods rows the timestep's sinusoid (16), a zero guidance's (16) and its own
// index's (32, of 1000 k), zeros past them.
func ModInput(t float64, mods, rows, in int) []float32 {
	x := make([]float32, rows*in)
	ts, g := sinusoid(t, 16), sinusoid(0, 16)
	for k := 0; k < mods; k++ {
		row := x[k*in:]
		copy(row, ts)
		copy(row[16:], g)
		copy(row[32:], sinusoid(1000*float64(k), 32))
	}
	return x
}

func (p *Pipeline) modInput(t float64) []float32 {
	return ModInput(t, p.C.Mods, p.C.ModRows, p.C.ApproxIn)
}

// Sampling is how Generate steps: the solver ("heun", "euler" or "dpm2m": DPM-Solver++ 2M,
// anima.DPM2M) and the classifier-free guidance: its scale (1: none) and the sigmas it runs at
// (CFGMax 0: every step).
type Sampling struct {
	Solver         string
	Guidance       float64
	CFGMin, CFGMax float64
}

func (s Sampling) guided(sigma float64) bool {
	return s.Guidance != 1 && (s.CFGMax <= 0 || sigma >= s.CFGMin && sigma <= s.CFGMax)
}

// Generate runs the steps over sigmas (Sigmas) from the noise (Noise's layout) under cond, guided
// by neg at the steps smp guides: Heun's (the DiT twice a step but the last, as
// FlowMatchHeunDiscreteScheduler), Euler's or DPM-Solver++ 2M's.
func (p *Pipeline) Generate(ctx context.Context, cond Cond, neg *Cond, noise []float32, sigmas []float64, smp Sampling, step StepFunc) ([]float32, error) {
	c := p.C
	if len(noise) != c.Tokens()*c.Latent {
		return nil, fmt.Errorf("chroma: %d noise values for latents of %d", len(noise), c.Tokens()*c.Latent)
	}
	if cond.ctx == nil && cond.rows == nil {
		return nil, errors.New("chroma: the DiT runs without a prompt (Encode)")
	}
	switch smp.Solver {
	case "heun", "euler", "dpm2m":
	default:
		return nil, fmt.Errorf("chroma: no solver %q (heun, euler, dpm2m)", smp.Solver)
	}
	if smp.Guidance == 0 {
		smp.Guidance = 1
	}
	if smp.Guidance != 1 && neg == nil {
		return nil, errors.New("chroma: guidance without a negative prompt's conditioning")
	}
	n := len(sigmas) - 1
	// the sigmas the DiT runs at: each step's, and Heun's next but the last
	var at []float64
	for i := 0; i < n; i++ {
		at = append(at, sigmas[i])
		if smp.Solver == "heun" && sigmas[i+1] != 0 {
			at = append(at, sigmas[i+1])
		}
	}
	if err := p.eng.openDiT(ctx, at); err != nil {
		return nil, err
	}
	if !p.KeepDiT {
		defer p.eng.closeDiT()
	}
	vel := func(x []float32, sigma float64) ([]float32, error) {
		conds := []Cond{cond}
		if smp.guided(sigma) {
			conds = append(conds, *neg)
		}
		vs, err := p.eng.velocity(ctx, conds, x, sigma)
		if err != nil {
			return nil, err
		}
		v := vs[0]
		if len(vs) == 2 { // neg + g (cond - neg)
			g := float32(smp.Guidance)
			for j := range v {
				v[j] = vs[1][j] + g*(v[j]-vs[1][j])
			}
		}
		return v, nil
	}
	x := append([]float32(nil), noise...)
	var d0, d1 []float32
	var hLast float64
	for i := 0; i < n; i++ {
		t0 := time.Now()
		dt := float32(sigmas[i+1] - sigmas[i])
		v1, err := vel(x, sigmas[i])
		if err != nil {
			return nil, fmt.Errorf("chroma: step %d: %w", i, err)
		}
		switch {
		case smp.Solver == "dpm2m":
			hLast = anima.DPM2M(x, v1, &d0, &d1, sigmas[i], sigmas[i+1], hLast)
		case smp.Solver == "heun" && sigmas[i+1] != 0:
			pred := make([]float32, len(x))
			for j := range x {
				pred[j] = x[j] + dt*v1[j]
			}
			v2, err := vel(pred, sigmas[i+1])
			if err != nil {
				return nil, fmt.Errorf("chroma: step %d: %w", i, err)
			}
			for j := range x {
				x[j] += dt * (v1[j] + v2[j]) / 2
			}
		default:
			for j := range x {
				x[j] += dt * v1[j]
			}
		}
		if step != nil {
			step(i, n, time.Since(t0))
		}
	}
	return x, nil
}

// Unpack is the packed latents (Noise's layout) as the VAE takes them, (16, H/8, W/8): FLUX's
// _unpack_latents.
func Unpack(x []float32, gh, gw, latent int) []float32 {
	H, W := 2*gh, 2*gw
	z := make([]float32, 16*H*W)
	for py := 0; py < gh; py++ {
		for px := 0; px < gw; px++ {
			t := (py*gw + px) * latent
			for ch := 0; ch < 16; ch++ {
				for dy := 0; dy < 2; dy++ {
					for dx := 0; dx < 2; dx++ {
						z[(ch*H+2*py+dy)*W+2*px+dx] = x[t+ch*4+dy*2+dx]
					}
				}
			}
		}
	}
	return z
}

// Decode turns the latents (Noise's layout) into the picture.
func (p *Pipeline) Decode(ctx context.Context, x []float32) (*image.RGBA, error) {
	c := p.C
	return p.eng.decode(ctx, Unpack(x, c.GH(), c.GW(), c.Latent), p.Stage)
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
