// Package chroma runs Chroma1 (lodestones/Chroma1-Flash), an image model, on AMD's NPU: the
// graphs tools/amd/iron_chroma.py (the DiT) and iron_chroma_parts.py (T5-XXL and the VAE) export.
//
// Chroma is FLUX.1 schnell's DiT (19 double-stream and 38 single-stream blocks) whose blocks'
// modulations come from one FFN of the timestep (the approximator: 344 rows), with T5-XXL reading
// the prompt (512 rows, the pads masked past the prompt and one pad) and FLUX's VAE. A picture is:
//
//  1. the prompt's T5 tokens (</s> last), their embedding rows, the t5 graph and the ctx graph
//     (context_embedder);
//  2. from Gaussian noise, flow-matching steps (Heun by default: Chroma1-Flash's 8 steps, the DiT
//     twice a step but the last): each DiT call the approximator's rows of its timestep (mods),
//     the latents' patches (dit_in), the double and single blocks over [image rows, text rows],
//     a group at a time, and dit_out;
//  3. the VAE's decoder on the latents.
//
// The export is for one image size. Chroma1-HD (the full model, of which Flash bakes the guidance
// in) runs classifier-free guidance with Generate's Sampling.
package chroma

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/imagegen/anima"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
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

// IsExport reports whether dir is an export of tools/amd/iron_chroma.py.
func IsExport(dir string) bool {
	for _, p := range []string{"doubles/manifest.json", "singles/manifest.json", "mods/manifest.json", "t5/manifest.json"} {
		if _, err := os.Stat(filepath.Join(dir, p)); err != nil {
			return false
		}
	}
	return true
}

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

// Pipeline is a loaded export.
type Pipeline struct {
	C     Config
	Model Model
	dev   *hwnpu.Device
	dir   string
	tok   *tokenizer.Unigram
	embed *os.File // T5's bf16 rows
	t5dim int
	vae   *anima.VAE

	// Logf, when set, is told what is done.
	Logf func(format string, args ...any)
	// Stage, when set, is told of each stage of the VAE as it ends.
	Stage anima.StageFunc
}

// Open reads the export in dir.
func Open(ctx context.Context, dev *hwnpu.Device, dir string) (*Pipeline, error) {
	p := &Pipeline{dev: dev, dir: dir}
	fail := func(err error) (*Pipeline, error) {
		p.Close()
		return nil, fmt.Errorf("chroma: %w", err)
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
	p.Model = Model{Name: "Chroma1-Flash", Steps: 8, Guidance: 1, Sampler: "heun"}
	if b, err := os.ReadFile(filepath.Join(dir, "model.json")); err == nil {
		if err := json.Unmarshal(b, &p.Model); err != nil {
			return fail(err)
		}
	}
	tm, err := iron.ReadGraphManifest(filepath.Join(dir, "t5"))
	if err != nil {
		return fail(err)
	}
	var t struct {
		Config struct {
			Hidden int `json:"hidden"`
			Rows   int `json:"rows"`
		} `json:"config"`
	}
	if err := json.Unmarshal(tm.Meta, &t); err != nil || t.Config.Rows != p.C.TextRows {
		return fail(fmt.Errorf("t5: %d rows for the DiT's %d (%v)", t.Config.Rows, p.C.TextRows, err))
	}
	p.t5dim = t.Config.Hidden
	if p.tok, err = tokenizer.LoadUnigram(filepath.Join(dir, "t5", "tokenizer.json")); err != nil {
		return fail(err)
	}
	if p.embed, err = os.Open(filepath.Join(dir, "t5", "embed.bin")); err != nil {
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

// Close releases what Open holds.
func (p *Pipeline) Close() error {
	if p.embed != nil {
		p.embed.Close()
	}
	return nil
}

func (p *Pipeline) logf(format string, args ...any) {
	if p.Logf != nil {
		p.Logf(format, args...)
	}
}

func (p *Pipeline) open(ctx context.Context, name string, swap bool) (*iron.Graph, error) {
	return iron.OpenGraphWith(ctx, p.dev, filepath.Join(p.dir, name), iron.GraphOptions{SwapVariants: swap, Barrier: true})
}

// Cond is a prompt's conditioning: the DiT's text rows (bf16, TextRows x hidden) and the keys
// they give (the prompt's tokens and one pad).
type Cond struct {
	ctx  []byte
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

// EncodeAll is Encode of each prompt with each graph loaded once (a prompt and its negative
// prompt: T5's 5 GB load is paid once).
func (p *Pipeline) EncodeAll(ctx context.Context, prompts ...string) ([]Cond, error) {
	t0 := time.Now()
	R, E := p.C.TextRows, p.t5dim
	pad := make([]byte, 2*E) // T5's <pad>, token 0's row
	if _, err := p.embed.ReadAt(pad, 0); err != nil {
		return nil, err
	}
	g, err := p.open(ctx, "t5", false)
	if err != nil {
		return nil, err
	}
	hs := make([][]byte, len(prompts))
	keys := make([]int, len(prompts))
	for k, pr := range prompts {
		ids := p.tok.Encode(pr)
		if len(ids) > R { // transformers truncates the text and keeps </s> last
			ids = append(ids[:R-1], ids[len(ids)-1])
		}
		x := make([]byte, 2*R*E)
		mask := make([]float32, R)
		for i := 0; i < R; i++ {
			if i < len(ids) {
				if _, err := p.embed.ReadAt(x[2*i*E:2*(i+1)*E], int64(ids[i])*int64(2*E)); err != nil {
					g.Close()
					return nil, err
				}
				mask[i] = 1
			} else {
				copy(x[2*i*E:], pad)
			}
		}
		if err := run(ctx, g, map[string][]byte{"x": x, "mask": toBF16(mask)}, nil); err == nil {
			hs[k], err = output(g, 0)
		} else {
			g.Close()
			return nil, fmt.Errorf("chroma: t5: %w", err)
		}
		// the prompt's tokens and one pad (the pipeline's mask)
		keys[k] = min(len(ids)+1, R)
		if n, err := strconv.Atoi(os.Getenv("HWNPU_CHROMA_KEYS")); err == nil && n > 0 { // a test of the text's mask
			keys[k] = min(n, R)
		}
		p.logf("text: %d tokens", len(ids))
	}
	g.Close()
	t1 := time.Now()
	c, err := p.open(ctx, "ctx", false)
	if err != nil {
		return nil, err
	}
	defer c.Close()
	conds := make([]Cond, len(prompts))
	for k := range prompts {
		if err := run(ctx, c, map[string][]byte{"x": hs[k]}, nil); err != nil {
			return nil, err
		}
		out, err := output(c, 0)
		if err != nil {
			return nil, err
		}
		conds[k] = Cond{ctx: out, Keys: keys[k]}
	}
	p.logf("T5-XXL %.2f s, the projection %.2f s", t1.Sub(t0).Seconds(), time.Since(t1).Seconds())
	return conds, nil
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
	b, err := g.Output(g.Outputs()[i].Name)
	return append([]byte(nil), b...), err
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

// dit holds the DiT's graphs while the steps run.
type dit struct {
	mods, in, doubles, singles, out *iron.Graph
}

func (d *dit) close() {
	for _, g := range []*iron.Graph{d.mods, d.in, d.doubles, d.singles, d.out} {
		if g != nil {
			g.Close()
		}
	}
}

// modInput is the approximator's input of timestep t (1000 sigma): each row the timestep's
// sinusoid (16), a zero guidance's (16) and its own index's (32, of 1000 k), zeros past the mods.
func (p *Pipeline) modInput(t float64) []float32 {
	c := p.C
	x := make([]float32, c.ModRows*c.ApproxIn)
	ts, g := sinusoid(t, 16), sinusoid(0, 16)
	for k := 0; k < c.Mods; k++ {
		row := x[k*c.ApproxIn:]
		copy(row, ts)
		copy(row[16:], g)
		copy(row[32:], sinusoid(1000*float64(k), 32))
	}
	return x
}

// velocity runs the DiT on the latents x (tokens x 64) at sigma for each conditioning, back to
// back in each group of blocks (the weights read once): the velocities.
func (p *Pipeline) velocity(ctx context.Context, d *dit, conds []Cond, x []float32, sigma float64) ([][]float32, error) {
	c := p.C
	E, N := c.Hidden, c.Tokens()
	if err := run(ctx, d.mods, map[string][]byte{"x": toBF16(p.modInput(1000 * sigma))}, nil); err != nil {
		return nil, err
	}
	mods, err := output(d.mods, 0)
	if err != nil {
		return nil, err
	}
	row := func(k int) []byte { return mods[2*k*E : 2*(k+1)*E] }
	if err := run(ctx, d.in, map[string][]byte{"x": toBF16(x)}, nil); err != nil {
		return nil, err
	}
	img, err := output(d.in, 0)
	if err != nil {
		return nil, err
	}
	hs := make([][]byte, len(conds))
	for k, cd := range conds {
		hs[k] = append(append([]byte(nil), img...), cd.ctx...) // [image rows, text rows]
	}
	blocks := func(g *iron.Graph, group, nmods int, rows func(b int) []int, out int, what string) error {
		for gi := 0; gi < g.Variants(); gi++ {
			var m []byte
			for j := 0; j < group; j++ {
				for _, r := range rows(gi*group + j) {
					m = append(m, row(r)...)
				}
			}
			if err := g.Use(gi); err != nil {
				return err
			}
			for k, cd := range conds {
				if err := run(ctx, g, map[string][]byte{"x": hs[k], "mods": m}, map[string]int64{"nk": int64(N + cd.Keys)}); err != nil {
					return fmt.Errorf("%s group %d: %w", what, gi, err)
				}
				o := out
				if o < 0 {
					o = len(g.Outputs()) - 1
				}
				if hs[k], err = output(g, o); err != nil {
					return err
				}
			}
		}
		return nil
	}
	err = blocks(d.doubles, c.DoubleGroup, 12, func(b int) []int {
		var r []int // the image's rows, then the text's
		for i := 0; i < 6; i++ {
			r = append(r, 3*c.Singles+6*b+i)
		}
		for i := 0; i < 6; i++ {
			r = append(r, 3*c.Singles+6*c.Doubles+6*b+i)
		}
		return r
	}, -1, "double blocks")
	if err == nil {
		err = blocks(d.singles, c.SingleGroup, 3, func(b int) []int { return []int{3 * b, 3*b + 1, 3*b + 2} }, 0, "single blocks")
	}
	if err != nil {
		return nil, err
	}
	last := c.Mods - 2
	vs := make([][]float32, len(conds))
	for k := range conds {
		if err := run(ctx, d.out, map[string][]byte{"x": hs[k][:2*N*E], "shift": row(last), "scale": row(last + 1)}, nil); err != nil {
			return nil, err
		}
		v, err := output(d.out, 0)
		if err != nil {
			return nil, err
		}
		vs[k] = fromBF16s(v)
	}
	return vs, nil
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
	if cond.ctx == nil {
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
	var d dit
	defer d.close()
	var err error
	for _, x := range []struct {
		g    **iron.Graph
		name string
		swap bool
	}{{&d.mods, "mods", false}, {&d.in, "dit_in", false}, {&d.doubles, "doubles", true}, {&d.singles, "singles", true}, {&d.out, "dit_out", false}} {
		if *x.g, err = p.open(ctx, x.name, x.swap); err != nil {
			return nil, err
		}
	}
	vel := func(x []float32, sigma float64) ([]float32, error) {
		conds := []Cond{cond}
		if smp.guided(sigma) {
			conds = append(conds, *neg)
		}
		vs, err := p.velocity(ctx, &d, conds, x, sigma)
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
	n := len(sigmas) - 1
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

// Decode turns the latents (Noise's layout) into the picture.
func (p *Pipeline) Decode(ctx context.Context, x []float32) (*image.RGBA, error) {
	c := p.C
	gh, gw := c.GH(), c.GW()
	H, W := 2*gh, 2*gw
	z := make([]float32, 16*H*W) // (16, H/8, W/8): FLUX's _unpack_latents
	for py := 0; py < gh; py++ {
		for px := 0; px < gw; px++ {
			t := (py*gw + px) * c.Latent
			for ch := 0; ch < 16; ch++ {
				for dy := 0; dy < 2; dy++ {
					for dx := 0; dx < 2; dx++ {
						z[(ch*H+2*py+dy)*W+2*px+dx] = x[t+ch*4+dy*2+dx]
					}
				}
			}
		}
	}
	p.vae.Stage = p.Stage
	return p.vae.Decode(ctx, z)
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
