// Package anima runs Anima (circlestone-labs/Anima), an image model, on AMD's NPU: the graphs
// tools/amd/iron_anima.py exports, with the host doing what is not a tensor computation.
//
// Anima is NVIDIA's Cosmos-Predict2 2B DiT with Qwen3-0.6B as its text encoder, an LLM adapter
// that maps Qwen3's hidden states onto T5-token queries, and the Qwen-Image VAE. A picture is:
//
//  1. the prompt's Qwen3 tokens and T5 tokens (internal/tokenizer), their embedding rows read
//     from the export's tables;
//  2. the text graph (Qwen3, then the adapter): the conditioning, 512 rows of 1024;
//  3. from Gaussian noise in the latent space (16 channels, an eighth of the image's sides), a
//     flow-matching Euler step per sigma: the DiT (dit_in, the dit_blocks groups, dit_out)
//     gives the velocity, x += (sigma_next - sigma) * v;
//  4. the VAE decoder on the latents, unnormalized by the VAE's per-channel mean and std.
//
// The export is for one image size; Anima-Turbo wants about 8 steps and no classifier-free
// guidance (CFG 1), which is all this package runs.
package anima

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
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// DitConfig is the DiT's shape in dit_blocks' meta (iron_anima.DitConfig).
type DitConfig struct {
	Width          int `json:"width"`
	Height         int `json:"height"`
	Hidden         int `json:"hidden"`
	Layers         int `json:"layers"`
	TextDim        int `json:"text_dim"`
	TextTokens     int `json:"text_tokens"`
	LatentChannels int `json:"latent_channels"`
	Patch          int `json:"patch"`
	VAEScale       int `json:"vae_scale"`
	Group          int `json:"group"`
}

// Grid is the patches down and across.
func (c DitConfig) Grid() (int, int) {
	s := c.VAEScale * c.Patch
	return c.Height / s, c.Width / s
}

// LatentSize is the latents' height and width.
func (c DitConfig) LatentSize() (int, int) { return c.Height / c.VAEScale, c.Width / c.VAEScale }

// TextConfig is the text graph's shape (iron_anima.TextConfig).
type TextConfig struct {
	Rows      int `json:"rows"`
	Hidden    int `json:"hidden"`
	T5Vocab   int `json:"t5_vocab"`
	QwenVocab int `json:"qwen_vocab"`
}

// Pipeline is a loaded Anima export.
type Pipeline struct {
	Dit  DitConfig
	Text TextConfig

	text, ditIn, ditBlocks, ditOut *iron.Graph
	vae                            *VAE
	qwen                           *tokenizer.Tokenizer
	t5                             *tokenizer.Unigram
	qwenEmbed, t5Embed             []byte // bf16 rows
	// Logf, when set, is told the time each stage takes.
	Logf func(format string, args ...any)
	// Dump, when set, is a directory the conditioning (cond.bin) and each step's velocity
	// (v<i>.bin, the DiT's output rows) are written to, raw bf16, to check against a reference.
	Dump string
}

func (p *Pipeline) dump(name string, b []byte) {
	if p.Dump == "" {
		return
	}
	if err := os.WriteFile(filepath.Join(p.Dump, name), b, 0o644); err != nil {
		p.logf("dump %s: %v", name, err)
	}
}

func meta[T any](g *iron.Graph) (T, error) {
	var m struct {
		Config T `json:"config"`
	}
	err := json.Unmarshal(g.Manifest.Meta, &m)
	return m.Config, err
}

// Open loads the export in dir (text/, dit_in/, dit_blocks/, dit_out/, vae/) onto dev.
func Open(ctx context.Context, dev *hwnpu.Device, dir string) (*Pipeline, error) {
	p := &Pipeline{}
	fail := func(err error) (*Pipeline, error) {
		p.Close()
		return nil, fmt.Errorf("anima: %w", err)
	}
	var err error
	for _, x := range []struct {
		g    **iron.Graph
		name string
	}{{&p.text, "text"}, {&p.ditIn, "dit_in"}, {&p.ditBlocks, "dit_blocks"}, {&p.ditOut, "dit_out"}} {
		if *x.g, err = iron.OpenGraph(ctx, dev, filepath.Join(dir, x.name)); err != nil {
			return fail(err)
		}
	}
	if p.Dit, err = meta[DitConfig](p.ditBlocks); err != nil {
		return fail(err)
	}
	if p.Text, err = meta[TextConfig](p.text); err != nil {
		return fail(err)
	}
	if n := p.Dit.Layers / p.Dit.Group; p.ditBlocks.Variants() != n {
		return fail(fmt.Errorf("dit_blocks holds %d groups, the DiT %d", p.ditBlocks.Variants(), n))
	}
	if _, err := os.Stat(filepath.Join(dir, "vae", "manifest.json")); err == nil {
		if p.vae, err = OpenVAE(ctx, dev, filepath.Join(dir, "vae")); err != nil {
			return fail(err)
		}
	}
	td := filepath.Join(dir, "text")
	if p.qwen, err = tokenizer.Load(filepath.Join(td, "qwen_tokenizer.json")); err != nil {
		return fail(err)
	}
	if p.t5, err = tokenizer.LoadUnigram(filepath.Join(td, "t5_tokenizer.json")); err != nil {
		return fail(err)
	}
	if p.qwenEmbed, err = os.ReadFile(filepath.Join(td, "qwen_embed.bin")); err != nil {
		return fail(err)
	}
	if p.t5Embed, err = os.ReadFile(filepath.Join(td, "t5_embed.bin")); err != nil {
		return fail(err)
	}
	return p, nil
}

// Close releases the device memory.
func (p *Pipeline) Close() error {
	var errs []error
	for _, g := range []*iron.Graph{p.text, p.ditIn, p.ditBlocks, p.ditOut} {
		if g != nil {
			errs = append(errs, g.Close())
		}
	}
	if p.vae != nil {
		errs = append(errs, p.vae.Close())
	}
	return errors.Join(errs...)
}

func (p *Pipeline) logf(format string, args ...any) {
	if p.Logf != nil {
		p.Logf(format, args...)
	}
}

// Tokens returns the prompt's Qwen3 tokens and T5 tokens (</s> last), each cut to the text
// graph's rows as the pipeline's max_sequence_length cuts them.
func (p *Pipeline) Tokens(prompt string) (qwen, t5 []int32) {
	qwen = p.qwen.Encode(prompt)
	t5 = p.t5.Encode(prompt)
	n := p.Text.Rows
	if len(qwen) > n {
		qwen = qwen[:n]
	}
	if len(t5) > n {
		// transformers truncates the text and keeps </s> last
		t5 = append(t5[:n-1], t5[len(t5)-1])
	}
	return qwen, t5
}

// Encode runs the text graph on the prompt: its conditioning, Text.Rows rows of bf16 (zeros
// past the T5 tokens, as the adapter's mask and its padding to 512 leave them).
func (p *Pipeline) Encode(ctx context.Context, prompt string) ([]byte, error) {
	t0 := time.Now()
	qwen, t5 := p.Tokens(prompt)
	if len(qwen) == 0 {
		// diffusers runs an empty prompt as one masked token; its rows are then all masked
		return nil, errors.New("anima: an empty prompt")
	}
	row := 2 * p.Text.Hidden
	qx, err := p.text.Input("qx")
	if err != nil {
		return nil, err
	}
	tx, err := p.text.Input("tx")
	if err != nil {
		return nil, err
	}
	clear(qx)
	clear(tx)
	for i, id := range qwen {
		copy(qx[i*row:(i+1)*row], p.qwenEmbed[int(id)*row:])
	}
	for i, id := range t5 {
		copy(tx[i*row:(i+1)*row], p.t5Embed[int(id)*row:])
	}
	if err := p.text.Run(ctx, map[string]int64{"nq": int64(len(qwen)), "nt": int64(len(t5))}); err != nil {
		return nil, err
	}
	out, err := p.text.Output(p.text.Outputs()[0].Name)
	if err != nil {
		return nil, err
	}
	cond := make([]byte, len(out))
	copy(cond, out[:len(t5)*row])
	p.logf("text: %d Qwen3 tokens, %d T5 tokens in %.2f s", len(qwen), len(t5), time.Since(t0).Seconds())
	p.dump("cond.bin", cond)
	return cond, nil
}

// Sigmas are the flow-matching sigmas of steps Euler steps: linspace(1, 1/steps) shifted by
// shift (FlowMatchEulerDiscreteScheduler's shift, 3 for Anima), then 0.
func Sigmas(steps int, shift float64) []float64 {
	s := make([]float64, steps+1)
	for i := 0; i < steps; i++ {
		x := 1.0
		if steps > 1 {
			x = 1 - float64(i)*(1-1/float64(steps))/float64(steps-1)
		}
		s[i] = shift * x / (1 + (shift-1)*x)
	}
	return s
}

// Noise returns Gaussian latents for seed: (channels, h, w), channel major.
func (p *Pipeline) Noise(seed uint64) []float32 {
	h, w := p.Dit.LatentSize()
	r := rand.New(rand.NewPCG(seed, 0x616e696d61))
	out := make([]float32, p.Dit.LatentChannels*h*w)
	for i := range out {
		out[i] = float32(r.NormFloat64())
	}
	return out
}

// Denoise runs the Euler steps from the latents x (Noise's layout) under cond: the latents
// after the last step. step, when set, is told each step as it ends.
func (p *Pipeline) Denoise(ctx context.Context, cond []byte, x []float32, sigmas []float64, step func(i int, d time.Duration)) ([]float32, error) {
	c := p.Dit
	gh, gw := c.Grid()
	h, w := c.LatentSize()
	C := c.LatentChannels
	if len(x) != C*h*w {
		return nil, fmt.Errorf("anima: %d latent values for %dx%dx%d", len(x), C, h, w)
	}
	x = append([]float32(nil), x...)
	if err := p.setAll("ctx", cond); err != nil {
		return nil, err
	}
	tokens := gh * gw
	pd := C * c.Patch * c.Patch
	patches := make([]byte, 2*tokens*pd)
	for i := 0; i+1 < len(sigmas); i++ {
		t0 := time.Now()
		// patches: token py*gw+px, feature c*4 + dy*2 + dx
		for py := 0; py < gh; py++ {
			for px := 0; px < gw; px++ {
				o := 2 * (py*gw + px) * pd
				for ch := 0; ch < C; ch++ {
					for dy := 0; dy < 2; dy++ {
						for dx := 0; dx < 2; dx++ {
							v := x[(ch*h+2*py+dy)*w+2*px+dx]
							binary.LittleEndian.PutUint16(patches[o+2*(ch*4+dy*2+dx):], bf16(v))
						}
					}
				}
			}
		}
		v, err := p.velocity(ctx, patches, sigmas[i])
		if err != nil {
			return nil, fmt.Errorf("anima: step %d: %w", i, err)
		}
		p.dump(fmt.Sprintf("v%d.bin", i), v)
		// v: token py*gw+px, feature dy*32 + dx*16 + c
		dt := float32(sigmas[i+1] - sigmas[i])
		for py := 0; py < gh; py++ {
			for px := 0; px < gw; px++ {
				o := 2 * (py*gw + px) * pd
				for dy := 0; dy < 2; dy++ {
					for dx := 0; dx < 2; dx++ {
						for ch := 0; ch < C; ch++ {
							vv := fromBF16(binary.LittleEndian.Uint16(v[o+2*(dy*2*C+dx*C+ch):]))
							x[(ch*h+2*py+dy)*w+2*px+dx] += dt * vv
						}
					}
				}
			}
		}
		if step != nil {
			step(i, time.Since(t0))
		}
	}
	return x, nil
}

// setAll copies data into input name of every DiT graph that takes it.
func (p *Pipeline) setAll(name string, data []byte) error {
	for _, g := range []*iron.Graph{p.ditBlocks} {
		if err := g.SetInput(name, data); err != nil {
			return err
		}
	}
	return nil
}

// velocity runs the DiT on the patches at sigma: dit_in, each group of dit_blocks, dit_out.
func (p *Pipeline) velocity(ctx context.Context, patches []byte, sigma float64) ([]byte, error) {
	c := p.Dit
	if err := p.ditIn.SetInput("patches", patches); err != nil {
		return nil, err
	}
	tp := make([]byte, 2*c.Hidden)
	for i, v := range TimestepSinusoid(sigma, c.Hidden) {
		binary.LittleEndian.PutUint16(tp[2*i:], bf16(v))
	}
	if err := p.ditIn.SetInput("tproj", tp); err != nil {
		return nil, err
	}
	if err := p.ditIn.Run(ctx, nil); err != nil {
		return nil, err
	}
	outs := p.ditIn.Outputs() // x, temb, et
	var x, temb, et []byte
	var err error
	if x, err = p.ditIn.Output(outs[0].Name); err != nil {
		return nil, err
	}
	if temb, err = p.ditIn.Output(outs[1].Name); err != nil {
		return nil, err
	}
	if et, err = p.ditIn.Output(outs[2].Name); err != nil {
		return nil, err
	}
	for _, in := range []struct {
		name string
		b    []byte
	}{{"temb", temb}, {"et", et}} {
		if err := p.ditBlocks.SetInput(in.name, in.b); err != nil {
			return nil, err
		}
		if err := p.ditOut.SetInput(in.name, in.b); err != nil {
			return nil, err
		}
	}
	for g := 0; g < p.ditBlocks.Variants(); g++ {
		if err := p.ditBlocks.Use(g); err != nil {
			return nil, err
		}
		if err := p.ditBlocks.SetInput("x", x); err != nil {
			return nil, err
		}
		if err := p.ditBlocks.Run(ctx, nil); err != nil {
			return nil, fmt.Errorf("blocks %d..%d: %w", g*c.Group, (g+1)*c.Group-1, err)
		}
		if x, err = p.ditBlocks.Output(p.ditBlocks.Outputs()[0].Name); err != nil {
			return nil, err
		}
	}
	if err := p.ditOut.SetInput("x", x); err != nil {
		return nil, err
	}
	if err := p.ditOut.Run(ctx, nil); err != nil {
		return nil, err
	}
	return p.ditOut.Output(p.ditOut.Outputs()[0].Name)
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

// bf16 rounds x to bfloat16 (to nearest, ties to even).
func bf16(x float32) uint16 {
	b := math.Float32bits(x)
	if b&0x7fffffff > 0x7f800000 {
		return uint16(b>>16) | 0x40
	}
	return uint16((b + 0x7fff + (b>>16)&1) >> 16)
}

func fromBF16(h uint16) float32 { return math.Float32frombits(uint32(h) << 16) }
