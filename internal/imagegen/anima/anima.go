// Package anima runs Anima (circlestone-labs/Anima), an image model, on an NPU: on AMD's, the
// graphs tools/amd/iron_anima.py exports; on Apple's Neural Engine and Intel's NPU, the Core ML
// packages and OpenVINO IR of internal/imagegen/anima/mil (hwnpu-image -convert). The host does
// what is not a tensor computation.
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
	"errors"
	"fmt"
	"image"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// DitConfig is the DiT's shape (iron_anima.DitConfig; for the Core ML and OpenVINO exports, the
// manifest's size and the constants of the checkpoint).
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

// engine runs the models of an export: the IRON graphs on AMD's NPU (ironEngine), or the
// programs hwnpu-image -convert writes for Apple's Neural Engine and Intel's NPU (progEngine).
type engine interface {
	// text runs the text encoder and the adapter on the embedding rows qx and tx (Text.Rows of
	// Text.Hidden each, zeros past nq and nt): the conditioning, the same shape.
	text(ctx context.Context, qx, tx []float32, nq, nt int) ([]float32, error)
	// setCond gives the DiT the conditioning of the next velocity calls.
	setCond(cond []float32) error
	// velocity runs the DiT on the patches (tokens x 64, c * 4 + dy * 2 + dx) at the timestep
	// whose sinusoid is tproj: each patch's velocity (tokens x 64, dy * 32 + dx * 16 + c).
	velocity(ctx context.Context, patches, tproj []float32) ([]float32, error)
	// decode runs the VAE on the latents (16, h, w), telling stage (when not nil) of each stage
	// as it ends; nil when the export has no VAE.
	decode(ctx context.Context, latents []float32, stage StageFunc) (*image.RGBA, error)
	hasVAE() bool
	close() error
}

// Pipeline is a loaded Anima export.
type Pipeline struct {
	Dit  DitConfig
	Text TextConfig

	eng                engine
	qwen               *tokenizer.Tokenizer
	t5                 *tokenizer.Unigram
	qwenEmbed, t5Embed []byte // bf16 rows
	// Logf, when set, is told what is done (the text's tokens and time).
	Logf func(format string, args ...any)
	// Stage, when set, is told of each stage of the VAE as it ends.
	Stage StageFunc
	// Dump, when set, is a directory the conditioning (cond.bin) and each step's velocity
	// (v<i>.bin, the DiT's output rows) are written to, raw bf16, to check against a reference.
	Dump string
}

func (p *Pipeline) dump(name string, xs []float32) {
	if p.Dump == "" {
		return
	}
	b := make([]byte, 2*len(xs))
	for i, x := range xs {
		binary.LittleEndian.PutUint16(b[2*i:], bf16(x))
	}
	if err := os.WriteFile(filepath.Join(p.Dump, name), b, 0o644); err != nil {
		p.logf("dump %s: %v", name, err)
	}
}

// StageFunc is told that stage i of n took d.
type StageFunc func(i, n int, d time.Duration)

// Open loads the export in dir onto dev: an IRON export (text/, dit_in/, dit_blocks/, dit_out/,
// vae/, AMD's NPU), or one of hwnpu-image -convert (manifest.json and the programs, Apple's
// Neural Engine or Intel's NPU; with dev nil, the CPU interpreter, Core ML packages only).
//
// log, when not nil, is shown each program of an export of hwnpu-image -convert as it loads (on a
// terminal, a line with a spinner: the first load compiles it for the NPU, minutes at 1024 x 1024).
func Open(ctx context.Context, dev *hwnpu.Device, dir string, log io.Writer) (*Pipeline, error) {
	p := &Pipeline{}
	fail := func(err error) (*Pipeline, error) {
		p.Close()
		return nil, fmt.Errorf("anima: %w", err)
	}
	var err error
	if IsProgramExport(dir) {
		var pe *progEngine
		pe, p.Dit, p.Text, err = openPrograms(ctx, dev, dir, log)
		if pe != nil {
			p.eng = pe
		}
	} else {
		if dev == nil {
			return nil, errors.New("anima: an IRON export runs on AMD's NPU only")
		}
		var ie *ironEngine
		ie, p.Dit, p.Text, err = openIron(ctx, dev, dir)
		if ie != nil {
			p.eng = ie
		}
	}
	if err != nil {
		return fail(err)
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
	if p.eng == nil {
		return nil
	}
	return p.eng.close()
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

// rows returns the bf16 rows of table for ids, as float32, zeros past them to Text.Rows.
func (p *Pipeline) rows(table []byte, ids []int32) []float32 {
	E := p.Text.Hidden
	out := make([]float32, p.Text.Rows*E)
	for i, id := range ids {
		src := table[int(id)*2*E:][:2*E]
		for j := 0; j < E; j++ {
			out[i*E+j] = fromBF16(binary.LittleEndian.Uint16(src[2*j:]))
		}
	}
	return out
}

// Encode runs the text encoder and the adapter on the prompt: its conditioning, Text.Rows rows
// of Text.Hidden (zeros past the T5 tokens, as the adapter's mask and its padding to 512 leave
// them).
func (p *Pipeline) Encode(ctx context.Context, prompt string) ([]float32, error) {
	t0 := time.Now()
	qwen, t5 := p.Tokens(prompt)
	if len(qwen) == 0 {
		// diffusers runs an empty prompt as one masked token; its rows are then all masked
		return nil, errors.New("anima: an empty prompt")
	}
	cond, err := p.eng.text(ctx, p.rows(p.qwenEmbed, qwen), p.rows(p.t5Embed, t5), len(qwen), len(t5))
	if err != nil {
		return nil, err
	}
	clear(cond[len(t5)*p.Text.Hidden:])
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
func (p *Pipeline) Denoise(ctx context.Context, cond []float32, x []float32, sigmas []float64, step func(i int, d time.Duration)) ([]float32, error) {
	c := p.Dit
	gh, gw := c.Grid()
	h, w := c.LatentSize()
	C := c.LatentChannels
	if len(x) != C*h*w {
		return nil, fmt.Errorf("anima: %d latent values for %dx%dx%d", len(x), C, h, w)
	}
	x = append([]float32(nil), x...)
	if err := p.eng.setCond(cond); err != nil {
		return nil, err
	}
	tokens := gh * gw
	pd := C * c.Patch * c.Patch
	patches := make([]float32, tokens*pd)
	for i := 0; i+1 < len(sigmas); i++ {
		t0 := time.Now()
		// patches: token py*gw+px, feature c*4 + dy*2 + dx
		for py := 0; py < gh; py++ {
			for px := 0; px < gw; px++ {
				o := (py*gw + px) * pd
				for ch := 0; ch < C; ch++ {
					for dy := 0; dy < 2; dy++ {
						for dx := 0; dx < 2; dx++ {
							patches[o+ch*4+dy*2+dx] = x[(ch*h+2*py+dy)*w+2*px+dx]
						}
					}
				}
			}
		}
		v, err := p.eng.velocity(ctx, patches, TimestepSinusoid(sigmas[i], c.Hidden))
		if err != nil {
			return nil, fmt.Errorf("anima: step %d: %w", i, err)
		}
		p.dump(fmt.Sprintf("v%d.bin", i), v)
		// v: token py*gw+px, feature dy*32 + dx*16 + c
		dt := float32(sigmas[i+1] - sigmas[i])
		for py := 0; py < gh; py++ {
			for px := 0; px < gw; px++ {
				o := (py*gw + px) * pd
				for dy := 0; dy < 2; dy++ {
					for dx := 0; dx < 2; dx++ {
						for ch := 0; ch < C; ch++ {
							x[(ch*h+2*py+dy)*w+2*px+dx] += dt * v[o+dy*2*C+dx*C+ch]
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
