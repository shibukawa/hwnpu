package anima

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
)

// GridShape is a padded channel-last grid of the VAE (iron_anima.GridShape): H rows of W
// pixels of C channels, each row padded to Wp = W + Pad pixels (one zero pixel before the
// image), a zero row above and below and a zero pixel before the first, rounded up to 256 rows.
type GridShape struct {
	H   int `json:"H"`
	W   int `json:"W"`
	C   int `json:"C"`
	Pad int `json:"-"`
}

// Wp is a padded row's pixels.
func (g GridShape) Wp() int { return g.W + g.Pad }

// Rows is the grid's rows (pixels).
func (g GridShape) Rows() int { return ((g.H+2)*g.Wp() + 2 + 255) / 256 * 256 }

// Pixel is the row of pixel (y, x).
func (g GridShape) Pixel(y, x int) int { return 1 + g.Wp() + y*g.Wp() + 1 + x }

// VAE is the Qwen-Image VAE decoder in stages (vae/s0, vae/s1, ...), each a graph from one
// padded grid to the next; the first takes the DiT's latents (unnormalized and post_quant_conv
// in its first 1 x 1 convolution), the last gives the pixels.
type VAE struct {
	stages []*iron.Graph
	in     GridShape
	Width  int
	Height int
	// Logf, when set, is told each stage's time.
	Logf func(format string, args ...any)
}

// OpenVAE loads the stages in dir onto dev.
func OpenVAE(ctx context.Context, dev *hwnpu.Device, dir string) (*VAE, error) {
	b, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return nil, err
	}
	var m struct {
		Stages int `json:"stages"`
		Width  int `json:"width"`
		Height int `json:"height"`
		Pad    int `json:"pad"`
	}
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, fmt.Errorf("anima: vae: %w", err)
	}
	v := &VAE{Width: m.Width, Height: m.Height}
	for i := 0; i < m.Stages; i++ {
		g, err := iron.OpenGraph(ctx, dev, filepath.Join(dir, fmt.Sprintf("s%d", i)))
		if err != nil {
			v.Close()
			return nil, err
		}
		v.stages = append(v.stages, g)
	}
	if len(v.stages) == 0 {
		return nil, errors.New("anima: vae: no stages")
	}
	var sm struct {
		Grid GridShape `json:"grid_in"`
	}
	if err := json.Unmarshal(v.stages[0].Manifest.Meta, &sm); err != nil {
		v.Close()
		return nil, err
	}
	v.in = sm.Grid
	v.in.Pad = m.Pad
	return v, nil
}

// Close releases the device memory.
func (v *VAE) Close() error {
	var errs []error
	for _, g := range v.stages {
		errs = append(errs, g.Close())
	}
	return errors.Join(errs...)
}

// Decode turns the DiT's latents (16, H/8, W/8), channel major, into the image.
func (v *VAE) Decode(ctx context.Context, latents []float32) (*image.RGBA, error) {
	g := v.in
	h, w := g.H, g.W
	const z = 16
	if len(latents) != z*h*w {
		return nil, fmt.Errorf("anima: vae: %d latent values for %dx%dx%d", len(latents), z, h, w)
	}
	in, err := v.stages[0].Input("x")
	if err != nil {
		return nil, err
	}
	clear(in)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			o := 2 * g.Pixel(y, x) * g.C
			for c := 0; c < z; c++ {
				binary.LittleEndian.PutUint16(in[o+2*c:], bf16(latents[(c*h+y)*w+x]))
			}
		}
	}
	var out []byte
	for i, s := range v.stages {
		t0 := time.Now()
		if i > 0 {
			if err := s.SetInput("x", out); err != nil {
				return nil, fmt.Errorf("anima: vae stage %d: %w", i, err)
			}
		}
		if err := s.Run(ctx, nil); err != nil {
			return nil, fmt.Errorf("anima: vae stage %d: %w", i, err)
		}
		if out, err = s.Output(s.Outputs()[0].Name); err != nil {
			return nil, err
		}
		if v.Logf != nil {
			v.Logf("vae stage %d: %.2f s", i, time.Since(t0).Seconds())
		}
	}
	W, H := v.Width, v.Height
	if len(out) < 2*64*W*H {
		return nil, fmt.Errorf("anima: vae: %d bytes of pixels for %dx%d", len(out), W, H)
	}
	img := image.NewRGBA(image.Rect(0, 0, W, H))
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			o := 2 * 64 * (y*W + x)
			var px [3]uint8
			for c := 0; c < 3; c++ {
				px[c] = pixel(fromBF16(binary.LittleEndian.Uint16(out[o+2*c:])))
			}
			img.SetRGBA(x, y, color.RGBA{px[0], px[1], px[2], 255})
		}
	}
	return img, nil
}

// pixel is the decoder's output, clamped to [-1, 1], as an 8-bit value (diffusers'
// VaeImageProcessor: (x / 2 + 0.5) * 255, rounded).
func pixel(v float32) uint8 {
	x := (math.Max(-1, math.Min(1, float64(v))) + 1) * 127.5
	return uint8(math.Round(x))
}

// Decode decodes latents with the export's VAE.
func (p *Pipeline) Decode(ctx context.Context, latents []float32) (*image.RGBA, error) {
	if !p.eng.hasVAE() {
		return nil, errors.New("anima: the export has no VAE")
	}
	return p.eng.decode(ctx, latents, p.Logf)
}
