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
// in its first 1 x 1 convolution), the last gives the pixels. Each stage is on the device only
// while it runs: the stages' banks are mostly their feature maps (8.2 GiB together at 1024 x
// 1024, the largest stage 1.2 GiB).
type VAE struct {
	dev    *hwnpu.Device
	stages []string // each stage's directory
	in     GridShape
	Width  int
	Height int
	// Logf, when set, is told each stage's time.
	Logf func(format string, args ...any)
}

// OpenVAE reads the stages' manifests in dir; Decode loads each stage onto dev as it runs.
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
	if m.Stages == 0 {
		return nil, errors.New("anima: vae: no stages")
	}
	v := &VAE{dev: dev, Width: m.Width, Height: m.Height}
	for i := 0; i < m.Stages; i++ {
		d := filepath.Join(dir, fmt.Sprintf("s%d", i))
		gm, err := iron.ReadGraphManifest(d)
		if err != nil {
			return nil, err
		}
		if i == 0 {
			var sm struct {
				Grid GridShape `json:"grid_in"`
			}
			if err := json.Unmarshal(gm.Meta, &sm); err != nil {
				return nil, err
			}
			v.in = sm.Grid
			v.in.Pad = m.Pad
		}
		v.stages = append(v.stages, d)
	}
	return v, nil
}

// Close releases nothing: no stage stays on the device after Decode.
func (v *VAE) Close() error { return nil }

// Decode turns the DiT's latents (16, H/8, W/8), channel major, into the image.
func (v *VAE) Decode(ctx context.Context, latents []float32) (*image.RGBA, error) {
	g := v.in
	h, w := g.H, g.W
	const z = 16
	if len(latents) != z*h*w {
		return nil, fmt.Errorf("anima: vae: %d latent values for %dx%dx%d", len(latents), z, h, w)
	}
	var out []byte
	for i, dir := range v.stages {
		t0 := time.Now()
		var err error
		if out, err = v.stage(ctx, i, dir, latents, out); err != nil {
			return nil, fmt.Errorf("anima: vae stage %d: %w", i, err)
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

// stage loads stage i, runs it on the previous stage's output (the latents for the first) and
// returns a copy of its output.
func (v *VAE) stage(ctx context.Context, i int, dir string, latents []float32, prev []byte) ([]byte, error) {
	s, err := iron.OpenGraph(ctx, v.dev, dir)
	if err != nil {
		return nil, err
	}
	defer s.Close()
	if i == 0 {
		in, err := s.Input("x")
		if err != nil {
			return nil, err
		}
		g := v.in
		clear(in)
		for y := 0; y < g.H; y++ {
			for x := 0; x < g.W; x++ {
				o := 2 * g.Pixel(y, x) * g.C
				for c := 0; c < 16; c++ {
					binary.LittleEndian.PutUint16(in[o+2*c:], bf16(latents[(c*g.H+y)*g.W+x]))
				}
			}
		}
	} else if err := s.SetInput("x", prev); err != nil {
		return nil, err
	}
	if err := s.Run(ctx, nil); err != nil {
		return nil, err
	}
	out, err := s.Output(s.Outputs()[0].Name)
	if err != nil {
		return nil, err
	}
	return append([]byte(nil), out...), nil
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
