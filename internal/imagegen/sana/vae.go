package sana

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

// GridShape is a padded channel-last grid of the VAE (iron_anima.GridShape, which
// iron_sana's VAE uses): H rows of W pixels of C channels, each row padded to Wp, W + 1
// rounded up to 16 (W + Pad in exports that wrote pad)
// pixels (one zero pixel before the image), a zero row above and below and a zero pixel before
// the first, rounded up to 256 rows.
type GridShape struct {
	H   int `json:"H"`
	W   int `json:"W"`
	C   int `json:"C"`
	Pad int `json:"-"`
}

// Wp is a padded row's pixels.
func (g GridShape) Wp() int {
	if g.Pad != 0 {
		return g.W + g.Pad // exports that wrote pad
	}
	return (g.W + 1 + 15) / 16 * 16
}

// Pixel is the row of pixel (y, x).
func (g GridShape) Pixel(y, x int) int { return 1 + g.Wp() + y*g.Wp() + 1 + x }

// VAE is DC-AE's decoder in stages (vae/s0, vae/s1, ...), each a graph from one padded grid to
// the next; the first takes the latents (its convolution divides them by the scaling factor),
// the last gives the pixels. Each stage is on the device only while it runs.
type VAE struct {
	dev    *hwnpu.Device
	stages []string
	in     GridShape
	Width  int
	Height int
	Latent int
	// Stage, when set, is told of each stage as it ends.
	Stage StageFunc
}

// OpenVAE reads the stages' manifests in dir.
func OpenVAE(dev *hwnpu.Device, dir string) (*VAE, error) {
	b, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return nil, err
	}
	var m struct {
		Stages int `json:"stages"`
		Width  int `json:"width"`
		Height int `json:"height"`
		Pad    int `json:"pad"`
		Latent int `json:"latent_channels"`
	}
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, fmt.Errorf("vae: %w", err)
	}
	if m.Stages == 0 {
		return nil, errors.New("vae: no stages")
	}
	v := &VAE{dev: dev, Width: m.Width, Height: m.Height, Latent: m.Latent}
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

// Decode turns the latents (Latent, H/32, W/32), channel major, into the image.
func (v *VAE) Decode(ctx context.Context, latents []float32) (*image.RGBA, error) {
	g := v.in
	if len(latents) != v.Latent*g.H*g.W {
		return nil, fmt.Errorf("vae: %d latent values for %dx%dx%d", len(latents), v.Latent, g.H, g.W)
	}
	// Each stage's graph is loaded while the one before runs (two stages on the device at
	// most): the loads, about a third of the decoder's time, are hidden.
	type opened struct {
		g   *iron.Graph
		err error
	}
	load := func(i int) chan opened {
		ch := make(chan opened, 1)
		go func() {
			g, err := iron.OpenGraph(ctx, v.dev, v.stages[i])
			ch <- opened{g, err}
		}()
		return ch
	}
	var out []byte
	next := load(0)
	cur := <-next
	if cur.err != nil {
		return nil, fmt.Errorf("vae stage 0: %w", cur.err)
	}
	if err := v.writeLatents(cur.g, latents); err != nil {
		cur.g.Close()
		return nil, err
	}
	for i := range v.stages {
		t0 := time.Now()
		next = nil
		if i+1 < len(v.stages) {
			next = load(i + 1)
		}
		fail := func(err error) (*image.RGBA, error) {
			cur.g.Close()
			if next != nil {
				if n := <-next; n.g != nil {
					n.g.Close()
				}
			}
			return nil, fmt.Errorf("vae stage %d: %w", i, err)
		}
		if err := cur.g.Run(ctx, nil); err != nil {
			return fail(err)
		}
		o, err := cur.g.Output(cur.g.Outputs()[0].Name)
		if err != nil {
			return fail(err)
		}
		if next == nil {
			out = append([]byte(nil), o...)
			cur.g.Close()
		} else {
			// the output straight into the next stage's input (one copy)
			n := <-next
			next = nil
			if n.err != nil {
				return fail(n.err)
			}
			in, err := n.g.Input("x")
			if err != nil || len(in) != len(o) {
				n.g.Close()
				return fail(fmt.Errorf("the next stage's input (%d bytes) for %d bytes: %v", len(in), len(o), err))
			}
			copy(in, o)
			cur.g.Close()
			cur = n
		}
		if v.Stage != nil {
			v.Stage(i, len(v.stages), time.Since(t0))
		}
	}
	W, H := v.Width, v.Height
	if len(out) < 2*64*W*H {
		return nil, fmt.Errorf("vae: %d bytes of pixels for %dx%d", len(out), W, H)
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

// writeLatents writes the latents into the first stage's input grid (bf16, zeros around).
func (v *VAE) writeLatents(s *iron.Graph, latents []float32) error {
	in, err := s.Input("x")
	if err != nil {
		return err
	}
	g := v.in
	clear(in)
	for y := 0; y < g.H; y++ {
		for x := 0; x < g.W; x++ {
			o := 2 * g.Pixel(y, x) * g.C
			for c := 0; c < v.Latent; c++ {
				binary.LittleEndian.PutUint16(in[o+2*c:], bf16(latents[(c*g.H+y)*g.W+x]))
			}
		}
	}
	return nil
}

// pixel is the decoder's output, clamped to [-1, 1], as an 8-bit value (diffusers'
// VaeImageProcessor: (x / 2 + 0.5) * 255, rounded).
func pixel(v float32) uint8 {
	x := (math.Max(-1, math.Min(1, float64(v))) + 1) * 127.5
	return uint8(math.Round(x))
}

// Decode releases the DiT and decodes latents with the export's VAE.
func (p *Pipeline) Decode(ctx context.Context, latents []float32) (*image.RGBA, error) {
	if err := p.eng.closeDiT(); err != nil {
		return nil, err
	}
	return p.eng.decode(ctx, latents, p.Stage)
}
