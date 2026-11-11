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
	// Latent is the latents' channels (16: Anima's; 64: Qwen-Image-2.1's), Channels the
	// picture's (3, RGB; 4, RGBA: Qwen-Image-2.1's): the manifest's latent and channels.
	Latent, Channels int
	// Stage, when set, is told of each stage as it ends.
	Stage StageFunc
	// PrefetchLimit, when not 0, loads a stage while the one before runs only when the two hold
	// at most this many bytes on the device together (otherwise after it closes): the peak
	// against a stage's load time. Qwen-Image-2.1's 1024 x 1024 stages 4 and 5 are 5.9 GiB
	// together.
	PrefetchLimit int64
	sizes         []int64 // each stage's banks and arguments
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
		// Latent and Channels: 16 and 3 when the manifest leaves them out (Anima's exports)
		Latent   int `json:"latent"`
		Channels int `json:"channels"`
	}
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, fmt.Errorf("anima: vae: %w", err)
	}
	if m.Stages == 0 {
		return nil, errors.New("anima: vae: no stages")
	}
	v := &VAE{dev: dev, Width: m.Width, Height: m.Height, Latent: m.Latent, Channels: m.Channels}
	if v.Latent == 0 {
		v.Latent = 16
	}
	if v.Channels == 0 {
		v.Channels = 3
	}
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
		var n int64
		for _, b := range gm.Banks {
			n += int64(b.Bytes)
		}
		for _, a := range gm.Graph.Arguments {
			if _, ok := gm.Banks[a.Kind]; !ok {
				n += int64(a.Bytes)
			}
		}
		v.sizes = append(v.sizes, n)
	}
	return v, nil
}

// Close releases nothing: no stage stays on the device after Decode.
func (v *VAE) Close() error { return nil }

// Decode turns the DiT's latents (Latent, H/8, W/8 for Anima), channel major, into the image.
func (v *VAE) Decode(ctx context.Context, latents []float32) (*image.RGBA, error) {
	g := v.in
	h, w := g.H, g.W
	z := v.Latent
	if len(latents) != z*h*w {
		return nil, fmt.Errorf("anima: vae: %d latent values for %dx%dx%d", len(latents), z, h, w)
	}
	// Each stage's graph is loaded while the one before runs (two stages on the device at
	// most), and its output is copied straight into the next one's input.
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
		return nil, fmt.Errorf("anima: vae stage 0: %w", cur.err)
	}
	if err := v.writeLatents(cur.g, latents); err != nil {
		cur.g.Close()
		return nil, err
	}
	for i := range v.stages {
		t0 := time.Now()
		next = nil
		early := v.PrefetchLimit == 0 || v.sizes[i]+v.sizes[i+min(1, len(v.sizes)-1-i)] <= v.PrefetchLimit
		if i+1 < len(v.stages) && early {
			next = load(i + 1)
		}
		fail := func(err error) (*image.RGBA, error) {
			cur.g.Close()
			if next != nil {
				if n := <-next; n.g != nil {
					n.g.Close()
				}
			}
			return nil, fmt.Errorf("anima: vae stage %d: %w", i, err)
		}
		if err := cur.g.Run(ctx, nil); err != nil {
			return fail(err)
		}
		o, err := cur.g.Output(cur.g.Outputs()[0].Name)
		if err != nil {
			return fail(err)
		}
		if next == nil && i+1 < len(v.stages) { // the next stage after this one closes
			o = append([]byte(nil), o...)
			cur.g.Close()
			if n := <-load(i + 1); n.err != nil {
				return nil, fmt.Errorf("anima: vae stage %d: %w", i+1, n.err)
			} else {
				in, err := n.g.Input("x")
				if err != nil || len(in) != len(o) {
					n.g.Close()
					return nil, fmt.Errorf("anima: vae stage %d: the input (%d bytes) for %d bytes: %v", i+1, len(in), len(o), err)
				}
				copy(in, o)
				cur = n
			}
		} else if next == nil {
			out = append([]byte(nil), o...)
			cur.g.Close()
		} else {
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
		return nil, fmt.Errorf("anima: vae: %d bytes of pixels for %dx%d", len(out), W, H)
	}
	img := image.NewRGBA(image.Rect(0, 0, W, H))
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			o := 2 * 64 * (y*W + x)
			px := [4]uint8{3: 255}
			for c := 0; c < v.Channels; c++ {
				px[c] = pixel(fromBF16(binary.LittleEndian.Uint16(out[o+2*c:])))
			}
			if v.Channels == 4 { // straight alpha into image.RGBA's premultiplied
				for c := 0; c < 3; c++ {
					px[c] = uint8((int(px[c])*int(px[3]) + 127) / 255)
				}
			}
			img.SetRGBA(x, y, color.RGBA{px[0], px[1], px[2], px[3]})
		}
	}
	return img, nil
}

// writeLatents writes the DiT's latents into the first stage's input grid (bf16, zeros around).
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

// Decode decodes latents with the export's VAE.
func (p *Pipeline) Decode(ctx context.Context, latents []float32) (*image.RGBA, error) {
	if !p.eng.hasVAE() {
		return nil, errors.New("anima: the export has no VAE")
	}
	return p.eng.decode(ctx, latents, p.Stage)
}
