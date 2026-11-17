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
	"slices"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
)

// GridShape is a padded channel-last grid of the VAE (iron_anima.GridShape): H rows of W
// pixels of C channels, each row padded to Wp pixels (one zero pixel before the image), a zero
// row above and below (and below the rows up to Ha) and a zero pixel before the first, rounded
// up to 256 rows.
type GridShape struct {
	H   int `json:"H"`
	W   int `json:"W"`
	C   int `json:"C"`
	Pad int `json:"-"`
}

// Wp is a padded row's pixels: W and Pad, and as many more as make it a multiple of 16.
func (g GridShape) Wp() int { return g.W + g.Pad + (16-g.W%16)%16 }

// Ha is the image rows the convolutions run over: H rounded up to 16.
func (g GridShape) Ha() int { return (g.H + 15) / 16 * 16 }

// Rows is the grid's rows (pixels).
func (g GridShape) Rows() int { return ((g.Ha()+2)*g.Wp() + 2 + 255) / 256 * 256 }

// Pixel is the row of pixel (y, x).
func (g GridShape) Pixel(y, x int) int { return 1 + g.Wp() + y*g.Wp() + 1 + x }

// VAE is a VAE's decoder (or encoder) in stages (vae/s0, vae/s1, ...), each a graph from one
// padded grid to the next; the first takes the DiT's latents (unnormalized and post_quant_conv
// in its first 1 x 1 convolution), the last gives the pixels. A stage is on the device only
// while it runs.
//
// The stages share their device buffers. A stage's banks are mostly its feature maps (1 to 3
// GiB at 1024 x 1024), and allocating a device buffer pins and zeroes its pages in the kernel
// (0.33 s a GiB, and 0.11 s a GiB to free): the ten stages of FLUX.2's decoder allocated 17.6
// GiB, 7.6 of its 14 s. So each bank kind has one buffer, as large as the largest stage's, lent to
// every stage (iron.GraphOptions.Buffers: filled from the stage's image, zeroed past it), and a
// stage's output buffer is the next stage's input buffer, with no copy.
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
	info  []stageInfo
}

// stageInfo is what a stage's graph holds on the device: its banks by kind, and the arguments
// of its input and output (each an argument's only handle, or the buffers are not shared).
type stageInfo struct {
	banks             map[string]int
	inKind, outKind   string
	inBytes, outBytes int
	args              int64 // the arguments' bytes
	norms             []HostNorm
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
		si := stageInfo{banks: map[string]int{}}
		for kind, b := range gm.Banks {
			si.banks[kind] = b.Bytes
		}
		argBytes := map[string]int{}
		for _, a := range gm.Graph.Arguments {
			if _, ok := gm.Banks[a.Kind]; !ok {
				si.args += int64(a.Bytes)
				argBytes[a.Kind] = a.Bytes
			}
		}
		// an argument that is one whole handle can be handed from a stage to the next
		whole := func(hs []iron.Handle) (string, int) {
			if len(hs) != 1 || hs[0].Offset != 0 || argBytes[hs[0].Argument] != hs[0].Bytes {
				return "", 0
			}
			for _, o := range append(append([]iron.Handle(nil), gm.Graph.Inputs...), gm.Graph.Outputs...) {
				if o.Argument == hs[0].Argument && o.Name != hs[0].Name {
					return "", 0
				}
			}
			return hs[0].Argument, hs[0].Bytes
		}
		var nm struct {
			Norms []HostNorm `json:"norms"`
		}
		if len(gm.Meta) > 0 {
			if err := json.Unmarshal(gm.Meta, &nm); err != nil {
				return nil, fmt.Errorf("anima: vae stage %d: %w", i, err)
			}
		}
		si.norms = nm.Norms
		si.inKind, si.inBytes = whole(gm.Graph.Inputs)
		si.outKind, si.outBytes = whole(gm.Graph.Outputs)
		v.info = append(v.info, si)
	}
	return v, nil
}

// Peak is the most a run pins at once: each bank kind's largest (the buffers the stages share)
// and the largest stage's arguments.
func (v *VAE) Peak() int64 {
	banks := map[string]int{}
	var args int64
	for _, si := range v.info {
		for kind, n := range si.banks {
			banks[kind] = max(banks[kind], n)
		}
		args = max(args, si.args)
	}
	peak := args
	for _, n := range banks {
		peak += int64(n)
	}
	return peak
}

// PrefetchFloor is the pinned memory a stage may always reach with its second set of banks: at
// 512 x 512 the VAE's peak is about 1 GiB (its stages share their buffers), under a DiT stage with
// the set, whose reads would then wait (FLUX.2 klein: 1.0 of a picture's 6.0 s of steps).
const PrefetchFloor = 2 << 30

// prefetchMargin is what a stage pins beside the buffers PinnedBytes counts (its programs, the
// commands' control code): FLUX.2 klein's edit with a half-size reference counted 3.2 GiB with the
// second set of banks and pinned 3.53, over the VAE's 3.39.
const prefetchMargin = 384 << 20

// PrefetchFits says whether the swapped graphs among the graphs in dirs (open together, under
// root) may keep a second set of banks (iron.GraphOptions.Prefetch) without the stage pinning
// more than the VAE's peak, or than PrefetchFloor: a 1024 x 1024 run's peak is then unchanged.
// Off when iron.Prefetch is.
func PrefetchFits(v *VAE, root string, dirs []string, swapped ...string) bool {
	if !iron.Prefetch || v == nil {
		return false
	}
	var total int64
	for _, d := range dirs {
		pinned, banks, err := iron.PinnedBytes(filepath.Join(root, d))
		if err != nil {
			return false
		}
		total += pinned
		if slices.Contains(swapped, d) {
			total += banks
		}
	}
	return total+prefetchMargin <= max(v.Peak(), PrefetchFloor)
}

// Close releases nothing: no stage stays on the device after Decode.
func (v *VAE) Close() error { return nil }

// Decode turns the DiT's latents (Latent, H/8, W/8 for Anima), channel major, into the image.
func (v *VAE) Decode(ctx context.Context, latents []float32) (*image.RGBA, error) {
	out, err := v.run(ctx, latents, v.Latent)
	if err != nil {
		return nil, err
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

// Encode runs an encoder's stages (FLUX.2's vae_enc: the first grid is the picture's, the last
// stage gives the latents' pixels, dense, 64 channels wide) on the picture px, (Channels, Height,
// Width), channel major, in [-1, 1]: the latents (Latent, h, w), channel major, h and w the
// last stage's.
func (v *VAE) Encode(ctx context.Context, px []float32) ([]float32, error) {
	out, err := v.run(ctx, px, v.Channels)
	if err != nil {
		return nil, err
	}
	n := len(out) / (2 * 64)
	z := make([]float32, v.Latent*n)
	for i := 0; i < n; i++ {
		for c := 0; c < v.Latent; c++ {
			z[c*n+i] = fromBF16(binary.LittleEndian.Uint16(out[2*(64*i+c):]))
		}
	}
	return z, nil
}

// run writes x, (channels, H, W) of the first stage's grid, channel major, into it and runs the
// stages in turn: the last stage's output.
func (v *VAE) run(ctx context.Context, latents []float32, channels int) (out []byte, err error) {
	g := v.in
	h, w := g.H, g.W
	z := channels
	if len(latents) != z*h*w {
		return nil, fmt.Errorf("anima: vae: %d values for %dx%dx%d", len(latents), z, h, w)
	}
	// the buffers the stages share: a bank kind's (the largest stage's size), and the input,
	// which is the stage before's output
	banks := map[string]*hwnpu.Buffer{}
	var in *hwnpu.Buffer
	defer func() {
		for _, b := range banks {
			b.Close()
		}
		if in != nil {
			in.Close()
		}
	}()
	for _, si := range v.info {
		for kind := range si.banks {
			if banks[kind] != nil {
				continue
			}
			n := 0
			for _, sj := range v.info {
				n = max(n, sj.banks[kind])
			}
			if banks[kind], err = v.dev.AllocBuffer(max(n, 4096)); err != nil {
				return nil, fmt.Errorf("anima: vae: the stages' %s bank (%d MiB): %w", kind, n>>20, err)
			}
		}
	}
	for i, dir := range v.stages {
		t0 := time.Now()
		si := v.info[i]
		lend := map[string]*hwnpu.Buffer{}
		for kind := range si.banks {
			lend[kind] = banks[kind]
		}
		handed := in != nil && si.inKind != "" && in.Size() >= si.inBytes
		if handed {
			lend[si.inKind] = in
		}
		var next *hwnpu.Buffer // this stage's output, the next one's input
		if i+1 < len(v.stages) && si.outKind != "" && v.info[i+1].inKind != "" && v.info[i+1].inBytes == si.outBytes {
			if next, err = v.dev.AllocBuffer(max(si.outBytes, 4096)); err != nil {
				return nil, fmt.Errorf("anima: vae stage %d: %w", i, err)
			}
			lend[si.outKind] = next
		}
		fail := func(err error) ([]byte, error) {
			if next != nil {
				next.Close()
			}
			return nil, fmt.Errorf("anima: vae stage %d: %w", i, err)
		}
		sg, err := iron.OpenGraphWith(ctx, v.dev, dir, iron.GraphOptions{Buffers: lend})
		if err != nil {
			return fail(err)
		}
		switch {
		case i == 0:
			err = v.writeLatents(sg, latents, channels)
		case !handed: // the stage before's output, copied
			var x []byte
			if x, err = sg.Input("x"); err == nil && len(x) != len(out) {
				err = fmt.Errorf("the input (%d bytes) for %d bytes", len(x), len(out))
			}
			copy(x, out)
		}
		var normsDone func() error
		if err == nil {
			normsDone, err = hostNorms(sg, si.norms)
		}
		if err == nil {
			err = sg.Run(ctx, nil)
		}
		if err == nil {
			err = normsDone()
		}
		var o []byte
		if err == nil {
			o, err = sg.Output(sg.Outputs()[0].Name)
		}
		if err != nil {
			sg.Close()
			return fail(err)
		}
		if next == nil { // the last stage's, or one the next stage copies
			out = append(out[:0], o...)
		}
		sg.Close()
		if in != nil {
			in.Close()
		}
		in = next
		if v.Stage != nil {
			v.Stage(i, len(v.stages), time.Since(t0))
		}
	}
	return out, nil
}

// writeLatents writes the latents (channels, H, W) into the first stage's input grid (bf16,
// zeros around).
func (v *VAE) writeLatents(s *iron.Graph, latents []float32, channels int) error {
	in, err := s.Input("x")
	if err != nil {
		return err
	}
	g := v.in
	clear(in)
	for y := 0; y < g.H; y++ {
		for x := 0; x < g.W; x++ {
			o := 2 * g.Pixel(y, x) * g.C
			for c := 0; c < channels; c++ {
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
