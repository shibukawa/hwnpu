// Package mil describes Depth Anything 3 (depth-anything/DA3*) as MIL programs, which hwnpu
// writes as Core ML packages for Apple's Neural Engine (llm.WritePart): the two graphs of
// tools/amd/iron_da3.py, the ViT (the patch embedding and the blocks up to the last feature)
// and the head (DPT, or DualDPT's depth branch), for one picture size each.
//
// They run in fp16; every layer norm first divides each row by its largest magnitude (milpart's
// ScaledLayerNorm), so that no square overflows.
package mil

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu/internal/imagegen/milpart"
	"github.com/shibukawa/hwnpu/internal/safetensors"
)

// Patch is DINOv2's patch size.
const Patch = 14

// PatchIn is the patch embedding's input width: 3 x 14 x 14 values, padded (the host's rows,
// depth.PatchRows).
const PatchIn = 640

// Part is one program (milpart.Part).
type Part = milpart.Part

type graph = milpart.Graph

func part(name string, body func(g *graph)) *Part {
	return &Part{Name: name, Model: "depth", Body: body}
}

// Config is a DA3's shape at one picture size (iron_da3.Config).
type Config struct {
	Width, Height int
	Hidden, Heads int
	Layers        int // the blocks the ViT runs: up to the last feature's
	MLP           int
	OutLayers     [4]int
	Features      int
	OutChannels   [4]int
	// The any-view models' (DA3-SMALL, DA3-BASE) settings, -1 when off: from AltStart the odd
	// blocks attend "globally" (one picture: the same tokens, every patch at one RoPE
	// position) and the camera token replaces the class token's row; from QKNormStart
	// per-head layer norms of q and k; from RopeStart 2-D RoPE. CatToken: each feature is the
	// last local block's output beside the normed one.
	AltStart, QKNormStart, RopeStart int
	CatToken                         bool
	// Dual: the head is DualDPT (a layer norm of the features, UV embeddings; depth and
	// confidence); else DPT (depth and sky).
	Dual bool
}

// GH and GW are the patch grid's rows and columns.
func (c Config) GH() int { return c.Height / Patch }

// GW is the patch grid's columns.
func (c Config) GW() int { return c.Width / Patch }

// Tokens is the class token and the patches.
func (c Config) Tokens() int { return 1 + c.GH()*c.GW() }

// FeatDim is a feature's width.
func (c Config) FeatDim() int {
	if c.CatToken {
		return 2 * c.Hidden
	}
	return c.Hidden
}

// ReadConfig reads root/config.json (the official format) for a picture of w x h pixels.
func ReadConfig(root string, w, h int) (Config, error) {
	b, err := os.ReadFile(filepath.Join(root, "config.json"))
	if err != nil {
		return Config{}, err
	}
	var f struct {
		Config struct {
			Net struct {
				Name        string `json:"name"`
				OutLayers   []int  `json:"out_layers"`
				AltStart    int    `json:"alt_start"`
				QKNormStart int    `json:"qknorm_start"`
				RopeStart   int    `json:"rope_start"`
				CatToken    bool   `json:"cat_token"`
			} `json:"net"`
			Head struct {
				Object struct {
					Name string `json:"name"`
				} `json:"__object__"`
				OutputDim   int   `json:"output_dim"`
				Features    int   `json:"features"`
				OutChannels []int `json:"out_channels"`
			} `json:"head"`
		} `json:"config"`
	}
	if err := json.Unmarshal(b, &f); err != nil {
		return Config{}, fmt.Errorf("depth: config.json: %w", err)
	}
	net, head := f.Config.Net, f.Config.Head
	vit := map[string][3]int{"vits": {384, 6, 12}, "vitb": {768, 12, 12}, "vitl": {1024, 16, 24}}
	v, ok := vit[net.Name]
	dual := head.Object.Name == "DualDPT"
	wantDim := 1
	if dual {
		wantDim = 2
	}
	if head.OutputDim == 0 {
		head.OutputDim = 1
	}
	switch {
	case !ok || (!dual && head.Object.Name != "DPT") || head.OutputDim != wantDim || len(net.OutLayers) != 4 || len(head.OutChannels) != 4:
		return Config{}, fmt.Errorf("depth: %s: a DA3 of ViT-S, B or L with a DPT or DualDPT head is supported", root)
	case w <= 0 || h <= 0 || w%Patch != 0 || h%Patch != 0:
		return Config{}, fmt.Errorf("depth: %dx%d is not whole %d-pixel patches", w, h, Patch)
	}
	c := Config{Width: w, Height: h, Hidden: v[0], Heads: v[1], MLP: 4 * v[0], Features: head.Features,
		AltStart: net.AltStart, QKNormStart: net.QKNormStart, RopeStart: net.RopeStart, CatToken: net.CatToken, Dual: dual}
	copy(c.OutLayers[:], net.OutLayers)
	copy(c.OutChannels[:], head.OutChannels)
	c.Layers = c.OutLayers[3] + 1
	return c, nil
}

// Weights reads a DA3's model.safetensors; a 1 x 1 convolution's weight (out, in, 1, 1) reads
// as the matrix (out, in).
type Weights struct {
	f    *safetensors.File
	path string
}

// OpenWeights opens root/model.safetensors.
func OpenWeights(root string) (*Weights, error) {
	p := filepath.Join(root, "model.safetensors")
	f, err := safetensors.Open(p)
	if err != nil {
		return nil, err
	}
	return &Weights{f: f, path: p}, nil
}

// Get returns a tensor as float32 and its shape.
func (w *Weights) Get(name string) ([]float32, []int, error) {
	t, ok := w.f.Lookup(name)
	if !ok {
		return nil, nil, fmt.Errorf("depth: %s has no tensor %s", w.path, name)
	}
	v, err := w.f.Float32(name)
	shape := t.Shape
	if len(shape) == 4 && shape[2] == 1 && shape[3] == 1 {
		shape = shape[:2]
	}
	return v, shape, err
}

// Rows returns rows [begin, end) of a matrix.
func (w *Weights) Rows(name string, begin, end int) ([]float32, error) {
	t, ok := w.f.Lookup(name)
	if !ok {
		return nil, fmt.Errorf("depth: %s has no tensor %s", w.path, name)
	}
	cols := 1
	for _, d := range t.Shape[1:] {
		cols *= d
	}
	return w.f.Float32Range(name, begin*cols, end*cols)
}

// Source names the checkpoint.
func (w *Weights) Source() string { return "safetensors " + filepath.Base(w.path) }

// Close closes the file.
func (w *Weights) Close() error { return w.f.Close() }

// bicubic is torch's bicubic interpolation (A = -0.75, align_corners False, the given scale
// factor, borders clamped) as a matrix (out, in).
func bicubic(out, in int, scale float64) []float64 {
	const A = -0.75
	cubic := func(x float64) float64 {
		x = math.Abs(x)
		switch {
		case x <= 1:
			return ((A+2)*x-(A+3))*x*x + 1
		case x < 2:
			return ((A*x-5*A)*x+8*A)*x - 4*A
		}
		return 0
	}
	m := make([]float64, out*in)
	for o := 0; o < out; o++ {
		src := (float64(o)+0.5)/scale - 0.5
		i0 := int(math.Floor(src))
		t := src - float64(i0)
		for k := -1; k < 3; k++ {
			i := min(max(i0+k, 0), in-1)
			m[o*in+i] += cubic(float64(k) - t)
		}
	}
	return m
}

// PosEmbed is DINOv2's interpolate_pos_encoding of the table pe ((1 + M M) x E): the patches'
// rows bicubically resized to gh x gw (scale (n + 0.1) / M), the class token's row first.
func PosEmbed(pe []float32, E, gh, gw int) []float32 {
	M := int(math.Round(math.Sqrt(float64(len(pe)/E - 1))))
	if gh == M && gw == M {
		return append([]float32(nil), pe...)
	}
	ah := bicubic(gh, M, (float64(gh)+0.1)/float64(M))
	aw := bicubic(gw, M, (float64(gw)+0.1)/float64(M))
	out := make([]float32, (1+gh*gw)*E)
	copy(out, pe[:E])
	tmp := make([]float64, gh*M*E) // the rows resized first: (gh, M, E)
	for a := 0; a < gh; a++ {
		for i := 0; i < M; i++ {
			wt := ah[a*M+i]
			if wt == 0 {
				continue
			}
			for j := 0; j < M; j++ {
				src := pe[(1+i*M+j)*E:]
				dst := tmp[(a*M+j)*E:]
				for c := 0; c < E; c++ {
					dst[c] += wt * float64(src[c])
				}
			}
		}
	}
	for a := 0; a < gh; a++ {
		for b := 0; b < gw; b++ {
			dst := out[(1+a*gw+b)*E:]
			for j := 0; j < M; j++ {
				wt := aw[b*M+j]
				if wt == 0 {
					continue
				}
				src := tmp[(a*M+j)*E:]
				for c := 0; c < E; c++ {
					dst[c] += float32(wt * src[c])
				}
			}
		}
	}
	return out
}
