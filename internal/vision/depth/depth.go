// Package depth estimates a picture's depth on AMD's NPU with Depth Anything 3's monocular
// model (depth-anything/DA3MONO-LARGE: DINOv2 ViT-L/14 and a DPT head), the graphs
// tools/amd/iron_da3.py exports. The host resizes and normalizes the picture, cuts its 14 x 14
// patches, takes exp of the depth logit and sets the sky to the far depth.
//
// The depth is relative (larger is farther, up to an unknown scale), at the export's size
// (504 x 504 by default; the picture is stretched to it).
package depth

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"
	"slices"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3vl"
)

// Config is model.json of an export (iron_da3.py).
type Config struct {
	Name   string    `json:"name"`
	Kind   string    `json:"kind"`
	Mean   []float32 `json:"mean"`
	Std    []float32 `json:"std"`
	Patch  int       `json:"patch"`
	Tokens int       `json:"tokens"` // the ViT's rows (the class token and the patches, padded)
	Rows   int       `json:"rows"`   // the rows that are tokens
	Model  struct {
		Width  int `json:"width"`
		Height int `json:"height"`
		Hidden int `json:"hidden"`
	} `json:"config"`
}

// patchIn is the patch embedding's input width (iron_da3.PE_IN): 3 x 14 x 14 values, padded.
const patchIn = 640

// SkyThreshold is DA3's: a pixel whose sky output is at least it is sky.
const SkyThreshold = 0.3

// IsExport reports whether dir is an export of tools/amd/iron_da3.py.
func IsExport(dir string) bool {
	b, err := os.ReadFile(filepath.Join(dir, "model.json"))
	if err != nil {
		return false
	}
	var c Config
	return json.Unmarshal(b, &c) == nil && c.Kind == "depth"
}

// Estimator is a loaded export, its graphs on the device.
type Estimator struct {
	Config
	vit, head *iron.Graph
	// Logf, when set, is told what is done.
	Logf func(format string, args ...any)
}

// Open loads the export in dir onto dev.
func Open(ctx context.Context, dev *hwnpu.Device, dir string) (*Estimator, error) {
	b, err := os.ReadFile(filepath.Join(dir, "model.json"))
	if err != nil {
		return nil, err
	}
	e := &Estimator{}
	if err := json.Unmarshal(b, &e.Config); err != nil {
		return nil, fmt.Errorf("depth: %w", err)
	}
	if e.Kind != "depth" || e.Patch == 0 || e.Model.Width == 0 {
		return nil, fmt.Errorf("depth: %s is not an export of tools/amd/iron_da3.py", dir)
	}
	if e.vit, err = iron.OpenGraph(ctx, dev, filepath.Join(dir, "vit")); err != nil {
		return nil, err
	}
	if e.head, err = iron.OpenGraph(ctx, dev, filepath.Join(dir, "head")); err != nil {
		e.vit.Close()
		return nil, err
	}
	return e, nil
}

// Close releases the device memory.
func (e *Estimator) Close() error {
	err := e.vit.Close()
	if err2 := e.head.Close(); err == nil {
		err = err2
	}
	return err
}

// Size is the depth map's width and height (the export's).
func (e *Estimator) Size() (int, int) { return e.Model.Width, e.Model.Height }

// Result is a depth map and the sky's, row major at Size.
type Result struct {
	W, H  int
	Depth []float32 // relative; the sky set to the 99th percentile of the rest
	Sky   []float32 // >= SkyThreshold is sky
}

// Patches returns the ViT's input for img: row 0 zeros (the class token's), then each 14 x
// 14 patch's normalized values (channel, row, column), the picture stretched to Size with
// PIL's antialiased bicubic (the official processor takes OpenCV's cubic or area).
func (e *Estimator) Patches(img image.Image) []float32 {
	W, H, P := e.Model.Width, e.Model.Height, e.Patch
	px := lfm3vl.Resize(img, W, H, "bicubic")
	gw, gh := W/P, H/P
	out := make([]float32, e.Tokens*patchIn)
	for py := 0; py < gh; py++ {
		for pxi := 0; pxi < gw; pxi++ {
			row := out[(1+py*gw+pxi)*patchIn:]
			for c := 0; c < 3; c++ {
				for ky := 0; ky < P; ky++ {
					for kx := 0; kx < P; kx++ {
						v := float32(px[((py*P+ky)*W+pxi*P+kx)*3+c]) / 255
						row[(c*P+ky)*P+kx] = (v - e.Mean[c]) / e.Std[c]
					}
				}
			}
		}
	}
	return out
}

// Estimate runs the ViT and the head on img.
func (e *Estimator) Estimate(ctx context.Context, img image.Image) (*Result, error) {
	t0 := time.Now()
	if err := e.vit.SetInput("patches", toBF16(e.Patches(img))); err != nil {
		return nil, err
	}
	t1 := time.Now()
	if err := e.vit.Run(ctx, map[string]int64{"nk": int64(e.Rows)}); err != nil {
		return nil, fmt.Errorf("depth: vit: %w", err)
	}
	for i, h := range e.vit.Outputs() {
		f, err := e.vit.Output(h.Name)
		if err != nil {
			return nil, err
		}
		if err := e.head.SetInput(fmt.Sprintf("f%d", i), f); err != nil {
			return nil, err
		}
	}
	t2 := time.Now()
	if err := e.head.Run(ctx, nil); err != nil {
		return nil, fmt.Errorf("depth: head: %w", err)
	}
	out, err := e.head.Output(e.head.Outputs()[0].Name)
	if err != nil {
		return nil, err
	}
	t3 := time.Now()
	W, H := e.Size()
	r := &Result{W: W, H: H, Depth: make([]float32, W*H), Sky: make([]float32, W*H)}
	for i := range r.Depth {
		r.Depth[i] = float32(math.Exp(float64(fromBF16(binary.LittleEndian.Uint16(out[2*64*i:])))))
		r.Sky[i] = max(0, fromBF16(binary.LittleEndian.Uint16(out[2*64*i+2:])))
	}
	r.fillSky()
	if e.Logf != nil {
		e.Logf("depth: input %.2f s, vit %.2f s, head %.2f s", t1.Sub(t0).Seconds(), t2.Sub(t1).Seconds(), t3.Sub(t2).Seconds())
	}
	return r, nil
}

// fillSky sets the sky's depth to the 99th percentile of the rest (DA3's
// _process_mono_sky_estimation, over all the other pixels where it samples 100000), when
// both are more than 10 pixels.
func (r *Result) fillSky() {
	var rest []float32
	for i, s := range r.Sky {
		if s < SkyThreshold {
			rest = append(rest, r.Depth[i])
		}
	}
	if len(rest) <= 10 || len(r.Sky)-len(rest) <= 10 {
		return
	}
	far := Quantile(rest, 0.99)
	for i, s := range r.Sky {
		if s >= SkyThreshold {
			r.Depth[i] = far
		}
	}
}

// Quantile is torch.quantile's (linear interpolation) of xs, which it sorts.
func Quantile(xs []float32, q float64) float32 {
	slices.Sort(xs)
	pos := q * float64(len(xs)-1)
	i := int(math.Floor(pos))
	if i+1 >= len(xs) {
		return xs[len(xs)-1]
	}
	f := float32(pos - float64(i))
	return xs[i] + f*(xs[i+1]-xs[i])
}

func toBF16(xs []float32) []byte {
	b := make([]byte, 2*len(xs))
	for i, x := range xs {
		bits := math.Float32bits(x)
		var h uint16
		if bits&0x7fffffff > 0x7f800000 {
			h = uint16(bits>>16) | 0x40
		} else {
			h = uint16((bits + 0x7fff + (bits>>16)&1) >> 16)
		}
		binary.LittleEndian.PutUint16(b[2*i:], h)
	}
	return b
}

func fromBF16(h uint16) float32 { return math.Float32frombits(uint32(h) << 16) }
