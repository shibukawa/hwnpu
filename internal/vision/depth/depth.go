// Package depth estimates a picture's depth on AMD's NPU with Depth Anything 3 (DA3MONO-LARGE
// and DA3METRIC-LARGE: DINOv2 ViT-L/14 and a DPT head; DA3-SMALL and DA3-BASE on one picture:
// ViT-S or B and DualDPT's depth and confidence), the graphs tools/amd/iron_da3.py exports. The host resizes and normalizes the picture, cuts its 14 x 14
// patches, takes exp of the depth logit and sets the sky to the far depth.
//
// The depth is relative (larger is farther, up to an unknown scale), or with DA3METRIC-LARGE
// metric given the focal length (Result.Meters), at the export's size nearest the picture's
// aspect (the longest side 504; the picture is stretched to it).
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

// Config is model.json of one size of an export (iron_da3.py): its directory's graphs.
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

// Export is model.json of an export: its sizes, each a directory WxH.
type Export struct {
	Name  string   `json:"name"`
	Kind  string   `json:"kind"`
	Sizes [][2]int `json:"sizes"`
	// Metric: the depth times the focal length in pixels over MetricScale is meters
	// (DA3METRIC-LARGE); else the depth is relative.
	Metric      bool    `json:"metric"`
	MetricScale float64 `json:"metric_scale"`
	// Second is what the head's second output is: "sky" (DPT: DA3MONO, DA3METRIC) or
	// "confidence" (DualDPT: DA3-SMALL, DA3-BASE).
	Second string `json:"second"`
}

// patchIn is the patch embedding's input width (iron_da3.PE_IN): 3 x 14 x 14 values, padded.
const patchIn = 640

// SkyThreshold is DA3's: a pixel whose sky output is at least it is sky.
const SkyThreshold = 0.3

func readExport(dir string) (*Export, error) {
	b, err := os.ReadFile(filepath.Join(dir, "model.json"))
	if err != nil {
		return nil, err
	}
	var x Export
	if err := json.Unmarshal(b, &x); err != nil {
		return nil, fmt.Errorf("depth: %w", err)
	}
	if x.Kind != "depth" || len(x.Sizes) == 0 {
		return nil, fmt.Errorf("depth: %s is not an export of tools/amd/iron_da3.py", dir)
	}
	return &x, nil
}

// IsExport reports whether dir is an export of tools/amd/iron_da3.py.
func IsExport(dir string) bool {
	_, err := readExport(dir)
	return err == nil
}

// Estimator is a loaded export. One size's graphs are on the device at a time: the one nearest
// the last picture's aspect.
type Estimator struct {
	Export
	dev *hwnpu.Device
	dir string
	cur *sized
	// Logf, when set, is told what is done.
	Logf func(format string, args ...any)
}

// sized is one size's graphs.
type sized struct {
	Config
	vit, head *iron.Graph
}

// Open reads the export in dir; a size's graphs are loaded onto dev for the first picture
// that takes it.
func Open(ctx context.Context, dev *hwnpu.Device, dir string) (*Estimator, error) {
	x, err := readExport(dir)
	if err != nil {
		return nil, err
	}
	return &Estimator{Export: *x, dev: dev, dir: dir}, nil
}

// Close releases the device memory.
func (e *Estimator) Close() error {
	if e.cur == nil {
		return nil
	}
	err := e.cur.close()
	e.cur = nil
	return err
}

func (s *sized) close() error {
	err := s.vit.Close()
	if err2 := s.head.Close(); err == nil {
		err = err2
	}
	return err
}

// Nearest is the export's size nearest the aspect of a w x h picture (in log ratio).
func (e *Estimator) Nearest(w, h int) (int, int) {
	best, bw, bh := math.Inf(1), 0, 0
	r := math.Log(float64(w) / float64(h))
	for _, s := range e.Sizes {
		if d := math.Abs(math.Log(float64(s[0])/float64(s[1])) - r); d < best {
			best, bw, bh = d, s[0], s[1]
		}
	}
	return bw, bh
}

// Official is DA3's input processor's size of a w x h picture (upper_bound_resize): the
// longest side to longest, the other scaled and rounded, then each to the nearest multiple of
// 14 (up on a tie).
func Official(w, h, longest int) (int, int) {
	scale := float64(longest) / float64(max(w, h))
	nw, nh := max(1, int(math.RoundToEven(float64(w)*scale))), max(1, int(math.RoundToEven(float64(h)*scale)))
	near := func(x int) int {
		down := x / 14 * 14
		if up := down + 14; up-x <= x-down {
			return up
		}
		return down
	}
	return near(nw), near(nh)
}

// Choose is the size the export runs a w x h picture at: DA3's own size of it when the export
// holds it, else the size nearest its aspect.
func (e *Estimator) Choose(w, h int) (int, int) {
	ow, oh := Official(w, h, 504)
	for _, s := range e.Sizes {
		if s[0] == ow && s[1] == oh {
			return ow, oh
		}
	}
	return e.Nearest(w, h)
}

// use loads the graphs of size w x h, releasing another's.
func (e *Estimator) use(ctx context.Context, w, h int) (*sized, error) {
	if e.cur != nil && e.cur.Model.Width == w && e.cur.Model.Height == h {
		return e.cur, nil
	}
	if err := e.Close(); err != nil {
		return nil, err
	}
	d := filepath.Join(e.dir, fmt.Sprintf("%dx%d", w, h))
	b, err := os.ReadFile(filepath.Join(d, "model.json"))
	if err != nil {
		return nil, err
	}
	s := &sized{}
	if err := json.Unmarshal(b, &s.Config); err != nil {
		return nil, fmt.Errorf("depth: %w", err)
	}
	if s.Model.Width != w || s.Model.Height != h || s.Patch == 0 {
		return nil, fmt.Errorf("depth: %s: not the graphs of %dx%d", d, w, h)
	}
	if s.vit, err = iron.OpenGraph(ctx, e.dev, filepath.Join(d, "vit")); err != nil {
		return nil, err
	}
	if s.head, err = iron.OpenGraph(ctx, e.dev, filepath.Join(d, "head")); err != nil {
		s.vit.Close()
		return nil, err
	}
	e.cur = s
	return s, nil
}

// Result is a depth map and the sky's or the confidence's, row major, at the size the picture
// was stretched to.
type Result struct {
	W, H  int
	Depth []float32 // with a sky output, the sky set to the 99th percentile of the rest
	Sky   []float32 // >= SkyThreshold is sky (nil without a sky output)
	Conf  []float32 // the confidence, exp + 1 (nil without one)
	// Metric: Depth times the focal length in pixels at W x H, over MetricScale, is meters.
	Metric      bool
	MetricScale float64
}

// Meters converts a metric model's depth to meters in place, for a picture of ow x oh pixels
// whose focal lengths in its own pixels are fx and fy: the focal length at W x H is the mean
// of fx W / ow and fy H / oh (DA3's README: the mean of fx and fy).
func (r *Result) Meters(fx, fy float64, ow, oh int) {
	f := (fx*float64(r.W)/float64(ow) + fy*float64(r.H)/float64(oh)) / 2
	k := float32(f / r.MetricScale)
	for i := range r.Depth {
		r.Depth[i] *= k
	}
}

// Patches returns the ViT's input for img: row 0 zeros (the class token's), then each 14 x
// 14 patch's normalized values (channel, row, column), the picture stretched to the size with
// PIL's antialiased bicubic (the official processor takes OpenCV's cubic or area, and keeps
// the aspect: the export's sizes are near it).
func (e *sized) Patches(img image.Image) []float32 {
	W, H := e.Model.Width, e.Model.Height
	return PatchRows(lfm3vl.Resize(img, W, H, "bicubic"), W, H, e.Patch, e.Tokens, e.Mean, e.Std)
}

// PatchRows is a ViT's input of the RGB pixels px (W x H, raster order): row 0 zeros (the class
// token's), then each patch x patch square's normalized values (channel, row, column), padded
// to 640 columns, tokens rows in all.
func PatchRows(px []uint8, W, H, P, tokens int, mean, std []float32) []float32 {
	gw, gh := W/P, H/P
	out := make([]float32, tokens*patchIn)
	for py := 0; py < gh; py++ {
		for pxi := 0; pxi < gw; pxi++ {
			row := out[(1+py*gw+pxi)*patchIn:]
			for c := 0; c < 3; c++ {
				for ky := 0; ky < P; ky++ {
					for kx := 0; kx < P; kx++ {
						v := float32(px[((py*P+ky)*W+pxi*P+kx)*3+c]) / 255
						row[(c*P+ky)*P+kx] = (v - mean[c]) / std[c]
					}
				}
			}
		}
	}
	return out
}

// Estimate runs the ViT and the head on img at the export's size nearest its aspect.
func (e *Estimator) Estimate(ctx context.Context, img image.Image) (*Result, error) {
	b := img.Bounds()
	t0 := time.Now()
	w, h := e.Choose(b.Dx(), b.Dy())
	s, err := e.use(ctx, w, h)
	if err != nil {
		return nil, err
	}
	tl := time.Now()
	if err := s.vit.SetInput("patches", toBF16(s.Patches(img))); err != nil {
		return nil, err
	}
	t1 := time.Now()
	if err := s.vit.Run(ctx, map[string]int64{"nk": int64(s.Rows)}); err != nil {
		return nil, fmt.Errorf("depth: vit: %w", err)
	}
	for i, h := range s.vit.Outputs() {
		f, err := s.vit.Output(h.Name)
		if err != nil {
			return nil, err
		}
		if err := s.head.SetInput(fmt.Sprintf("f%d", i), f); err != nil {
			return nil, err
		}
	}
	t2 := time.Now()
	if err := s.head.Run(ctx, nil); err != nil {
		return nil, fmt.Errorf("depth: head: %w", err)
	}
	out, err := s.head.Output(s.head.Outputs()[0].Name)
	if err != nil {
		return nil, err
	}
	t3 := time.Now()
	W, H := s.Model.Width, s.Model.Height
	r := &Result{W: W, H: H, Depth: make([]float32, W*H), Metric: e.Metric, MetricScale: e.MetricScale}
	second := make([]float32, W*H)
	for i := range r.Depth {
		r.Depth[i] = float32(math.Exp(float64(fromBF16(binary.LittleEndian.Uint16(out[2*64*i:])))))
		second[i] = fromBF16(binary.LittleEndian.Uint16(out[2*64*i+2:]))
	}
	if e.Second == "confidence" {
		for i, v := range second {
			second[i] = float32(math.Exp(float64(v))) + 1
		}
		r.Conf = second
	} else {
		for i, v := range second {
			second[i] = max(0, v)
		}
		r.Sky = second
		r.fillSky()
	}
	if e.Logf != nil {
		e.Logf("depth: %dx%d, load %.2f s, input %.2f s, vit %.2f s, head %.2f s", W, H, tl.Sub(t0).Seconds(),
			t1.Sub(tl).Seconds(), t2.Sub(t1).Seconds(), t3.Sub(t2).Seconds())
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

// ToBF16 rounds xs to bfloat16 bytes (to nearest, ties to even).
func ToBF16(xs []float32) []byte { return toBF16(xs) }

// FromBF16 is the float32 of a bfloat16.
func FromBF16(h uint16) float32 { return fromBF16(h) }

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
