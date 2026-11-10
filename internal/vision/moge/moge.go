// Package moge estimates a picture's metric geometry on an NPU with MoGe-2
// (Ruicheng/moge-2-vit{s,b,l}-normal, MIT): on AMD's, the graphs tools/amd/iron_moge.py exports
// (the ViT, the neck and the points, normal and mask heads); on Apple's Neural Engine, the Core
// ML packages of internal/vision/moge/mil (Prepare). The host stretches the picture to the
// export's size nearest its aspect, cuts the patches, and does MoGe's post-processing: the
// heads' outputs resized to the picture, the points' z taken exp of (and xy times it), the mask
// through a sigmoid, the normals normalized, the focal length and z shift solved on the point
// map, everything times the metric scale: depth in meters.
package moge

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3vl"
	"github.com/shibukawa/hwnpu/internal/vision/depth"
)

// Export is model.json of an export: its sizes, each a directory WxH.
type Export struct {
	Name   string   `json:"name"`
	Kind   string   `json:"kind"` // "moge"
	Sizes  [][2]int `json:"sizes"`
	Tokens int      `json:"tokens"` // MoGe's num_tokens the sizes were made for
	// Format is "coreml" for an export of Prepare (a size's programs in WxH/, converted from
	// Checkpoint, model.pt, when a picture first takes the size).
	Format     string `json:"format,omitempty"`
	Checkpoint string `json:"checkpoint,omitempty"`
}

// sizeConfig is model.json of one size.
type sizeConfig struct {
	Mean   []float32 `json:"mean"`
	Std    []float32 `json:"std"`
	Patch  int       `json:"patch"`
	Tokens int       `json:"tokens"`
	Rows   int       `json:"rows"`
	Width  int       `json:"width"`
	Height int       `json:"height"`
	Config struct {
		BaseW int `json:"base_w"`
		BaseH int `json:"base_h"`
	} `json:"config"`
}

// IsExport reports whether dir is an export of tools/amd/iron_moge.py.
func IsExport(dir string) bool {
	x, err := readExport(dir)
	return err == nil && x.Kind == "moge"
}

func readExport(dir string) (*Export, error) {
	b, err := os.ReadFile(filepath.Join(dir, "model.json"))
	if err != nil {
		return nil, err
	}
	var x Export
	if err := json.Unmarshal(b, &x); err != nil {
		return nil, fmt.Errorf("moge: %w", err)
	}
	if x.Kind != "moge" || len(x.Sizes) == 0 {
		return nil, fmt.Errorf("moge: %s is not an export of tools/amd/iron_moge.py", dir)
	}
	return &x, nil
}

// Estimator is a loaded export; one size's graphs are on the device at a time.
type Estimator struct {
	Export
	dev *hwnpu.Device
	dir string
	cur *sized
	// FovX, when not 0, is the camera's horizontal field of view in degrees: the focal length
	// is not solved but taken from it (infer's fov_x), only the z shift is.
	FovX float64
	// Logf, when set, is told what is done.
	Logf func(format string, args ...any)
}

type sized struct {
	sizeConfig
	vit, neck *iron.Graph
	heads     *iron.Graph // points, normal, mask: variants 0, 1, 2 read into one bank in turn
	prog      *progSized
}

var headNames = [3]string{"points", "normal", "mask"}

func (s *sized) close() error {
	if s.prog != nil {
		return s.prog.close()
	}
	var err error
	for _, g := range []*iron.Graph{s.vit, s.neck, s.heads} {
		if g != nil {
			if e := g.Close(); err == nil {
				err = e
			}
		}
	}
	return err
}

// Open reads the export in dir.
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

// Nearest is the export's size nearest the aspect of a w x h picture.
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

// Choose is the size the export runs a w x h picture at: MoGe's own (the patch grid of Tokens
// at the picture's aspect, rounded) when the export holds it, else the nearest by aspect.
func (e *Estimator) Choose(w, h int) (int, int) {
	if e.Tokens > 0 {
		a := float64(w) / float64(h)
		bw := int(math.RoundToEven(math.Sqrt(float64(e.Tokens) * a)))
		bh := int(math.RoundToEven(math.Sqrt(float64(e.Tokens) / a)))
		for _, s := range e.Sizes {
			if s[0] == 14*bw && s[1] == 14*bh {
				return s[0], s[1]
			}
		}
	}
	return e.Nearest(w, h)
}

func (e *Estimator) use(ctx context.Context, w, h int) (*sized, error) {
	if e.cur != nil && e.cur.Width == w && e.cur.Height == h {
		return e.cur, nil
	}
	if err := e.Close(); err != nil {
		return nil, err
	}
	d := filepath.Join(e.dir, fmt.Sprintf("%dx%d", w, h))
	if e.Format == "coreml" {
		return e.useProgram(ctx, d, w, h)
	}
	b, err := os.ReadFile(filepath.Join(d, "model.json"))
	if err != nil {
		return nil, err
	}
	s := &sized{}
	if err := json.Unmarshal(b, &s.sizeConfig); err != nil {
		return nil, fmt.Errorf("moge: %w", err)
	}
	open := func(name string) (*iron.Graph, error) { return iron.OpenGraph(ctx, e.dev, filepath.Join(d, name)) }
	if s.vit, err = open("vit"); err == nil {
		if s.neck, err = open("neck"); err == nil {
			s.heads, err = iron.OpenGraphWith(ctx, e.dev, filepath.Join(d, "heads"), iron.GraphOptions{SwapVariants: true})
		}
	}
	if err != nil {
		s.close()
		return nil, err
	}
	e.cur = s
	return s, nil
}

// Result is MoGe's geometry at the picture's size, row major.
type Result struct {
	W, H   int
	Depth  []float32 // meters; +Inf where the mask is off
	Mask   []bool    // a valid pixel (not sky or background beyond the depth)
	Normal []float32 // 3 a pixel, unit length (camera space: x right, y down, z forward)
	Points []float32 // 3 a pixel, meters (x, y, z = Depth)
	// FovX is the horizontal field of view the point map implies, in degrees; Fx and Fy the
	// focal lengths in pixels.
	FovX, Fx, Fy float64
	Scale        float32 // the metric scale
}

// Estimate runs the graphs on img and post-processes their outputs at img's size.
func (e *Estimator) Estimate(ctx context.Context, img image.Image) (*Result, error) {
	b := img.Bounds()
	ow, oh := b.Dx(), b.Dy()
	t0 := time.Now()
	w, h := e.Choose(ow, oh)
	s, err := e.use(ctx, w, h)
	if err != nil {
		return nil, err
	}
	tl := time.Now()
	// MoGe's encoder resizes bilinearly with antialias
	px := lfm3vl.Resize(img, s.Width, s.Height, "bilinear")
	if s.prog != nil {
		return e.estimateProgram(ctx, s, px, ow, oh, t0, tl)
	}
	if err := s.vit.SetInput("patches", depth.ToBF16(depth.PatchRows(px, s.Width, s.Height, s.Patch, s.Tokens, s.Mean, s.Std))); err != nil {
		return nil, err
	}
	if err := s.vit.Run(ctx, map[string]int64{"nk": int64(s.Rows)}); err != nil {
		return nil, fmt.Errorf("moge: vit: %w", err)
	}
	vo := s.vit.Outputs()
	f, err := s.vit.Output(vo[0].Name)
	if err != nil {
		return nil, err
	}
	sc, err := s.vit.Output(vo[1].Name)
	if err != nil {
		return nil, err
	}
	scaleLogit := depth.FromBF16(binary.LittleEndian.Uint16(sc))
	if err := s.neck.SetInput("f", f); err != nil {
		return nil, err
	}
	t1 := time.Now()
	if err := s.neck.Run(ctx, nil); err != nil {
		return nil, fmt.Errorf("moge: neck: %w", err)
	}
	t2 := time.Now()
	var levels [][]byte
	for _, o := range s.neck.Outputs() {
		x, err := s.neck.Output(o.Name)
		if err != nil {
			return nil, err
		}
		levels = append(levels, append([]byte(nil), x...))
	}
	H4, W4 := s.Config.BaseH<<4, s.Config.BaseW<<4
	raw := make([][]float32, 7) // points 3, normal 3, mask 1 at H4 x W4
	for i := range raw {
		raw[i] = make([]float32, H4*W4)
	}
	ch := 0
	g := s.heads
	for hi := range headNames {
		if err := g.Use(hi); err != nil {
			return nil, err
		}
		for li, x := range levels {
			if err := g.SetInput(fmt.Sprintf("n%d", li), x); err != nil {
				return nil, err
			}
		}
		if err := g.Run(ctx, nil); err != nil {
			return nil, fmt.Errorf("moge: %s: %w", headNames[hi], err)
		}
		out, err := g.Output(g.Outputs()[0].Name)
		if err != nil {
			return nil, err
		}
		n := []int{3, 3, 1}[hi]
		for c := 0; c < n; c++ {
			for i := 0; i < H4*W4; i++ {
				raw[ch+c][i] = depth.FromBF16(binary.LittleEndian.Uint16(out[2*(64*i+c):]))
			}
		}
		ch += n
	}
	t3 := time.Now()
	r := Post(raw, W4, H4, ow, oh, scaleLogit, e.FovX)
	if e.Logf != nil {
		e.Logf("moge: %dx%d, load %.2f s, vit %.2f s, neck %.2f s, heads %.2f s, post %.2f s; scale %.2f, fov %.1f°",
			w, h, tl.Sub(t0).Seconds(), t1.Sub(tl).Seconds(), t2.Sub(t1).Seconds(), t3.Sub(t2).Seconds(),
			time.Since(t3).Seconds(), r.Scale, r.FovX)
	}
	return r, nil
}

// Post is MoGe's infer() on the heads' raw outputs (7 maps of w x h: points' 3, normal's 3,
// the mask's logit) for a picture of ow x oh: resized (bilinear, align_corners=False), the
// points' remap (x, y times exp z; z = exp z), the focal length and shift solved on the
// masked point map (only the shift, given fovX in degrees), the depth (z + shift) times
// exp(scaleLogit).
func Post(raw [][]float32, w, h, ow, oh int, scaleLogit float32, fovX float64) *Result {
	m := make([][]float32, len(raw))
	for i := range raw {
		m[i] = Bilinear(raw[i], w, h, ow, oh)
	}
	n := ow * oh
	r := &Result{W: ow, H: oh, Depth: make([]float32, n), Mask: make([]bool, n), Normal: make([]float32, 3*n),
		Points: make([]float32, 3*n), Scale: float32(math.Exp(float64(scaleLogit)))}
	for i := 0; i < n; i++ {
		z := float32(math.Exp(float64(m[2][i])))
		r.Points[3*i], r.Points[3*i+1], r.Points[3*i+2] = m[0][i]*z, m[1][i]*z, z
		nx, ny, nz := m[3][i], m[4][i], m[5][i]
		l := float32(math.Sqrt(float64(nx*nx + ny*ny + nz*nz)))
		if l < 1e-12 {
			l = 1e-12
		}
		r.Normal[3*i], r.Normal[3*i+1], r.Normal[3*i+2] = nx/l, ny/l, nz/l
		r.Mask[i] = m[6][i] > 0 // sigmoid > 0.5
	}
	a := float64(ow) / float64(oh)
	var given float64
	if fovX > 0 {
		// infer's focal of fov_x, relative to the half diagonal
		given = a / math.Sqrt(1+a*a) / math.Tan(fovX*math.Pi/360)
	}
	focal, shift := RecoverFocalShift(r.Points, r.Mask, ow, oh, given)
	fx := focal / 2 * math.Sqrt(1+a*a) / a
	fy := focal / 2 * math.Sqrt(1+a*a)
	r.Fx, r.Fy = fx*float64(ow), fy*float64(oh)
	r.FovX = 2 * math.Atan(0.5/fx) * 180 / math.Pi
	// force_projection: the points from the depth and the intrinsics, then the metric scale
	for i := 0; i < n; i++ {
		d := r.Points[3*i+2] + float32(shift)
		if d <= 0 {
			r.Mask[i] = false
		}
		u := (float64(i%ow)+0.5)/float64(ow) - 0.5
		v := (float64(i/ow)+0.5)/float64(oh) - 0.5
		x, y := float32(u/fx)*d, float32(v/fy)*d
		r.Points[3*i], r.Points[3*i+1], r.Points[3*i+2] = x*r.Scale, y*r.Scale, d*r.Scale
		r.Depth[i] = d * r.Scale
		if !r.Mask[i] {
			r.Depth[i] = float32(math.Inf(1))
			r.Normal[3*i], r.Normal[3*i+1], r.Normal[3*i+2] = 0, 0, 0
		}
	}
	return r
}

// Bilinear resizes the map xs (w x h) to ow x oh as torch's interpolate does with
// align_corners=False and no antialias.
func Bilinear(xs []float32, w, h, ow, oh int) []float32 {
	if w == ow && h == oh {
		return append([]float32(nil), xs...)
	}
	src := func(o, n, no int) (int, int, float32) {
		s := math.Max(0, (float64(o)+0.5)*float64(n)/float64(no)-0.5)
		i0 := min(int(s), n-1)
		return i0, min(i0+1, n-1), float32(s - float64(i0))
	}
	out := make([]float32, ow*oh)
	for y := 0; y < oh; y++ {
		y0, y1, fy := src(y, h, oh)
		for x := 0; x < ow; x++ {
			x0, x1, fx := src(x, w, ow)
			a := xs[y0*w+x0] + fx*(xs[y0*w+x1]-xs[y0*w+x0])
			b := xs[y1*w+x0] + fx*(xs[y1*w+x1]-xs[y1*w+x0])
			out[y*ow+x] = a + fy*(b-a)
		}
	}
	return out
}

// ViewUV is MoGe's normalized_view_plane_uv of a w x h map: u, v a pixel.
func ViewUV(w, h int) []float64 {
	a := float64(w) / float64(h)
	sx, sy := a/math.Sqrt(1+a*a), 1/math.Sqrt(1+a*a)
	lin := func(lo float64, n, i int) float64 {
		if n == 1 {
			return lo
		}
		return lo + (-2*lo)*float64(i)/float64(n-1)
	}
	out := make([]float64, 2*w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			out[2*(y*w+x)] = lin(-sx*float64(w-1)/float64(w), w, x)
			out[2*(y*w+x)+1] = lin(-sy*float64(h-1)/float64(h), h, y)
		}
	}
	return out
}

// RecoverFocalShift is MoGe's recover_focal_shift: the point map (3 a pixel, w x h) and its
// mask resized to 64 x 64 (nearest), the shift minimizing |focal xy / (z + shift) - uv| with
// the best focal for each shift (least squares from 0, as scipy's LM), then that focal (relative
// to the half diagonal). A given focal (not 0) is kept and only the shift solved.
func RecoverFocalShift(points []float32, mask []bool, w, h int, given float64) (float64, float64) {
	const S = 64
	uv := ViewUV(w, h)
	var xs, ys, zs, us, vs []float64
	for y := 0; y < S; y++ {
		sy := min(y*h/S, h-1) // torch's nearest: floor(dst * in / out)
		for x := 0; x < S; x++ {
			sx := min(x*w/S, w-1)
			i := sy*w + sx
			if mask != nil && !mask[i] {
				continue
			}
			xs = append(xs, float64(points[3*i]))
			ys = append(ys, float64(points[3*i+1]))
			zs = append(zs, float64(points[3*i+2]))
			us = append(us, uv[2*i])
			vs = append(vs, uv[2*i+1])
		}
	}
	if len(xs) < 2 {
		return 1, 0
	}
	focalAt := func(s float64) float64 {
		if given > 0 {
			return given
		}
		var num, den float64
		for i := range xs {
			px, py := xs[i]/(zs[i]+s), ys[i]/(zs[i]+s)
			num += px*us[i] + py*vs[i]
			den += px*px + py*py
		}
		return num / den
	}
	cost := func(s float64) float64 {
		f := focalAt(s)
		var c float64
		for i := range xs {
			ex := f*xs[i]/(zs[i]+s) - us[i]
			ey := f*ys[i]/(zs[i]+s) - vs[i]
			c += ex*ex + ey*ey
		}
		return c
	}
	// Levenberg-Marquardt in one variable on the cost (a sum of squares) with numeric
	// derivatives: from 0 as scipy's least_squares(method="lm")
	s, lambda := 0.0, 1e-3
	c := cost(s)
	for it := 0; it < 200; it++ {
		eps := 1e-6 * math.Max(1, math.Abs(s))
		g := (cost(s+eps) - cost(s-eps)) / (2 * eps)
		hh := (cost(s+eps) - 2*c + cost(s-eps)) / (eps * eps)
		if hh <= 0 {
			hh = math.Abs(g) + 1e-12
		}
		step := -g / (hh * (1 + lambda))
		ns := s + step
		nc := cost(ns)
		if nc < c {
			done := (c-nc) <= 1e-10*c || math.Abs(step) < 1e-9*math.Max(1, math.Abs(s))
			s, c = ns, nc
			lambda /= 10
			if done {
				break
			}
		} else {
			lambda *= 10
			if lambda > 1e12 {
				break
			}
		}
	}
	return focalAt(s), s
}
