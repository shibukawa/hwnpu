package main

import (
	"image"
	"image/color"
	"image/draw"
	"math"
	"slices"
	"time"

	"github.com/shibukawa/hwnpu/ai/vision"
)

// View is where the eye is before the picture.
type View struct {
	// X and Y are across the view, -1 to 1: to the viewer's right, and up.
	X, Y float64
}

// Params are how deep the picture looks.
type Params struct {
	// Strength is how far the nearest point moves when the eye is at the edge of the view, as a
	// share of the picture's width (0.03: 31 pixels of 1024).
	Strength float64
	// Focus is the nearness (0 the farthest, 1 the nearest) that stays where it is: 0 moves the
	// far points least and the near ones most, 0.5 turns the picture about its middle.
	Focus float64
}

// steps are the layers a ray is followed through, and refine the halvings that find where
// between two layers it enters the picture, as in parallax.kage.
const (
	steps  = 32
	refine = 5
)

// uniforms are what the shader is given for a view: where the eye is on the screen (y down),
// the nearest point's move in pixels, and how much of the picture is shown.
func (p Params) uniforms(v View, width int) (viewX, viewY, shift, focus, zoom float64) {
	return v.X, -v.Y, p.Strength * float64(width), p.Focus, 1 - 2*p.Strength
}

// Scene is a picture with how near each of its points is.
//
// A picture in layers is a scene for each layer, the nearer ones over the ones behind (Behind):
// each has its own nearness, so that a layer has its relief, and where a near layer moves aside
// the layer behind shows what it has there instead of the near layer's edge stretched. A layer's
// nearness goes on past the layer's own part (where the layer is transparent), so that the
// layer ends at its edge.
type Scene struct {
	W, H int
	// Pix is the picture; a layer is transparent where it has nothing.
	Pix *image.RGBA
	// Near is the nearness of each pixel, row major: 1 the nearest, 0 the farthest.
	Near []float32
	// Behind is the layer behind this one, drawn before it; nil for the farthest layer, and for
	// a picture that is not in layers.
	Behind *Scene
}

// Layers are the scene's layers from the farthest to this one.
func (s *Scene) layers() []*Scene {
	if s.Behind == nil {
		return []*Scene{s}
	}
	return append(s.Behind.layers(), s)
}

// NewScene is the scene of a picture and its nearness (of the picture's size).
//
// The far layers are first pulled apart (Layers, by layers). Then the near things are made edge
// pixels larger in the nearness (Spread), so that a thing's outline moves with the thing: an
// estimated depth is soft at an outline and a little off it, and the outline would otherwise
// tear. What is stretched when a near thing moves aside is then the background just outside
// it.
func NewScene(img image.Image, near []float32, edge int, layers float64) *Scene {
	b := img.Bounds()
	rgba := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(rgba, rgba.Bounds(), img, b.Min, draw.Src) // a layer's colours come to hold their alpha
	return &Scene{W: b.Dx(), H: b.Dy(), Pix: rgba, Near: Spread(Layers(near, layers), b.Dx(), b.Dy(), edge)}
}

// Layers pulls the far layers of a nearness apart: n becomes log(1 + k n) / log(1 + k).
//
// A nearness is as the eye would see it: a thing twice as far moves half as much. With grass
// right before the eye, a bridge, houses behind it and mountains then all move as one, a
// hundredth of the grass's move apart. The curve gives the far layers a larger share of the
// move and the near ones a smaller: with k of 30, a tenth of the nearness becomes four tenths.
// k of 0 leaves the nearness as it is.
func Layers(near []float32, k float64) []float32 {
	if k <= 0 {
		return near
	}
	out := make([]float32, len(near))
	scale := 1 / math.Log1p(k)
	for i, n := range near {
		out[i] = float32(math.Log1p(k*float64(max(n, 0))) * scale)
	}
	return out
}

// look is how a picture in layers is made deep from its depth: the near things made edge
// pixels larger (Spread), the far layers pulled apart (Layers, by far), and the near layers'
// relief and roundness (Relief).
type look struct {
	edge               int
	far, relief, round float64
}

// Relief is the nearness of a near layer of w x h, a thing cut out of the picture (in, where the
// layer has its colour), from its depth.
//
// The layer as a whole sits where the far curve (Layers, by far) puts the middle of its depth.
// Inside it the curve would flatten it (with far of 30, a nearness of 0.6 to 0.7 becomes 0.86
// to 0.89, a pixel or two of move), and the layer would move as a card: its nearness goes on
// from there as the depth says, times relief. A depth model sees an animal or a figure as
// nearly flat too, so the layer is also made round, round higher in its middle than at its
// outline: as much as a tenth of the picture's width from it, a parabola. The picture's own
// border is no outline (the ground goes on past it). The layer is then brought down until its
// nearest points (a hundredth of it) are at most 1, the nearest the window can draw.
func Relief(near []float32, in []bool, w, h int, far, relief, round float64) []float32 {
	var inside []float32
	for i, ok := range in {
		if ok {
			inside = append(inside, near[i])
		}
	}
	if len(inside) == 0 {
		return Layers(near, far)
	}
	slices.Sort(inside)
	mid := float64(inside[len(inside)/2])
	base := mid
	if far > 0 {
		base = math.Log1p(far*mid) / math.Log1p(far)
	}
	d := outline(in, w, h)
	r := float64(max(w, h)) / 10
	out := make([]float32, len(near))
	vals := inside[:0]
	for i, n := range near {
		v := base + relief*(float64(n)-mid)
		if in[i] {
			t := min(d[i]/r, 1)
			v += round * t * (2 - t)
		}
		out[i] = float32(v)
		if in[i] {
			vals = append(vals, out[i])
		}
	}
	slices.Sort(vals)
	down := max(float64(vals[len(vals)*99/100])-1, 0)
	for i, v := range out {
		out[i] = float32(min(max(float64(v)-down, 0), 1))
	}
	return out
}

// outline is how far each pixel of a mask of w x h is from the nearest pixel outside it, in
// pixels (a chamfer of 1 and √2, two passes); 0 outside. Past the picture's border is inside.
func outline(in []bool, w, h int) []float64 {
	const diag = math.Sqrt2
	d := make([]float64, len(in))
	for i, ok := range in {
		if ok {
			d[i] = math.Inf(1)
		}
	}
	at := func(x, y int) float64 {
		if x < 0 || y < 0 || x >= w || y >= h {
			return math.Inf(1)
		}
		return d[y*w+x]
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if i := y*w + x; d[i] > 0 {
				d[i] = min(d[i], at(x-1, y)+1, at(x, y-1)+1, at(x-1, y-1)+diag, at(x+1, y-1)+diag)
			}
		}
	}
	for y := h - 1; y >= 0; y-- {
		for x := w - 1; x >= 0; x-- {
			if i := y*w + x; d[i] > 0 {
				d[i] = min(d[i], at(x+1, y)+1, at(x, y+1)+1, at(x+1, y+1)+diag, at(x-1, y+1)+diag)
			}
		}
	}
	for i, v := range d {
		if math.IsInf(v, 1) { // a mask with nothing outside it
			d[i] = float64(max(w, h))
		}
	}
	return d
}

// Spread makes every pixel of a nearness of w x h as near as the nearest within r pixels of it
// (across, then down), and then smooths the steps that leaves over the same distance.
func Spread(near []float32, w, h, r int) []float32 {
	if r <= 0 {
		return near
	}
	pass := func(src []float32, n, stride, lines, lineStride int, op func(win []float32) float32) []float32 {
		dst := make([]float32, len(src))
		win := make([]float32, 0, 2*r+1)
		for l := 0; l < lines; l++ {
			for i := 0; i < n; i++ {
				win = win[:0]
				for k := max(i-r, 0); k <= min(i+r, n-1); k++ {
					win = append(win, src[l*lineStride+k*stride])
				}
				dst[l*lineStride+i*stride] = op(win)
			}
		}
		return dst
	}
	most := func(win []float32) float32 { return slices.Max(win) }
	mean := func(win []float32) float32 {
		var sum float32
		for _, v := range win {
			sum += v
		}
		return sum / float32(len(win))
	}
	out := pass(near, w, 1, h, w, most) // across
	out = pass(out, h, w, w, 1, most)   // down
	out = pass(out, w, 1, h, w, mean)
	return pass(out, h, w, w, 1, mean)
}

// NearImage is the nearness as a picture the shader reads: its red is the nearness.
func (s *Scene) NearImage() *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, s.W, s.H))
	for i, n := range s.Near {
		v := uint8(n*255 + 0.5)
		img.Pix[4*i], img.Pix[4*i+1], img.Pix[4*i+2], img.Pix[4*i+3] = v, v, v, 255
	}
	return img
}

// nearAt is the nearness at the pixel that holds (x, y), as the shader reads it: quantised to
// the 8 bits of the picture it is given.
func (s *Scene) nearAt(x, y float64) float64 {
	xi := min(max(int(x), 0), s.W-1)
	yi := min(max(int(y), 0), s.H-1)
	return float64(uint8(s.Near[yi*s.W+xi]*255+0.5)) / 255
}

// colorAt is the picture's colour at (x, y), between its four nearest pixels.
func (s *Scene) colorAt(x, y float64) color.RGBA {
	qx, qy := x-0.5, y-0.5
	x0, y0 := math.Floor(qx), math.Floor(qy)
	fx, fy := qx-x0, qy-y0
	at := func(xi, yi int) [4]float64 {
		xi, yi = min(max(xi, 0), s.W-1), min(max(yi, 0), s.H-1)
		i := s.Pix.PixOffset(xi, yi)
		return [4]float64{float64(s.Pix.Pix[i]), float64(s.Pix.Pix[i+1]), float64(s.Pix.Pix[i+2]), float64(s.Pix.Pix[i+3])}
	}
	c00, c10, c01, c11 := at(int(x0), int(y0)), at(int(x0)+1, int(y0)), at(int(x0), int(y0)+1), at(int(x0)+1, int(y0)+1)
	var out [4]uint8
	for k := range out {
		top := c00[k] + (c10[k]-c00[k])*fx
		bot := c01[k] + (c11[k]-c01[k])*fx
		out[k] = uint8(top + (bot-top)*fy + 0.5)
	}
	return color.RGBA{out[0], out[1], out[2], out[3]}
}

// Render draws the scene as seen from v into dst (of the scene's size), on the CPU: what
// parallax.kage draws on the GPU, written to be read and to be checked against. For each pixel
// of the screen, the ray from the eye through it is followed from the nearest layer to the
// farthest; at layer n it is over the point of the picture that a point of nearness n would be
// drawn from, shifted by the eye's place times n - Focus. The first layer at which the picture
// is at least that near is what the eye sees there, so a near thing hides what is behind it.
//
// A scene in layers is drawn from the farthest layer, each over what is drawn already: where
// the ray meets a layer at a point it does not cover, what is behind stays.
func (s *Scene) Render(dst *image.RGBA, v View, p Params) {
	if s.Behind != nil {
		s.Behind.Render(dst, v, p)
	}
	vx, vy, shift, focus, zoom := p.uniforms(v, s.W)
	sx, sy := vx*shift, vy*shift
	cx, cy := float64(s.W)/2, float64(s.H)/2
	clampX := func(x float64) float64 { return min(max(x, 0.5), float64(s.W)-0.5) }
	clampY := func(y float64) float64 { return min(max(y, 0.5), float64(s.H)-0.5) }
	for y := 0; y < s.H; y++ {
		for x := 0; x < s.W; x++ {
			px := cx + (float64(x)+0.5-cx)*zoom
			py := cy + (float64(y)+0.5-cy)*zoom
			in := func(n float64) bool { // is the ray, at layer n, in the picture?
				return n <= s.nearAt(clampX(px+sx*(n-focus)), clampY(py+sy*(n-focus)))
			}
			above, below := 1.0, 0.0
			for i := 0; i < steps; i++ {
				n := 1 - float64(i)/float64(steps-1)
				if in(n) {
					below = n
					break
				}
				above = n
			}
			for i := 0; i < refine; i++ {
				if m := (above + below) / 2; in(m) {
					below = m
				} else {
					above = m
				}
			}
			hx, hy := clampX(px+sx*(below-focus)), clampY(py+sy*(below-focus))
			c := s.colorAt(hx, hy)
			if s.Behind != nil && c.A < 255 { // over what is behind (the colours hold their alpha)
				under := dst.RGBAAt(x, y)
				keep := 255 - int(c.A)
				c.R, c.G = uint8(min(int(c.R)+(int(under.R)*keep+127)/255, 255)), uint8(min(int(c.G)+(int(under.G)*keep+127)/255, 255))
				c.B, c.A = uint8(min(int(c.B)+(int(under.B)*keep+127)/255, 255)), uint8(min(int(c.A)+(int(under.A)*keep+127)/255, 255))
			}
			dst.SetRGBA(x, y, c)
		}
	}
}

// Follow moves a view towards where the eye was last seen, a share of the way each frame, so
// that the picture glides between the places a detector finds 15 times a second.
func (v View) Follow(to View, share float64) View {
	return View{v.X + (to.X-v.X)*share, v.Y + (to.Y-v.Y)*share}
}

// sweep is the view of a head that goes around the middle once as turn goes from 0 to 1.
func sweep(turn float64) View {
	return View{X: math.Cos(2 * math.Pi * turn), Y: 0.5 * math.Sin(2*math.Pi*turn)}
}

// windowOptions are how the window runs.
type windowOptions struct {
	title string
	// target is where the view is to go and a line of what follows the head (nil: the pointer).
	target func() (View, string)
	// sweep moves the view around by itself.
	sweep bool
	// dump writes the first frames frames into this directory and ends (to check the shader).
	dump   string
	frames int
	quiet  bool
	// record writes the window as a PNG into this directory every 2 seconds.
	record string
	// closeAfter closes the window after this long (0: never).
	closeAfter time.Duration
	// progress, when not nil, is the making of the deep picture (its depth, its layers), which
	// runs beside the window: the window shows each step until the scene is ready. picture is
	// the picture being made deep.
	progress *progress
	picture  image.Image
	// camera, when not nil, is what the tracker sees: the newest frame, the faces found, the
	// head and whether one is seen now.
	camera func() (image.Image, vision.Found, vision.Head, bool)
	// gain, when not nil, changes how far the view turns for the head's move by a step (the
	// arrow keys left and right, the wheel) and returns it.
	gain func(step float64) float64
}
