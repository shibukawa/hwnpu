package ocr

import (
	"math"
	"slices"
)

// MinSize is the shortest side of a box kept (PaddleOCR's DBPostProcess: 3 pixels of the map,
// and 2 more once grown).
const MinSize = 3

// MaxCandidates is the most regions of the map looked at (PP-OCRv6's inference.yml).
const MaxCandidates = 3000

type pt struct{ x, y float64 }

// Boxes are the boxes of text in the detector's probability map (W x H, the picture's nw x nh
// at the top left), in the map's pixels, as PaddleOCR's DBPostProcess finds them: the regions
// over Thresh (8-connected), each one's rotated rectangle of least area, kept when its mean
// probability is at least BoxThresh, grown by Unclip and kept when its shorter side is at
// least MinSize + 2. Each box is four corners clockwise from the top left.
func (r *Reader) Boxes(prob []float32, W, H, nw, nh int) [][4][2]float32 {
	label := make([]int32, nw*nh)
	var boxes [][4][2]float32
	var stack []int32
	var pts []pt
	n := 0
	for start := range nw * nh {
		if label[start] != 0 || prob[start/nw*W+start%nw] <= r.Thresh {
			continue
		}
		n++
		if n > MaxCandidates {
			break
		}
		// the region's pixels
		pts = pts[:0]
		stack = append(stack[:0], int32(start))
		label[start] = int32(n)
		for len(stack) > 0 {
			i := int(stack[len(stack)-1])
			stack = stack[:len(stack)-1]
			x, y := i%nw, i/nw
			pts = append(pts, pt{float64(x), float64(y)})
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					xx, yy := x+dx, y+dy
					if xx < 0 || yy < 0 || xx >= nw || yy >= nh {
						continue
					}
					j := yy*nw + xx
					if label[j] == 0 && prob[yy*W+xx] > r.Thresh {
						label[j] = int32(n)
						stack = append(stack, int32(j))
					}
				}
			}
		}
		if len(pts) < 4 {
			continue
		}
		rect := minAreaRect(hull(pts))
		if min(rect.w, rect.h) < MinSize {
			continue
		}
		if score := meanIn(prob, W, nw, nh, rect.corners()); score < float64(r.BoxThresh) {
			continue
		}
		// unclip: the polygon offset by its area times Unclip over its perimeter (pyclipper's
		// round joins); the least rectangle around it is the rectangle grown by that on each side
		d := rect.w * rect.h * float64(r.Unclip) / (2 * (rect.w + rect.h))
		rect.w, rect.h = rect.w+2*d, rect.h+2*d
		if min(rect.w, rect.h) < MinSize+2 {
			continue
		}
		c := order(rect.corners())
		var box [4][2]float32
		for k := range 4 {
			box[k] = [2]float32{float32(math.Max(0, math.Min(float64(nw), c[k].x))), float32(math.Max(0, math.Min(float64(nh), c[k].y)))}
		}
		boxes = append(boxes, box)
	}
	return boxes
}

// hull is the convex hull of points (Andrew's monotone chain), counterclockwise.
func hull(p []pt) []pt {
	p = slices.Clone(p)
	slices.SortFunc(p, func(a, b pt) int {
		if a.x != b.x {
			if a.x < b.x {
				return -1
			}
			return 1
		}
		switch {
		case a.y < b.y:
			return -1
		case a.y > b.y:
			return 1
		}
		return 0
	})
	p = slices.Compact(p)
	if len(p) < 3 {
		return p
	}
	cross := func(o, a, b pt) float64 { return (a.x-o.x)*(b.y-o.y) - (a.y-o.y)*(b.x-o.x) }
	h := make([]pt, 0, 2*len(p))
	for _, q := range p {
		for len(h) >= 2 && cross(h[len(h)-2], h[len(h)-1], q) <= 0 {
			h = h[:len(h)-1]
		}
		h = append(h, q)
	}
	lo := len(h) + 1
	for i := len(p) - 2; i >= 0; i-- {
		q := p[i]
		for len(h) >= lo && cross(h[len(h)-2], h[len(h)-1], q) <= 0 {
			h = h[:len(h)-1]
		}
		h = append(h, q)
	}
	return h[:len(h)-1]
}

// rrect is a rotated rectangle: its center, its sides along (cos, sin) and across, and the angle.
type rrect struct {
	cx, cy, w, h float64
	ux, uy       float64 // the direction of w
}

func (r rrect) corners() [4]pt {
	vx, vy := -r.uy, r.ux
	hw, hh := r.w/2, r.h/2
	return [4]pt{
		{r.cx - r.ux*hw - vx*hh, r.cy - r.uy*hw - vy*hh},
		{r.cx + r.ux*hw - vx*hh, r.cy + r.uy*hw - vy*hh},
		{r.cx + r.ux*hw + vx*hh, r.cy + r.uy*hw + vy*hh},
		{r.cx - r.ux*hw + vx*hh, r.cy - r.uy*hw + vy*hh},
	}
}

// minAreaRect is the rectangle of least area around a convex polygon: one of its sides lies on
// one of the polygon's edges (rotating calipers, here over every edge).
func minAreaRect(h []pt) rrect {
	if len(h) == 0 {
		return rrect{}
	}
	if len(h) < 3 {
		// a segment or a point: a rectangle of no width
		a, b := h[0], h[len(h)-1]
		l := math.Hypot(b.x-a.x, b.y-a.y)
		ux, uy := 1.0, 0.0
		if l > 0 {
			ux, uy = (b.x-a.x)/l, (b.y-a.y)/l
		}
		return rrect{cx: (a.x + b.x) / 2, cy: (a.y + b.y) / 2, w: l, ux: ux, uy: uy}
	}
	best := rrect{w: math.Inf(1), h: math.Inf(1)}
	for i := range h {
		a, b := h[i], h[(i+1)%len(h)]
		l := math.Hypot(b.x-a.x, b.y-a.y)
		if l == 0 {
			continue
		}
		ux, uy := (b.x-a.x)/l, (b.y-a.y)/l
		minU, maxU, minV, maxV := math.Inf(1), math.Inf(-1), math.Inf(1), math.Inf(-1)
		for _, q := range h {
			u := q.x*ux + q.y*uy
			v := -q.x*uy + q.y*ux
			minU, maxU, minV, maxV = math.Min(minU, u), math.Max(maxU, u), math.Min(minV, v), math.Max(maxV, v)
		}
		w, hh := maxU-minU, maxV-minV
		if w*hh < best.w*best.h {
			cu, cv := (minU+maxU)/2, (minV+maxV)/2
			best = rrect{cx: cu*ux - cv*uy, cy: cu*uy + cv*ux, w: w, h: hh, ux: ux, uy: uy}
		}
	}
	return best
}

// meanIn is the mean probability over the map's pixels inside a convex quadrilateral (the
// pixels' centers on it or within half a pixel, as cv2.fillPoly marks them).
func meanIn(prob []float32, W, nw, nh int, q [4]pt) float64 {
	x0, y0, x1, y1 := q[0].x, q[0].y, q[0].x, q[0].y
	for _, p := range q[1:] {
		x0, y0, x1, y1 = math.Min(x0, p.x), math.Min(y0, p.y), math.Max(x1, p.x), math.Max(y1, p.y)
	}
	xa, xb := max(0, int(math.Floor(x0))), min(nw-1, int(math.Ceil(x1)))
	ya, yb := max(0, int(math.Floor(y0))), min(nh-1, int(math.Ceil(y1)))
	// the quadrilateral's orientation, for inside tests by the edges' cross products
	sign := 1.0
	if area(q) < 0 {
		sign = -1
	}
	var sum float64
	n := 0
	for y := ya; y <= yb; y++ {
		for x := xa; x <= xb; x++ {
			in := true
			for k := range 4 {
				a, b := q[k], q[(k+1)%4]
				l := math.Hypot(b.x-a.x, b.y-a.y)
				if l == 0 {
					continue
				}
				// the signed distance of the pixel from the edge, inward positive
				if sign*((b.x-a.x)*(float64(y)-a.y)-(b.y-a.y)*(float64(x)-a.x))/l < -0.5 {
					in = false
					break
				}
			}
			if in {
				sum += float64(prob[y*W+x])
				n++
			}
		}
	}
	if n == 0 {
		return 0
	}
	return sum / float64(n)
}

func area(q [4]pt) float64 {
	var s float64
	for k := range 4 {
		a, b := q[k], q[(k+1)%4]
		s += a.x*b.y - b.x*a.y
	}
	return s / 2
}

// order puts a rectangle's corners as PaddleOCR's get_mini_boxes does: of the two leftmost, the
// upper is the first and the lower the fourth; of the two others the upper is the second and the
// lower the third.
func order(c [4]pt) [4]pt {
	s := c[:]
	slices.SortStableFunc(s, func(a, b pt) int {
		switch {
		case a.x < b.x:
			return -1
		case a.x > b.x:
			return 1
		}
		return 0
	})
	var out [4]pt
	if s[1].y > s[0].y {
		out[0], out[3] = s[0], s[1]
	} else {
		out[0], out[3] = s[1], s[0]
	}
	if s[3].y > s[2].y {
		out[1], out[2] = s[2], s[3]
	} else {
		out[1], out[2] = s[3], s[2]
	}
	return out
}

// SortBoxes puts boxes in reading order as PaddleOCR's sorted_boxes does: by their first
// corner's y and then x, and boxes whose first corners are within 10 pixels in y by x.
func SortBoxes(b [][4][2]float32) {
	slices.SortStableFunc(b, func(p, q [4][2]float32) int {
		switch {
		case p[0][1] != q[0][1]:
			if p[0][1] < q[0][1] {
				return -1
			}
			return 1
		case p[0][0] < q[0][0]:
			return -1
		case p[0][0] > q[0][0]:
			return 1
		}
		return 0
	})
	for i := 0; i+1 < len(b); i++ {
		for j := i; j >= 0; j-- {
			if math.Abs(float64(b[j+1][0][1]-b[j][0][1])) < 10 && b[j+1][0][0] < b[j][0][0] {
				b[j], b[j+1] = b[j+1], b[j]
			} else {
				break
			}
		}
	}
}

// Crop is a line's box of a picture's RGB pixels (w x h) as the recognizer reads it: the box
// straightened (PaddleOCR's get_rotate_crop_image; turned a quarter counterclockwise when it is
// half again as high as wide), scaled to Height rows and as many columns as keep its aspect, at
// most maxW. It returns the RGB pixels and the width.
func Crop(rgb []uint8, w, h int, box [4][2]float32, maxW int) ([]uint8, int) {
	p := [4]pt{}
	for k := range 4 {
		p[k] = pt{float64(box[k][0]), float64(box[k][1])}
	}
	dist := func(a, b pt) float64 { return math.Hypot(b.x-a.x, b.y-a.y) }
	cw := math.Floor(math.Max(dist(p[0], p[1]), dist(p[2], p[3])))
	ch := math.Floor(math.Max(dist(p[0], p[3]), dist(p[1], p[2])))
	if ch >= 1.5*cw {
		p = [4]pt{p[1], p[2], p[3], p[0]}
		cw, ch = ch, cw
	}
	cw, ch = math.Max(cw, 1), math.Max(ch, 1)
	ow := min(maxW, max(1, int(math.Ceil(Height*cw/ch))))
	out := make([]uint8, Height*ow*3)
	// a point of the straightened crop (s along the top edge, t down the left one, in its
	// pixels) is p0 + s/cw (p1 - p0) + t/ch (p3 - p0): the box is a rectangle
	ex, ey := (p[1].x-p[0].x)/cw, (p[1].y-p[0].y)/cw
	fx, fy := (p[3].x-p[0].x)/ch, (p[3].y-p[0].y)/ch
	sx, sy := cw/float64(ow), ch/Height
	for v := range Height {
		t := (float64(v)+0.5)*sy - 0.5
		for u := range ow {
			s := (float64(u)+0.5)*sx - 0.5
			// the crop's pixel (s, t) has its center at s + 0.5, t + 0.5 from the box's corner
			X := p[0].x + (s+0.5)*ex + (t+0.5)*fx - 0.5
			Y := p[0].y + (s+0.5)*ey + (t+0.5)*fy - 0.5
			bilinear(out[(v*ow+u)*3:], rgb, w, h, X, Y)
		}
	}
	return out, ow
}

// bilinear writes the RGB of a picture (w x h) at (x, y), the pixels' centers at integers, the
// edges repeated past it.
func bilinear(dst, rgb []uint8, w, h int, x, y float64) {
	x = math.Max(0, math.Min(float64(w-1), x))
	y = math.Max(0, math.Min(float64(h-1), y))
	x0, y0 := int(x), int(y)
	x1, y1 := min(w-1, x0+1), min(h-1, y0+1)
	ax, ay := x-float64(x0), y-float64(y0)
	for c := range 3 {
		a := float64(rgb[(y0*w+x0)*3+c])*(1-ax) + float64(rgb[(y0*w+x1)*3+c])*ax
		b := float64(rgb[(y1*w+x0)*3+c])*(1-ax) + float64(rgb[(y1*w+x1)*3+c])*ax
		dst[c] = uint8(math.Round(a*(1-ay) + b*ay))
	}
}
