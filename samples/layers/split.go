package main

import (
	"image"
	"image/color"
	"math"
	"slices"

	"golang.org/x/image/draw"
)

// curve pulls the far end of a nearness apart: n becomes log(1 + k n) / log(1 + k). A nearness
// is the inverse of the depth, so behind something right before the eye all the rest is almost
// 0; on the curve the layers behind get a share of the range, and a cut can go between them.
func curve(near []float32, k float64) []float32 {
	out := make([]float32, len(near))
	scale := 1 / math.Log1p(k)
	for i, n := range near {
		out[i] = float32(math.Log1p(k*float64(min(max(n, 0), 1))) * scale)
	}
	return out
}

// bins are the steps cuts counts the nearness in.
const bins = 256

// cuts are the n - 1 values that part the values of v (0 to 1) into n classes with the least
// spread inside each (natural breaks: the sum of the classes' squared distances from their
// means is the smallest there is), ascending. A cut is between two classes' values.
func cuts(v []float32, n int) []float32 {
	// the counts and the sums of the bins, then of every run of bins from the first
	var count, sum, sq [bins + 1]float64
	for _, x := range v {
		b := min(int(min(max(x, 0), 1)*bins), bins-1)
		count[b+1]++
	}
	for b := 1; b <= bins; b++ {
		mid := (float64(b) - 0.5) / bins
		sum[b] = sum[b-1] + count[b]*mid
		sq[b] = sq[b-1] + count[b]*mid*mid
		count[b] += count[b-1]
	}
	// spread is the sum of squared distances from the mean of the bins lo (included) to hi (not)
	spread := func(lo, hi int) float64 {
		c := count[hi] - count[lo]
		if c == 0 {
			return 0
		}
		s := sum[hi] - sum[lo]
		return sq[hi] - sq[lo] - s*s/c
	}
	n = min(max(n, 1), bins)
	// best[k][b] is the least spread of the bins before b in k + 1 classes; from[k][b] where the
	// last of them starts
	best := make([][bins + 1]float64, n)
	from := make([][bins + 1]int, n)
	for b := 0; b <= bins; b++ {
		best[0][b] = spread(0, b)
	}
	for k := 1; k < n; k++ {
		for b := k + 1; b <= bins; b++ {
			best[k][b] = math.Inf(1)
			for s := k; s < b; s++ {
				if t := best[k-1][s] + spread(s, b); t < best[k][b] {
					best[k][b], from[k][b] = t, s
				}
			}
		}
	}
	// a cut goes in the middle of the empty bins between two classes
	out := make([]float32, n-1)
	for k, b := n-1, bins; k >= 1; k-- {
		b = from[k][b]
		lo, hi := b, b
		for lo > 0 && count[lo] == count[lo-1] {
			lo--
		}
		for hi < bins && count[hi+1] == count[hi] {
			hi++
		}
		out[k-1] = float32(lo+hi) / (2 * bins)
	}
	return out
}

// above is 1 where v is over cut and 0 elsewhere.
func above(v []float32, cut float32) []float32 {
	out := make([]float32, len(v))
	for i, x := range v {
		if x > cut {
			out[i] = 1
		}
	}
	return out
}

// window runs op over every pixel's neighbours within r of it, across and then down.
func window(m []float32, w, h, r int, op func(run []float32) float32) []float32 {
	if r <= 0 {
		return m
	}
	pass := func(src []float32, n, stride, lines, lineStride int) []float32 {
		dst := make([]float32, len(src))
		run := make([]float32, 0, 2*r+1)
		for l := 0; l < lines; l++ {
			for i := 0; i < n; i++ {
				run = run[:0]
				for k := max(i-r, 0); k <= min(i+r, n-1); k++ {
					run = append(run, src[l*lineStride+k*stride])
				}
				dst[l*lineStride+i*stride] = op(run)
			}
		}
		return dst
	}
	return pass(pass(m, w, 1, h, w), h, w, w, 1)
}

// grow makes a mask r pixels larger: every pixel takes the largest value within r of it.
func grow(m []float32, w, h, r int) []float32 {
	return window(m, w, h, r, func(run []float32) float32 {
		v := run[0]
		for _, x := range run[1:] {
			v = max(v, x)
		}
		return v
	})
}

// soften makes a mask's edge a ramp 2r pixels wide: every pixel takes the mean within r of it.
func soften(m []float32, w, h, r int) []float32 {
	return window(m, w, h, r, func(run []float32) float32 {
		var sum float32
		for _, x := range run {
			sum += x
		}
		return sum / float32(len(run))
	})
}

// covered is the share of a mask that is over a half.
func covered(m []float32) float64 {
	n := 0
	for _, x := range m {
		if x > 0.5 {
			n++
		}
	}
	return float64(n) / float64(max(len(m), 1))
}

// key is a colour for the part of a picture to be drawn again, with its name for the prompt:
// of a few plain colours, the one the picture has the least of.
func key(img *image.RGBA) (color.RGBA, string) {
	type named struct {
		c    color.RGBA
		name string
	}
	keys := []named{
		{color.RGBA{0, 255, 0, 255}, "green"}, {color.RGBA{255, 0, 255, 255}, "magenta"},
		{color.RGBA{0, 0, 255, 255}, "blue"}, {color.RGBA{255, 0, 0, 255}, "red"},
	}
	near := make([]int, len(keys))
	for i := 0; i < len(img.Pix); i += 4 * 7 { // every seventh pixel is enough
		for k, c := range keys {
			dr, dg, db := int(img.Pix[i])-int(c.c.R), int(img.Pix[i+1])-int(c.c.G), int(img.Pix[i+2])-int(c.c.B)
			if dr*dr+dg*dg+db*db < 150*150 {
				near[k]++
			}
		}
	}
	best := 0
	for k := range keys {
		if near[k] < near[best] {
			best = k
		}
	}
	return keys[best].c, keys[best].name
}

// paint is img with the pixels of a mask (over a half) in colour c.
func paint(img *image.RGBA, m []float32, c color.RGBA) *image.RGBA {
	out := image.NewRGBA(img.Rect)
	copy(out.Pix, img.Pix)
	for i, x := range m {
		if x > 0.5 {
			out.Pix[4*i], out.Pix[4*i+1], out.Pix[4*i+2], out.Pix[4*i+3] = c.R, c.G, c.B, 255
		}
	}
	return out
}

// mix is a where the weight is 0 and b where it is 1, and between them between.
func mix(a, b *image.RGBA, weight []float32) *image.RGBA {
	out := image.NewRGBA(a.Rect)
	for i, t := range weight {
		for k := 0; k < 4; k++ {
			x, y := float32(a.Pix[4*i+k]), float32(b.Pix[4*i+k])
			out.Pix[4*i+k] = uint8(x + (y-x)*t + 0.5)
		}
	}
	return out
}

// cutOut is img with a mask as its alpha: a layer.
func cutOut(img *image.RGBA, m []float32) *image.NRGBA {
	out := image.NewNRGBA(img.Rect)
	for i, a := range m {
		out.Pix[4*i], out.Pix[4*i+1], out.Pix[4*i+2] = img.Pix[4*i], img.Pix[4*i+1], img.Pix[4*i+2]
		out.Pix[4*i+3] = uint8(min(max(a, 0), 1)*255 + 0.5)
	}
	return out
}

// fit is the a and b for which a*x + b is nearest y over the pixels where the mask is 0 (least
// squares): how a second estimate of the nearness, x, is brought to the scale of the first, y,
// from the part of the picture both saw. Without enough of it, x is taken as it is.
func fit(x, y, mask []float32) (a, b float64) {
	var n, sx, sy, sxx, sxy float64
	for i := range x {
		if mask[i] > 0 {
			continue
		}
		u, v := float64(x[i]), float64(y[i])
		n++
		sx += u
		sy += v
		sxx += u * u
		sxy += u * v
	}
	den := n*sxx - sx*sx
	if n < 16 || math.Abs(den) < 1e-9 {
		return 1, 0
	}
	a = (n*sxy - sx*sy) / den
	return a, (sy - a*sx) / n
}

// stretch is img at w x h (its shape is not kept).
func stretch(img image.Image, w, h int) *image.RGBA {
	out := image.NewRGBA(image.Rect(0, 0, w, h))
	b := img.Bounds()
	if b.Dx() == w && b.Dy() == h {
		draw.Draw(out, out.Rect, img, b.Min, draw.Src)
		return out
	}
	draw.CatmullRom.Scale(out, out.Rect, img, b, draw.Src, nil)
	return out
}

// seam is how much a picture differs across the edge of a mask, 0 to 441: the picture is
// looked at in blocks along the edge, each block's mean colour just inside the mask against
// that just outside, and the difference three quarters of the blocks stay under is the answer.
// A part drawn again that goes on from its surroundings scores about 20; one left as a flat
// shape (the colour it was painted in, or a shadow of what was there) over 80.
func seam(img *image.RGBA, m []float32, w, h int) float64 {
	band := max(2, w/128)
	out := make([]float32, len(m))
	for i, x := range m {
		if x <= 0.5 {
			out[i] = 1
		}
	}
	nearOut, nearIn := grow(out, w, h, band), grow(m, w, h, band)
	block := 4 * band
	var diffs []float64
	for by := 0; by < h; by += block {
		for bx := 0; bx < w; bx += block {
			var in, ex [3]float64
			var nIn, nEx float64
			for y := by; y < min(by+block, h); y++ {
				for x := bx; x < min(bx+block, w); x++ {
					i := y*w + x
					switch {
					case m[i] > 0.5 && nearOut[i] > 0.5: // inside, by the edge
						nIn++
						for k := range in {
							in[k] += float64(img.Pix[4*i+k])
						}
					case m[i] <= 0.5 && nearIn[i] > 0.5: // outside, by the edge
						nEx++
						for k := range ex {
							ex[k] += float64(img.Pix[4*i+k])
						}
					}
				}
			}
			if nIn < 16 || nEx < 16 {
				continue
			}
			var d float64
			for k := range in {
				d += (in[k]/nIn - ex[k]/nEx) * (in[k]/nIn - ex[k]/nEx)
			}
			diffs = append(diffs, math.Sqrt(d))
		}
	}
	if len(diffs) == 0 {
		return 0
	}
	slices.Sort(diffs)
	return diffs[len(diffs)*3/4]
}
