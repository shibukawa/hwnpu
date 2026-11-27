// Package pixels resizes pictures as PIL and torchvision do, for the models' preprocessing
// (lfm3vl, the vision models) and for the clients that draw on what the models give back
// (hwnpu-image depth --ply). It reads nothing of hwnpu's, so that a client without the engine
// links it (docs/design/hwnpud.md, "Clients without the engine").
package pixels

import (
	"image"
	"image/color"
	"math"
	"runtime"
	"sync"
)

// Resize scales img to w x h (stretching) with an antialiased bilinear or bicubic filter, as PIL
// and torchvision's resize with antialias do: a horizontal pass, then a vertical one, each output
// rounded to 8 bits. It returns the pixels as RGB bytes in raster order. The filters are computed
// in floating point; the libraries' fixed point arithmetic can differ by one in a few pixels.
func Resize(img image.Image, w, h int, filter string) []uint8 {
	b := img.Bounds()
	return ResizeRGB(RGB(img), b.Dx(), b.Dy(), w, h, filter)
}

// RGB is img's pixels as RGB bytes in raster order (alpha dropped, as color.NRGBA's R, G, B).
// RGBA, NRGBA and 4:2:0 / 4:4:4 YCbCr pictures (PNG, JPEG) take a fast path.
func RGB(img image.Image) []uint8 {
	b := img.Bounds()
	sw, sh := b.Dx(), b.Dy()
	src := make([]uint8, sw*sh*3)
	// four bytes a pixel to three, a row at a time on the cores
	drop := func(pix []uint8, stride, x0, y0 int) {
		rows(sh, func(lo, hi int) {
			for y := lo; y < hi; y++ {
				in := pix[(y+y0)*stride+4*x0:][:4*sw]
				out := src[y*sw*3:][:3*sw]
				for x := range sw {
					p, q := in[4*x:4*x+3:4*x+3], out[3*x:3*x+3:3*x+3]
					q[0], q[1], q[2] = p[0], p[1], p[2]
				}
			}
		})
	}
	switch m := img.(type) {
	case *image.NRGBA:
		drop(m.Pix, m.Stride, b.Min.X-m.Rect.Min.X, b.Min.Y-m.Rect.Min.Y)
		return src
	case *image.RGBA:
		if opaque(m) {
			drop(m.Pix, m.Stride, b.Min.X-m.Rect.Min.X, b.Min.Y-m.Rect.Min.Y)
			return src
		}
	case *image.YCbCr:
		rows(sh, func(lo, hi int) {
			for y := lo; y < hi; y++ {
				for x := 0; x < sw; x++ {
					// RGBA's conversion, as NRGBAModel's (YCbCrToRGB can differ by one)
					px, py := b.Min.X+x, b.Min.Y+y
					ci := m.COffset(px, py)
					r, g, bl := ycbcrRGBA(m.Y[m.YOffset(px, py)], m.Cb[ci], m.Cr[ci])
					i := (y*sw + x) * 3
					src[i], src[i+1], src[i+2] = uint8(r>>8), uint8(g>>8), uint8(bl>>8)
				}
			}
		})
		return src
	}
	for y := 0; y < sh; y++ {
		for x := 0; x < sw; x++ {
			c := color.NRGBAModel.Convert(img.At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA)
			i := (y*sw + x) * 3
			src[i], src[i+1], src[i+2] = c.R, c.G, c.B
		}
	}
	return src
}

// opaque reports whether every pixel of m has alpha 255 (its R, G, B are then NRGBA's).
func opaque(m *image.RGBA) bool {
	for y := m.Rect.Min.Y; y < m.Rect.Max.Y; y++ {
		row := m.Pix[(y-m.Rect.Min.Y)*m.Stride:]
		for x := 0; x < m.Rect.Dx(); x++ {
			if row[4*x+3] != 255 {
				return false
			}
		}
	}
	return true
}

// ResizeRGB is Resize of RGB bytes src (sw x sh); the same size is src itself.
func ResizeRGB(src []uint8, sw, sh, w, h int, filter string) []uint8 {
	if sw == w && sh == h {
		return src
	}
	support, kernel := 1.0, func(x float64) float64 { return math.Max(0, 1-math.Abs(x)) }
	if filter == "bicubic" {
		const a = -0.5
		support = 2
		kernel = func(x float64) float64 {
			x = math.Abs(x)
			switch {
			case x < 1:
				return ((a+2)*x-(a+3))*x*x + 1
			case x < 2:
				return (((x-5)*x+8)*x - 4) * a
			}
			return 0
		}
	}
	// taps are each output pixel's source pixels from lo and their weights, PIL's
	taps := func(in, out int) (los []int, ws [][]float64, sums []float64) {
		scale := float64(in) / float64(out)
		fs := math.Max(scale, 1)
		sup := support * fs
		for o := 0; o < out; o++ {
			center := (float64(o) + 0.5) * scale
			lo := max(int(center-sup+0.5), 0)
			hi := min(int(center+sup+0.5), in)
			wt := make([]float64, hi-lo)
			var sum float64
			for j := range wt {
				wt[j] = kernel((float64(j+lo) - center + 0.5) / fs)
				sum += wt[j]
			}
			los, ws, sums = append(los, lo), append(ws, wt), append(sums, sum)
		}
		return los, ws, sums
	}
	// the built-in min and max: math.Max and Min (calls, for NaN and signed zeros, which the sums
	// are not) were a third of the CPU time (512² to 640²: 4.2 ms on an M2's 8 cores)
	clamp := func(acc, sum float64) uint8 {
		if sum != 0 {
			acc /= sum
		}
		return uint8(min(max(math.Round(acc), 0), 255))
	}
	// across: each row of sw pixels to w (the sums in the order of the taps, as PIL's)
	los, ws, sums := taps(sw, w)
	tmp := make([]uint8, sh*w*3)
	rows(sh, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			in, out := src[y*sw*3:(y+1)*sw*3], tmp[y*w*3:(y+1)*w*3]
			for o, lo := range los {
				var r, g, b float64
				for j, wt := range ws[o] {
					p := in[(lo+j)*3:]
					r += wt * float64(p[0])
					g += wt * float64(p[1])
					b += wt * float64(p[2])
				}
				out[o*3], out[o*3+1], out[o*3+2] = clamp(r, sums[o]), clamp(g, sums[o]), clamp(b, sums[o])
			}
		}
	})
	// down: each column of sh pixels to h, a row of the result at a time
	los, ws, sums = taps(sh, h)
	dst := make([]uint8, h*w*3)
	rows(h, func(o0, o1 int) {
		acc := make([]float64, w*3)
		for o := o0; o < o1; o++ {
			clear(acc)
			for j, wt := range ws[o] {
				in := tmp[(los[o]+j)*w*3:][:w*3]
				for i, v := range in {
					acc[i] += wt * float64(v)
				}
			}
			out := dst[o*w*3:][:w*3]
			for i, a := range acc {
				out[i] = clamp(a, sums[o])
			}
		}
	})
	return dst
}

// rows calls f over [0, n) in parts, at once on the CPU's cores (parts of at least 16 rows).
func rows(n int, f func(lo, hi int)) {
	parts := min(runtime.GOMAXPROCS(0), max(1, n/16))
	if parts <= 1 {
		f(0, n)
		return
	}
	var wg sync.WaitGroup
	for k := range parts {
		lo, hi := n*k/parts, n*(k+1)/parts
		wg.Add(1)
		go func() {
			defer wg.Done()
			f(lo, hi)
		}()
	}
	wg.Wait()
}

// ycbcrRGBA is color.YCbCr.RGBA's R, G, B (0 to 0xffff).
func ycbcrRGBA(y, cb, cr uint8) (uint32, uint32, uint32) {
	yy1 := int32(y) * 0x10101
	cb1 := int32(cb) - 128
	cr1 := int32(cr) - 128
	r := yy1 + 91881*cr1
	if uint32(r)&0xff000000 == 0 {
		r >>= 8
	} else {
		r = ^(r >> 31) & 0xffff
	}
	g := yy1 - 22554*cb1 - 46802*cr1
	if uint32(g)&0xff000000 == 0 {
		g >>= 8
	} else {
		g = ^(g >> 31) & 0xffff
	}
	b := yy1 + 116130*cb1
	if uint32(b)&0xff000000 == 0 {
		b >>= 8
	} else {
		b = ^(b >> 31) & 0xffff
	}
	return uint32(r), uint32(g), uint32(b)
}
