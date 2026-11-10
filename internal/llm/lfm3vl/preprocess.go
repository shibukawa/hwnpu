package lfm3vl

import (
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	_ "image/gif"  // decoders for image.Decode
	_ "image/jpeg" //
	_ "image/png"  //
)

// Processor turns images into the vision encoder's input, as the checkpoint's image processor
// (preprocessor_config.json) and llama.cpp's mtmd_image_preprocessor_lfm3vl do: resize (stretch)
// to a size in multiples of twice the patch size, rescale to [0, 1], normalize by channel and cut
// into patches.
type Processor struct {
	Mean, Std  [3]float64
	Rescale    float64
	Resample   string // "bilinear" or "bicubic"
	MinPixels  int
	MaxPixels  int
	PatchSize  int
	DoNorm     bool
	DoRescaled bool
}

// LoadProcessor reads the image processor's settings from preprocessor_config.json and, over it,
// processor_config.json's image_processor, in dir (llama.cpp merges them so).
func LoadProcessor(dir string, c *Config) (*Processor, error) {
	var f struct {
		Mean     []float64 `json:"image_mean"`
		Std      []float64 `json:"image_std"`
		Rescale  *float64  `json:"rescale_factor"`
		Resample *int      `json:"resample"`
		Size     struct {
			Shortest int `json:"shortest_edge"`
			Longest  int `json:"longest_edge"`
		} `json:"size"`
		DoNorm    *bool `json:"do_normalize"`
		DoRescale *bool `json:"do_rescale"`
	}
	found := false
	for _, name := range []string{"preprocessor_config.json", "processor_config.json"} {
		raw, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			continue
		}
		found = true
		if name == "processor_config.json" {
			var wrap struct {
				ImageProcessor json.RawMessage `json:"image_processor"`
			}
			if json.Unmarshal(raw, &wrap) != nil || wrap.ImageProcessor == nil {
				continue
			}
			raw = wrap.ImageProcessor
		}
		if err := json.Unmarshal(raw, &f); err != nil {
			return nil, fmt.Errorf("lfm3vl: %s: %w", name, err)
		}
	}
	if !found {
		return nil, fmt.Errorf("lfm3vl: no preprocessor_config.json in %s", dir)
	}
	p := &Processor{Rescale: 1.0 / 255, Resample: "bilinear", MinPixels: f.Size.Shortest, MaxPixels: f.Size.Longest,
		PatchSize: c.Vision.PatchSize, DoNorm: true, DoRescaled: true}
	if len(f.Mean) != 3 || len(f.Std) != 3 || p.MaxPixels <= 0 {
		return nil, fmt.Errorf("lfm3vl: the image processor needs image_mean, image_std and size.longest_edge")
	}
	copy(p.Mean[:], f.Mean)
	copy(p.Std[:], f.Std)
	if f.Rescale != nil {
		p.Rescale = *f.Rescale
	}
	if f.DoNorm != nil {
		p.DoNorm = *f.DoNorm
	}
	if f.DoRescale != nil {
		p.DoRescaled = *f.DoRescale
	}
	if f.Resample != nil {
		// PIL's enum, as llama.cpp reads it (the torchvision backend has no lanczos)
		switch *f.Resample {
		case 2:
			p.Resample = "bilinear"
		case 3:
			p.Resample = "bicubic"
		default:
			return nil, fmt.Errorf("lfm3vl: resample %d is not supported", *f.Resample)
		}
	}
	return p, nil
}

// SmartResize is the size the processor gives a w x h image: each edge rounded to a multiple of
// twice the patch size, then scaled down or up, keeping the aspect, to fit between MinPixels and
// MaxPixels (llama.cpp's smart_resize for a still image).
func (p *Processor) SmartResize(w, h int) (int, int) {
	f := 2 * p.PatchSize
	fw, fh := float64(w), float64(h)
	ff := float64(f)
	hb := int(math.RoundToEven(fh/ff)) * f
	wb := int(math.RoundToEven(fw/ff)) * f
	if p.MaxPixels > 0 && hb*wb > p.MaxPixels {
		beta := math.Sqrt(fh * fw / float64(p.MaxPixels))
		hb = int(math.Floor(fh/beta/ff)) * f
		wb = int(math.Floor(fw/beta/ff)) * f
	} else if hb*wb < p.MinPixels {
		beta := math.Sqrt(float64(p.MinPixels) / (fh * fw))
		hb = int(math.Ceil(fh*beta/ff)) * f
		wb = int(math.Ceil(fw*beta/ff)) * f
	}
	return max(f, wb), max(f, hb)
}

// Patches resizes img to w x h and returns the normalized patches (patches, 3*patch*patch) in
// raster order, each channel then row then column: the input x of the vision encoder's program
// built for that size.
func (p *Processor) Patches(img image.Image, w, h int) []float32 {
	rgb := Resize(img, w, h, p.Resample)
	ps := p.PatchSize
	gw, gh := w/ps, h/ps
	dim := 3 * ps * ps
	out := make([]float32, gw*gh*dim)
	for py := 0; py < gh; py++ {
		for px := 0; px < gw; px++ {
			patch := out[(py*gw+px)*dim:]
			for c := 0; c < 3; c++ {
				for y := 0; y < ps; y++ {
					for x := 0; x < ps; x++ {
						v := float64(rgb[((py*ps+y)*w+px*ps+x)*3+c])
						if p.DoRescaled {
							v *= p.Rescale
						}
						if p.DoNorm {
							v = (v - p.Mean[c]) / p.Std[c]
						}
						patch[(c*ps+y)*ps+x] = float32(v)
					}
				}
			}
		}
	}
	return out
}

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
	switch m := img.(type) {
	case *image.NRGBA:
		for y := 0; y < sh; y++ {
			row := m.Pix[(y+b.Min.Y-m.Rect.Min.Y)*m.Stride+(b.Min.X-m.Rect.Min.X)*4:]
			for x := 0; x < sw; x++ {
				copy(src[(y*sw+x)*3:(y*sw+x)*3+3], row[4*x:4*x+3])
			}
		}
		return src
	case *image.RGBA:
		if opaque(m) {
			for y := 0; y < sh; y++ {
				row := m.Pix[(y+b.Min.Y-m.Rect.Min.Y)*m.Stride+(b.Min.X-m.Rect.Min.X)*4:]
				for x := 0; x < sw; x++ {
					copy(src[(y*sw+x)*3:(y*sw+x)*3+3], row[4*x:4*x+3])
				}
			}
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
	clamp := func(acc, sum float64) uint8 {
		if sum != 0 {
			acc /= sum
		}
		return uint8(math.Max(0, math.Min(255, math.Round(acc))))
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
