package lfm3vl

import (
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"math"
	"os"
	"path/filepath"

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
	sw, sh := b.Dx(), b.Dy()
	src := make([]uint8, sw*sh*3)
	for y := 0; y < sh; y++ {
		for x := 0; x < sw; x++ {
			c := color.NRGBAModel.Convert(img.At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA)
			i := (y*sw + x) * 3
			src[i], src[i+1], src[i+2] = c.R, c.G, c.B
		}
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
	// pass resamples n lines of in pixels to out pixels; at and put give the offset of pixel i
	// of a line in the source and in the result
	pass := func(data []uint8, in, out, n int, at, put func(line, i int) int) []uint8 {
		dst := make([]uint8, n*out*3)
		scale := float64(in) / float64(out)
		fs := math.Max(scale, 1)
		sup := support * fs
		for o := 0; o < out; o++ {
			center := (float64(o) + 0.5) * scale
			lo := max(int(center-sup+0.5), 0)
			hi := min(int(center+sup+0.5), in)
			ws := make([]float64, hi-lo)
			var sum float64
			for j := range ws {
				ws[j] = kernel((float64(j+lo) - center + 0.5) / fs)
				sum += ws[j]
			}
			for line := 0; line < n; line++ {
				for c := 0; c < 3; c++ {
					var acc float64
					for j, wt := range ws {
						acc += wt * float64(data[at(line, lo+j)+c])
					}
					if sum != 0 {
						acc /= sum
					}
					dst[put(line, o)+c] = uint8(math.Max(0, math.Min(255, math.Round(acc))))
				}
			}
		}
		return dst
	}
	tmp := pass(src, sw, w, sh, func(y, x int) int { return (y*sw + x) * 3 }, func(y, x int) int { return (y*w + x) * 3 })
	return pass(tmp, sh, h, w, func(x, y int) int { return (y*w + x) * 3 }, func(x, y int) int { return (y*w + x) * 3 })
}
