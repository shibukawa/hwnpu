package lfm3vl

import (
	"encoding/json"
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"

	_ "image/gif"  // decoders for image.Decode
	_ "image/jpeg" //
	_ "image/png"  //

	"github.com/shibukawa/hwnpu/internal/pixels"
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
// and torchvision's resize with antialias do (pixels.Resize): the pixels as RGB bytes in raster
// order.
func Resize(img image.Image, w, h int, filter string) []uint8 { return pixels.Resize(img, w, h, filter) }

// RGB is img's pixels as RGB bytes in raster order (pixels.RGB).
func RGB(img image.Image) []uint8 { return pixels.RGB(img) }

// ResizeRGB is Resize of RGB bytes src (sw x sh); the same size is src itself (pixels.ResizeRGB).
func ResizeRGB(src []uint8, sw, sh, w, h int, filter string) []uint8 {
	return pixels.ResizeRGB(src, sw, sh, w, h, filter)
}
