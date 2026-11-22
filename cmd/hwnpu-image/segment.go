package main

import (
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/shibukawa/hwnpu/ai/vision"
	"github.com/shibukawa/hwnpu/internal/console"
)

// SegmentCmd is "hwnpu-image segment": the mask of the thing a prompt picks in pictures, as
// Segment Anything does, with EfficientSAM (Meta's, Apache-2.0; ai/vision's Segment). The
// prompt is points in the thing (--point), out of it (--negative), or a box (--box), in the
// picture's pixels. The image encoder runs on AMD's NPU (an export of
// tools/amd/iron_efficientsam.py, made with IRON on the first run), on Apple's Neural Engine and
// Intel's NPU (its ONNX file converted to a Core AI program or OpenVINO IR on the first run),
// elsewhere or with --device cpu on the CPU (ONNX Runtime when its library is there, else
// hwnpu's evaluator: slow); the mask decoder on the host.
//
// For each picture it prints one line of JSON: the mask's score (the IoU the decoder predicts),
// its area and box, and the files it wrote: NAME.mask.png (white on the thing) and
// NAME.cutout.png (the picture, transparent off the thing). The decoder gives three candidates,
// from the whole thing to a part; the best scored is kept, all three with --all.
type SegmentCmd struct {
	Pictures    []string `arg:"" type:"existingfile" help:"The pictures (JPEG or PNG)."`
	Model       string   `short:"m" default:"segment" placeholder:"NAME" help:"The model: a name, a use or a family of models.yaml (segment, efficientsam, efficientsam:s)."`
	Point       []string `short:"p" sep:"none" placeholder:"X,Y" help:"A point on the thing, in the picture's pixels (repeatable)."`
	Negative    []string `sep:"none" placeholder:"X,Y" help:"A point off the thing (repeatable)."`
	Box         string   `placeholder:"X1,Y1,X2,Y2" help:"A box around the thing."`
	All         bool     `help:"Write all three candidate masks (NAME.mask0.png ...) instead of the best scored."`
	Out         string   `short:"o" type:"path" placeholder:"DIR" help:"Where to write the masks (default: next to each picture)."`
	Device      string   `enum:",npu,cpu" default:"" help:"Where to run the encoder: npu or cpu (default: models.yaml's device for this machine's NPU, or HWNPU_DEVICE)."`
	Threads     int      `default:"1" help:"The CPU's threads with --device cpu (1: the least CPU time and energy)."`
	backendFlag `embed:""`
}

// Run cuts out the prompt's thing in each picture.
func (c *SegmentCmd) Run(ctx context.Context) error {
	if err := outDir(c.Out); err != nil {
		return err
	}
	lw := console.Log(os.Stderr)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	prompt, err := c.prompt()
	if err != nil {
		return err
	}
	useBackend(c.Backend)
	how := made{device: c.Device}
	how.threads(c.Threads)
	sm, rt, loaded, err := load(ctx, c.Model, vision.Segment, how, lw)
	if err != nil {
		return err
	}
	defer rt.Close()
	logf("%s, the encoder on %s", loaded, sm.Where())
	console.Running()
	enc := json.NewEncoder(os.Stdout)
	for _, p := range c.Pictures {
		img, err := readPicture(p)
		if err != nil {
			return err
		}
		e, err := sm.Encode(ctx, img)
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		masks, err := sm.Masks(e, prompt)
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		logf("%s: %.0f ms (resize %.0f, encoder %.0f, decoder %.0f)", filepath.Base(p), ms(e.Took+masks.Took), ms(e.Resize), ms(e.Encode), ms(masks.Took))
		keep := []int{masks.Best()}
		if c.All {
			keep = []int{0, 1, 2}[:len(masks.Masks)]
		}
		d := c.Out
		if d == "" {
			d = filepath.Dir(p)
		}
		base := filepath.Join(d, strings.TrimSuffix(filepath.Base(p), filepath.Ext(p)))
		type out struct {
			Score  float64 `json:"score"`
			Area   int     `json:"area"`
			Box    [4]int  `json:"box"`
			Mask   string  `json:"mask"`
			Cutout string  `json:"cutout"`
		}
		var outs []out
		for _, j := range keep {
			suffix := ""
			if c.All {
				suffix = strconv.Itoa(j)
			}
			k := &masks.Masks[j]
			r := k.Bounds()
			o := out{Score: k.Score, Area: k.Area(), Box: [4]int{r.Min.X, r.Min.Y, r.Max.X, r.Max.Y}, Mask: base + ".mask" + suffix + ".png", Cutout: base + ".cutout" + suffix + ".png"}
			if err := writeMask(o.Mask, o.Cutout, img, k); err != nil {
				return err
			}
			outs = append(outs, o)
		}
		if err := enc.Encode(struct {
			File   string `json:"file"`
			Width  int    `json:"width"`
			Height int    `json:"height"`
			Masks  []out  `json:"masks"`
		}{p, img.Bounds().Dx(), img.Bounds().Dy(), outs}); err != nil {
			return err
		}
	}
	return nil
}

// prompt is the points of --point, --negative and --box.
func (c *SegmentCmd) prompt() (vision.Prompt, error) {
	nums := func(s string, n int) ([]float64, error) {
		f := strings.Split(s, ",")
		if len(f) != n {
			return nil, fmt.Errorf("%q is not %d numbers separated by commas", s, n)
		}
		out := make([]float64, n)
		for i, v := range f {
			x, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
			if err != nil {
				return nil, fmt.Errorf("%q: %w", s, err)
			}
			out[i] = x
		}
		return out, nil
	}
	var p vision.Prompt
	for _, list := range []struct {
		flags []string
		to    *[]vision.Point
	}{{c.Point, &p.Points}, {c.Negative, &p.Negative}} {
		for _, s := range list.flags {
			v, err := nums(s, 2)
			if err != nil {
				return p, err
			}
			*list.to = append(*list.to, vision.Point{X: v[0], Y: v[1]})
		}
	}
	if c.Box != "" {
		v, err := nums(c.Box, 4)
		if err != nil {
			return p, err
		}
		p.Box = &vision.Box{Min: vision.Point{X: v[0], Y: v[1]}, Max: vision.Point{X: v[2], Y: v[3]}}
	}
	if len(p.Points) == 0 && len(p.Negative) == 0 && p.Box == nil {
		return p, fmt.Errorf("no prompt: give --point X,Y (on the thing), --negative X,Y or --box X1,Y1,X2,Y2")
	}
	return p, nil
}

// writeMask writes the mask (white on the thing, black off it) and the cutout (the picture,
// transparent off the thing) as PNGs.
func writeMask(maskPath, cutoutPath string, img image.Image, k *vision.Mask) error {
	for path, im := range map[string]image.Image{maskPath: k.Image(), cutoutPath: k.Cutout(img)} {
		f, err := os.Create(path)
		if err != nil {
			return err
		}
		if err := png.Encode(f, im); err != nil {
			f.Close()
			return err
		}
		if err := f.Close(); err != nil {
			return err
		}
	}
	return nil
}
