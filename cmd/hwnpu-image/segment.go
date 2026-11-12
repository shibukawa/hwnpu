package main

import (
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/vision/segment"
)

// SegmentCmd is "hwnpu-image segment": the mask of the thing a prompt picks in pictures, as
// Segment Anything does, with EfficientSAM (Meta's, Apache-2.0; internal/vision/segment). The
// prompt is points in the thing (--point), out of it (--negative), or a box (--box), in the
// picture's pixels. The image encoder runs on AMD's NPU (an export of
// tools/amd/iron_efficientsam.py, made with IRON on the first run), on Intel's NPU (its ONNX
// file converted to OpenVINO IR on the first run), elsewhere or with --device cpu on the CPU (ONNX Runtime when its library is there, else hwnpu's evaluator: slow); the mask
// decoder on the host.
//
// For each picture it prints one line of JSON: the mask's score (the IoU the decoder predicts),
// its area and box, and the files it wrote: NAME.mask.png (white on the thing) and
// NAME.cutout.png (the picture, transparent off the thing). The decoder gives three candidates,
// from the whole thing to a part; the best scored is kept, all three with --all.
type SegmentCmd struct {
	Pictures []string `arg:"" type:"existingfile" help:"The pictures (JPEG or PNG)."`
	Model    string   `short:"m" default:"segment" placeholder:"NAME" help:"The model: a name, a use or a family of models.yaml (segment, efficientsam, efficientsam:s)."`
	Point    []string `short:"p" sep:"none" placeholder:"X,Y" help:"A point on the thing, in the picture's pixels (repeatable)."`
	Negative []string `sep:"none" placeholder:"X,Y" help:"A point off the thing (repeatable)."`
	Box      string   `placeholder:"X1,Y1,X2,Y2" help:"A box around the thing."`
	All      bool     `help:"Write all three candidate masks (NAME.mask0.png ...) instead of the best scored."`
	Out      string   `short:"o" type:"path" placeholder:"DIR" help:"Where to write the masks (default: next to each picture)."`
	Dir      string   `type:"path" placeholder:"DIR" help:"Where the models are kept (as hwnpu-chat's --dir)."`
	Device   string   `enum:",npu,cpu" default:"" help:"Where to run the encoder: npu or cpu (default: models.yaml's device for this machine's NPU, or HWNPU_DEVICE)."`
	Threads  int      `default:"1" help:"The CPU's threads with --device cpu (1: the least CPU time and energy)."`
}

// Run cuts out the prompt's thing in each picture.
func (c *SegmentCmd) Run(ctx context.Context) error {
	lw := console.Log(os.Stderr)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	prompt, err := c.prompt()
	if err != nil {
		return err
	}
	m, err := resolve(c.Model, chat.KindSegment)
	if err != nil {
		return err
	}
	var encFile, decFile string
	for f := range m.SHA256 {
		switch {
		case strings.HasSuffix(f, "_encoder.onnx"):
			encFile = f
		case strings.HasSuffix(f, "_decoder.onnx"):
			decFile = f
		}
	}
	if encFile == "" || decFile == "" {
		return fmt.Errorf("%s: models.yaml names no *_encoder.onnx and *_decoder.onnx in its sha256", m.Name)
	}
	dir := c.Dir
	if dir == "" {
		dir = chat.ModelsDir()
	}
	setup := segment.Setup{Dir: filepath.Join(dir, m.DirName()), Repo: m.Source(), Encoder: encFile, Decoder: decFile, SHA256: m.SHA256, Log: lw}
	t0 := time.Now()
	var sm *segment.Model
	fetchEnc := func() (string, error) { e, _, err := segment.Fetch(ctx, setup); return e, err }
	have := func(dir string) error { _, err := os.Stat(filepath.Join(dir, "model.json")); return err }
	if export := amdFirstRunExport(ctx, m, c.Device, false, logf, have, fetchEnc, func(python []string, model, dir string) error {
		return segment.ExportNPU(ctx, python, m.Name, model, dir, lw)
	}); export != "" {
		_, dec, err := segment.Fetch(ctx, setup)
		if err != nil {
			return err
		}
		dev, err := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
		if err != nil {
			return err
		}
		defer dev.Close()
		if sm, err = segment.OpenNPU(ctx, dev, export, dec); err != nil {
			return err
		}
	} else if vendor := chat.HereVendor(); c.Device != "cpu" && (vendor == "intel" || vendor == "apple" && c.Device == "npu") {
		// Intel's NPU runs the encoder converted from its ONNX file to OpenVINO IR; Apple's the
		// same as Core ML only when asked (--device npu: not yet measured on the Neural Engine)
		enc, dec, err := segment.Fetch(ctx, setup)
		if err != nil {
			return err
		}
		var opts []hwnpu.Option
		if vendor == "apple" {
			opts = append(opts, hwnpu.WithPlacement(hwnpu.PlacementStrict))
		}
		dev, backend, err := llm.OpenDevice(ctx, "", cacheDirOf(""), opts...)
		if err != nil {
			return err
		}
		defer dev.Close()
		if sm, err = segment.OpenProgram(ctx, dev, backend, enc, dec, filepath.Dir(enc), lw); err != nil {
			return err
		}
	} else {
		enc, dec, err := segment.Fetch(ctx, setup)
		if err != nil {
			return err
		}
		if sm, err = segment.OpenCPU(enc, dec, c.Threads); err != nil {
			return err
		}
	}
	defer sm.Close()
	logf("loaded %s in %.1f s, the encoder on %s", m.Name, time.Since(t0).Seconds(), sm.Where())
	console.Running()
	enc := json.NewEncoder(os.Stdout)
	for _, p := range c.Pictures {
		img, err := readPicture(p)
		if err != nil {
			return err
		}
		start := time.Now()
		e, err := sm.Embed(ctx, img)
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		t1 := time.Now()
		masks, err := sm.Segment(e, prompt)
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		logf("%s: %.0f ms (resize %.0f, encoder %.0f, decoder %.0f)", filepath.Base(p), ms(time.Since(start)), ms(sm.Times[0]), ms(sm.Times[1]), ms(time.Since(t1)))
		keep := []int{best(masks)}
		if c.All {
			keep = []int{0, 1, 2}[:len(masks)]
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
			o := out{Score: masks[j].Score, Area: masks[j].Area(), Box: masks[j].Box(), Mask: base + ".mask" + suffix + ".png", Cutout: base + ".cutout" + suffix + ".png"}
			if err := writeMask(o.Mask, o.Cutout, img, &masks[j]); err != nil {
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
func (c *SegmentCmd) prompt() ([]segment.Point, error) {
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
	var pts []segment.Point
	for label, list := range map[int][]string{1: c.Point, 0: c.Negative} {
		for _, s := range list {
			v, err := nums(s, 2)
			if err != nil {
				return nil, err
			}
			pts = append(pts, segment.Point{X: v[0], Y: v[1], Label: label})
		}
	}
	if c.Box != "" {
		v, err := nums(c.Box, 4)
		if err != nil {
			return nil, err
		}
		pts = append(pts, segment.Point{X: v[0], Y: v[1], Label: 2}, segment.Point{X: v[2], Y: v[3], Label: 3})
	}
	if len(pts) == 0 {
		return nil, fmt.Errorf("no prompt: give --point X,Y (on the thing), --negative X,Y or --box X1,Y1,X2,Y2")
	}
	return pts, nil
}

// best is the mask the decoder scores highest.
func best(masks []segment.Mask) int {
	j := 0
	for i, k := range masks {
		if k.Score > masks[j].Score {
			j = i
		}
	}
	return j
}

// writeMask writes the mask (white on the thing, black off it) and the cutout (the picture,
// transparent off the thing) as PNGs.
func writeMask(maskPath, cutoutPath string, img image.Image, k *segment.Mask) error {
	b := img.Bounds()
	mask := image.NewGray(image.Rect(0, 0, k.W, k.H))
	cut := image.NewNRGBA(image.Rect(0, 0, k.W, k.H))
	for y := range k.H {
		for x := range k.W {
			if !k.In[y*k.W+x] {
				continue
			}
			mask.SetGray(x, y, color.Gray{255})
			r, g, bl, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
			cut.SetNRGBA(x, y, color.NRGBA{uint8(r >> 8), uint8(g >> 8), uint8(bl >> 8), 255})
		}
	}
	for path, im := range map[string]image.Image{maskPath: mask, cutoutPath: cut} {
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
