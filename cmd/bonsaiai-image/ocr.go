package main

import (
	"context"
	"encoding/json"
	"fmt"
	"image/color"
	"os"
	"path/filepath"
	"time"

	"github.com/shibukawa/bonsaiai/ai/vision"
	"github.com/shibukawa/bonsaiai/internal/console"
	"github.com/shibukawa/bonsaiai/internal/imagedev"
)

// OCRCmd is "bonsaiai-image ocr": the text in pictures with PaddleOCR's PP-OCRv6 (Apache-2.0;
// ai/vision's OCR), a text detector and a recognizer of 50 languages (Japanese, Chinese,
// English, ...), on Apple's Neural Engine or Intel's NPU, and elsewhere (AMD's NPU has no form of
// them yet) or with --device cpu on the CPU with ONNX Runtime (--cpu: the converted programs on
// the CPU interpreter). The models are downloaded the first time, and for an NPU each size of
// their programs converted the first time a picture takes it.
//
// For each picture it prints one line of JSON: each line of text's box (four corners clockwise
// from the top left, in the picture's pixels), text and score, in reading order. --text prints
// the text alone, a line for each. With --draw it also writes NAME.text.png with the boxes.
type OCRCmd struct {
	Pictures    []string `arg:"" type:"existingfile" help:"The pictures (JPEG or PNG)."`
	Model       string   `short:"m" default:"ocr" placeholder:"NAME" help:"The model: a name, a use or a family of models.yaml (ocr, ppocr, ppocr:small)."`
	Text        bool     `help:"Print the text alone: its lines, a blank line between pictures."`
	Size        string   `placeholder:"WxH" help:"The detector's size, multiples of 32 (default: of 960x960, 1280x736 and 736x1280 the one the picture is scaled down least to fit). Larger finds smaller text."`
	MinScore    float32  `default:"0.5" help:"The least recognition score of a line to report."`
	Draw        bool     `help:"Also write NAME.text.png: the picture with the lines' boxes."`
	Out         string   `short:"o" type:"path" placeholder:"DIR" help:"The directory --draw writes into (default: next to each picture)."`
	Repeat      int      `default:"1" help:"Run each picture this many times and report the mean time."`
	CPU         bool     `name:"cpu" help:"Run the converted programs on the CPU interpreter instead of the NPU (slow; for checking a conversion)."`
	Device      string   `enum:",npu,cpu" default:"" help:"Where to run: npu or cpu (ONNX Runtime; default: models.yaml's device for this machine's NPU, or BONSAIAI_DEVICE)."`
	Threads     int      `default:"1" help:"The CPU's threads with --device cpu (1: the least CPU time and energy)."`
	backendFlag `embed:""`
}

// Run reads the text of each picture.
func (c *OCRCmd) Run(ctx context.Context) error {
	if err := outDir(c.Out); err != nil {
		return err
	}
	lw := console.Log(os.Stderr)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	o := vision.OCROptions{MinScore: c.MinScore}
	if c.Size != "" {
		if _, err := fmt.Sscanf(c.Size, "%dx%d", &o.Size[0], &o.Size[1]); err != nil || o.Size[0]%32 != 0 || o.Size[1]%32 != 0 || o.Size[0] <= 0 || o.Size[1] <= 0 {
			return fmt.Errorf("--size %q: not WxH in multiples of 32", c.Size)
		}
	}
	useBackend(c.Backend)
	how := made{device: c.Device}
	how.threads(c.Threads)
	how.interpreter(c.CPU)
	r, rt, _, err := load(ctx, c.Model, vision.OCR, how, lw)
	if err != nil {
		return err
	}
	defer rt.Close()
	defer imagedev.NPUHeld(logf)()
	console.Running()
	enc := json.NewEncoder(os.Stdout)
	for n, p := range c.Pictures {
		img, err := readPicture(p)
		if err != nil {
			return err
		}
		var text *vision.Text
		start := time.Now()
		for i := range max(1, c.Repeat) {
			if i == 1 {
				start = time.Now()
			}
			if text, err = r.Read(ctx, img, o); err != nil {
				return fmt.Errorf("%s: %w", p, err)
			}
		}
		runs := 1
		if c.Repeat > 1 {
			runs = c.Repeat - 1
		}
		logf("%s: %d lines in %.0f ms (detector at %dx%d: resize %.1f, network %.1f, boxes %.1f; crops %.1f, recognizer %.1f in %d calls, decoding %.1f)%s",
			filepath.Base(p), len(text.Lines), ms(time.Since(start))/float64(runs), text.Size[0], text.Size[1], ms(text.Resize), ms(text.Detect), ms(text.Boxes),
			ms(text.Crop), ms(text.Recognize), text.Calls, ms(text.Decode), repeated(c.Repeat))
		lines := []lineJSON{}
		for _, l := range text.Lines {
			lj := lineJSON{Text: l.Text, Score: float32(l.Score)}
			for i, q := range l.Box {
				lj.Box[i] = [2]float32{float32(q.X), float32(q.Y)}
			}
			lines = append(lines, lj)
		}
		if c.Text {
			if n > 0 {
				fmt.Println()
			}
			for _, l := range lines {
				fmt.Println(l.Text)
			}
		} else {
			if err := enc.Encode(struct {
				File   string     `json:"file"`
				Width  int        `json:"width"`
				Height int        `json:"height"`
				Lines  []lineJSON `json:"lines"`
			}{p, img.Bounds().Dx(), img.Bounds().Dy(), lines}); err != nil {
				return err
			}
		}
		if c.Draw {
			out := drawPath(c.Out, p, ".text.png")
			var boxes []box
			for _, l := range lines {
				boxes = append(boxes, box{corners: l.Box, color: color.RGBA{255, 40, 80, 255}})
			}
			if err := drawBoxes(out, img, boxes); err != nil {
				return err
			}
			logf("wrote %s", out)
		}
	}
	return nil
}

// lineJSON is a line of text as ocr prints it: its box's four corners, its text and score.
type lineJSON struct {
	Box   [4][2]float32 `json:"box"`
	Text  string        `json:"text"`
	Score float32       `json:"score"`
}
