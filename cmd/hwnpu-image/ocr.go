package main

import (
	"context"
	"encoding/json"
	"fmt"
	"image/color"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/vision/ocr"
)

// OCRCmd is "hwnpu-image ocr": the text in pictures with PaddleOCR's PP-OCRv6 (Apache-2.0;
// internal/vision/ocr), a text detector and a recognizer of 50 languages (Japanese, Chinese,
// English, ...), on Apple's Neural Engine or Intel's NPU (--cpu: the CPU interpreter). The models
// are downloaded the first time, and each size of their programs converted the first time a
// picture takes it.
//
// For each picture it prints one line of JSON: each line of text's box (four corners clockwise
// from the top left, in the picture's pixels), text and score, in reading order. --text prints
// the text alone, a line for each. With --draw it also writes NAME.text.png with the boxes.
type OCRCmd struct {
	Pictures []string `arg:"" type:"existingfile" help:"The pictures (JPEG or PNG)."`
	Model    string   `short:"m" default:"ocr" placeholder:"NAME" help:"The model: a name, a use or a family of models.yaml (ocr, ppocr, ppocr:small)."`
	Text     bool     `help:"Print the text alone: its lines, a blank line between pictures."`
	Size     string   `placeholder:"WxH" help:"The detector's size, multiples of 32 (default: of 960x960, 1280x736 and 736x1280 the one the picture is scaled down least to fit). Larger finds smaller text."`
	MinScore float32  `default:"0.5" help:"The least recognition score of a line to report."`
	Draw     bool     `help:"Also write NAME.text.png: the picture with the lines' boxes."`
	Out      string   `short:"o" type:"path" placeholder:"DIR" help:"The directory --draw writes into (default: next to each picture)."`
	Repeat   int      `default:"1" help:"Run each picture this many times and report the mean time."`
	Dir      string   `type:"path" placeholder:"DIR" help:"Where the models are kept (as hwnpu-chat's --dir)."`
	Cache    string   `placeholder:"DIR" help:"Compile cache (default: hwnpu in the user cache directory)."`
	CPU      bool     `name:"cpu" help:"Run the programs on the CPU interpreter instead of the NPU."`
}

// Run reads the text of each picture.
func (c *OCRCmd) Run(ctx context.Context) error {
	lw := console.Log(os.Stderr)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	m, err := resolve(c.Model, chat.KindOCR)
	if err != nil {
		return err
	}
	det := m.With["detector"]
	if det == "" {
		return fmt.Errorf("%s: models.yaml gives no detector (with: {detector: hf:OWNER/NAME})", m.Name)
	}
	dev, backend, closeDev, err := openProgramDevice(ctx, c.CPU, c.Cache, "PP-OCR")
	if err != nil {
		return err
	}
	defer closeDev()
	defer npuHeld(logf)()
	dir := c.Dir
	if dir == "" {
		dir = chat.ModelsDir()
	}
	detSHA, recSHA := map[string]string{}, map[string]string{}
	for f, d := range m.SHA256 {
		if rest, ok := strings.CutPrefix(f, "detector/"); ok {
			detSHA[rest] = d
		} else {
			recSHA[f] = d
		}
	}
	r, err := ocr.Open(ctx, dev, ocr.Setup{Dir: dir, Det: ocr.Source{Repo: det, SHA256: detSHA}, Rec: ocr.Source{Repo: m.Source(), SHA256: recSHA}, Backend: backend, Log: lw})
	if err != nil {
		return err
	}
	defer r.Close()
	r.Logf, r.MinScore = logf, c.MinScore
	if c.Size != "" {
		if _, err := fmt.Sscanf(c.Size, "%dx%d", &r.Size[0], &r.Size[1]); err != nil || r.Size[0]%32 != 0 || r.Size[1]%32 != 0 || r.Size[0] <= 0 || r.Size[1] <= 0 {
			return fmt.Errorf("--size %q: not WxH in multiples of 32", c.Size)
		}
	}
	enc := json.NewEncoder(os.Stdout)
	for n, p := range c.Pictures {
		img, err := readPicture(p)
		if err != nil {
			return err
		}
		var lines []ocr.Line
		start := time.Now()
		for i := range max(1, c.Repeat) {
			if i == 1 {
				start = time.Now()
			}
			if lines, err = r.Read(ctx, img); err != nil {
				return fmt.Errorf("%s: %w", p, err)
			}
		}
		runs := 1
		if c.Repeat > 1 {
			runs = c.Repeat - 1
		}
		W, H := r.Choose(img.Bounds().Dx(), img.Bounds().Dy())
		t := r.Times
		logf("%s: %d lines in %.0f ms (detector at %dx%d: resize %.1f, network %.1f, boxes %.1f; crops %.1f, recognizer %.1f in %d calls, decoding %.1f)%s",
			filepath.Base(p), len(lines), ms(time.Since(start))/float64(runs), W, H, ms(t.Resize), ms(t.Detect), ms(t.Boxes), ms(t.Crop), ms(t.Recognize), r.Calls, ms(t.Decode), repeated(c.Repeat))
		if c.Text {
			if n > 0 {
				fmt.Println()
			}
			for _, l := range lines {
				fmt.Println(l.Text)
			}
		} else {
			if lines == nil {
				lines = []ocr.Line{}
			}
			if err := enc.Encode(struct {
				File   string     `json:"file"`
				Width  int        `json:"width"`
				Height int        `json:"height"`
				Lines  []ocr.Line `json:"lines"`
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
