package main

import (
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/shibukawa/bonsaiai/ai/vision"
	"github.com/shibukawa/bonsaiai/internal/console"
)

// PoseCmd is "bonsaiai-image pose": the people in pictures and their 17 COCO keypoints with RTMO
// (OpenMMLab MMPose; ai/vision's Pose), on Apple's Neural Engine or Intel's NPU, converted the
// first time, on AMD's NPU from an export of tools/amd/iron_onnxcnn.py (made with IRON on the
// first run), and elsewhere or with --device cpu on the CPU with ONNX Runtime (--cpu: the
// converted program on the CPU interpreter). The network up to the head's outputs for every
// anchor runs on the NPU (on AMD's, up to the head's convolutions); top-k, non-maximum
// suppression and each person's keypoint classifier on the host.
//
// For each picture it prints one line of JSON: each person's box, score and keypoints in the
// picture's pixels. With --draw it also writes NAME.pose.png with them and the skeleton drawn.
type PoseCmd struct {
	Pictures    []string `arg:"" type:"existingfile" help:"The pictures (JPEG or PNG)."`
	Model       string   `short:"m" default:"pose" placeholder:"NAME" help:"The model: a name, a use or a family of models.yaml (pose, rtmo, rtmo:s)."`
	Score       float64  `default:"0.3" help:"The least score of a person to report."`
	Draw        bool     `help:"Also write NAME.pose.png: the picture with the boxes, keypoints and skeletons."`
	Out         string   `short:"o" type:"path" placeholder:"DIR" help:"The directory --draw writes into (default: next to each picture)."`
	CPU         bool     `name:"cpu" help:"Run the converted program on the CPU interpreter instead of the NPU (slow; for checking a conversion)."`
	Device      string   `enum:",npu,cpu" default:"" help:"Where to run: npu or cpu (ONNX Runtime; default: models.yaml's device for this machine's NPU, or BONSAIAI_DEVICE)."`
	Threads     int      `default:"1" help:"The CPU's threads with --device cpu (1: the least CPU time and energy)."`
	backendFlag `embed:""`
}

// Run finds the people of each picture.
func (c *PoseCmd) Run(ctx context.Context) error {
	if err := outDir(c.Out); err != nil {
		return err
	}
	lw := console.Log(os.Stderr)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	useBackend(c.Backend)
	how := made{device: c.Device}
	how.threads(c.Threads)
	how.interpreter(c.CPU)
	e, rt, loaded, err := load(ctx, c.Model, vision.Pose, how, lw)
	if err != nil {
		return err
	}
	defer rt.Close()
	logf("%s, %s", loaded, e.Where())
	score := c.Score
	if score == 0 {
		score = -1 // every person
	}
	console.Running()
	enc := json.NewEncoder(os.Stdout)
	for _, p := range c.Pictures {
		img, err := readPicture(p)
		if err != nil {
			return err
		}
		found, err := e.Estimate(ctx, img, vision.PoseOptions{Score: score})
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		logf("%s: %d people in %.0f ms (resize %.0f, network %.0f, people and keypoints on the host %.0f)", filepath.Base(p), len(found.People),
			ms(found.Took), ms(found.Resize), ms(found.Network), ms(found.Host))
		var people []personJSON // none: null, as before
		for _, q := range found.People {
			pj := personJSON{Box: [4]float64{q.Box.Min.X, q.Box.Min.Y, q.Box.Max.X, q.Box.Max.Y}, Score: q.Score}
			for _, k := range q.Keypoints {
				pj.Keypoints = append(pj.Keypoints, keypointJSON{k.Name, k.X, k.Y, k.Score})
			}
			people = append(people, pj)
		}
		if err := enc.Encode(struct {
			File   string       `json:"file"`
			Width  int          `json:"width"`
			Height int          `json:"height"`
			People []personJSON `json:"people"`
		}{p, img.Bounds().Dx(), img.Bounds().Dy(), people}); err != nil {
			return err
		}
		if c.Draw {
			d := c.Out
			if d == "" {
				d = filepath.Dir(p)
			}
			out := filepath.Join(d, strings.TrimSuffix(filepath.Base(p), filepath.Ext(p))+".pose.png")
			if err := drawPeople(out, img, found.People); err != nil {
				return err
			}
			logf("wrote %s", out)
		}
	}
	return nil
}

// personJSON is a person as pose prints it: the box (x1, y1, x2, y2), its score, the keypoints.
type personJSON struct {
	Box       [4]float64     `json:"box"`
	Score     float64        `json:"score"`
	Keypoints []keypointJSON `json:"keypoints"`
}

type keypointJSON struct {
	Name  string  `json:"name"`
	X     float64 `json:"x"`
	Y     float64 `json:"y"`
	Score float64 `json:"score"`
}

// drawPeople writes img with the people's boxes, limbs and keypoints drawn to a PNG at path.
func drawPeople(path string, img image.Image, people []vision.Person) error {
	b := img.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(dst, dst.Bounds(), img, b.Min, draw.Src)
	line := func(x0, y0, x1, y1 float64, c color.Color) {
		n := int(math.Max(math.Abs(x1-x0), math.Abs(y1-y0))) + 1
		for i := 0; i <= n; i++ {
			t := float64(i) / float64(n)
			x, y := int(x0+(x1-x0)*t), int(y0+(y1-y0)*t)
			for dx := -1; dx <= 1; dx++ {
				for dy := -1; dy <= 1; dy++ {
					dst.Set(x+dx, y+dy, c)
				}
			}
		}
	}
	box, limb, joint := color.RGBA{255, 200, 0, 255}, color.RGBA{0, 220, 255, 255}, color.RGBA{255, 40, 80, 255}
	for _, p := range people {
		x0, y0, x1, y1 := p.Box.Min.X, p.Box.Min.Y, p.Box.Max.X, p.Box.Max.Y
		line(x0, y0, x1, y0, box)
		line(x1, y0, x1, y1, box)
		line(x1, y1, x0, y1, box)
		line(x0, y1, x0, y0, box)
		for _, s := range vision.Skeleton {
			a, c := p.Keypoints[s[0]], p.Keypoints[s[1]]
			if a.Score >= 0.3 && c.Score >= 0.3 {
				line(a.X, a.Y, c.X, c.Y, limb)
			}
		}
		for _, k := range p.Keypoints {
			if k.Score >= 0.3 {
				line(k.X-2, k.Y, k.X+2, k.Y, joint)
			}
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := png.Encode(f, dst); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
