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
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/pose"
)

// PoseCmd is "hwnpu-image pose": the people in pictures and their 17 COCO keypoints with RTMO
// (OpenMMLab MMPose), on Apple's Neural Engine or Intel's NPU (--cpu: the CPU interpreter). The
// model is downloaded and converted the first time. The network up to the head's outputs for
// every anchor runs on the NPU; top-k, non-maximum suppression and each person's keypoint
// classifier on the host.
//
// For each picture it prints one line of JSON: each person's box, score and keypoints in the
// picture's pixels. With --draw it also writes NAME.pose.png with them and the skeleton drawn.
type PoseCmd struct {
	Pictures []string `arg:"" type:"existingfile" help:"The pictures (JPEG or PNG)."`
	Model    string   `short:"m" default:"pose" placeholder:"NAME" help:"The model: a name, a use or a family of models.yaml (pose, rtmo, rtmo:s)."`
	Score    float64  `default:"0.3" help:"The least score of a person to report."`
	Draw     bool     `help:"Also write NAME.pose.png: the picture with the boxes, keypoints and skeletons."`
	Out      string   `short:"o" type:"path" placeholder:"DIR" help:"The directory --draw writes into (default: next to each picture)."`
	Dir      string   `type:"path" placeholder:"DIR" help:"Where the models are kept (as hwnpu-chat's --dir)."`
	Cache    string   `placeholder:"DIR" help:"Compile cache (default: hwnpu in the user cache directory)."`
	CPU      bool     `name:"cpu" help:"Run the program on the CPU interpreter instead of the NPU."`
}

// Run finds the people of each picture.
func (c *PoseCmd) Run(ctx context.Context) error {
	lw := console.Log(os.Stderr)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	m, err := resolve(c.Model, chat.KindPose)
	if err != nil {
		return err
	}
	cache := c.Cache
	if cache == "" {
		if d, err := os.UserCacheDir(); err == nil {
			cache = filepath.Join(d, "hwnpu")
		}
	}
	dir := c.Dir
	if dir == "" {
		dir = chat.ModelsDir()
	}
	backend := llm.BackendCoreML
	var dev *hwnpu.Device
	if !c.CPU {
		var device string
		if dev, device, err = llm.OpenDevice(ctx, "", cache); err != nil {
			return err
		}
		defer dev.Close()
		if backend = llm.ProgramBackend(device); backend == "" {
			return fmt.Errorf("%s runs neither Core ML packages nor OpenVINO IR (AMD's NPU has no RTMO export yet; --cpu runs it on the CPU)", device)
		}
	}
	model, program, err := pose.Prepare(ctx, pose.Setup{Dir: filepath.Join(dir, m.DirName()), Repo: m.Source(), SHA256: m.SHA256[pose.ModelFile], Backend: backend, Log: lw})
	if err != nil {
		return err
	}
	t0 := time.Now()
	e, err := pose.Open(ctx, dev, model, program)
	if err != nil {
		return err
	}
	defer e.Close()
	where := "on the CPU interpreter"
	if !c.CPU {
		where = "all on the NPU"
		if !e.OnNPU() {
			where = "NOT all on the NPU"
		}
	}
	logf("loaded %s in %.1f s, %s", m.Name, time.Since(t0).Seconds(), where)
	enc := json.NewEncoder(os.Stdout)
	ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }
	for _, p := range c.Pictures {
		img, err := readPicture(p)
		if err != nil {
			return err
		}
		start := time.Now()
		people, err := e.Estimate(ctx, img, c.Score)
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		logf("%s: %d people in %.0f ms (resize %.0f, network %.0f, people and keypoints on the host %.0f)", filepath.Base(p), len(people), ms(time.Since(start)), ms(e.Times[0]), ms(e.Times[1]), ms(e.Times[2]))
		if err := enc.Encode(struct {
			File   string        `json:"file"`
			Width  int           `json:"width"`
			Height int           `json:"height"`
			People []pose.Person `json:"people"`
		}{p, img.Bounds().Dx(), img.Bounds().Dy(), people}); err != nil {
			return err
		}
		if c.Draw {
			d := c.Out
			if d == "" {
				d = filepath.Dir(p)
			}
			out := filepath.Join(d, strings.TrimSuffix(filepath.Base(p), filepath.Ext(p))+".pose.png")
			if err := drawPeople(out, img, people); err != nil {
				return err
			}
			logf("wrote %s", out)
		}
	}
	return nil
}

// skeleton are COCO's limbs, by keypoint.
var skeleton = [][2]int{{15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12}, {5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10}, {1, 2}, {0, 1}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}}

// drawPeople writes img with the people's boxes, limbs and keypoints drawn to a PNG at path.
func drawPeople(path string, img image.Image, people []pose.Person) error {
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
		x0, y0, x1, y1 := p.Box[0], p.Box[1], p.Box[2], p.Box[3]
		line(x0, y0, x1, y0, box)
		line(x1, y0, x1, y1, box)
		line(x1, y1, x0, y1, box)
		line(x0, y1, x0, y0, box)
		for _, s := range skeleton {
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
