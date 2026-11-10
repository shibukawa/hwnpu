// hwnpu-pose finds the people in images and their 17 COCO keypoints with RTMO (OpenMMLab
// MMPose) on the NPU.
//
//	hwnpu-pose photo.jpg                 # JSON: each person's box, score and keypoints
//	hwnpu-pose -draw out.png photo.jpg   # also the keypoints and the skeleton drawn on the image
//
// The model is downloaded and converted the first time. The network up to the head's outputs
// for every anchor runs on the NPU; top-k, non-maximum suppression and each person's keypoint
// classifier on the host.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	"image/png"
	"math"
	"os"
	"os/signal"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/pose"
)

func main() {
	use := flag.String("use", "pose", "the model: a name, a use or a family of -list (pose, rtmo, rtmo:s)")
	dir := flag.String("dir", defaultDir(), "where the models are kept, each in a directory of its name")
	minScore := flag.Float64("score", 0.3, "the least score of a person to report")
	out := flag.String("draw", "", "write the image with the keypoints and the skeleton drawn to this PNG (one input)")
	cpu := flag.Bool("cpu", false, "run the program on the CPU interpreter instead of the NPU")
	cacheDir := flag.String("cache", "", "compile cache (default: hwnpu in the user cache directory)")
	list := flag.Bool("list", false, "list the pose models of the definitions and exit")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: hwnpu-pose [flags] IMAGE...\n\nThe people in each image (JPEG or PNG) and their keypoints, as JSON.\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()
	if *list {
		chat.List(os.Stdout, *dir, chat.KindPose)
		return
	}
	if flag.NArg() == 0 || *out != "" && flag.NArg() != 1 {
		flag.Usage()
		os.Exit(2)
	}
	if *cacheDir == "" {
		if d, err := os.UserCacheDir(); err == nil {
			*cacheDir = filepath.Join(d, "hwnpu")
		}
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if err := run(ctx, *use, *dir, *cacheDir, *cpu, *minScore, *out, flag.Args()); err != nil {
		fmt.Fprintln(os.Stderr, "hwnpu-pose:", err)
		os.Exit(1)
	}
}

func defaultDir() string {
	d, err := os.UserCacheDir()
	if err != nil {
		return "models"
	}
	return filepath.Join(d, "hwnpu", "models")
}

func run(ctx context.Context, use, dir, cacheDir string, cpu bool, minScore float64, out string, paths []string) error {
	m, err := chat.ResolveFor(use, chat.KindPose, os.Stderr)
	if err != nil {
		return err
	}
	backend := llm.BackendCoreML
	var dev *hwnpu.Device
	if !cpu {
		var device string
		if dev, device, err = llm.OpenDevice(ctx, "", cacheDir); err != nil {
			return err
		}
		defer dev.Close()
		if backend = llm.ProgramBackend(device); backend == "" {
			return fmt.Errorf("%s runs neither Core ML packages nor OpenVINO IR (AMD's NPU has no RTMO export yet; -cpu runs it on the CPU)", device)
		}
	}
	model, program, err := pose.Prepare(ctx, pose.Setup{Dir: filepath.Join(dir, m.DirName()), Repo: m.Source(), SHA256: m.SHA256[pose.ModelFile], Backend: backend, Log: os.Stderr})
	if err != nil {
		return err
	}
	t0 := time.Now()
	e, err := pose.Open(ctx, dev, model, program)
	if err != nil {
		return err
	}
	defer e.Close()
	fmt.Fprintf(os.Stderr, "loaded %s in %.1f s (on the NPU: %v)\n", m.Name, time.Since(t0).Seconds(), e.OnNPU())
	enc := json.NewEncoder(os.Stdout)
	for _, p := range paths {
		f, err := os.Open(p)
		if err != nil {
			return err
		}
		img, _, err := image.Decode(f)
		f.Close()
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		start := time.Now()
		people, err := e.Estimate(ctx, img, minScore)
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }
		fmt.Fprintf(os.Stderr, "%s: %d people in %.0f ms (resize %.0f, network %.0f, people and keypoints on the host %.0f)\n", p, len(people), ms(time.Since(start)), ms(e.Times[0]), ms(e.Times[1]), ms(e.Times[2]))
		if err := enc.Encode(struct {
			File   string        `json:"file"`
			Width  int           `json:"width"`
			Height int           `json:"height"`
			People []pose.Person `json:"people"`
		}{p, img.Bounds().Dx(), img.Bounds().Dy(), people}); err != nil {
			return err
		}
		if out != "" {
			if err := drawPeople(out, img, people); err != nil {
				return err
			}
		}
	}
	return nil
}

// skeleton are COCO's limbs, by keypoint.
var skeleton = [][2]int{{15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12}, {5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10}, {1, 2}, {0, 1}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}}

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
