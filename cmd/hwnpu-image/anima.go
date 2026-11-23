package main

import (
	"cmp"
	"encoding/binary"
	"fmt"
	hcache "github.com/shibukawa/hwnpu/internal/cache"
	"math"
	"os"
	"time"

	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/imagegen/anima"
	"github.com/shibukawa/hwnpu/internal/llm"
)

// convertFlags are what an Anima export for Apple's or Intel's NPU is made with (generate's first
// run, and convert).
type convertFlags struct {
	Backend    string `help:"The program format: coreai (Apple's Neural Engine through Core AI, macOS 27: a Mac's only one) or intel (Intel's NPU); convert also takes coreml, the Core ML export --cpu runs. Default: HWNPU_BACKEND, else the NPU found (convert: coreai)."`
	Size       string `default:"1:1" help:"The picture's aspect ratio: 1:1 (1024 x 1024), 1:1s (512 x 512), 2:3, 3:4, 9:16, 3:2, 4:3 or 16:9 (Anima and FLUX.2 klein; Anima an export each, on AMD's NPU for 1:1 and 1:1s only; FLUX.2 klein on AMD's NPU one export of all eight). The other models take 1:1 and 1:1s, and still a pixel count."`
	Dit        string `placeholder:"FILE" help:"The diffusion model, a file of split_files/diffusion_models/ in circlestone-labs/Anima (default: the model's of models.yaml; convert: anima-turbo-v1.1.safetensors)."`
	TextFormat string `help:"The text encoder's weights (fp16; lut8, lut6 on Apple's; int8 on Intel; default int8 on Apple's, where fp16's 1.0 GB is over a program's limit, fp16 on Intel)."`
	DitFormat  string `help:"The DiT's weights (fp16; lut8, lut6 on Apple's; int8, int4 on Intel; default int8 on Apple's: faster than fp16 at half its memory, and nearer it than lut8; fp16 on Intel. Qwen-Image: lut6, its text pass the same and a group's two passes one program with the weights once)."`
	Group      int    `help:"The DiT's blocks a program holds (default 4; Anima on Intel: fewer above 512 x 512, 1 at 1024 x 1024, as a program running past Windows' 2 s limit loses the NPU)."`
	QChunk     int    `name:"qchunk" default:"512" help:"The queries an attention scores at a time (M3: 512 is 25 % faster than 1024 at 1024 x 1024)."`
}

func (f convertFlags) options() (anima.ConvertOptions, error) {
	w, h, err := anima.ParseAspect(f.Size)
	return anima.ConvertOptions{Backend: f.Backend, Width: w, Height: h, Dit: f.Dit,
		TextFormat: f.TextFormat, DitFormat: f.DitFormat, Group: f.Group, QChunk: f.QChunk}, err
}

// ConvertCmd is "hwnpu-image convert": an Anima directory into a Core AI export (Core ML's for --cpu, OpenVINO IR for Intel's NPU).
type ConvertCmd struct {
	Anima string `arg:"" name:"dir" type:"existingdir" help:"The Anima directory (diffusion_models/, text_encoders/, diffusers/)."`
	Model string `short:"m" required:"" type:"path" placeholder:"EXPORT" help:"The export to write."`
	convertFlags
}

// Run converts the checkpoint.
func (c *ConvertCmd) Run() error {
	t0 := time.Now()
	o, err := c.options()
	if err != nil {
		return err
	}
	o.Backend = cmp.Or(o.Backend, llm.BackendCoreAI)
	text, dit := anima.Formats(o.Backend)
	o.TextFormat, o.DitFormat = cmp.Or(o.TextFormat, text), cmp.Or(o.DitFormat, dit)
	if err := anima.Convert(anima.LayoutCheckpoint(c.Anima), c.Model, o, console.Log(os.Stderr)); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "%s: converted in %s\n", c.Model, console.Duration(time.Since(t0)))
	return nil
}

// cacheDirOf is the compile cache of --cache: the user cache directory's hwnpu by default, "" for
// "-".
func cacheDirOf(flag string) string {
	switch flag {
	case "-":
		return ""
	case "":
		return hcache.Root()
	}
	return flag
}

func readF32(path string) ([]float32, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out, nil
}

func writeF32(path string, xs []float32) error {
	b := make([]byte, 4*len(xs))
	for i, x := range xs {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return os.WriteFile(path, b, 0o644)
}
