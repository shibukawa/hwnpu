package main

import (
	"cmp"
	"encoding/binary"
	"fmt"
	hcache "github.com/shibukawa/hwnpu/internal/cache"
	"math"
	"os"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/imagegen/anima"
	"github.com/shibukawa/hwnpu/internal/imagegen/aspect"
	"github.com/shibukawa/hwnpu/internal/imagegen/chroma"
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

// side is the picture's side in pixels for the models whose sizes are still squares: 1:1 and
// 1:1s, or a pixel count as before.
func (f convertFlags) side() (int, error) {
	w, h, err := aspect.ParseOrSide(f.Size, 16)
	if err != nil {
		return 0, err
	}
	if w != h {
		return 0, fmt.Errorf("--size %s: this model takes 1:1 or 1:1s for now", f.Size)
	}
	return w, nil
}

// ConvertCmd is "hwnpu-image convert": an Anima directory or a Chroma repository into a Core AI export (Core ML's for --cpu, OpenVINO IR for Intel's NPU).
type ConvertCmd struct {
	Anima string `arg:"" name:"dir" type:"existingdir" help:"The Anima directory (diffusion_models/, text_encoders/, diffusers/), or a Chroma repository of lodestones."`
	Model string `short:"m" required:"" type:"path" placeholder:"EXPORT" help:"The export to write."`
	Text  string `type:"existingdir" placeholder:"DIR" help:"Chroma: the directory of T5's text_encoder/ and the VAE's vae/ (default: the repository)."`
	// Chroma's
	Tokenizer     string   `type:"existingfile" placeholder:"FILE" help:"Chroma: T5's tokenizer.json (Anima's t5_tokenizer/tokenizer.json: Chroma's repository has only spiece.model)."`
	TextRows      int      `name:"text-rows" help:"Chroma: the text's rows (default 512, its pipeline's; fewer cut longer prompts)."`
	Only          string   `placeholder:"PARTS" help:"Chroma: write these parts alone (text, mods, doubles, singles, vae), comma separated."`
	DitFormatFor  []string `name:"dit-format-for" sep:"," placeholder:"PART=FORMAT" help:"Chroma: weight formats of the DiT's weights whose names hold PART (proj_mlp=lut4)."`
	TextFormatFor []string `name:"text-format-for" sep:"," placeholder:"PART=FORMAT" help:"Chroma: weight formats of T5's weights whose names hold PART (wo=fp16)."`
	DitQuant      string   `name:"dit-quant" placeholder:"NAME" help:"Chroma: the quantizer of the DiT's lut formats (uniform, kmeans, kmeans-minmax)."`
	LinkFrom      []string `name:"link-from" sep:"," type:"path" placeholder:"EXPORT" help:"Chroma: exports to hard link the parts that come out the same from (T5's programs, the VAE's of the same size): compiled once for all of them."`
	convertFlags
}

// Run converts the checkpoint.
func (c *ConvertCmd) Run() error {
	t0 := time.Now()
	if chroma.IsRepository(c.Anima) {
		backend, err := appleFormat("Chroma", c.Backend)
		if err != nil {
			return err
		}
		side, err := c.side()
		if err != nil {
			return err
		}
		o := chroma.ConvertOptions{Backend: backend, Width: side, Height: side, TextFormat: c.TextFormat, DitFormat: c.DitFormat,
			Tokenizer: c.Tokenizer, TextRows: c.TextRows}
		if c.Only != "" {
			o.Only = map[string]bool{}
			for _, k := range strings.Split(c.Only, ",") {
				o.Only[k] = true
			}
		}
		if len(c.DitFormatFor) > 0 {
			o.DitFormatFor = c.DitFormatFor
		}
		o.LinkFrom = c.LinkFrom
		o.DitQuant = c.DitQuant
		o.TextFormatFor = c.TextFormatFor
		if c.Group > 0 { // single blocks a program; a double block weighs two
			o.SingleGroup, o.DoubleGroup = c.Group, max(1, c.Group/2)
		}
		if err := chroma.Convert(c.Anima, cmp.Or(c.Text, c.Anima), c.Model, o, console.Log(os.Stderr)); err != nil {
			return err
		}
	} else {
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
	}
	fmt.Fprintf(os.Stderr, "%s: converted in %s\n", c.Model, console.Duration(time.Since(t0)))
	return nil
}

// appleFormat is the program format --backend names for a model that converts for Apple's Neural
// Engine alone: coreai (the default, what hwnpu-image runs) or coreml (what --cpu runs).
func appleFormat(model, backend string) (string, error) {
	switch backend {
	case "", llm.BackendCoreAI:
		return llm.BackendCoreAI, nil
	case llm.BackendCoreML:
		return llm.BackendCoreML, nil
	}
	return "", fmt.Errorf("%s converts for coreml or coreai (--backend %s)", model, backend)
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
