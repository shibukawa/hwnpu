package main

import (
	"cmp"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	hcache "github.com/shibukawa/hwnpu/internal/cache"
	"image/png"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/imagegen/anima"
	"github.com/shibukawa/hwnpu/internal/imagegen/aspect"
	"github.com/shibukawa/hwnpu/internal/imagegen/chroma"
	"github.com/shibukawa/hwnpu/internal/imagegen/flux2klein"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
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

// picture is the picture's width and height for a model whose pictures are multiples of
// multiple pixels: an aspect ratio of --size (aspect.Of), or a pixel count as before (a square).
func (f convertFlags) picture(multiple int) (w, h int, err error) {
	return aspect.ParseOrSide(f.Size, multiple)
}

// pictureIn is picture for a model whose aspect ratios are the sizes of as (FLUX.2 klein's own).
func (f convertFlags) pictureIn(as []aspect.Aspect, multiple int) (w, h int, err error) {
	return aspect.ParseOrSideIn(f.Size, as, multiple)
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

// GenerateCmd is "hwnpu-image generate": a picture from a prompt with FLUX.2 klein, Anima or Chroma1.
type GenerateCmd struct {
	Prompts    []string  `arg:"" name:"prompt" help:"What to draw (Anima: Danbooru tags and natural language). Chroma and FLUX.2 klein take several: a picture each, the model loaded once (on Apple's Neural Engine --batch of them are drawn together)."`
	Model      string    `short:"m" default:"image" placeholder:"NAME|EXPORT" help:"The model: a name, a family or a use of models.yaml (image: FLUX.2 klein 4B from 8 GB; flux2-klein, chroma, anima-turbo and anima (Anima-Aesthetic) with HWNPU_MODEL_ALLOWLIST=+nsfw; a family alone picks for the machine's memory, or for the memory a tag names: image:8g), or an export's directory (AMD: of tools/amd/iron_anima.py, iron_chroma.py or iron_flux2klein.py). On AMD's NPU a name runs NAME.amd-npu2 (NAME-SIZE.amd-npu2 for another --size) in the cache's entries/amd-xdna2 or ~/models, else one the sources hold, downloaded once (FLUX.2 klein's exported with IRON when they have none; one made by hand in entries/amd-xdna2 goes in the sources written back to on its first use); on Apple's and Intel's (Anima; Chroma1 and FLUX.2 klein on Apple's) the model is downloaded and converted on its first run."`
	Out        string    `short:"o" type:"path" placeholder:"FILE" help:"The PNG to write (default: image.png, else image (2).png, ... here, then shown in the system's viewer); with several prompts NAME-1.png, NAME-2.png, ..."`
	Batch      int       `name:"batch" default:"4" placeholder:"N" help:"Chroma and FLUX.2 klein on Apple's Neural Engine with several prompts: the pictures drawn together, N at a time (their prompts are read, then their steps run, then the VAE's stages over all of them, so that a program is loaded once for the N pictures; they are written together, when the last is drawn); FLUX.2 klein on AMD's NPU too, its steps together (a group of the DiT's weights read once for the N pictures; the VAE a picture after another); 1 draws one picture after another."`
	KeepDiT    bool      `name:"keep-dit" default:"true" negatable:"" help:"Chroma on Apple's Neural Engine with several prompts: keep the DiT loaded between the pictures (the text encoder and the VAE are loaded for each; Chroma on macOS 27 loads the DiT's groups for each step either way)."`
	Steps      int       `help:"Steps (default: Anima-Turbo's Euler 8, 8 to 12 wanted; Anima-Aesthetic 25 with dpm2m, the model card's 30 to 50 with euler)."`
	Negative   string    `help:"The negative prompt of the classifier-free guidance (default empty; Anima's model card recommends \"worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration\", without the score tags for Anima-Aesthetic)."`
	Guidance   float64   `placeholder:"SCALE" help:"The classifier-free guidance's scale, each step the DiT twice (default: Anima 4, 4 to 5 wanted; Anima-Turbo 1, none, or 1.5 with a --negative)."`
	Shift      float64   `placeholder:"SHIFT" default:"3.0" help:"Anima: the sigma schedule's shift (Anima's scheduler: 3)."`
	Sampler    string    `placeholder:"NAME" help:"Anima: the solver, euler (Anima's scheduler; Anima-Turbo's default) or dpm2m (DPM-Solver++ 2M, the default with guidance: the same DiT runs a step, closer to many steps' result in fewer)."`
	CFGSigmas  []float64 `name:"cfg-sigmas" sep:"," placeholder:"LO,HI" help:"Run the classifier-free guidance only at steps whose sigma is in [LO, HI] (1 is pure noise; 0,1 for every step; default 0.6,1 for Anima); the other steps run the DiT once."`
	Seed       uint64    `placeholder:"N" help:"The noise's seed (0: from the time)."`
	Noise      string    `type:"path" placeholder:"FILE" help:"The initial latents as raw float32 in place of the seed's (Anima: (16, H/8, W/8))."`
	Latents    string    `type:"path" placeholder:"FILE" help:"Also write the final latents as raw float32 here."`
	Image      []string  `name:"image" short:"i" placeholder:"FILE[@half]" help:"FLUX.2 klein: a reference image to edit or to compose from (PNG or JPEG; its centre cropped to the picture's shape and scaled to its size); repeat for up to 4, named image 1, image 2, ... in the prompt. FILE@half reads that reference at half the picture's sides (a quarter of its rows in each step: an edit at 1024 x 1024 in about two thirds of the time, with less of the reference's fine detail)."`
	RefLatents []string  `name:"ref-latents" type:"existingfile" placeholder:"FILE" help:"FLUX.2 klein: a reference's latents as raw float32 (tokens x 128, normalized), for checks against tools/ref/ref_flux2klein.py; repeat for several."`
	Cache      string    `help:"Compile cache (default: hwnpu in the user cache directory; \"-\" for none)."`
	CPU        bool      `name:"cpu" help:"Run a Core ML export with the CPU interpreter instead of the NPU (slow; for checking)."`
	Resident   bool      `name:"resident-dit" help:"AMD: keep every group of the DiT's blocks on the NPU instead of reading each group's weights into one bank before it runs (Anima: pinned memory 7 banks, 3.4 GB at q8)."`
	Dump       string    `type:"path" placeholder:"DIR" help:"A directory to write the conditioning and each step's velocity to (raw bf16), to check against tools/ref/ref_anima.py."`
	convertFlags
}

// Run draws the picture into --out, or into a new image.png here which it then shows.
func (c *GenerateCmd) Run(ctx context.Context) error {
	show := c.Out == ""
	var outs []string
	for i := range c.Prompts {
		switch {
		case show:
			outs = append(outs, freeName("image.png", outs...))
		case len(c.Prompts) == 1:
			outs = append(outs, c.Out)
		default:
			ext := filepath.Ext(c.Out)
			outs = append(outs, fmt.Sprintf("%s-%d%s", strings.TrimSuffix(c.Out, ext), i+1, ext))
		}
	}
	c.Out = outs[0]
	model := expandHome(c.Model)
	var kleinAs []aspect.Aspect // FLUX.2 klein's sizes on this machine's export (nil: the directory's own)
	if !isPath(c.Model) {
		m, err := resolve(c.Model, chat.KindImage)
		if err != nil {
			return err
		}
		model = ""
		if strings.HasPrefix(m.Family, "flux2-klein") {
			if model, kleinAs, err = c.kleinModel(ctx, m); err != nil {
				return err
			}
		} else if strings.HasPrefix(m.Family, "chroma") {
			if model, err = c.chromaModel(ctx, m); err != nil {
				return err
			}
		} else if !strings.HasPrefix(m.Family, "anima") {
			return fmt.Errorf("%s: hwnpu-image generate draws with Anima, Chroma1 and FLUX.2 klein", m.Name)
		} else {
			if c.Dit == "" {
				c.Dit = m.File
			}
			if model, err = c.animaModel(ctx, m); err != nil {
				return err
			}
		}
	}
	var err error
	if flux2klein.IsExport(model) {
		err = c.klein(ctx, model, kleinAs, outs)
	} else if chroma.IsExport(model) {
		err = c.chroma(ctx, model, outs)
	} else if len(c.Prompts) > 1 {
		err = errors.New("Anima takes one prompt")
	} else {
		err = c.anima(ctx, model)
	}
	if err == nil && show {
		for _, o := range outs {
			if e := openFile(o); e != nil {
				fmt.Fprintf(os.Stderr, "%s: not shown (%v)\n", o, e)
			}
		}
	}
	return err
}

// animaModel is the Anima export m runs here: on AMD's NPU (no Apple or Intel NPU found, no
// --backend) the export of tools/amd/iron_anima.py that draws --size (animaAMD), "" otherwise
// (anima prepares the one prepared here).
func (c *GenerateCmd) animaModel(ctx context.Context, m chat.Model) (string, error) {
	if c.CPU || c.Backend != "" {
		return "", nil
	}
	if _, err := llm.ProgramFormat(ctx, ""); err == nil {
		return "", nil // Apple's or Intel's NPU
	}
	w, h, err := anima.ParseAspect(c.Size)
	if err != nil {
		return "", err
	}
	return animaAMD(ctx, m, w, h)
}

// animaAMD is the export of Anima model m for AMD's NPU that draws w x h (aspectsAMD).
func animaAMD(ctx context.Context, m chat.Model, w, h int) (string, error) {
	return aspectsAMD(ctx, m, w, h, "tools/amd/iron_anima.py", 16, anima.IronSizes)
}

// aspectsAMD is the export of model m for AMD's NPU that draws w x h, of every aspect ratio by the
// name NAME.amd-npu2 (an older one of 1024 x 1024 alone by that name draws that size only; sizes are
// the sizes an export draws), here or in the sources (chat.AMDNamed): made by hand with tool,
// which the error tells how to run for every aspect ratio of pictures of multiple pixels
// (aspect.Of) when there is none.
func aspectsAMD(ctx context.Context, m chat.Model, w, h int, tool string, multiple int, sizes func(dir string) [][2]int) (string, error) {
	names := []string{strings.ToLower(m.Name) + ".amd-npu2"}
	draws := func(d string) bool { return isDir(d) && slices.Contains(sizes(d), [2]int{w, h}) }
	if d := chat.AMDNamed(ctx, m, names, draws, console.Log(os.Stderr)); d != "" {
		return d, nil
	}
	flags := ""
	if m.File != "" {
		flags = " --dit " + m.File
	}
	var args []string
	for _, a := range aspect.Of(multiple) {
		if a.Width == a.Height {
			args = append(args, fmt.Sprint(a.Width))
		} else {
			args = append(args, fmt.Sprintf("%dx%d", a.Width, a.Height))
		}
	}
	return "", fmt.Errorf("AMD's NPU runs exports of %s, and neither this machine (%s) nor the sources have one of %s that draws %dx%d: download %s and run\n  python %s ~/models/%s %s --size %s%s\n(one export of all eight aspect ratios; its first run here puts it in the sources written back to; see docs/amd-npu-models.md)",
		tool, strings.Join(hcache.ExportRoots(), ", "), m.Name, w, h, strings.TrimPrefix(m.Source(), "hf:"), tool, filepath.Base(m.Repo),
		shortHome(filepath.Join(chat.AMDRoot(), names[0])), strings.Join(args, " "), flags)
}

// turboNegativeGuidance is the guidance scale a --negative without --guidance gets on a model that
// runs none by default (Anima-Turbo): weak, as a model distilled to run without guidance burns
// at the 4 the others take.
const turboNegativeGuidance = 1.5

// anima draws the picture with Anima: the export model, or "" for the one prepared here (downloaded
// and converted when missing).
func (c *GenerateCmd) anima(ctx context.Context, model string) error {
	anima.ResidentDiT = c.Resident
	o, err := c.options()
	if err != nil {
		return err
	}
	cacheDir := c.Cache
	dir := chat.ModelsDir()
	// what goes on in gray (with a spinner or a bar on a terminal), the picture's line as it is
	lw := console.Log(os.Stderr)
	live := console.Live(lw)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }

	if model == "" {
		format := llm.BackendCoreML // the CPU interpreter runs Core ML exports
		var err error
		if !c.CPU {
			format, err = imageFormat(ctx, c.Backend)
		}
		if err != nil {
			return err
		}
		o.Backend = format
		if model, err = anima.Prepare(ctx, anima.Setup{Dir: filepath.Join(dir, "Anima"), ConvertOptions: o, Log: lw}); err != nil {
			return err
		}
	}
	t0 := time.Now()
	var dev *hwnpu.Device
	if !c.CPU {
		var opts []hwnpu.Option
		if cacheDir = cacheDirOf(cacheDir); cacheDir != "" {
			// without it, Core ML compiles each package to a new temporary path and the Neural
			// Engine's compiled bundles are never reused: every start compiled the whole DiT
			// again (118 s at 512 x 512 on M3)
			opts = append(opts, hwnpu.WithCacheDir(cacheDir))
		}
		if !anima.IsProgramExport(model) {
			opts = append(opts, hwnpu.WithVendor(hwnpu.VendorAMD)) // IRON graphs
		} else if m, err := anima.ReadProgramManifest(model); err == nil && llm.AppleFormat(m.Backend) {
			// on Core AI (an older Core ML export is lowered to Core AI as it loads). One hwnpu
			// process at a time uses the Neural Engine, as the other models' devices are opened
			// (llm.OpenDevice): beside a chat a stage of the VAE failed, "Inference failed —
			// request aborted" (design.md 16.215)
			if err := llm.LockNPU(ctx); err != nil {
				return err
			}
			opts = append(opts, hwnpu.WithBackend(exportDevice(m.Backend, c.Backend)))
		} else if b := imageDevice(c.Backend); b != "" {
			opts = append(opts, hwnpu.WithBackend(b))
		}
		d, err := hwnpu.Open(ctx, opts...)
		if err != nil {
			return err
		}
		defer d.Close()
		dev = d
	}
	if !anima.IsProgramExport(model) {
		iron.ShareExport(model, logf)
	}
	pw, ph := 0, 0
	if !anima.IsProgramExport(model) && len(anima.IronSizes(model)) > 1 {
		pw, ph = o.Width, o.Height // an export of several sizes draws at --size (one of a size alone at its own)
	}
	p, err := anima.OpenAt(ctx, dev, model, pw, ph, lw)
	if err != nil {
		return err
	}
	defer p.Close()
	p.Logf = logf
	p.Dump = c.Dump
	logf("loaded %s (%dx%d) in %s", filepath.Base(model), p.Dit.Width, p.Dit.Height, console.Duration(time.Since(t0)))
	console.Running()

	steps, guidance := cmp.Or(c.Steps, p.DefaultSteps()), cmp.Or(c.Guidance, p.DefaultGuidance())
	if c.Negative != "" && guidance == 1 {
		if c.Guidance != 0 {
			return fmt.Errorf("--negative: --guidance 1 runs no classifier-free guidance; give a --guidance over 1 to use it")
		}
		// Anima-Turbo runs no guidance by default: a negative prompt asks for some, and a weak
		// scale, as a distilled model burns at a strong one
		guidance = turboNegativeGuidance
		logf("--negative: %s runs no guidance by default, so --guidance %g (give --guidance to change it)", cmp.Or(p.Checkpoint, "Anima-Turbo"), guidance)
	}
	cond, err := p.Encode(ctx, c.Prompts[0])
	if err != nil {
		return err
	}
	var uncond []float32
	if guidance != 1 {
		if uncond, err = p.EncodeNegative(ctx, c.Negative); err != nil {
			return err
		}
		logf("classifier-free guidance %g: the DiT twice a step", guidance)
	}
	var x []float32
	if c.Noise != "" {
		if x, err = readF32(c.Noise); err != nil {
			return err
		}
	} else {
		s := c.Seed
		if s == 0 {
			s = rand.Uint64()
		}
		logf("seed %d", s)
		x = p.Noise(s)
	}
	sig := anima.Sigmas(steps, c.Shift)
	bar := console.NewBar(lw, "denoising")
	bar.Set(0, fmt.Sprintf("step 1/%d", steps))
	smp := p.DefaultSampling()
	if c.Sampler != "" {
		smp.Solver = c.Sampler
	}
	if len(c.CFGSigmas) > 0 {
		if len(c.CFGSigmas) != 2 || uncond == nil {
			return fmt.Errorf("--cfg-sigmas: give LO,HI, with a --guidance over 1")
		}
		smp.CFGMin, smp.CFGMax = c.CFGSigmas[0], c.CFGSigmas[1]
	}
	if uncond != nil && smp.CFGMax > 0 {
		n := 0
		for _, sg := range sig[:steps] {
			if sg >= smp.CFGMin && sg <= smp.CFGMax {
				n++
			}
		}
		logf("%s, guidance at %d of %d steps (sigma %g to %g)", cmp.Or(smp.Solver, "euler"), n, steps, smp.CFGMin, smp.CFGMax)
	}
	x, err = p.Denoise(ctx, cond, uncond, guidance, x, sig, smp, func(i int, d time.Duration) {
		per := bar.Elapsed() / time.Duration(i+1)
		left := time.Duration(steps-i-1) * per
		bar.Set(float64(i+1)/float64(steps), fmt.Sprintf("step %d/%d, %.1f s a step, %s left", min(i+2, steps), steps, per.Seconds(), console.Duration(left)))
		if !live {
			logf("step %d/%d: sigma %.3f -> %.3f, %.1f s", i+1, steps, sig[i], sig[i+1], d.Seconds())
		}
	})
	if err != nil {
		bar.Done("failed")
		return err
	}
	bar.Done(fmt.Sprintf("%d steps in %s (%.1f s a step)", steps, console.Duration(bar.Elapsed()), bar.Elapsed().Seconds()/float64(steps)))
	if c.Latents != "" {
		if err := writeF32(c.Latents, x); err != nil {
			return err
		}
	}
	bar = console.NewBar(lw, "decoding")
	p.Stage = func(i, n int, d time.Duration) {
		bar.Set(float64(i+1)/float64(n), fmt.Sprintf("VAE stage %d/%d", min(i+2, n), n))
		if !live {
			logf("vae stage %d: %.2f s", i, d.Seconds())
		}
	}
	img, err := p.Decode(ctx, x)
	if err != nil {
		bar.Done("failed")
		return err
	}
	bar.Done("in " + console.Duration(bar.Elapsed()))
	f, err := os.Create(c.Out)
	if err != nil {
		return err
	}
	if err := png.Encode(f, img); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "%s: %s in all\n", c.Out, console.Duration(time.Since(t0)))
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

func isDir(path string) bool {
	fi, err := os.Stat(path)
	return err == nil && fi.IsDir()
}
