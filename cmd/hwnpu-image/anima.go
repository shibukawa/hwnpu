package main

import (
	"cmp"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"image/png"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/imagegen/anima"
	"github.com/shibukawa/hwnpu/internal/imagegen/sana"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
)

// convertFlags are what an Anima export for Apple's or Intel's NPU is made with (generate's first
// run, and convert).
type convertFlags struct {
	Backend    string `help:"The program format: coreml (Apple's Neural Engine) or intel (Intel's NPU); default: the NPU found (convert: coreml)."`
	Size       int    `default:"1024" help:"The image's width and height in pixels, a multiple of 16 (an export each)."`
	Dit        string `placeholder:"FILE" help:"The diffusion model, a file of split_files/diffusion_models/ in circlestone-labs/Anima (default: the model's of models.yaml; convert: anima-turbo-v1.1.safetensors)."`
	TextFormat string `help:"The text encoder's weights (fp16; lut8, lut6 on Core ML; int8 on Intel; default int8 on Core ML, where fp16's 1.0 GB is over a program's limit, fp16 on Intel)."`
	DitFormat  string `help:"The DiT's weights (fp16; lut8, lut6 on Core ML; int8, int4 on Intel; default int8 on Core ML: faster than fp16 at half its memory, and nearer it than lut8; fp16 on Intel)."`
	Group      int    `help:"The DiT's blocks a program holds (default 4; Anima on Intel: fewer above 512 x 512, 1 at 1024 x 1024, as a program running past Windows' 2 s limit loses the NPU)."`
	QChunk     int    `name:"qchunk" default:"512" help:"The queries an attention scores at a time (M3: 512 is 25 % faster than 1024 at 1024 x 1024)."`
}

func (f convertFlags) options() anima.ConvertOptions {
	return anima.ConvertOptions{Backend: f.Backend, Width: f.Size, Height: f.Size, Dit: f.Dit,
		TextFormat: f.TextFormat, DitFormat: f.DitFormat, Group: f.Group, QChunk: f.QChunk}
}

// ConvertCmd is "hwnpu-image convert": an Anima directory into a Core ML or OpenVINO export, or
// a Sana repository into a Core ML export.
type ConvertCmd struct {
	Anima string `arg:"" name:"dir" type:"existingdir" help:"The Anima directory (diffusion_models/, text_encoders/, diffusers/), or a Sana repository of Efficient-Large-Model (transformer/, vae/, scheduler/)."`
	Model string `short:"m" required:"" type:"path" placeholder:"EXPORT" help:"The export to write."`
	Text  string `type:"existingdir" placeholder:"DIR" help:"Sana: the directory of Gemma 2's text_encoder/ and tokenizer/ (default: the repository)."`
	convertFlags
}

// Run converts the checkpoint.
func (c *ConvertCmd) Run() error {
	t0 := time.Now()
	if isDir(filepath.Join(c.Anima, "transformer")) {
		if c.Backend != "" && c.Backend != llm.BackendCoreML {
			return fmt.Errorf("Sana converts for coreml only (--backend %s)", c.Backend)
		}
		o := sana.ConvertOptions{Width: c.Size, Height: c.Size, Group: c.Group, TextFormat: c.TextFormat, DitFormat: c.DitFormat}
		if err := sana.Convert(c.Anima, cmp.Or(c.Text, c.Anima), c.Model, o, console.Log(os.Stderr)); err != nil {
			return err
		}
	} else {
		o := c.options()
		o.Backend = cmp.Or(o.Backend, llm.BackendCoreML)
		text, dit := anima.Formats(o.Backend)
		o.TextFormat, o.DitFormat = cmp.Or(o.TextFormat, text), cmp.Or(o.DitFormat, dit)
		if err := anima.Convert(c.Anima, c.Model, o, console.Log(os.Stderr)); err != nil {
			return err
		}
	}
	fmt.Fprintf(os.Stderr, "%s: converted in %s\n", c.Model, console.Duration(time.Since(t0)))
	return nil
}

// GenerateCmd is "hwnpu-image generate": a picture from a prompt with Anima or Sana.
type GenerateCmd struct {
	Prompts   []string  `arg:"" name:"prompt" help:"What to draw (Anima: Danbooru tags and natural language; Sana: English). Sana takes several: a picture each, the model loaded once."`
	Model     string    `short:"m" default:"image" placeholder:"NAME|EXPORT" help:"The model: a name, a family or a use of models.yaml (image: anima-turbo; anima (Anima-Aesthetic), sana, sana:1.6b; a family alone picks for the machine's memory, or for the memory a tag names: sana:8g), or an export's directory (AMD: of tools/amd/iron_anima.py or iron_sana.py). On AMD's NPU a name runs ~/models/NAME.amd-npu2; on Apple's (Anima and Sana) and Intel's (Anima) the model is downloaded and converted into --dir on its first run."`
	Dir       string    `type:"path" placeholder:"DIR" help:"Where the models are kept (as hwnpu-chat's --dir): the checkpoints and exports in Anima/ and Sana/."`
	Out       string    `short:"o" type:"path" placeholder:"FILE" help:"The PNG to write (default: image.png, else image (2).png, ... here, then shown in the system's viewer); with several prompts NAME-1.png, NAME-2.png, ..."`
	KeepDiT   bool      `name:"keep-dit" default:"true" negatable:"" help:"Sana on Apple's Neural Engine with several prompts: keep the DiT loaded between the pictures (Gemma 2 and the VAE are loaded for each)."`
	Steps     int       `help:"Steps (default: Anima-Turbo's Euler 8, 8 to 12 wanted; Anima-Aesthetic 25 with dpm2m, the model card's 30 to 50 with euler; Sana 20)."`
	Negative  string    `help:"The negative prompt of the classifier-free guidance (default empty; Anima's model card recommends \"worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration\", without the score tags for Anima-Aesthetic)."`
	Guidance  float64   `placeholder:"SCALE" help:"The classifier-free guidance's scale, each step the DiT twice (default: Anima 4, 4 to 5 wanted; Anima-Turbo 1, none, or 1.5 with a --negative; Sana 4.5)."`
	Shift     float64   `placeholder:"SHIFT" default:"3.0" help:"Anima: the sigma schedule's shift (Anima's scheduler: 3)."`
	Sampler   string    `placeholder:"NAME" help:"Anima: the solver, euler (Anima's scheduler; Anima-Turbo's default) or dpm2m (DPM-Solver++ 2M, the default with guidance: the same DiT runs a step, closer to many steps' result in fewer)."`
	CFGSigmas []float64 `name:"cfg-sigmas" sep:"," placeholder:"LO,HI" help:"Run the classifier-free guidance only at steps whose sigma is in [LO, HI] (1 is pure noise; 0,1 for every step; default 0.6,1 for Anima and Sana); the other steps run the DiT once."`
	Seed      uint64    `placeholder:"N" help:"The noise's seed (0: from the time)."`
	Noise     string    `type:"path" placeholder:"FILE" help:"The initial latents as raw float32 in place of the seed's (Anima: (16, H/8, W/8); Sana: (32, H/32, W/32))."`
	Latents   string    `type:"path" placeholder:"FILE" help:"Also write the final latents as raw float32 here."`
	Cache     string    `help:"Compile cache (default: hwnpu in the user cache directory; \"-\" for none)."`
	CPU       bool      `name:"cpu" help:"Run a Core ML export with the CPU interpreter instead of the NPU."`
	Resident  bool      `name:"resident-dit" help:"AMD: keep every group of the DiT's blocks on the NPU instead of reading each group's weights into one bank before it runs (Anima: pinned memory 7 banks, 3.4 GB at q8)."`
	Dump      string    `type:"path" placeholder:"DIR" help:"A directory to write the conditioning and each step's velocity to (raw bf16), to check against tools/ref/ref_anima.py or ref_sana.py."`
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
	if !isPath(c.Model) {
		m, err := resolve(c.Model, chat.KindImage)
		if err != nil {
			return err
		}
		model = ""
		if isAnima := strings.HasPrefix(m.Family, "anima"); !isAnima {
			if model, err = c.sanaModel(ctx, m); err != nil {
				return err
			}
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
	if sana.IsExport(model) {
		err = runSana(ctx, sanaRun{model: model, prompts: c.Prompts, outs: outs, keepDiT: c.KeepDiT, negative: c.Negative, noise: c.Noise, latents: c.Latents,
			steps: c.Steps, guidance: c.Guidance, cfgSigmas: c.CFGSigmas, seed: c.Seed, resident: c.Resident, dump: c.Dump, cpu: c.CPU, cache: c.Cache})
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

// sanaModel is the Sana export m runs here: on Apple's Neural Engine (or with --cpu) the Core ML
// export of --dir, downloaded and converted when missing; on AMD's NPU ~/models/NAME.amd-npu2
// of tools/amd/iron_sana.py.
func (c *GenerateCmd) sanaModel(ctx context.Context, m chat.Model) (string, error) {
	format, err := llm.ProgramFormat(ctx, c.Backend)
	switch {
	case c.CPU || err == nil && format == llm.BackendCoreML:
	case err == nil:
		return "", fmt.Errorf("%s runs on AMD's NPU and Apple's Neural Engine; not yet on %s", m.Name, format)
	default:
		return amdExport(m, "tools/amd/iron_sana.py")
	}
	if c.Size%32 != 0 {
		return "", fmt.Errorf("--size %d: Sana's pictures are multiples of 32 pixels", c.Size)
	}
	o := sana.ConvertOptions{Width: c.Size, Height: c.Size, Group: c.Group}
	if c.TextFormat != "" || c.DitFormat != "" {
		o.TextFormat, o.DitFormat = c.TextFormat, c.DitFormat
	}
	return sana.Prepare(ctx, sana.Setup{Dir: filepath.Join(cmp.Or(c.Dir, chat.ModelsDir()), "Sana"), Repo: m.Repo, Name: m.Name,
		ConvertOptions: o, Log: console.Log(os.Stderr)})
}

// animaModel is the Anima export m runs here: on AMD's NPU (no Apple or Intel NPU found, no
// --backend) ~/models/NAME.amd-npu2 of tools/amd/iron_anima.py, "" otherwise (anima prepares
// the one of --dir).
func (c *GenerateCmd) animaModel(ctx context.Context, m chat.Model) (string, error) {
	if c.CPU || c.Backend != "" {
		return "", nil
	}
	if _, err := llm.ProgramFormat(ctx, ""); err == nil {
		return "", nil // Apple's or Intel's NPU
	}
	return amdExport(m, "tools/amd/iron_anima.py")
}

// turboNegativeGuidance is the guidance scale a --negative without --guidance gets on a model that
// runs none by default (Anima-Turbo): weak, as a model distilled to run without guidance burns
// at the 4 the others take.
const turboNegativeGuidance = 1.5

// anima draws the picture with Anima: the export model, or "" for the one of --dir (downloaded
// and converted when missing).
func (c *GenerateCmd) anima(ctx context.Context, model string) error {
	anima.ResidentDiT = c.Resident
	o := c.options()
	cacheDir := c.Cache
	dir := cmp.Or(c.Dir, chat.ModelsDir())
	// what goes on in gray (with a spinner or a bar on a terminal), the picture's line as it is
	lw := console.Log(os.Stderr)
	live := console.Live(lw)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }

	if model == "" {
		format := llm.BackendCoreML // the CPU interpreter runs Core ML exports
		var err error
		if !c.CPU {
			format, err = llm.ProgramFormat(ctx, c.Backend)
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
	p, err := anima.Open(ctx, dev, model, lw)
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
		if d, err := os.UserCacheDir(); err == nil {
			return filepath.Join(d, "hwnpu")
		}
		return ""
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
