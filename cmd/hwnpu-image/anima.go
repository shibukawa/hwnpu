package main

import (
	"cmp"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"image/png"
	"math"
	"os"
	"path/filepath"
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
	Dit        string `default:"anima-turbo-v1.1.safetensors" help:"The diffusion model, a file of split_files/diffusion_models/ in circlestone-labs/Anima."`
	TextFormat string `help:"The text encoder's weights (fp16; lut8, lut6 on Core ML; int8 on Intel; default int8 on Core ML, where fp16's 1.0 GB is over a program's limit, fp16 on Intel)."`
	DitFormat  string `help:"The DiT's weights (fp16; lut8, lut6 on Core ML; int8, int4 on Intel; default int8 on Core ML: faster than fp16 at half its memory, and nearer it than lut8; fp16 on Intel)."`
	Group      int    `default:"4" help:"The DiT's blocks a program holds."`
	QChunk     int    `name:"qchunk" default:"512" help:"The queries an attention scores at a time (M3: 512 is 25 % faster than 1024 at 1024 x 1024)."`
}

func (f convertFlags) options() anima.ConvertOptions {
	return anima.ConvertOptions{Backend: f.Backend, Width: f.Size, Height: f.Size, Dit: f.Dit,
		TextFormat: f.TextFormat, DitFormat: f.DitFormat, Group: f.Group, QChunk: f.QChunk}
}

// ConvertCmd is "hwnpu-image convert": an Anima directory into a Core ML or OpenVINO export.
type ConvertCmd struct {
	Anima string `arg:"" type:"existingdir" help:"The Anima directory (diffusion_models/, text_encoders/, diffusers/)."`
	Model string `short:"m" required:"" type:"path" placeholder:"EXPORT" help:"The export to write."`
	convertFlags
}

// Run converts the checkpoint.
func (c *ConvertCmd) Run() error {
	o := c.options()
	o.Backend = cmp.Or(o.Backend, llm.BackendCoreML)
	text, dit := anima.Formats(o.Backend)
	o.TextFormat, o.DitFormat = cmp.Or(o.TextFormat, text), cmp.Or(o.DitFormat, dit)
	t0 := time.Now()
	if err := anima.Convert(c.Anima, c.Model, o, console.Log(os.Stderr)); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "%s: converted in %s\n", c.Model, console.Duration(time.Since(t0)))
	return nil
}

// GenerateCmd is "hwnpu-image generate": a picture from a prompt with Anima, Sana or Sana-Sprint.
type GenerateCmd struct {
	Prompt   string  `arg:"" help:"What to draw (Anima: Danbooru tags and natural language; Sana: English)."`
	Model    string  `short:"m" default:"image" placeholder:"NAME|EXPORT" help:"The model: a name, a family or a use of models.yaml (anima, sana, sana:1.6b, sana-full; a family alone picks for the machine's memory, or for the memory a tag names: sana:8g), or an export's directory (AMD: of tools/amd/iron_anima.py or iron_sana.py). On AMD's NPU a name runs ~/models/NAME.amd-npu2; on Apple's and Intel's Anima is downloaded and converted into --dir on its first run."`
	Dir      string  `type:"path" placeholder:"DIR" help:"Where the models are kept (as hwnpu-chat's -dir): Anima's checkpoint and exports in Anima/."`
	Out      string  `short:"o" type:"path" placeholder:"FILE" help:"The PNG to write (default: image.png, else image (2).png, ... here, then shown in the system's viewer)."`
	Steps    int     `help:"Steps (default: Anima-Turbo's Euler 8, 8 to 12 wanted; Sana-Sprint 2, 1 to 4; Sana 20)."`
	Negative string  `help:"Sana: the negative prompt of the classifier-free guidance (default empty)."`
	Guidance float64 `placeholder:"SCALE" help:"Sana: the guidance scale (default 4.5; Sana-Sprint embeds it, Sana runs classifier-free guidance)."`
	Shift    float64 `placeholder:"SHIFT" default:"3.0" help:"Anima: the sigma schedule's shift (Anima's scheduler: 3)."`
	Seed     uint64  `placeholder:"N" help:"The noise's seed (0: from the time)."`
	Noise    string  `type:"path" placeholder:"FILE" help:"The initial latents as raw float32 in place of the seed's (Anima: (16, H/8, W/8); Sana-Sprint: steps+1 of (32, H/32, W/32))."`
	Latents  string  `type:"path" placeholder:"FILE" help:"Also write the final latents as raw float32 here."`
	Cache    string  `help:"Compile cache (default: hwnpu in the user cache directory; \"-\" for none)."`
	CPU      bool    `name:"cpu" help:"Run a Core ML export with the CPU interpreter instead of the NPU."`
	Resident bool    `name:"resident-dit" help:"AMD: keep every group of the DiT's blocks on the NPU instead of reading each group's weights into one bank before it runs (Anima: pinned memory 7 banks, 3.4 GB at q8)."`
	Dump     string  `type:"path" placeholder:"DIR" help:"A directory to write the conditioning and each step's velocity to (raw bf16), to check against tools/ref/ref_anima.py or ref_sana.py."`
	convertFlags
}

// Run draws the picture into --out, or into a new image.png here which it then shows.
func (c *GenerateCmd) Run(ctx context.Context) error {
	show := c.Out == ""
	if show {
		c.Out = freeName("image.png")
	}
	model := expandHome(c.Model)
	isAnima := false
	if !isPath(c.Model) {
		m, err := resolve(c.Model, chat.KindImage)
		if err != nil {
			return err
		}
		model, isAnima = "", m.Family == "anima"
		if !isAnima { // Sana: AMD's NPU only
			if model, err = amdExport(m, "tools/amd/iron_sana.py"); err != nil {
				return err
			}
		} else if model, err = c.animaModel(ctx, m); err != nil {
			return err
		}
	}
	var err error
	if sana.IsExport(model) {
		err = runSana(ctx, sanaRun{model: model, prompt: c.Prompt, negative: c.Negative, out: c.Out, noise: c.Noise, latents: c.Latents,
			steps: c.Steps, guidance: c.Guidance, seed: c.Seed, resident: c.Resident, dump: c.Dump})
	} else {
		err = c.anima(ctx, model)
	}
	if err == nil && show {
		if e := openFile(c.Out); e != nil {
			fmt.Fprintf(os.Stderr, "%s: not shown (%v)\n", c.Out, e)
		}
	}
	return err
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

// anima draws the picture with Anima: the export model, or "" for the one of --dir (downloaded
// and converted when missing).
func (c *GenerateCmd) anima(ctx context.Context, model string) error {
	anima.ResidentDiT = c.Resident
	o := c.options()
	if c.Negative != "" || c.Guidance != 0 {
		return errors.New("--negative and --guidance are for Sana (Anima-Turbo runs no classifier-free guidance)")
	}
	steps, cacheDir := c.Steps, c.Cache
	if steps == 0 {
		steps = 8
	}
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
		if cacheDir == "" {
			if d, err := os.UserCacheDir(); err == nil {
				cacheDir = filepath.Join(d, "hwnpu")
			}
		}
		if cacheDir != "" && cacheDir != "-" {
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

	cond, err := p.Encode(ctx, c.Prompt)
	if err != nil {
		return err
	}
	var x []float32
	if c.Noise != "" {
		if x, err = readF32(c.Noise); err != nil {
			return err
		}
	} else {
		s := c.Seed
		if s == 0 {
			s = uint64(time.Now().UnixNano())
		}
		logf("seed %d", s)
		x = p.Noise(s)
	}
	sig := anima.Sigmas(steps, c.Shift)
	bar := console.NewBar(lw, "denoising")
	bar.Set(0, fmt.Sprintf("step 1/%d", steps))
	x, err = p.Denoise(ctx, cond, x, sig, func(i int, d time.Duration) {
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
