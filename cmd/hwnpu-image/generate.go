package main

import (
	"context"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	"image/png"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/ai"
	aicli "github.com/shibukawa/hwnpu/ai/cli"
	"github.com/shibukawa/hwnpu/ai/imagegen"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	devenv "github.com/shibukawa/hwnpu/internal/dev"
	"github.com/shibukawa/hwnpu/internal/llm"
)

// generate runs its models through the ai package, as an app would: the model --model names is
// a use of the definitions (ai.Catalog), the flags that change how it is made or run a model of
// its own made from it (ai.Definition), prepared when it is not here (aicli.Ensure) and loaded by
// a Runtime that compiles it at its first load (ai.CompileOnLoad); ai/imagegen draws.

// GenerateCmd is "hwnpu-image generate": a picture from a prompt with FLUX.2 klein, FLUX.3 klein or Anima.
type GenerateCmd struct {
	Prompts    []string  `arg:"" name:"prompt" help:"What to draw (Anima: Danbooru tags and natural language). Several draw a picture each, the model loaded once (FLUX.2 klein on Apple's Neural Engine, --batch of them together)."`
	Model      string    `short:"m" default:"image" placeholder:"NAME|EXPORT" help:"The model: a name, a family or a use of models.yaml (image: FLUX.2 klein 4B from 8 GB; flux2-klein, flux3-klein, anima-turbo and anima (Anima-Aesthetic) with HWNPU_MODEL_ALLOWLIST=+nsfw; a family alone picks for the machine's memory, or for the memory a tag names: image:8g), or an export's directory (AMD: of tools/amd/iron_anima.py, iron_flux2klein.py or iron_flux3klein.py). On AMD's NPU a name runs NAME.amd-npu2 in the cache's entries/amd-xdna2 or ~/models, else one the sources hold, downloaded once (FLUX.2 and FLUX.3 klein's exported with IRON when they have none; one made by hand in entries/amd-xdna2 goes in the sources written back to on its first use); on Apple's and Intel's (Anima; FLUX.2 and FLUX.3 klein on Apple's) the model is downloaded and converted on its first run."`
	Out        string    `short:"o" type:"path" placeholder:"FILE" help:"The PNG to write (default: image.png, else image (2).png, ... here, then shown in the system's viewer); with several prompts NAME-1.png, NAME-2.png, ..."`
	Batch      int       `name:"batch" default:"4" placeholder:"N" help:"FLUX.2 klein on Apple's Neural Engine with several prompts: the pictures drawn together, N at a time (their prompts are read, then their steps run, then the VAE's stages over all of them, so that a program is loaded once for the N pictures; they are written together, when the last is drawn); FLUX.2 klein on AMD's NPU too, its steps together (a group of the DiT's weights read once for the N pictures; the VAE a picture after another); 1 draws one picture after another."`
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
	CPU        bool      `name:"cpu" help:"Run a Core ML export with the CPU interpreter instead of the NPU (Anima; slow; for checking)."`
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
	if c.Backend != "" {
		// the device the models are prepared for and run on (ai's NPU is HWNPU_BACKEND's)
		os.Setenv(llm.BackendEnv, c.Backend)
	}
	lw := console.Log(os.Stderr)
	o, err := c.options()
	if err != nil {
		return err
	}
	st := ai.Store{}
	use, defs, err := c.use(st)
	if err != nil {
		return err
	}
	st.Definitions = defs
	app := ai.NewApp("hwnpu-image", use)
	app.Store = st
	if err := aicli.Ensure(ctx, app, aicli.PrepareOptions{AcceptLicenses: acceptLicenses, Log: os.Stderr}); err != nil {
		return err
	}
	t0 := time.Now()
	rt, err := app.Open(ctx, ai.CompileOnLoad(), ai.WithLog(lw))
	if err != nil {
		return err
	}
	defer rt.Close()
	g, err := use.Get(ctx, rt)
	if err != nil {
		return err
	}
	console.Running()
	size, err := g.SizeOf(c.Size)
	if err != nil {
		return err
	}
	if sizes := g.Sizes(); len(sizes) > 1 && !containsSize(sizes, size) {
		return fmt.Errorf("--size %s: %s here draws %s, not %dx%d", c.Size, g.Model(), sizeList(sizes), size.X, size.Y)
	}
	o.Width, o.Height = size.X, size.Y
	if err := c.references(&o); err != nil {
		return err
	}
	c.progress(&o, lw, outs, t0)
	if _, err := g.GenerateAll(ctx, c.Prompts, o); err != nil {
		return err
	}
	if show {
		for _, out := range outs {
			if e := openFile(out); e != nil {
				fmt.Fprintf(os.Stderr, "%s: not shown (%v)\n", out, e)
			}
		}
	}
	return nil
}

// options are the picture's options of the flags, but its size and references.
func (c *GenerateCmd) options() (imagegen.Options, error) {
	o := imagegen.Options{Seed: c.Seed, Steps: c.Steps, Negative: c.Negative, Guidance: c.Guidance, CFGSigmas: c.CFGSigmas,
		Sampler: c.Sampler, Shift: c.Shift, Batch: c.Batch}
	var ck imagegen.Checks
	if c.Noise != "" {
		x, err := readF32(c.Noise)
		if err != nil {
			return o, err
		}
		ck.Noise = x
	}
	ck.NoSteps = devenv.On("klein.decode") // a check of the VAE alone: --noise holds final latents
	if c.Latents != "" {
		ck.Latents = func(_ int, x []float32) {
			if err := writeF32(c.Latents, x); err != nil {
				fmt.Fprintf(os.Stderr, "--latents: %v\n", err)
			}
		}
	}
	for _, f := range c.RefLatents {
		r, err := readF32(f)
		if err != nil {
			return o, err
		}
		ck.RefLatents = append(ck.RefLatents, r)
	}
	ck.Dump = c.Dump
	if ck.Noise != nil || ck.NoSteps || ck.Latents != nil || ck.RefLatents != nil || ck.Dump != "" {
		o.Checks = &ck
	}
	return o, nil
}

// references reads --image into o.
func (c *GenerateCmd) references(o *imagegen.Options) error {
	for _, arg := range c.Image {
		file, div, err := referenceArg(arg)
		if err != nil {
			return err
		}
		r, err := os.Open(file)
		if err != nil {
			return err
		}
		img, _, err := image.Decode(r)
		r.Close()
		if err != nil {
			return fmt.Errorf("%s: %w", file, err)
		}
		o.References = append(o.References, img)
		o.ReferenceScales = append(o.ReferenceScales, div)
	}
	return nil
}

// use is the use of the model --model names, as the flags make it: a model of its own for the
// weight formats, Anima's conversion, --cpu, or an export's directory.
func (c *GenerateCmd) use(st ai.Store) (ai.Use[*imagegen.Generator], []ai.Definition, error) {
	mk := imagegen.Generate
	if (len(c.Image) > 0 || len(c.RefLatents) > 0) && c.Size != "1:1s" && c.Size != "512" {
		// on Apple's Neural Engine with the export of the edits of 1024 beside the model's (the
		// edits at 512 x 512 are in the model's)
		mk = imagegen.Edit
	}
	var d ai.Definition
	name := c.Model
	if isPath(c.Model) {
		var err error
		if d, err = imagegen.Export(expandHome(c.Model)); err != nil {
			return ai.Use[*imagegen.Generator]{}, nil, err
		}
		name = d.Like
	}
	cat, err := st.Catalog()
	if err != nil {
		return ai.Use[*imagegen.Generator]{}, nil, err
	}
	use, defs, err := imageUse(cat, mk(name))
	if err != nil {
		return use, nil, err
	}
	info, _ := cat.Model(use.Name())
	set := func(k, v string) {
		if d.Settings == nil {
			d.Settings = map[string]string{}
		}
		d.Settings[k] = v
	}
	if c.TextFormat != "" {
		set(imagegen.SettingTextFormat, c.TextFormat)
	}
	if c.DitFormat != "" {
		set(imagegen.SettingDitFormat, c.DitFormat)
	}
	vendor := chat.HereVendor()
	switch fam := info.Family; {
	case strings.HasPrefix(fam, "anima"):
		if c.Dit != "" {
			set(imagegen.SettingDit, c.Dit)
		}
		if c.Group > 0 {
			set(imagegen.SettingGroup, strconv.Itoa(c.Group))
		}
		if c.QChunk != 0 && c.QChunk != 512 {
			set(imagegen.SettingQChunk, strconv.Itoa(c.QChunk))
		}
		if c.Size != "1:1" && (vendor == "intel" || c.CPU) { // an export a size on Intel's NPU and Core ML's
			set(imagegen.SettingSize, c.Size)
		}
	}
	if c.CPU {
		d.Device = "cpu"
	}
	if d.Programs == nil && d.Settings == nil && d.Device == "" {
		return use, defs, nil
	}
	d.Like = use.Name()
	return mk(d.ID()), append(defs, d), nil
}

// imageUse is the use of the model u names (ai.Normalize), and the definitions it needs (a
// Hugging Face repository the definitions do not have). A name of none is an error after the
// image models are listed.
func imageUse[T io.Closer](cat *ai.Catalog, u ai.Use[T]) (ai.Use[T], []ai.Definition, error) {
	n, def, err := ai.Normalize(cat, u)
	if e := (*ai.NoModelError)(nil); errors.As(err, &e) {
		fmt.Fprintf(os.Stderr, "no image model %q; the image models of the definitions:\n\n", e.Name)
		listModels(os.Stderr, chat.KindImage)
		fmt.Fprintln(os.Stderr)
		return u, nil, fmt.Errorf("no image model, family or use %q (hwnpu-image list shows them)", e.Name)
	}
	if err != nil {
		return u, nil, err
	}
	if def != nil {
		return n, []ai.Definition{*def}, nil
	}
	return n, nil, nil
}

// progress shows the drawing on lw as the bars of hwnpu's commands, and writes each picture to
// its file of outs as it is drawn.
func (c *GenerateCmd) progress(o *imagegen.Options, lw io.Writer, outs []string, t0 time.Time) {
	live := console.Live(lw)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	var (
		bar   *console.Bar
		took  stepTimes
		batch imagegen.Batch
		t1    time.Time
		times func(time.Duration) string
	)
	o.OnBatch = func(b imagegen.Batch) {
		batch, t1, took = b, time.Now(), nil
		for _, s := range b.Seeds {
			logf("seed %d", s)
		}
		name := "denoising"
		if b.N > 1 {
			name = fmt.Sprintf("denoising %d pictures", b.N)
		}
		// times are a step's time and, with several pictures, each one's share of it
		times = func(per time.Duration) string {
			if b.N > 1 {
				return fmt.Sprintf("%.1f s a step, %.1f s a picture", per.Seconds(), per.Seconds()/float64(b.N))
			}
			return fmt.Sprintf("%.1f s a step", per.Seconds())
		}
		bar = console.NewBar(lw, name)
		bar.Set(0, fmt.Sprintf("step 1/%d", b.Steps))
	}
	o.OnStep = func(s imagegen.Step) {
		took = append(took, s.Took)
		i, total := s.Step, s.Steps
		per := bar.Elapsed() / time.Duration(i+1)
		bar.Set(float64(i+1)/float64(total), fmt.Sprintf("step %d/%d, %s, %s left", min(i+2, total), total, times(per),
			console.Duration(time.Duration(total-i-1)*per)))
		if !live {
			logf("step %d/%d: sigma %.3f -> %.3f, %.1f s", i+1, total, s.From, s.To, s.Took.Seconds())
		}
	}
	o.Phase = func(phase string) {
		if phase == imagegen.PhaseDecode && bar != nil {
			bar.Done(took.summary(bar.Elapsed(), times))
			bar = console.NewBar(lw, "decoding")
		}
	}
	o.OnStage = func(s imagegen.Stage) {
		bar.Set(float64(s.Stage+1)/float64(s.Stages), fmt.Sprintf("VAE stage %d/%d", min(s.Stage+2, s.Stages), s.Stages))
		if !live {
			logf("vae stage %d: %.2f s", s.Stage, s.Took.Seconds())
		}
	}
	o.Drawn = func(i int, img *image.RGBA) {
		if i == batch.First && bar != nil {
			bar.Done("in " + console.Duration(bar.Elapsed()))
			bar = nil
		}
		out := outs[i]
		if err := writePNG(out, img); err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", out, err)
			return
		}
		if n := batch.N; n > 1 {
			fmt.Fprintf(os.Stderr, "%s: %s a picture, %d together (%s since the start)\n", out, console.Duration(time.Since(t1)/time.Duration(n)), n,
				console.Duration(time.Since(t0)))
		} else {
			fmt.Fprintf(os.Stderr, "%s: %s (%s since the start)\n", out, console.Duration(time.Since(t1)), console.Duration(time.Since(t0)))
		}
	}
}

// writePNG writes img to path.
func writePNG(path string, img image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := png.Encode(f, img); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

func containsSize(sizes []image.Point, s image.Point) bool {
	for _, x := range sizes {
		if x == s {
			return true
		}
	}
	return false
}

// sizeList is "1024x1024, 512x512, ...".
func sizeList(sizes []image.Point) string {
	var out []string
	for _, s := range sizes {
		out = append(out, fmt.Sprintf("%dx%d", s.X, s.Y))
	}
	return strings.Join(out, ", ")
}

// referenceArg is an --image argument's file and the reference's size as the picture's sides over
// div: FILE or FILE@full (1: the picture's size), FILE@half (2: half its sides). A file whose own
// name ends in @half or @full is read as it is, at the picture's size.
func referenceArg(arg string) (file string, div int, err error) {
	div = 1
	if _, err := os.Stat(arg); err == nil {
		return arg, div, nil
	}
	at := strings.LastIndex(arg, "@")
	if at < 0 {
		return "", 0, fmt.Errorf("--image: %s: no such file", arg)
	}
	switch strings.ToLower(arg[at+1:]) {
	case "half":
		div = 2
	case "full":
		div = 1
	default:
		return "", 0, fmt.Errorf("--image: %s: no such file (a reference's size is FILE@half or FILE@full)", arg)
	}
	file = arg[:at]
	if _, err := os.Stat(file); err != nil {
		return "", 0, fmt.Errorf("--image: %s: no such file", file)
	}
	return file, div, nil
}
