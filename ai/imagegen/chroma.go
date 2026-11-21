package imagegen

import (
	"cmp"
	"context"
	"fmt"
	"image"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagegen/aspect"
	"github.com/shibukawa/hwnpu/internal/imagegen/chroma"
	"github.com/shibukawa/hwnpu/internal/imagegen/flux2klein"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
)

// Chroma1 (Chroma1-Flash: Heun's 8 steps without guidance; Chroma1-HD: guidance 3 with
// DPM-Solver++ 2M): FLUX.1 schnell's DiT with an FFN for its modulations, T5-XXL and FLUX's VAE.
// AMD's NPU runs the export of tools/amd/iron_chroma.py (NAME.amd-npu2 of the eight aspect
// ratios, here or in the sources); Apple's Neural Engine a Core AI export of every size of
// models.yaml's npu.apple (one export, the weights once) converted from the checkpoint; a model
// of device cpu the Core ML one of a square (the setting size) on the CPU interpreter.

// chromaTool is the script that exports Chroma for AMD's NPU.
const chromaTool = "tools/amd/iron_chroma.py"

// chromaSide is the side of the square the setting size names for Chroma's conversions.
func chromaSide(m chat.Model) (int, error) {
	spec := setting(m, SettingSize, "1:1")
	w, h, err := aspect.ParseOrSide(spec, 16)
	if err != nil {
		return 0, err
	}
	if w != h {
		return 0, fmt.Errorf("--size %s: Chroma's exports for Apple's Neural Engine take 1:1 or 1:1s for now", spec)
	}
	if w%16 != 0 {
		return 0, fmt.Errorf("--size %d: Chroma's pictures are multiples of 16 pixels", w)
	}
	return w, nil
}

func prepareChroma(e airole.PrepareEnv) (f airole.Files, err error) {
	m, base := e.Model, e.Base
	if dir := first(m.Programs[ProgramExport]); dir != "" {
		f.Set("export", dir)
		return f, nil
	}
	cpu := e.NPU.CPU(m)
	format := e.NPU.Format
	switch {
	case cpu: // the CPU interpreter runs Core ML exports
		format = llm.BackendCoreML
	case e.NPU.AMD():
		// an export of every aspect ratio, or one of the square alone
		dir, err := aspectsAMD(e.Ctx, base, 1024, 1024, chromaTool, chroma.IronSizes, e.Log)
		if err != nil {
			return f, err
		}
		f.Set("export", dir)
		return f, nil
	case !llm.AppleFormat(format):
		return f, fmt.Errorf("ai/imagegen: %s runs on AMD's NPU and Apple's Neural Engine; not yet on %s", m.Name, cmp.Or(format, "this machine"))
	}
	o := chroma.ConvertOptions{Backend: format, TextFormat: m.Settings[SettingTextFormat], DitFormat: m.Settings[SettingDitFormat]}
	if format == llm.BackendCoreAI {
		// every size of models.yaml's npu.apple in one export (the weights once), 1:1 first
		as, err := flux2klein.NPUAspects(m, "apple", flux2klein.NPUGeneration("apple"))
		if err != nil {
			return f, err
		}
		o.Width, o.Height, o.Sizes = as[0].Width, as[0].Height, aspect.SizesOf(as)
	} else {
		side, err := chromaSide(m)
		if err != nil {
			return f, err
		}
		o.Width, o.Height = side, side
	}
	dir, err := chroma.Prepare(chat.WithModel(e.Ctx, base), chroma.Setup{Dir: filepath.Join(chat.ModelsDir(), "Chroma"), Repo: base.Repo, Name: base.Name,
		ConvertOptions: o, Log: e.Log})
	if err != nil {
		return f, err
	}
	f.Set("export", dir)
	return f, nil
}

// chromaGen draws with Chroma.
type chromaGen struct {
	g     *Generator
	dir   string
	sized bool // an export of several sizes, opened at a size each
	p     *chroma.Pipeline
	at    image.Point
}

func loadChroma(ctx context.Context, g *Generator, f airole.Files, e airole.LoadEnv) (family, error) {
	c := &chromaGen{g: g, dir: f.Path("export")}
	programs := chroma.IsProgramExport(c.dir)
	var err error
	switch {
	case programs:
		// the sizes the export holds: models.yaml's npu.apple in a Core AI export, else its square
		man, err := chroma.ReadProgramManifest(c.dir)
		if err != nil {
			return nil, err
		}
		if s := man.SizeList(); len(s) > 1 {
			c.sized = true
			g.sizes = points(s)
		}
		g.parse = func(spec string) (int, int, error) {
			w, h, err := aspect.ParseOrSide(spec, 16)
			if err == nil && !c.sized && w != h {
				err = fmt.Errorf("--size %s: this export holds a square alone", spec)
			}
			return w, h, err
		}
		if !e.NPU.CPU(e.Def) {
			// a Core AI export on Core AI
			if g.dev, _, err = llm.OpenDevice(ctx, exportDevice(chroma.ExportFormat(c.dir), e.NPU), e.Cache, placement(e)); err != nil {
				return nil, err
			}
		}
	default:
		g.amd = true
		g.parse = func(spec string) (int, int, error) { return aspect.ParseOrSide(spec, 16) }
		if g.dev, err = openAMD(ctx); err != nil {
			return nil, err
		}
		iron.ShareExport(c.dir, logfOf(e.Log))
		if s := chroma.IronSizes(c.dir); len(s) > 1 {
			c.sized = true
			g.sizes = points(s)
		}
	}
	if len(g.sizes) == 0 {
		g.sizes = []image.Point{{}}
	}
	p, err := c.open(ctx, g.sizes[0])
	if err != nil {
		return nil, err
	}
	if !c.sized {
		g.sizes = []image.Point{{p.C.Width, p.C.Height}}
		c.at = g.sizes[0]
	}
	return c, nil
}

func (c *chromaGen) open(ctx context.Context, size image.Point) (*chroma.Pipeline, error) {
	if c.p != nil && c.at == size {
		return c.p, nil
	}
	if c.p != nil {
		c.p.Close()
		c.p = nil
	}
	w, h := 0, 0 // the export's first size
	if c.sized {
		w, h = size.X, size.Y
	}
	t0 := time.Now()
	p, err := chroma.OpenAt(ctx, c.g.dev, c.dir, w, h, c.g.log)
	if err != nil {
		return nil, err
	}
	p.Logf = logfOf(c.g.log)
	c.g.logf("loaded %s (%dx%d) in %.1f s", filepath.Base(c.dir), p.C.Width, p.C.Height, time.Since(t0).Seconds())
	c.p, c.at = p, size
	return p, nil
}

func (c *chromaGen) warm(ctx context.Context, r *run) error { return c.draw(ctx, r) }

func (c *chromaGen) draw(ctx context.Context, r *run) error {
	o := r.o
	if c.p == nil || c.at != r.size {
		r.phase(PhaseOpen)
	}
	t0 := time.Now()
	p, err := c.open(ctx, r.size)
	if err != nil {
		return err
	}
	r.opened(t0)
	// with several pictures the DiT stays loaded between them
	p.KeepDiT = !o.UnloadDiT && len(r.prompts) > 1
	sig := p.Sigmas(o.Steps)
	steps := len(sig) - 1
	// the export's defaults: Chroma1-Flash Heun's 8 steps without guidance, Chroma1-HD 40 steps
	// with guidance 3 (model.json)
	smp := chroma.Sampling{Solver: cmp.Or(o.Sampler, p.Model.Sampler, "euler"), Guidance: cmp.Or(o.Guidance, p.Model.Guidance, 1)}
	if o.Negative != "" && smp.Guidance == 1 {
		return fmt.Errorf("--negative: %s runs no classifier-free guidance; give a --guidance over 1 to use it", p.Model.Name)
	}
	if len(o.CFGSigmas) > 0 {
		if len(o.CFGSigmas) != 2 || smp.Guidance == 1 {
			return fmt.Errorf("--cfg-sigmas: give LO,HI, with a --guidance over 1")
		}
		smp.CFGMin, smp.CFGMax = o.CFGSigmas[0], o.CFGSigmas[1]
	} else if cs := p.Model.CFGSigmas; len(cs) == 2 && o.Sampler == "" && o.Steps == 0 {
		smp.CFGMin, smp.CFGMax = cs[0], cs[1] // the export's default with its default sampler and steps
	}
	if smp.Guidance != 1 {
		n := 0
		for _, sg := range sig[:steps] {
			if smp.CFGMax <= 0 || sg >= smp.CFGMin && sg <= smp.CFGMax {
				n++
			}
		}
		c.g.logf("%s, classifier-free guidance %g at %d of %d steps", smp.Solver, smp.Guidance, n, steps)
	}
	// every prompt (and the negative one) in one pass of the text encoder: its groups are loaded
	// once for all the pictures, not once a picture (T5's twelve fp16 programs take 11 s to load on
	// the M1)
	texts := append(append([]string(nil), r.prompts...), o.Negative)
	if smp.Guidance == 1 {
		texts = texts[:len(r.prompts)]
	}
	r.phase(PhasePrompt)
	t := time.Now()
	conds, err := p.EncodeAll(ctx, texts...)
	if err != nil {
		return err
	}
	r.out.Prompts = time.Since(t)
	var neg *chroma.Cond
	if smp.Guidance != 1 {
		neg = &conds[len(r.prompts)]
	}
	// an export of programs: Batch pictures at a time, their steps together (a group of the DiT's
	// blocks runs every picture before the next group), then the VAE's stages over all of them.
	// AMD's NPU draws one picture after another
	batch := 1
	if p.Batched() {
		batch = max(1, o.Batch)
	}
	for lo := 0; lo < len(r.prompts); lo += batch {
		hi := min(lo+batch, len(r.prompts))
		xs := r.batch(lo, hi, steps, p.Noise)
		xs, err = p.GenerateAll(ctx, conds[lo:hi], neg, xs, sig, smp, r.stepsOf(sig))
		if err != nil {
			return err
		}
		r.latents(lo, xs...)
		r.phase(PhaseDecode)
		p.Stage = r.stage
		t := time.Now()
		imgs, err := p.DecodeAll(ctx, xs...)
		p.Stage = nil
		if err != nil {
			return err
		}
		if err := r.drawn(lo, time.Since(t), imgs...); err != nil {
			return err
		}
	}
	return nil
}

func (c *chromaGen) close() error {
	if c.p == nil {
		return nil
	}
	err := c.p.Close()
	c.p = nil
	return err
}
