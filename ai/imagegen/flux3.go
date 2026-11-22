package imagegen

import (
	"context"
	"errors"
	"fmt"
	"image"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagegen/aspect"
	"github.com/shibukawa/hwnpu/internal/imagegen/flux2klein"
	"github.com/shibukawa/hwnpu/internal/imagegen/flux3klein"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
)

// FLUX.3 klein: 16 flow-map steps without guidance, a prompt after another, from a prompt alone.
// AMD's NPU runs the export of tools/amd/iron_flux3klein.py of every size of models.yaml's
// npu.amd (one export, from the shared layers or made with IRON); Apple's Neural Engine a Core AI
// export converted from the checkpoint (with the DiT's lut8 tables fitted to the activations when
// the exports' directory or a source holds flux3klein.StatsFile).

func prepareFlux3(e airole.PrepareEnv) (f airole.Files, err error) {
	m, base := e.Model, e.Base
	if dir := first(m.Programs[ProgramExport]); dir != "" {
		f.Set("export", dir)
		return f, nil
	}
	if e.NPU.CPU(m) {
		return f, fmt.Errorf("ai/imagegen: %s runs on the NPU alone (not on the CPU)", m.Name)
	}
	ctx := chat.WithModel(e.Ctx, base)
	var dir string
	switch {
	case e.NPU.AMD():
		as, err := flux2klein.NPUAspects(m, "amd", flux2klein.NPUGeneration("amd"))
		if err != nil {
			return f, err
		}
		dir, err = flux3klein.PrepareAMD(ctx, base, aspect.SizesOf(as), as[0].Width, as[0].Height, e.Log)
		if err != nil {
			return f, err
		}
	case e.NPU.Format == llm.BackendCoreAI:
		o := flux3klein.ConvertOptions{TextFormat: m.Settings[SettingTextFormat], DitFormat: m.Settings[SettingDitFormat]}
		dir, err = flux3klein.Prepare(ctx, flux3klein.Setup{Dir: filepath.Join(chat.ModelsDir(), "FLUX.3-klein"), Repo: base.Repo,
			Name: base.Name, ConvertOptions: o, Log: e.Log})
		if err != nil {
			return f, err
		}
	default:
		return f, fmt.Errorf("ai/imagegen: %s runs on AMD's NPU and Apple's Neural Engine; not yet on %s", m.Name, e.NPU.Format)
	}
	f.Set("export", dir)
	return f, nil
}

// flux3 draws with FLUX.3 klein.
type flux3 struct {
	g   *Generator
	dir string
	p   *flux3klein.Pipeline
	at  image.Point
}

func loadFlux3(ctx context.Context, g *Generator, f airole.Files, e airole.LoadEnv) (family, error) {
	x := &flux3{g: g, dir: f.Path("export")}
	var err error
	g.amd = flux3klein.IsIronExport(x.dir)
	if g.amd {
		as, err := flux2klein.NPUAspects(g.model, "amd", flux2klein.NPUGeneration("amd"))
		if err != nil {
			return nil, err
		}
		g.parse = func(spec string) (int, int, error) { return aspect.ParseOrSideIn(spec, as, 16) }
		if s := flux3klein.IronSizes(x.dir); len(s) > 0 {
			g.sizes = points(s)
		} else {
			g.sizes = points(aspect.SizesOf(as))
		}
		if g.dev, err = openAMD(ctx); err != nil {
			return nil, err
		}
		iron.ShareExport(x.dir, logfOf(e.Log))
	} else {
		g.parse = func(spec string) (int, int, error) { return aspect.ParseOrSide(spec, 16) }
		man, err := flux3klein.ReadProgramManifest(x.dir)
		if err != nil {
			return nil, err
		}
		g.sizes = points(man.Sizes())
		if g.dev, _, err = llm.OpenDevice(ctx, llm.BackendCoreAI, e.Cache, placement(e)); err != nil {
			return nil, err
		}
	}
	if len(g.sizes) == 0 {
		return nil, fmt.Errorf("ai/imagegen: the export %s draws no size", x.dir)
	}
	p, err := x.open(ctx, g.sizes[0])
	if err != nil {
		return nil, err
	}
	if !p.Manifest().FittedTables {
		g.logf("note: the export's DiT tables are spaced evenly (no activation statistics when it was converted): fine detail may change")
	}
	return x, nil
}

func (x *flux3) open(ctx context.Context, size image.Point) (*flux3klein.Pipeline, error) {
	if x.p != nil && x.at == size {
		return x.p, nil
	}
	if x.p != nil {
		x.p.Close()
		x.p = nil
	}
	t0 := time.Now()
	p, err := flux3klein.Open(chat.WithModel(ctx, x.g.model), x.g.dev, x.dir, size.X, size.Y, x.g.log)
	if err != nil {
		return nil, err
	}
	p.Logf = logfOf(x.g.log)
	x.g.logf("loaded %s (%dx%d) in %.1f s", filepath.Base(x.dir), size.X, size.Y, time.Since(t0).Seconds())
	x.p, x.at = p, size
	return p, nil
}

func (x *flux3) warm(ctx context.Context, r *run) error {
	return x.drawSteps(ctx, r, flux3klein.Sigmas()[:2])
}

func (x *flux3) draw(ctx context.Context, r *run) error {
	o := r.o
	if o.Negative != "" || o.Guidance > 1 {
		return errors.New("FLUX.3 klein is distilled: it runs no classifier-free guidance (Negative, Guidance)")
	}
	if o.Steps != 0 && o.Steps != flux3klein.Steps {
		return fmt.Errorf("FLUX.3 klein is distilled for %d steps (Steps)", flux3klein.Steps)
	}
	if len(o.References) > 0 {
		return errors.New("ai/imagegen: FLUX.3 klein's edits are not supported yet (text to image only)")
	}
	return x.drawSteps(ctx, r, flux3klein.Sigmas())
}

// drawSteps draws r's pictures over sig, a prompt after another.
func (x *flux3) drawSteps(ctx context.Context, r *run, sig []float64) error {
	ctx = chat.WithModel(ctx, x.g.model)
	if x.p == nil || x.at != r.size {
		r.phase(PhaseOpen)
	}
	t0 := time.Now()
	p, err := x.open(ctx, r.size)
	if err != nil {
		return err
	}
	r.opened(t0)
	if p.Dump, err = r.dumpDir(); err != nil {
		return err
	}
	for i, prompt := range r.prompts {
		r.phase(PhasePrompt)
		t := time.Now()
		cond, err := p.Encode(ctx, prompt)
		if err != nil {
			return err
		}
		r.out.Prompts += time.Since(t)
		xs := r.batch(i, i+1, len(sig)-1, p.Noise)
		lat, err := p.Generate(ctx, cond, xs[0], sig, r.stepsOf(sig))
		if err != nil {
			return err
		}
		r.latents(i, lat)
		r.phase(PhaseDecode)
		t = time.Now()
		img, err := p.Decode(ctx, cond, lat, r.stage)
		if err != nil {
			return err
		}
		if err := r.drawn(i, time.Since(t), img); err != nil {
			return err
		}
	}
	return nil
}

func (x *flux3) close() error {
	if x.p == nil {
		return nil
	}
	err := x.p.Close()
	x.p = nil
	return err
}
