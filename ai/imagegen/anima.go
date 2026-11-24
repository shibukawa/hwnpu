package imagegen

import (
	"cmp"
	"context"
	"fmt"
	"image"
	"path/filepath"
	"strconv"
	"time"

	"github.com/shibukawa/bonsaiai"
	"github.com/shibukawa/bonsaiai/internal/airole"
	"github.com/shibukawa/bonsaiai/internal/chat"
	"github.com/shibukawa/bonsaiai/internal/imagegen/anima"
	"github.com/shibukawa/bonsaiai/internal/imagegen/aspect"
	"github.com/shibukawa/bonsaiai/internal/llm"
	"github.com/shibukawa/bonsaiai/internal/llm/iron"
)

// Anima (circlestone-labs/Anima): Qwen3-0.6B and Anima's LLM adapter read the prompt, the
// Cosmos-Predict2 2B DiT denoises the latents in flow-matching steps and the Qwen-Image VAE
// decodes them. AMD's NPU runs the export of tools/amd/iron_anima.py (NAME.amd-npu2 of the eight
// aspect ratios, here or in the sources); Apple's Neural Engine and Intel's NPU an export
// converted from the checkpoint (Core AI's of the eight aspect ratios; OpenVINO IR of the size of
// the setting size); a model of device cpu a Core ML export on the CPU interpreter.

// animaTool is the script that exports Anima for AMD's NPU.
const animaTool = "tools/amd/iron_anima.py"

// turboNegativeGuidance is the guidance scale a Negative without Guidance gets on a model that
// runs none by default (Anima-Turbo): weak, as a model distilled to run without guidance burns
// at the 4 the others take.
const turboNegativeGuidance = 1.5

// animaOptions are the conversion's options the model's settings give, for backend.
func animaOptions(m chat.Model, backend string) (anima.ConvertOptions, error) {
	w, h, err := anima.ParseAspect(setting(m, SettingSize, "1:1"))
	if err != nil {
		return anima.ConvertOptions{}, err
	}
	o := anima.ConvertOptions{Backend: backend, Width: w, Height: h, Dit: setting(m, SettingDit, m.File),
		TextFormat: m.Settings[SettingTextFormat], DitFormat: m.Settings[SettingDitFormat]}
	for name, p := range map[string]*int{SettingGroup: &o.Group, SettingQChunk: &o.QChunk} {
		if v := m.Settings[name]; v != "" {
			if *p, err = strconv.Atoi(v); err != nil {
				return o, fmt.Errorf("ai/imagegen: %s's setting %s: %w", m.Name, name, err)
			}
		}
	}
	if backend == llm.BackendCoreAI {
		// every aspect ratio in one export, the weights once (new-model-checklist.md 11.4), what
		// the settings leave out from the model's recipe
		o.Sizes = aspect.SizesOf(aspect.All)
		return o, anima.FromRecipe(m.Name, &o)
	}
	if o.QChunk == 0 {
		o.QChunk = 512 // M3: 25 % faster than 1024 at 1024 x 1024
	}
	return o, nil
}

func prepareAnima(e airole.PrepareEnv) (f airole.Files, err error) {
	m := e.Model
	if dir := first(m.Programs[ProgramExport]); dir != "" {
		f.Set("export", dir)
		return f, nil
	}
	cpu := e.NPU.CPU(m)
	if e.NPU.AMD() && !cpu {
		// the export of every aspect ratio, by the name of the model of the definitions
		dir, err := aspectsAMD(e.Ctx, e.Base, 1024, 1024, animaTool, anima.IronSizes, e.Log)
		if err != nil {
			return f, err
		}
		f.Set("export", dir)
		return f, nil
	}
	backend := e.NPU.Format
	if cpu {
		backend = llm.BackendCoreML // the CPU interpreter runs Core ML exports
	}
	if backend == "" {
		return f, fmt.Errorf("ai/imagegen: no NPU here converts %s (Apple's Neural Engine, Intel's NPU)", m.Name)
	}
	o, err := animaOptions(m, backend)
	if err != nil {
		return f, err
	}
	dir, err := anima.Prepare(e.Ctx, anima.Setup{Dir: filepath.Join(chat.ModelsDir(), "Anima"), ConvertOptions: o, Log: e.Log})
	if err != nil {
		return f, err
	}
	f.Set("export", dir)
	return f, nil
}

// animaGen draws with Anima.
type animaGen struct {
	g        *Generator
	dir      string
	resident bool
	sized    bool // an export of several sizes, opened at a size each
	p        *anima.Pipeline
	at       image.Point
}

func loadAnima(ctx context.Context, g *Generator, f airole.Files, e airole.LoadEnv) (family, error) {
	a := &animaGen{g: g, dir: f.Path("export"), resident: e.Def.Settings[SettingResidentDiT] == "true"}
	g.parse = func(spec string) (int, int, error) { return anima.ParseAspect(spec) }
	programs := anima.IsProgramExport(a.dir)
	var sizes [][2]int
	g.amd = !programs
	if programs {
		man, err := anima.ReadProgramManifest(a.dir)
		if err != nil {
			return nil, err
		}
		sizes = man.SizeList()
		a.sized = len(sizes) > 1
		if !e.NPU.CPU(e.Def) {
			var opts []bonsaiai.Option
			if e.Cache != "" {
				opts = append(opts, bonsaiai.WithCacheDir(e.Cache))
			}
			if llm.AppleFormat(man.Backend) {
				// on Core AI (an older Core ML export is lowered to Core AI as it loads). One bonsaiai
				// process at a time uses the Neural Engine, as the other models' devices are opened
				// (llm.OpenDevice): beside a chat a stage of the VAE failed
				if err := llm.LockNPU(ctx); err != nil {
					return nil, err
				}
				opts = append(opts, bonsaiai.WithBackend(exportDevice(man.Backend, e.NPU)))
			} else if e.NPU.Device != "" {
				opts = append(opts, bonsaiai.WithBackend(e.NPU.Device))
			}
			if g.dev, err = bonsaiai.Open(ctx, opts...); err != nil {
				return nil, err
			}
		}
	} else {
		sizes = anima.IronSizes(a.dir)
		a.sized = len(sizes) > 1
		var opts []bonsaiai.Option
		if e.Cache != "" {
			opts = append(opts, bonsaiai.WithCacheDir(e.Cache))
		}
		var err error
		if g.dev, err = openAMD(ctx, opts...); err != nil {
			return nil, err
		}
		iron.ShareExport(a.dir, logfOf(e.Log))
	}
	g.sizes = points(sizes)
	if len(g.sizes) == 0 {
		g.sizes = []image.Point{{}}
	}
	p, err := a.open(ctx, g.sizes[0])
	if err != nil {
		return nil, err
	}
	if !a.sized {
		g.sizes = []image.Point{{p.Dit.Width, p.Dit.Height}}
		a.at = g.sizes[0]
	}
	return a, nil
}

func (a *animaGen) open(ctx context.Context, size image.Point) (*anima.Pipeline, error) {
	if a.p != nil && a.at == size {
		return a.p, nil
	}
	if a.p != nil {
		a.p.Close()
		a.p = nil
	}
	w, h := 0, 0
	if a.sized {
		w, h = size.X, size.Y // an export of several sizes draws at the size (one of a size alone at its own)
	}
	anima.ResidentDiT = a.resident
	t0 := time.Now()
	p, err := anima.OpenAt(ctx, a.g.dev, a.dir, w, h, a.g.log)
	if err != nil {
		return nil, err
	}
	p.Logf = logfOf(a.g.log)
	a.g.logf("loaded %s (%dx%d) in %.1f s", filepath.Base(a.dir), p.Dit.Width, p.Dit.Height, time.Since(t0).Seconds())
	a.p, a.at = p, size
	return p, nil
}

func (a *animaGen) warm(ctx context.Context, r *run) error { return a.draw(ctx, r) }

func (a *animaGen) draw(ctx context.Context, r *run) error {
	o := r.o
	if a.p == nil || a.at != r.size {
		r.phase(PhaseOpen)
	}
	t0 := time.Now()
	p, err := a.open(ctx, r.size)
	if err != nil {
		return err
	}
	r.opened(t0)
	anima.ResidentDiT = a.resident // read as the DiT is opened, at the first step
	if p.Dump, err = r.dumpDir(); err != nil {
		return err
	}
	steps, guidance := cmp.Or(o.Steps, p.DefaultSteps()), cmp.Or(o.Guidance, p.DefaultGuidance())
	if o.Negative != "" && guidance == 1 {
		if o.Guidance != 0 {
			return fmt.Errorf("a negative prompt: guidance 1 runs no classifier-free guidance; give a guidance over 1 to use it")
		}
		// Anima-Turbo runs no guidance by default: a negative prompt asks for some, and a weak
		// scale, as a distilled model burns at a strong one
		guidance = turboNegativeGuidance
		a.g.logf("--negative: %s runs no guidance by default, so --guidance %g (give --guidance to change it)", cmp.Or(p.Checkpoint, "Anima-Turbo"), guidance)
	}
	sig := anima.Sigmas(steps, cmp.Or(o.Shift, 3))
	smp := p.DefaultSampling()
	if o.Sampler != "" {
		smp.Solver = o.Sampler
	}
	var uncond []float32
	for i, prompt := range r.prompts {
		r.phase(PhasePrompt)
		t := time.Now()
		cond, err := p.Encode(ctx, prompt)
		if err != nil {
			return err
		}
		if guidance != 1 && uncond == nil {
			if uncond, err = p.EncodeNegative(ctx, o.Negative); err != nil {
				return err
			}
			a.g.logf("classifier-free guidance %g: the DiT twice a step", guidance)
		}
		r.out.Prompts += time.Since(t)
		if i == 0 {
			if len(o.CFGSigmas) > 0 {
				if len(o.CFGSigmas) != 2 || uncond == nil {
					return fmt.Errorf("--cfg-sigmas: give LO,HI, with a --guidance over 1")
				}
				smp.CFGMin, smp.CFGMax = o.CFGSigmas[0], o.CFGSigmas[1]
			}
			if uncond != nil && smp.CFGMax > 0 {
				n := 0
				for _, sg := range sig[:steps] {
					if sg >= smp.CFGMin && sg <= smp.CFGMax {
						n++
					}
				}
				a.g.logf("%s, guidance at %d of %d steps (sigma %g to %g)", cmp.Or(smp.Solver, "euler"), n, steps, smp.CFGMin, smp.CFGMax)
			}
		}
		xs := r.batch(i, i+1, steps, p.Noise)
		step := r.stepsOf(sig)
		x, err := p.Denoise(ctx, cond, uncond, guidance, xs[0], sig, smp, func(i int, d time.Duration) { step(i, steps, d) })
		if err != nil {
			return err
		}
		r.latents(i, x)
		r.phase(PhaseDecode)
		p.Stage = r.stage
		t = time.Now()
		img, err := p.Decode(ctx, x)
		p.Stage = nil
		if err != nil {
			return err
		}
		if err := r.drawn(i, time.Since(t), img); err != nil {
			return err
		}
	}
	return nil
}

func (a *animaGen) close() error {
	if a.p == nil {
		return nil
	}
	err := a.p.Close()
	a.p = nil
	return err
}
