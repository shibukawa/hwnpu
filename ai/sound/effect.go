package sound

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/sfx"
)

// The sound effects' audio: 48 kHz mono, up to EffectMaxSeconds.
const (
	EffectRate       = sfx.SampleRate
	EffectMaxSeconds = sfx.MaxSeconds
	// EffectPromptTokens are the tokens of a description the text encoder reads; the rest is
	// cut.
	EffectPromptTokens = sfx.PromptRows
)

// Effect is the use of a model that makes sound effects from a description (MOSS-SoundEffect)
// for an app: a model, a family or a use of the definitions ("sfx", "moss-soundeffect:8gb").
func Effect(name string) ai.Use[*EffectGenerator] {
	return ai.Define[*EffectGenerator](effectRole, name)
}

// The settings a sound effects model's definition may give (ai.Definition):
//
//	settings: text-format  the text encoder's weight format, fp16 or lut8 (default: by the
//	                       model's machine, fp16 from 16 GB)
//	          dit-format   the DiT's weight format, lut8 (the default) or fp16
//
// A model of device cpu runs on hwnpu's CPU interpreter (slow). On AMD's NPU the model runs as
// the graphs of tools/amd/iron_moss_sfx.py (exported on the first preparation; dit-format does
// not change them).
var effectRole = &airole.Role{
	Remote: func(h *airpc.Handle) io.Closer { return &EffectGenerator{remote: h} },
	Name:   "sfx",
	Kinds:  []chat.Kind{chat.KindSFX},
	Prepare: func(e airole.PrepareEnv) (f airole.Files, err error) {
		cpu := e.NPU.CPU(e.Model)
		var p sfx.Paths
		if e.NPU.AMD() && !cpu {
			// the export of the model of the definitions (moss-soundeffect:8gb's own), its text
			// encoder in the format asked for
			text, _ := sfx.Formats(cmp.Or(e.Base.Machine, chat.MachineMemory()))
			if p, err = sfx.PrepareAMD(chat.WithModel(e.Ctx, e.Base), e.Base, cmp.Or(e.Model.Settings["text-format"], text), e.Log); err != nil {
				return f, err
			}
		} else {
			backend := llm.BackendCoreML // what the CPU interpreter runs
			if !cpu {
				if backend, err = programFormat(e.NPU); err != nil {
					return f, err
				}
			}
			if p, err = sfx.Prepare(e.Ctx, sfx.Setup{Dir: e.Dir, Repo: e.Model.Source(), Machine: e.Base.Machine,
				TextFormat: e.Model.Settings["text-format"], DitFormat: e.Model.Settings["dit-format"], Backend: backend, Log: e.Log}); err != nil {
				return f, err
			}
		}
		f.Set("dir", p.Dir)
		f.Set("text", p.Text...)
		f.Set("context", p.Context)
		f.Set("blocks", p.Blocks...)
		f.Set("stages", p.Stages...)
		f.Set("tables", p.Tables)
		f.Set("host-text", p.Host[""]...)
		f.Set("host-dit", p.Host["dit."]...)
		if p.AMD {
			f.SetValue("amd", "true")
		}
		return f, nil
	},
	Load: func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
		p := sfx.Paths{Dir: f.Path("dir"), Text: f.Paths["text"], Context: f.Path("context"), Blocks: f.Paths["blocks"],
			Stages: f.Paths["stages"], Tables: f.Path("tables"), AMD: f.Values["amd"] == "true"}
		if t, d := f.Paths["host-text"], f.Paths["host-dit"]; len(t) > 0 || len(d) > 0 {
			p.Host = map[string][]string{"": t, "dit.": d}
		}
		cfg, err := sfx.TextConfig(p.Dir)
		if err != nil {
			return nil, err
		}
		p.TextCfg = cfg
		g := &EffectGenerator{p: p, model: e.Model, cpu: e.NPU.CPU(e.Def), allow: e.AllowOffNPU, log: e.Log}
		switch {
		case g.cpu:
		case p.AMD:
			g.dev, err = hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
		default:
			g.dev, err = openNPU(ctx, e.NPU, e.Cache)
		}
		if err != nil {
			return nil, err
		}
		if g.g, err = sfx.Open(p, g.dev); err != nil {
			g.Close()
			return nil, err
		}
		return g, nil
	},
}

// EffectGenerator makes sound effects (MOSS-SoundEffect): the text encoder reads the
// description, the DiT denoises 30 seconds of latents, and the decoder turns the duration
// asked for into 48 kHz mono. The text encoder's programs are loaded for each description and
// let go before the DiT's, which are loaded at the first sound and kept. It runs one request at
// a time; others wait.
type EffectGenerator struct {
	remote *airpc.Handle // the model in hwnpud (ai's HWNPU_DAEMON), nil when it is here
	mu     sync.Mutex
	g      *sfx.Generator
	p      sfx.Paths
	dev    *hwnpu.Device
	loaded bool // the DiT's and the decoder's programs
	model  string
	// cpu: the model's device is cpu; allow: programs placed off the NPU are run
	cpu, allow bool
	log        io.Writer
}

// Model is the name of the model in the definitions.
func (g *EffectGenerator) Model() string {
	if g.remote != nil {
		return airpc.Value[string](g.remote, "Model")
	}
	return g.model
}

// OnCPU reports whether the model runs on hwnpu's CPU interpreter (its definition's device cpu).
func (g *EffectGenerator) OnCPU() bool {
	if g.remote != nil {
		return airpc.Value[bool](g.remote, "OnCPU")
	}
	return g.cpu
}

// Programs names the weight formats of the text encoder's and the DiT's programs, for a log.
func (g *EffectGenerator) Programs() string {
	if g.remote != nil {
		return airpc.Value[string](g.remote, "Programs")
	}
	return g.p.String()
}

// EffectOptions are a sound effect's settings.
type EffectOptions struct {
	// Duration is how long the sound lasts, to a tenth of a second, up to EffectMaxSeconds (0:
	// 10 seconds).
	Duration time.Duration
	// Steps are the flow-matching steps (0: 100; each two calls of the DiT: fewer is faster and
	// rougher), CFG how strongly the description guides (0: 4) and Shift the sigma schedule's
	// shift (0: 5).
	Steps      int
	CFG, Shift float64
	// Seed makes the sound repeatable (0: a new one, told in EffectStats.Seed).
	Seed uint64
	// Fade fades the last of the sound out (0: none; a sound effect may end where it should).
	Fade time.Duration
	// Progress, when not nil, is told what is being done, in the call's goroutine.
	Progress func(EffectProgress)
}

// EffectStage is what making a sound effect is doing.
type EffectStage int

const (
	EffectReading EffectStage = iota // the text encoder reads the description
	EffectRead                       // it has (Tokens, Took, Where)
	EffectLoading                    // the DiT's and the decoder's programs are loaded (the first sound)
	EffectLoaded                     // they are (Took, Where)
	EffectStep                       // a step of the DiT is done (Step of Steps, Took the step's)
)

// EffectProgress is how far making a sound effect is.
type EffectProgress struct {
	Stage       EffectStage
	Tokens      int
	Step, Steps int
	Took        time.Duration
	// Where says where the programs just loaded run: "all on the NPU", "NOT all on the NPU",
	// "placement not known" or "on the CPU interpreter".
	Where string
}

// EffectSound is a sound effect made.
type EffectSound struct {
	Samples []float32 // mono, in [-1, 1]
	Rate    int
	Stats   EffectStats
}

// EffectStats are what making a sound effect took.
type EffectStats struct {
	// Seed is the seed of the noise (the one drawn when EffectOptions had none: given again, it
	// makes the same sound).
	Seed uint64
	// Seconds is the sound's length; Steps the DiT's steps.
	Seconds float64
	Steps   int
	// Tokens are the description's tokens, Cut true when it was cut at EffectPromptTokens.
	Tokens int
	Cut    bool
	// Text is the time the text encoder took (its programs' loads with it), Load the DiT's and
	// the decoder's loads (0 after the first sound), Took the generation's.
	Text, Load, Took time.Duration
}

// placement is what loading programs says of where they run.
type placement struct {
	cpu          bool
	dev          *hwnpu.Device
	off, unknown []string
}

// note is sfx's loaded: what Core ML puts off the NPU, the programs it says run partly
// elsewhere (off) and those it could not tell about (unknown: MLComputePlan fails or times out
// while the Neural Engine is busy compiling, which is not a placement off the NPU).
func (pl *placement) note(name string, pr runner.Program, _ time.Duration) {
	if pl.dev == nil || pr.OnNPU() {
		return
	}
	p, ok := pr.(interface {
		Placement() (known bool, ops []string, reason string)
	})
	if !ok {
		pl.off = append(pl.off, name)
		return
	}
	known, ops, reason := p.Placement()
	switch {
	case !known:
		pl.unknown = append(pl.unknown, fmt.Sprintf("%s (%s)", name, reason))
	default:
		pl.off = append(pl.off, fmt.Sprintf("%s: %s", name, strings.Join(ops, ", ")))
	}
}

func (pl *placement) where() string {
	switch {
	case pl.cpu:
		return "on the CPU interpreter"
	case len(pl.off) > 0:
		return "NOT all on the NPU"
	case len(pl.unknown) > 0:
		return "placement not known"
	}
	return "all on the NPU"
}

// check logs the programs whose placement is not known and refuses those off the NPU unless
// allowed; it starts again for the next programs.
func (g *EffectGenerator) check(pl *placement) error {
	for _, u := range pl.unknown {
		logf(g.log)("  placement not known, going on: %s", u)
	}
	off := pl.off
	pl.off, pl.unknown = nil, nil
	return checkNPU(g.log, g.allow, off)
}

// Generate makes the sound of a description (English or Chinese).
func (g *EffectGenerator) Generate(ctx context.Context, description string, o EffectOptions) (EffectSound, error) {
	if g.remote != nil {
		return airpc.Call1[EffectSound](g.remote, ctx, "Generate", description, o)
	}
	prompt := strings.TrimSpace(description)
	if prompt == "" {
		return EffectSound{}, errors.New("sound: a description of the sound is needed")
	}
	d := cmp.Or(o.Duration, 10*time.Second)
	seconds := math.Round(d.Seconds()*10) / 10
	if seconds <= 0 || seconds > sfx.MaxSeconds {
		return EffectSound{}, fmt.Errorf("sound: a duration of %v: more than 0, up to %ds", d, sfx.MaxSeconds)
	}
	if o.Steps < 0 {
		return EffectSound{}, fmt.Errorf("sound: %d steps", o.Steps)
	}
	steps := cmp.Or(o.Steps, sfx.DefaultSteps)
	tell := func(p EffectProgress) {
		if o.Progress != nil {
			o.Progress(p)
		}
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	pl := &placement{cpu: g.cpu, dev: g.dev}
	st := EffectStats{Seconds: seconds, Steps: steps}
	// the description first: the text encoder's programs are let go before the DiT's are loaded
	tell(EffectProgress{Stage: EffectReading})
	t0 := time.Now()
	text, ids, err := g.g.Encode(ctx, prompt, seconds, pl.note)
	if err != nil {
		return EffectSound{}, err
	}
	st.Text, st.Tokens, st.Cut = time.Since(t0), len(ids), len(ids) >= sfx.PromptRows
	tell(EffectProgress{Stage: EffectRead, Tokens: len(ids), Took: st.Text, Where: pl.where()})
	if err := g.check(pl); err != nil {
		return EffectSound{}, err
	}
	if !g.loaded {
		tell(EffectProgress{Stage: EffectLoading})
		t0 = time.Now()
		if err := g.g.Load(ctx, pl.note); err != nil {
			return EffectSound{}, err
		}
		st.Load = time.Since(t0)
		tell(EffectProgress{Stage: EffectLoaded, Took: st.Load, Where: pl.where()})
		if err := g.check(pl); err != nil {
			return EffectSound{}, err
		}
		g.loaded = true
	}
	st.Seed = o.Seed
	if st.Seed == 0 {
		st.Seed = rand.Uint64()
	}
	t0 = time.Now()
	samples, err := g.g.Generate(ctx, text, seconds, sfx.Options{Steps: steps, CFG: o.CFG, Shift: o.Shift, Seed: st.Seed,
		Step: func(i, n int, took time.Duration) {
			tell(EffectProgress{Stage: EffectStep, Step: i + 1, Steps: n, Took: took})
		}})
	if err != nil {
		return EffectSound{}, err
	}
	st.Took = time.Since(t0)
	if o.Fade > 0 {
		fadeOut(samples, 1, int(o.Fade.Seconds()*sfx.SampleRate))
	}
	return EffectSound{Samples: samples, Rate: sfx.SampleRate, Stats: st}, nil
}

// Warm loads the DiT's and the decoder's programs and reads a description once, which compiles
// them all for the NPU (ai.Warmer: the load of "models prepare").
func (g *EffectGenerator) Warm(ctx context.Context) error {
	if g.remote != nil {
		return airpc.Call0(g.remote, ctx, "Warm")
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if _, _, err := g.g.Encode(ctx, "rain", 1, nil); err != nil {
		return err
	}
	if g.loaded {
		return nil
	}
	if err := g.g.Load(ctx, nil); err != nil {
		return err
	}
	g.loaded = true
	return nil
}

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (g *EffectGenerator) Close() error {
	if g.remote != nil {
		return g.remote.Release() // the Runtime in hwnpud unloads it
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	var err error
	if g.g != nil {
		err = g.g.Close()
		g.g = nil
	}
	if g.dev != nil {
		g.dev.Close()
		g.dev = nil
	}
	return err
}

func init() {
	airole.Register(effectRole)
	airpc.RegisterErrorType("sound.OffNPUError", (*OffNPUError)(nil))
}
