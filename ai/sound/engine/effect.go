package engine

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
	"github.com/shibukawa/hwnpu/ai/sound"
	"github.com/shibukawa/hwnpu/ai/sound/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/sfx"
)

// prepareEffect prepares a sound effects model (sound.Effect's settings).
func prepareEffect(e airole.PrepareEnv) (f airole.Files, err error) {
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
			TextFormat: e.Model.Settings["text-format"], DitFormat: e.Model.Settings["dit-format"], Backend: backend, Model: e.Base.Name, Log: e.Log}); err != nil {
			return f, err
		}
	}
	f.Set("dir", p.Dir)
	f.Set("text", p.Text...)
	f.Set("context", p.Context)
	f.Set("blocks", p.Blocks...)
	f.Set("stages", p.Stages...)
	f.Set("tables", p.Tables)
	f.Set("tokenizer", p.Tokenizer)
	f.Set("host-text", p.Host[""]...)
	f.Set("host-dit", p.Host["dit."]...)
	if p.AMD {
		f.SetValue("amd", "true")
	}
	return f, nil
}

func loadEffect(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
	p := sfx.Paths{Dir: f.Path("dir"), Text: f.Paths["text"], Context: f.Path("context"), Blocks: f.Paths["blocks"],
		Stages: f.Paths["stages"], Tables: f.Path("tables"), Tokenizer: f.Path("tokenizer"), AMD: f.Values["amd"] == "true"}
	if t, d := f.Paths["host-text"], f.Paths["host-dit"]; len(t) > 0 || len(d) > 0 {
		p.Host = map[string][]string{"": t, "dit.": d}
	}
	cfg, err := sfx.TextConfig(p.Dir)
	if err != nil {
		return nil, err
	}
	p.TextCfg = cfg
	g := &effectGenerator{p: p, model: e.Model, cpu: e.NPU.CPU(e.Def), allow: e.AllowOffNPU, log: e.Log}
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
	return bridge.Wrap["sfx"](g), nil
}

// effectGenerator is a sound effects model of the engine (sound.EffectGenerator's). It runs one
// request at a time; others wait.
type effectGenerator struct {
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

func (g *effectGenerator) Model() string    { return g.model }
func (g *effectGenerator) OnCPU() bool      { return g.cpu }
func (g *effectGenerator) Programs() string { return g.p.String() }

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
func (g *effectGenerator) check(pl *placement) error {
	for _, u := range pl.unknown {
		logf(g.log)("  placement not known, going on: %s", u)
	}
	off := pl.off
	pl.off, pl.unknown = nil, nil
	return checkNPU(g.log, g.allow, off)
}

func (g *effectGenerator) Generate(ctx context.Context, description string, o sound.EffectOptions) (sound.EffectSound, error) {
	prompt := strings.TrimSpace(description)
	if prompt == "" {
		return sound.EffectSound{}, errors.New("sound: a description of the sound is needed")
	}
	d := cmp.Or(o.Duration, 10*time.Second)
	seconds := math.Round(d.Seconds()*10) / 10
	if seconds <= 0 || seconds > sfx.MaxSeconds {
		return sound.EffectSound{}, fmt.Errorf("sound: a duration of %v: more than 0, up to %ds", d, sfx.MaxSeconds)
	}
	if o.Steps < 0 {
		return sound.EffectSound{}, fmt.Errorf("sound: %d steps", o.Steps)
	}
	steps := cmp.Or(o.Steps, sfx.DefaultSteps)
	tell := func(p sound.EffectProgress) {
		if o.Progress != nil {
			o.Progress(p)
		}
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	pl := &placement{cpu: g.cpu, dev: g.dev}
	st := sound.EffectStats{Seconds: seconds, Steps: steps}
	// the description first: the text encoder's programs are let go before the DiT's are loaded
	tell(sound.EffectProgress{Stage: sound.EffectReading})
	t0 := time.Now()
	text, ids, err := g.g.Encode(ctx, prompt, seconds, pl.note)
	if err != nil {
		return sound.EffectSound{}, err
	}
	st.Text, st.Tokens, st.Cut = time.Since(t0), len(ids), len(ids) >= sfx.PromptRows
	tell(sound.EffectProgress{Stage: sound.EffectRead, Tokens: len(ids), Took: st.Text, Where: pl.where()})
	if err := g.check(pl); err != nil {
		return sound.EffectSound{}, err
	}
	if !g.loaded {
		tell(sound.EffectProgress{Stage: sound.EffectLoading})
		t0 = time.Now()
		if err := g.g.Load(ctx, pl.note); err != nil {
			return sound.EffectSound{}, err
		}
		st.Load = time.Since(t0)
		tell(sound.EffectProgress{Stage: sound.EffectLoaded, Took: st.Load, Where: pl.where()})
		if err := g.check(pl); err != nil {
			return sound.EffectSound{}, err
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
			tell(sound.EffectProgress{Stage: sound.EffectStep, Step: i + 1, Steps: n, Took: took})
		}})
	if err != nil {
		return sound.EffectSound{}, err
	}
	st.Took = time.Since(t0)
	if o.Fade > 0 {
		fadeOut(samples, 1, int(o.Fade.Seconds()*sfx.SampleRate))
	}
	return sound.EffectSound{Samples: samples, Rate: sfx.SampleRate, Stats: st}, nil
}

func (g *effectGenerator) Warm(ctx context.Context) error {
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

func (g *effectGenerator) Close() error {
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
