package main

import (
	"context"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/shibukawa/bonsaiai/ai"
	"github.com/shibukawa/bonsaiai/ai/sound"
	"github.com/shibukawa/bonsaiai/internal/console"
)

// effectUse is the sound effects model --model names, as the flags make it: a model of its own
// for --text-format, --dit-format and --cpu.
func (c *GenerateCmd) effectUse(st ai.Store) (ai.Use[*sound.EffectGenerator], []ai.Definition, error) {
	use, defs, err := useFor(st, sound.Effect(c.Model))
	if err != nil {
		return use, nil, err
	}
	var d ai.Definition
	for k, v := range map[string]string{"text-format": c.TextFormat, "dit-format": c.DitFormat} {
		if v != "" {
			if d.Settings == nil {
				d.Settings = map[string]string{}
			}
			d.Settings[k] = v
		}
	}
	if c.CPU {
		d.Device = "cpu"
	}
	use, defs = defined(use, defs, d, sound.Effect)
	return use, defs, nil
}

// runSFX makes a sound effect from the description in the arguments with a model of kind sfx
// (MOSS-SoundEffect v2.0), written and played as stereo. The model's machine
// (moss-soundeffect:8gb, :16gb, or the one of this machine) picks the text encoder's format.
func (c *GenerateCmd) runSFX(ctx context.Context, st ai.Store) error {
	prompt := strings.TrimSpace(strings.Join(c.Args, " "))
	switch {
	case prompt == "":
		return fmt.Errorf("%s: a description of the sound is needed", c.Model)
	case c.Codes != "" || c.Continue != "" || c.Notes != "":
		return fmt.Errorf("%s makes sound effects: --codes, --continue and --notes are for music", c.Model)
	}
	o := sound.EffectOptions{Duration: 10 * time.Second, Steps: c.Steps, CFG: c.CFG, Shift: c.Shift, Seed: c.Seed}
	if c.Duration != nil {
		o.Duration = *c.Duration
	}
	if s := math.Round(o.Duration.Seconds()*10) / 10; s <= 0 || s > sound.EffectMaxSeconds {
		return fmt.Errorf("--duration %v: more than 0, up to %ds", o.Duration, sound.EffectMaxSeconds)
	}
	if c.Steps <= 0 {
		return fmt.Errorf("--steps %d: at least 1", c.Steps)
	}
	// faded out only when --fade is given, as a sound effect may end where it should
	if c.Fade != nil {
		o.Fade = *c.Fade
	}
	lw := console.Log(os.Stderr)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	use, defs, err := c.effectUse(st)
	if err != nil {
		return err
	}
	st.Definitions = defs
	app := ai.NewApp("bonsaiai-audio", use)
	app.Store = st
	var opts []ai.Option
	if c.OffNPU {
		opts = append(opts, ai.AllowOffNPU())
	}
	rt, err := open(ctx, app, lw, opts...)
	if err != nil {
		return err
	}
	defer rt.Close()
	g, err := use.Get(ctx, rt)
	if err != nil {
		return err
	}
	var done func(string)
	var bar *console.Bar
	var t0 time.Time
	o.Progress = func(p sound.EffectProgress) {
		switch p.Stage {
		case sound.EffectReading:
			console.Running()
			done = spinner(lw, "reading the description")
		case sound.EffectRead:
			done(fmt.Sprintf("%d tokens in %s, %s", p.Tokens, console.Duration(p.Took), p.Where))
			done = nil
			if p.Tokens >= sound.EffectPromptTokens {
				logf("  the description is cut at %d tokens", sound.EffectPromptTokens)
			}
		case sound.EffectLoading:
			console.Preparing()
			done = spinner(lw, fmt.Sprintf("loading %s (%s)", use.Name(), g.Programs()))
		case sound.EffectLoaded:
			done(fmt.Sprintf("in %s, %s", console.Duration(p.Took), p.Where))
			done = nil
		case sound.EffectStep:
			if bar == nil {
				console.Running()
				bar, t0 = console.NewBar(lw, "generating"), time.Now()
			}
			left := time.Duration(float64(time.Since(t0)) / float64(p.Step) * float64(p.Steps-p.Step))
			bar.Set(float64(p.Step)/float64(p.Steps), fmt.Sprintf("step %d of %d, %s a step, %s left", p.Step, p.Steps, console.Duration(p.Took), console.Duration(left)))
		}
	}
	s, err := g.Generate(ctx, prompt, o)
	if err != nil {
		switch {
		case bar != nil:
			bar.Done("failed")
		case done != nil:
			done("failed")
		}
		return err
	}
	t := s.Stats
	logf("seed %d", t.Seed)
	if bar == nil {
		bar = console.NewBar(lw, "generating")
	}
	bar.Done(fmt.Sprintf("%.1f s of sound in %s (%d steps, %s a step)", t.Seconds, console.Duration(t.Took), t.Steps, console.Duration(t.Took/time.Duration(t.Steps))))

	// the same in both channels, at full gain (the music's WAV is at 0.5)
	stereo := make([]float32, 2*len(s.Samples))
	for i, v := range s.Samples {
		stereo[2*i], stereo[2*i+1] = v, v
	}
	var out sink
	if c.Out != "" {
		if out.w, err = newWAVWriter(c.Out); err != nil {
			return err
		}
		out.w.gain = 1
	}
	if c.Play || c.Out == "" {
		if out.sp, err = newSpeaker(); err != nil {
			return fmt.Errorf("the speakers: %w", err)
		}
		out.sp.gain = 1
	}
	if err := out.write(stereo); err != nil {
		return err
	}
	if err := out.finish(ctx, lw); err != nil {
		return err
	}
	if out.w != nil && c.Out != "-" {
		fmt.Fprintf(os.Stderr, "%s: %.1f s of 48 kHz (mono in both channels)\n", c.Out, t.Seconds)
	}
	return nil
}
