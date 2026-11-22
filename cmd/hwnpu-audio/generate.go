package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/sound"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
)

// Run makes the music or the sound effect, as the model --model names is one or the other.
func (c *GenerateCmd) Run(ctx context.Context) error {
	st := store("")
	sfx, err := c.isEffect(st)
	if err != nil {
		return err
	}
	if sfx {
		return c.runSFX(ctx, st)
	}
	return c.runMusic(ctx, st)
}

// isEffect reports whether --model names a sound effects model rather than one of music; when
// it names neither, the models of both are listed before the error.
func (c *GenerateCmd) isEffect(st ai.Store) (bool, error) {
	cat, err := st.Catalog()
	if err != nil {
		return false, err
	}
	_, _, errM := ai.Normalize(cat, sound.Music(c.Model))
	if errM == nil {
		return false, nil
	}
	_, _, errS := ai.Normalize(cat, sound.Effect(c.Model))
	if errS == nil {
		return true, nil
	}
	none := (*ai.NoModelError)(nil)
	switch {
	case !errors.As(errM, &none):
		return false, errM
	case !errors.As(errS, &none):
		return false, errS
	}
	fmt.Fprintf(os.Stderr, "no model %q; the music and sound effects models of the definitions:\n\n", c.Model)
	chat.List(os.Stderr, chat.KindMusic, chat.KindSFX)
	fmt.Fprintln(os.Stderr)
	return false, fmt.Errorf("no music or sound effects model, family or use %q (hwnpu-audio list shows them)", c.Model)
}

// spinner shows label with a spinner on the log until the function it returns is called with
// the result.
func spinner(lw interface{ Write([]byte) (int, error) }, label string) func(result string) {
	stop := console.Spin(lw, label)
	return func(result string) {
		stop()
		fmt.Fprintf(lw, " %s\n", result)
	}
}

// parseStyles reads the STYLE arguments: texts or WAV files, each with an optional weight after
// @; names says each for the log.
func parseStyles(args []string) (styles []sound.Style, names []string, err error) {
	for _, a := range args {
		text, weight := a, 1.0
		if i := strings.LastIndex(a, "@"); i > 0 {
			if w, err := strconv.ParseFloat(a[i+1:], 64); err == nil {
				if w <= 0 {
					return nil, nil, fmt.Errorf("%q: the weight must be more than 0", a)
				}
				text, weight = a[:i], w
			}
		}
		if strings.HasSuffix(strings.ToLower(text), ".wav") {
			if _, err := os.Stat(text); err == nil {
				raw, err := os.ReadFile(text)
				if err != nil {
					return nil, nil, err
				}
				clip, err := sound.DecodeWAV(raw)
				if err != nil {
					return nil, nil, fmt.Errorf("%s: %w", text, err)
				}
				styles = append(styles, sound.Style{Audio: clip, Weight: weight})
				names = append(names, fmt.Sprintf("%s (%.0f s)", text, clip.Seconds()))
				continue
			}
		}
		styles = append(styles, sound.Style{Text: text, Weight: weight})
		names = append(names, strconv.Quote(text))
	}
	return styles, names, nil
}

// musicUse is the music model --model names, as the flags make it: a model of its own for
// --format and --cpu.
func (c *GenerateCmd) musicUse(st ai.Store) (ai.Use[*sound.MusicGenerator], []ai.Definition, error) {
	use, defs, err := useFor(st, sound.Music(c.Model))
	if err != nil {
		return use, nil, err
	}
	var d ai.Definition
	if c.Format != "" {
		d.Settings = map[string]string{"format": c.Format}
	}
	if c.CPU {
		d.Device = "cpu"
	}
	use, defs = defined(use, defs, d, sound.Music)
	return use, defs, nil
}

// runMusic makes music with Magenta RealTime 2.
func (c *GenerateCmd) runMusic(ctx context.Context, st ai.Store) error {
	if len(c.Args) == 0 && c.Codes == "" && c.Continue == "" {
		return errors.New("music wants a style: texts or WAV files, --codes or --continue")
	}
	drums := map[string]sound.Drums{"auto": sound.DrumsAuto, "on": sound.DrumsOn, "off": sound.DrumsOff}
	o := sound.MusicOptions{Onsets: c.Onsets, NoteWidth: c.NoteWidth, NewOnly: c.NewOnly, Temperature: c.Temperature, TopK: c.TopK,
		Guidance: &sound.Guidance{Style: c.CFGStyle, Notes: c.CFGNotes, Drums: c.CFGDrums}, Fade: 2 * time.Second, Seed: c.Seed}
	drum, ok := drums[c.Drums]
	if !ok {
		return fmt.Errorf("--drums %q: auto, on or off", c.Drums)
	}
	o.Drums = drum
	if c.Duration != nil {
		if *c.Duration <= 0 {
			return fmt.Errorf("--duration %v: more than 0", *c.Duration)
		}
		o.Duration = *c.Duration
	}
	if c.Fade != nil {
		o.Fade = *c.Fade
	}
	var styles []sound.Style
	var names []string
	var err error
	switch {
	case c.Codes != "":
		for _, s := range strings.Split(c.Codes, ",") {
			code, err := strconv.Atoi(strings.TrimSpace(s))
			if err != nil || code < 0 || code >= 1024 {
				return fmt.Errorf("--codes %q: 12 codes from 0 to 1023", c.Codes)
			}
			o.Codes = append(o.Codes, code)
		}
		if len(o.Codes) != sound.StyleCodes {
			return fmt.Errorf("--codes has %d codes, not %d", len(o.Codes), sound.StyleCodes)
		}
	default:
		if styles, names, err = parseStyles(c.Args); err != nil {
			return err
		}
	}
	lw := console.Log(os.Stderr)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	if c.Notes != "" {
		raw, err := os.ReadFile(c.Notes)
		if err != nil {
			return err
		}
		if o.Notes, err = sound.ReadMIDI(raw); err != nil {
			return fmt.Errorf("%s: %w", c.Notes, err)
		}
		end := 0.0
		for _, n := range o.Notes {
			end = math.Max(end, n.End)
		}
		logf("notes: %d of %s (%.1f s)", len(o.Notes), c.Notes, end)
	}
	if c.Continue != "" {
		raw, err := os.ReadFile(c.Continue)
		if err != nil {
			return err
		}
		if o.Continue, err = sound.DecodeWAV(raw); err != nil {
			return fmt.Errorf("%s: %w", c.Continue, err)
		}
	}

	use, defs, err := c.musicUse(st)
	if err != nil {
		return err
	}
	st.Definitions = defs
	app := ai.NewApp("hwnpu-audio", use)
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
	console.Preparing()
	t0 := time.Now()
	done := spinner(lw, "loading "+use.Name())
	g, err := use.Get(ctx, rt)
	if err != nil {
		done("failed")
		return err
	}
	done(fmt.Sprintf("in %s, %s", console.Duration(time.Since(t0)), g.Where()))
	console.Running()

	// the style
	if len(styles) > 0 {
		done := spinner(lw, "reading the style")
		res, err := g.Style(ctx, styles...)
		if err != nil {
			done("failed")
			return err
		}
		done(fmt.Sprintf("in %s", console.Duration(res.Took)))
		logf("style %s: %v (--codes takes them)", strings.Join(names, " + "), res.Codes)
		o.Codes = res.Codes
	}

	// the output, a file and the speakers: the clip first, unless --new-only
	var out sink
	if c.Out != "" {
		if out.w, err = newWAVWriter(c.Out); err != nil {
			return err
		}
	}
	if c.Play || c.Out == "" {
		if out.sp, err = newSpeaker(); err != nil {
			return fmt.Errorf("the speakers: %w", err)
		}
	}
	var bar *console.Bar
	var clipDone func(string)
	var stats *sound.MusicStats // Generate's, which the progress tells of
	o.Progress = func(p sound.MusicProgress) {
		switch p.Stage {
		case sound.MusicReadingClip:
			clipDone = spinner(lw, fmt.Sprintf("reading the last %.1f s of %s", p.ClipSeconds, c.Continue))
		case sound.MusicClipRead:
			clipDone(fmt.Sprintf("%d frames in %s", stats.ClipFrames, console.Duration(stats.Clip)))
			clipDone = nil
		case sound.MusicGenerating:
			if bar == nil {
				bar = console.NewBar(lw, "generating")
			}
			at, of := time.Duration(p.Frames)*time.Second/sound.MusicFrameRate, time.Duration(p.Of)*time.Second/sound.MusicFrameRate
			bar.Set(float64(p.Frames)/float64(p.Of), fmt.Sprintf("%s of %s, %.1f times real time", console.Duration(at), console.Duration(of), at.Seconds()/p.Elapsed.Seconds()))
		}
	}
	music, stats := g.Generate(ctx, o)
	logf("seed %d", stats.Seed)
	for ch, err := range music {
		if err != nil {
			if bar != nil {
				bar.Done("failed")
			}
			if clipDone != nil {
				clipDone("failed")
			}
			return err
		}
		if err := out.write(ch.Data); err != nil {
			return err
		}
	}
	if bar == nil {
		bar = console.NewBar(lw, "generating")
	}
	bar.Done(fmt.Sprintf("%s in %s (%.1f times real time; the decoder %s of it)", console.Duration(time.Duration(stats.Frames)*time.Second/sound.MusicFrameRate),
		console.Duration(stats.Took), stats.RealTime(), console.Duration(stats.Decode)))
	if err := out.finish(ctx, lw); err != nil {
		return err
	}
	if out.w != nil {
		fmt.Fprintf(os.Stderr, "%s: %.1f s of 48 kHz stereo\n", c.Out, float64(out.w.frames)/sound.MusicRate)
	}
	logf("the weights are CC-BY 4.0: credit \"Magenta RealTime 2 by Google DeepMind\" where you use the music")
	return nil
}

// sink is where the sound goes: a WAV file, the speakers, or both.
type sink struct {
	w  *wavWriter
	sp *speaker
}

func (s *sink) write(pcm []float32) error {
	if s.sp != nil {
		s.sp.write(pcm)
	}
	if s.w != nil {
		return s.w.write(pcm)
	}
	return nil
}

// finish plays the rest and closes the WAV.
func (s *sink) finish(ctx context.Context, lw interface{ Write([]byte) (int, error) }) error {
	if s.sp != nil {
		done := spinner(lw, "playing the rest")
		if err := s.sp.finish(ctx); err != nil {
			done("stopped")
			return err
		}
		done("done")
	}
	if s.w != nil {
		return s.w.close()
	}
	return nil
}
