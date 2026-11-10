package main

import (
	"context"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/sfx"
)

// runSFX makes a sound effect from the description in args with a model of kind sfx
// (MOSS-SoundEffect v2.0): the text encoder reads it, the DiT denoises 30 seconds of latents,
// and the decoder turns the duration asked for into 48 kHz mono, written and played as stereo.
// The model's machine (moss-soundeffect:8gb, :16gb, or the one of this machine) picks the text
// encoder's format (sfx.Formats).
func runSFX(ctx context.Context, o options, m chat.Model, args []string, lw io.Writer) error {
	prompt := strings.TrimSpace(strings.Join(args, " "))
	switch {
	case prompt == "":
		return fmt.Errorf("%s: a description of the sound is needed", m.Name)
	case o.codes != "" || o.cont != "" || o.notes != "":
		return fmt.Errorf("%s makes sound effects: --codes, --continue and --notes are for music", m.Name)
	}
	seconds := math.Round(o.duration.Seconds()*10) / 10
	if seconds <= 0 || seconds > sfx.MaxSeconds {
		return fmt.Errorf("--duration %v: more than 0, up to %ds", o.duration, sfx.MaxSeconds)
	}
	if o.steps <= 0 {
		return fmt.Errorf("--steps %d: at least 1", o.steps)
	}
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	spin := func(label string) func(result string) {
		stop := console.Spin(lw, label)
		return func(result string) {
			stop()
			fmt.Fprintf(lw, " %s\n", result)
		}
	}
	var dev *hwnpu.Device
	if !o.cpu {
		d, device, err := llm.OpenDevice(ctx, "", o.cacheDir)
		if err != nil {
			return err
		}
		defer d.Close()
		if backend := llm.ProgramBackend(device); backend != llm.BackendCoreML {
			return fmt.Errorf("%s: only Apple's Neural Engine runs MOSS-SoundEffect so far (--cpu runs it on the CPU)", device)
		}
		dev = d
	}
	p, err := sfx.Prepare(ctx, sfx.Setup{Dir: filepath.Join(o.dir, m.DirName()), Repo: m.Source(), Machine: m.Machine,
		TextFormat: o.textFormat, DitFormat: o.ditFormat, Log: lw})
	if err != nil {
		return err
	}
	g, err := sfx.Open(p, dev)
	if err != nil {
		return err
	}
	defer g.Close()
	// what Core ML puts off the NPU: programs it says run partly elsewhere (off), and those it
	// could not tell about (unknown: MLComputePlan fails or times out while the Neural Engine is
	// busy compiling, which is not a placement off the NPU)
	var off, unknown []string
	note := func(name string, p runner.Program, _ time.Duration) {
		if dev == nil || p.OnNPU() {
			return
		}
		pl, ok := p.(interface {
			Placement() (known bool, ops []string, reason string)
		})
		if !ok {
			off = append(off, name)
			return
		}
		known, ops, reason := pl.Placement()
		switch {
		case !known:
			unknown = append(unknown, fmt.Sprintf("%s (%s)", name, reason))
		default:
			off = append(off, fmt.Sprintf("%s: %s", name, strings.Join(ops, ", ")))
		}
	}
	where := func() string {
		switch {
		case o.cpu:
			return "on the CPU interpreter"
		case len(off) > 0:
			return "NOT all on the NPU"
		case len(unknown) > 0:
			return "placement not known"
		}
		return "all on the NPU"
	}
	report := func() error {
		for _, u := range unknown {
			logf("  placement not known, going on: %s", u)
		}
		unknown = nil
		return checkNPU(lw, o, off)
	}
	// the description first: the text encoder's programs are let go before the DiT's are loaded
	t0 := time.Now()
	done := spin("reading the description")
	text, ids, err := g.Encode(ctx, prompt, seconds, note)
	if err != nil {
		done("failed")
		return err
	}
	done(fmt.Sprintf("%d tokens in %s, %s", len(ids), console.Duration(time.Since(t0)), where()))
	if err := report(); err != nil {
		return err
	}
	if len(ids) >= sfx.PromptRows {
		logf("  the description is cut at %d tokens", sfx.PromptRows)
	}
	t0 = time.Now()
	off = nil
	done = spin(fmt.Sprintf("loading %s (%s)", m.Name, p))
	if err := g.Load(ctx, note); err != nil {
		done("failed")
		return err
	}
	done(fmt.Sprintf("in %s, %s", console.Duration(time.Since(t0)), where()))
	if err := report(); err != nil {
		return err
	}

	seed := o.seed
	if seed == 0 {
		seed = rand.Uint64()
	}
	logf("seed %d", seed)
	bar := console.NewBar(lw, "generating")
	t0 = time.Now()
	samples, err := g.Generate(ctx, text, seconds, sfx.Options{Steps: o.steps, CFG: o.cfg, Shift: o.shift, Seed: seed,
		Step: func(i, n int, took time.Duration) {
			left := time.Duration(float64(time.Since(t0)) / float64(i+1) * float64(n-i-1))
			bar.Set(float64(i+1)/float64(n), fmt.Sprintf("step %d of %d, %s a step, %s left", i+1, n, console.Duration(took), console.Duration(left)))
		}})
	if err != nil {
		bar.Done("failed")
		return err
	}
	bar.Done(fmt.Sprintf("%.1f s of sound in %s (%d steps, %s a step)", seconds, console.Duration(time.Since(t0)), o.steps, console.Duration(time.Since(t0)/time.Duration(o.steps))))

	// the same in both channels, at full gain (the music's WAV is at 0.5); faded out only when
	// -fade is given, as a sound effect may end where it should
	stereo := make([]float32, 2*len(samples))
	for i, v := range samples {
		stereo[2*i], stereo[2*i+1] = v, v
	}
	var out sink
	if o.out != "" {
		if out.w, err = newWAVWriter(o.out); err != nil {
			return err
		}
		out.w.gain = 1
	}
	if o.play {
		if out.sp, err = newSpeaker(); err != nil {
			return fmt.Errorf("the speakers: %w", err)
		}
		out.sp.gain = 1
	}
	if o.fadeSet {
		out.fade = 2 * int(o.fade.Seconds()*sfx.SampleRate)
	}
	if err := out.write(stereo); err != nil {
		return err
	}
	if err := out.end(); err != nil {
		return err
	}
	if out.sp != nil {
		done := spin("playing")
		if err := out.sp.finish(ctx); err != nil {
			done("stopped")
			return err
		}
		done("done")
	}
	if out.w != nil {
		if err := out.w.close(); err != nil {
			return err
		}
		if o.out != "-" {
			fmt.Fprintf(os.Stderr, "%s: %.1f s of 48 kHz (mono in both channels)\n", o.out, seconds)
		}
	}
	return nil
}
