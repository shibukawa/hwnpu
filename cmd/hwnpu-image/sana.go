package main

import (
	"context"
	"fmt"
	"image/png"
	"io"
	"math/rand/v2"
	"os"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/imagegen/sana"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
)

// sanaRun is what runSana takes from the command line.
type sanaRun struct {
	model, negative, noise, latents string
	// prompts are drawn one after another into outs, the pipeline loaded once
	prompts, outs []string
	keepDiT       bool
	steps         int
	guidance      float64
	cfgSigmas     []float64 // LO, HI: the steps that run the guidance (nil: the pipeline's)
	seed          uint64
	resident, cpu bool
	dump, cache   string
}

// runSana draws the picture with an export of Sana: of tools/amd/iron_sana.py on
// AMD's NPU, or of hwnpu-image convert (Core ML) on Apple's Neural Engine or, with cpu, the CPU
// interpreter.
func runSana(ctx context.Context, r sanaRun) error {
	lw := console.Log(os.Stderr)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	t0 := time.Now()
	programs := sana.IsProgramExport(r.model)
	var dev *hwnpu.Device
	switch {
	case r.cpu:
	case programs:
		// llm.OpenDevice takes the Neural Engine's lock (design.md 16.132): with another process
		// compiling for it, a placement check of Gemma 2's programs took over 2 minutes. Strict,
		// so that a program off the NPU, or one whose placement is not verified, says why
		d, _, err := llm.OpenDevice(ctx, llm.BackendCoreML, cacheDirOf(r.cache), hwnpu.WithPlacement(hwnpu.PlacementStrict))
		if err != nil {
			return err
		}
		defer d.Close()
		dev = d
	default:
		d, err := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
		if err != nil {
			return err
		}
		defer d.Close()
		dev = d
	}
	if !programs {
		iron.ShareExport(r.model, logf)
	}
	p, err := sana.Open(ctx, dev, r.model, lw)
	if err != nil {
		return err
	}
	defer p.Close()
	p.Logf = logf
	p.ResidentDiT = r.resident
	p.Dump = r.dump
	steps := r.steps
	if steps == 0 {
		steps = p.Model.Steps
	}
	guidance := r.guidance
	if guidance == 0 {
		guidance = p.Model.Guidance
	}
	if len(r.cfgSigmas) > 0 {
		if len(r.cfgSigmas) != 2 {
			return fmt.Errorf("--cfg-sigmas: give LO,HI")
		}
		p.CFGMin, p.CFGMax = r.cfgSigmas[0], r.cfgSigmas[1]
	}
	if guidance != 1 && p.CFGMax > 0 {
		n := 0
		for _, sg := range p.Sigmas(steps)[:steps] {
			if p.Guided(sg) {
				n++
			}
		}
		logf("guidance %g at %d of %d steps (sigma %g to %g)", guidance, n, steps, p.CFGMin, p.CFGMax)
	}
	if r.noise != "" && len(r.prompts) > 1 {
		return fmt.Errorf("--noise is for one prompt")
	}
	logf("loaded %s (%dx%d) in %s", filepath.Base(r.model), p.Dit.W(), p.Dit.H(), console.Duration(time.Since(t0)))
	console.Running()

	// with several pictures the DiT stays on the device between them (keepDiT): only Gemma 2
	// and the VAE's stages are loaded for each
	p.KeepDiT = r.keepDiT && len(r.prompts) > 1
	for i, prompt := range r.prompts {
		t1 := time.Now()
		seed := r.seed
		if seed != 0 {
			seed += uint64(i)
		}
		if err := drawSana(ctx, p, r, prompt, r.outs[i], seed, steps, guidance, lw, logf); err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "%s: %s (%s since the start)\n", r.outs[i], console.Duration(time.Since(t1)), console.Duration(time.Since(t0)))
	}
	return nil
}

// drawSana draws one picture of prompt into out with the loaded pipeline p.
func drawSana(ctx context.Context, p *sana.Pipeline, r sanaRun, prompt, out string, seed uint64, steps int, guidance float64,
	lw io.Writer, logf func(string, ...any)) error {
	live := console.Live(lw)
	cond, neg, err := p.EncodeBoth(ctx, prompt, r.negative)
	if err != nil {
		return err
	}
	var noise []float32
	if r.noise != "" {
		if noise, err = readF32(r.noise); err != nil {
			return err
		}
		if len(noise) != p.Dit.Latent*p.Dit.Tokens() {
			return fmt.Errorf("%s: %d values for latents of %d", r.noise, len(noise), p.Dit.Latent*p.Dit.Tokens())
		}
	} else {
		s := seed
		if s == 0 {
			s = rand.Uint64()
		}
		logf("seed %d", s)
		noise = p.Noise(s)
	}
	bar := console.NewBar(lw, "denoising")
	bar.Set(0, fmt.Sprintf("step 1/%d", steps))
	x, err := p.Generate(ctx, cond, neg, noise, steps, guidance, func(i, n int, d time.Duration) {
		per := bar.Elapsed() / time.Duration(i+1)
		bar.Set(float64(i+1)/float64(n), fmt.Sprintf("step %d/%d, %.1f s a step", min(i+2, n), n, per.Seconds()))
		if !live {
			logf("step %d/%d: %.2f s", i+1, n, d.Seconds())
		}
	})
	if err != nil {
		bar.Done("failed")
		return err
	}
	bar.Done(fmt.Sprintf("%d steps in %s", steps, console.Duration(bar.Elapsed())))
	if r.latents != "" {
		if err := writeF32(r.latents, x); err != nil {
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
	f, err := os.Create(out)
	if err != nil {
		return err
	}
	if err := png.Encode(f, img); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
