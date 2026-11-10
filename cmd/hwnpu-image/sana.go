package main

import (
	"context"
	"fmt"
	"image/png"
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
	model, prompt, negative, out, noise, latents string
	steps                                        int
	guidance                                     float64
	seed                                         uint64
	resident, cpu                                bool
	dump, cache                                  string
}

// runSana draws the picture with an export of Sana or Sana-Sprint: of tools/amd/iron_sana.py on
// AMD's NPU, or of hwnpu-image convert (Core ML) on Apple's Neural Engine or, with cpu, the CPU
// interpreter.
func runSana(ctx context.Context, r sanaRun) error {
	lw := console.Log(os.Stderr)
	live := console.Live(lw)
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
	m := p.Model
	steps := r.steps
	if steps == 0 {
		steps = m.Steps
	}
	guidance := r.guidance
	if guidance == 0 {
		guidance = m.Guidance
	}
	logf("loaded %s (%dx%d) in %s", filepath.Base(r.model), p.Dit.W(), p.Dit.H(), console.Duration(time.Since(t0)))

	cond, err := p.Encode(ctx, r.prompt)
	if err != nil {
		return err
	}
	var neg sana.Cond
	if r.negative != "" {
		if m.Sprint {
			return fmt.Errorf("Sana-Sprint runs no classifier-free guidance: --negative is for Sana")
		}
		if neg, err = p.EncodeNegative(ctx, r.negative, cond); err != nil {
			return err
		}
	}
	var noises [][]float32
	n := p.NoiseCount(steps)
	if r.noise != "" {
		x, err := readF32(r.noise)
		if err != nil {
			return err
		}
		size := len(x) / n
		if size*n != len(x) || size != p.Dit.Latent*p.Dit.Tokens() {
			return fmt.Errorf("%s: %d values for %d noises of %d", r.noise, len(x), n, p.Dit.Latent*p.Dit.Tokens())
		}
		for k := 0; k < n; k++ {
			noises = append(noises, x[k*size:(k+1)*size])
		}
	} else {
		s := r.seed
		if s == 0 {
			s = rand.Uint64()
		}
		logf("seed %d", s)
		noises = p.Noise(s, n)
	}
	bar := console.NewBar(lw, "denoising")
	bar.Set(0, fmt.Sprintf("step 1/%d", steps))
	x, err := p.Generate(ctx, cond, neg, noises, steps, guidance, func(i, n int, d time.Duration) {
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
	f, err := os.Create(r.out)
	if err != nil {
		return err
	}
	if err := png.Encode(f, img); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "%s: %s in all\n", r.out, console.Duration(time.Since(t0)))
	return nil
}
