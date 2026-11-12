package main

import (
	"context"
	"fmt"
	"image"
	"image/png"
	"math/rand/v2"
	"os"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/imagegen/qwenimage"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
)

// qwenImage draws the pictures with an export of tools/amd/iron_qwenimage.py on AMD's NPU, or of
// qwenimage.Convert (Core ML) on Apple's Neural Engine or, with --cpu, the CPU interpreter
// (Qwen-Image-2.1-Turbo: the checkpoint's eight steps, no classifier-free guidance).
func (c *GenerateCmd) qwenImage(ctx context.Context, model string, outs []string) error {
	lw := console.Log(os.Stderr)
	live := console.Live(lw)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	t0 := time.Now()
	programs := qwenimage.IsProgramExport(model)
	var dev *hwnpu.Device
	switch {
	case c.CPU:
	case programs:
		// strict, so that a program off the NPU, or one whose placement is not verified, says why
		d, _, err := llm.OpenDevice(ctx, llm.BackendCoreML, cacheDirOf(c.Cache), hwnpu.WithPlacement(hwnpu.PlacementStrict))
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
		iron.ShareExport(model, logf)
	}
	p, err := qwenimage.OpenWith(ctx, dev, model, lw)
	if err != nil {
		return err
	}
	defer p.Close()
	p.Logf = logf
	p.ResidentDiT = c.Resident
	logf("loaded %s (%dx%d) in %s", filepath.Base(model), p.Dit.Width, p.Dit.Height, console.Duration(time.Since(t0)))
	console.Running()
	sig := p.Sigmas(c.Steps)
	steps := len(sig) - 1
	// Turbo: its eight sigmas, no guidance. The base checkpoint: 40 steps, guidance only when
	// asked (the pipeline's true_cfg_scale 1 by default), as diffusers runs it
	smp := qwenimage.Sampling{Solver: c.Sampler, Guidance: c.Guidance}
	if smp.Guidance == 0 {
		smp.Guidance = 1
	}
	if c.Negative != "" && smp.Guidance == 1 {
		return fmt.Errorf("--negative: give a --guidance over 1 to use it (the pipeline's true_cfg_scale)")
	}
	if len(c.CFGSigmas) > 0 {
		if len(c.CFGSigmas) != 2 || smp.Guidance == 1 {
			return fmt.Errorf("--cfg-sigmas: give LO,HI, with a --guidance over 1")
		}
		smp.CFGMin, smp.CFGMax = c.CFGSigmas[0], c.CFGSigmas[1]
	}
	if smp.Guidance != 1 {
		n := 0
		for _, sg := range sig[:steps] {
			if smp.CFGMax <= 0 || sg >= smp.CFGMin && sg <= smp.CFGMax {
				n++
			}
		}
		logf("classifier-free guidance %g at %d of %d steps", smp.Guidance, n, steps)
	}
	noise := func(i int) ([]float32, error) {
		if c.Noise != "" {
			return readF32(c.Noise)
		}
		s := c.Seed
		if s == 0 {
			s = rand.Uint64()
		} else {
			s += uint64(i)
		}
		logf("seed %d", s)
		return p.Noise(s), nil
	}
	generate := func(cond qwenimage.Cond, neg *qwenimage.Cond, x []float32) ([]float32, error) {
		bar := console.NewBar(lw, "denoising")
		bar.Set(0, fmt.Sprintf("step 1/%d", steps))
		x, err := p.Generate(ctx, cond, neg, x, sig, smp, func(i, n int, d time.Duration) {
			per := bar.Elapsed() / time.Duration(i+1)
			bar.Set(float64(i+1)/float64(n), fmt.Sprintf("step %d/%d, %.1f s a step, %s left", min(i+2, n), n, per.Seconds(),
				console.Duration(time.Duration(n-i-1)*per)))
			if !live {
				logf("step %d/%d: sigma %.3f -> %.3f, %.1f s", i+1, n, sig[i], sig[i+1], d.Seconds())
			}
		})
		if err != nil {
			bar.Done("failed")
			return nil, err
		}
		bar.Done(fmt.Sprintf("%d steps in %s (%.1f s a step)", steps, console.Duration(bar.Elapsed()), bar.Elapsed().Seconds()/float64(steps)))
		if c.Latents != "" {
			if err := writeF32(c.Latents, x); err != nil {
				return nil, err
			}
		}
		return x, nil
	}
	decode := func(xs ...[]float32) ([]*image.RGBA, error) {
		bar := console.NewBar(lw, "decoding")
		p.Stage = func(i, n int, d time.Duration) {
			bar.Set(float64(i+1)/float64(n), fmt.Sprintf("VAE stage %d/%d", min(i+2, n), n))
			if !live {
				logf("vae stage %d: %.2f s", i, d.Seconds())
			}
		}
		imgs, err := p.DecodeAll(ctx, xs...)
		if err != nil {
			bar.Done("failed")
			return nil, err
		}
		bar.Done("in " + console.Duration(bar.Elapsed()))
		return imgs, nil
	}
	write := func(i int, img *image.RGBA, t1 time.Time) error {
		f, err := os.Create(outs[i])
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
		fmt.Fprintf(os.Stderr, "%s: %s (%s since the start)\n", outs[i], console.Duration(time.Since(t1)), console.Duration(time.Since(t0)))
		return nil
	}
	// the conditionings of the prompts (and of the negative prompt, after them)
	encode := func(prompts []string) ([]qwenimage.Cond, *qwenimage.Cond, error) {
		if smp.Guidance != 1 {
			prompts = append(append([]string(nil), prompts...), c.Negative)
		}
		cs, err := p.EncodeAll(ctx, prompts...)
		if err != nil {
			return nil, nil, err
		}
		if smp.Guidance != 1 {
			return cs[:len(cs)-1], &cs[len(cs)-1], nil
		}
		return cs, nil, nil
	}
	if p.Batched() {
		// a Core ML export loads its programs for each phase, which takes longer than running them:
		// every prompt's text, then every picture's steps with the DiT held, then the VAE's stages
		// over all the pictures
		t1 := time.Now()
		conds, neg, err := encode(c.Prompts)
		if err != nil {
			return err
		}
		xs := make([][]float32, len(c.Prompts))
		for i := range c.Prompts {
			x, err := noise(i)
			if err != nil {
				return err
			}
			if xs[i], err = generate(conds[i], neg, x); err != nil {
				return err
			}
		}
		imgs, err := decode(xs...)
		if err != nil {
			return err
		}
		for i, img := range imgs {
			if err := write(i, img, t1); err != nil {
				return err
			}
		}
		return nil
	}
	for i, prompt := range c.Prompts {
		t1 := time.Now()
		conds, neg, err := encode([]string{prompt})
		if err != nil {
			return err
		}
		x, err := noise(i)
		if err != nil {
			return err
		}
		if x, err = generate(conds[0], neg, x); err != nil {
			return err
		}
		imgs, err := decode(x)
		if err != nil {
			return err
		}
		if err := write(i, imgs[0], t1); err != nil {
			return err
		}
	}
	return nil
}
