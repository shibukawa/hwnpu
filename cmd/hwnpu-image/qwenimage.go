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
	"github.com/shibukawa/hwnpu/internal/imagegen/qwenimage"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
)

// qwenImage draws the pictures with an export of tools/amd/iron_qwenimage.py on AMD's NPU
// (Qwen-Image-2.1-Turbo: the checkpoint's eight steps, no classifier-free guidance).
func (c *GenerateCmd) qwenImage(ctx context.Context, model string, outs []string) error {
	lw := console.Log(os.Stderr)
	live := console.Live(lw)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	t0 := time.Now()
	dev, err := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
	if err != nil {
		return err
	}
	defer dev.Close()
	iron.ShareExport(model, logf)
	p, err := qwenimage.Open(ctx, dev, model)
	if err != nil {
		return err
	}
	defer p.Close()
	p.Logf = logf
	p.ResidentDiT = c.Resident
	logf("loaded %s (%dx%d) in %s", filepath.Base(model), p.Dit.Width, p.Dit.Height, console.Duration(time.Since(t0)))
	sig := p.Sigmas(c.Steps)
	steps := len(sig) - 1
	// Turbo: its eight sigmas, no guidance. The full model (Qwen-Image-2.1): 40 steps, guidance only when
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
	for i, prompt := range c.Prompts {
		t1 := time.Now()
		var cond qwenimage.Cond
		var neg *qwenimage.Cond
		if smp.Guidance != 1 {
			cs, err := p.EncodeAll(ctx, prompt, c.Negative)
			if err != nil {
				return err
			}
			cond, neg = cs[0], &cs[1]
		} else {
			cs, err := p.EncodeAll(ctx, prompt)
			if err != nil {
				return err
			}
			cond = cs[0]
		}
		var x []float32
		if c.Noise != "" {
			if x, err = readF32(c.Noise); err != nil {
				return err
			}
		} else {
			s := c.Seed
			if s == 0 {
				s = rand.Uint64()
			} else {
				s += uint64(i)
			}
			logf("seed %d", s)
			x = p.Noise(s)
		}
		bar := console.NewBar(lw, "denoising")
		bar.Set(0, fmt.Sprintf("step 1/%d", steps))
		x, err = p.Generate(ctx, cond, neg, x, sig, smp, func(i, n int, d time.Duration) {
			per := bar.Elapsed() / time.Duration(i+1)
			bar.Set(float64(i+1)/float64(n), fmt.Sprintf("step %d/%d, %.1f s a step, %s left", min(i+2, n), n, per.Seconds(),
				console.Duration(time.Duration(n-i-1)*per)))
			if !live {
				logf("step %d/%d: sigma %.3f -> %.3f, %.1f s", i+1, n, sig[i], sig[i+1], d.Seconds())
			}
		})
		if err != nil {
			bar.Done("failed")
			return err
		}
		bar.Done(fmt.Sprintf("%d steps in %s (%.1f s a step)", steps, console.Duration(bar.Elapsed()), bar.Elapsed().Seconds()/float64(steps)))
		if c.Latents != "" {
			if err := writeF32(c.Latents, x); err != nil {
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
	}
	return nil
}
