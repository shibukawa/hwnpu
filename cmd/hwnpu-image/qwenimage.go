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
// qwenimage.Convert on Apple's Neural Engine (Core AI's) or,
// with --cpu, the CPU interpreter
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
		// strict, so that a program off the NPU, or one whose placement is not verified, says why;
		// a Core AI export on Core AI
		d, _, err := llm.OpenDevice(ctx, exportDevice(qwenimage.ExportFormat(model), c.Backend), cacheDirOf(c.Cache),
			hwnpu.WithPlacement(hwnpu.PlacementStrict))
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
	// generate draws the pictures of conds from the noises xs: with an export of programs their
	// steps together (a step is then all of them through the DiT), else one picture after another
	generate := func(conds []qwenimage.Cond, neg *qwenimage.Cond, xs [][]float32) ([][]float32, error) {
		n := 1 // the pictures a step runs
		if p.Batched() {
			n = len(conds)
		}
		name := "denoising"
		if n > 1 {
			name = fmt.Sprintf("denoising %d pictures", n)
		}
		// times are a step's time and, with several pictures, each one's share of it
		times := func(per time.Duration) string {
			if n > 1 {
				return fmt.Sprintf("%.1f s a step, %.1f s a picture", per.Seconds(), per.Seconds()/float64(n))
			}
			return fmt.Sprintf("%.1f s a step", per.Seconds())
		}
		bar := console.NewBar(lw, name)
		bar.Set(0, fmt.Sprintf("step 1/%d", steps))
		xs, err := p.GenerateAll(ctx, conds, neg, xs, sig, smp, func(i, total int, d time.Duration) {
			per := bar.Elapsed() / time.Duration(i+1)
			bar.Set(float64(i+1)/float64(total), fmt.Sprintf("step %d/%d, %s, %s left", min(i+2, total), total, times(per),
				console.Duration(time.Duration(total-i-1)*per)))
			if !live {
				logf("step %d/%d: sigma %.3f -> %.3f, %.1f s", i+1, total, sig[i], sig[i+1], d.Seconds())
			}
		})
		if err != nil {
			bar.Done("failed")
			return nil, err
		}
		bar.Done(fmt.Sprintf("%d steps in %s (%s)", steps, console.Duration(bar.Elapsed()), times(bar.Elapsed()/time.Duration(steps))))
		if c.Latents != "" {
			for _, x := range xs {
				if err := writeF32(c.Latents, x); err != nil {
					return nil, err
				}
			}
		}
		return xs, nil
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
	// write writes picture i, one of n drawn together since t1
	write := func(i int, img *image.RGBA, t1 time.Time, n int) error {
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
		if n > 1 {
			fmt.Fprintf(os.Stderr, "%s: %s a picture, %d together (%s since the start)\n", outs[i], console.Duration(time.Since(t1)/time.Duration(n)), n,
				console.Duration(time.Since(t0)))
		} else {
			fmt.Fprintf(os.Stderr, "%s: %s (%s since the start)\n", outs[i], console.Duration(time.Since(t1)), console.Duration(time.Since(t0)))
		}
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
	// an export of programs loads them for each phase, which takes longer than running them, and
	// a group of the DiT's blocks that follows another one pays for that in its first run alone:
	// --batch pictures at a time, their prompts' text, then their steps together, then the VAE's
	// stages over all of them. AMD's NPU draws one picture after another
	batch := 1
	if p.Batched() {
		batch = max(1, c.Batch)
	}
	for lo := 0; lo < len(c.Prompts); lo += batch {
		hi := min(lo+batch, len(c.Prompts))
		t1 := time.Now()
		conds, neg, err := encode(c.Prompts[lo:hi])
		if err != nil {
			return err
		}
		xs := make([][]float32, hi-lo)
		for i := lo; i < hi; i++ {
			if xs[i-lo], err = noise(i); err != nil {
				return err
			}
		}
		if xs, err = generate(conds, neg, xs); err != nil {
			return err
		}
		imgs, err := decode(xs...)
		if err != nil {
			return err
		}
		for i, img := range imgs {
			if err := write(lo+i, img, t1, hi-lo); err != nil {
				return err
			}
		}
	}
	return nil
}
