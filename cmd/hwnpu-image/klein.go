package main

import (
	"context"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	"image/png"
	"math/rand/v2"
	"os"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/imagegen/flux2klein"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
)

// kleinModel is the FLUX.2 klein export m runs here: on AMD's NPU ~/models/NAME.amd-npu2 of
// tools/amd/iron_flux2klein.py (no other NPU runs it yet).
func (c *GenerateCmd) kleinModel(ctx context.Context, m chat.Model) (string, error) {
	format, err := imageFormat(ctx, c.Backend)
	if err == nil {
		return "", fmt.Errorf("%s runs on AMD's NPU; not yet on %s", m.Name, format)
	}
	return amdOr(err, m, "tools/amd/iron_flux2klein.py")
}

// klein draws the pictures with an export of tools/amd/iron_flux2klein.py on AMD's NPU
// (FLUX.2 klein 4B: 4 Euler steps, no guidance).
func (c *GenerateCmd) klein(ctx context.Context, model string, outs []string) error {
	lw := console.Log(os.Stderr)
	live := console.Live(lw)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	t0 := time.Now()
	if c.Negative != "" || c.Guidance > 1 {
		return errors.New("FLUX.2 klein is distilled: it runs no classifier-free guidance (--negative, --guidance)")
	}
	dev, err := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
	if err != nil {
		return err
	}
	defer dev.Close()
	iron.ShareExport(model, logf)
	p, err := flux2klein.Open(ctx, dev, model)
	if err != nil {
		return err
	}
	defer p.Close()
	p.Logf = logf
	if c.Dump != "" {
		if err := os.MkdirAll(c.Dump, 0o755); err != nil {
			return err
		}
		p.Dump = c.Dump
	}
	logf("loaded %s (%dx%d) in %s", filepath.Base(model), p.C.Width, p.C.Height, console.Duration(time.Since(t0)))
	console.Running()
	sig := p.Sigmas(c.Steps)
	steps := len(sig) - 1
	// the references' latents: for now as raw float32 rows (tools/ref/ref_flux2klein.py's "refs")
	var refs [][]float32
	for _, f := range c.RefLatents {
		r, err := readF32(f)
		if err != nil {
			return err
		}
		refs = append(refs, r)
	}
	div := 1
	if c.RefHalf {
		div = 2
	}
	if n := len(c.Image); n > 0 && !p.HasReferences(n, div) {
		return fmt.Errorf("--image: the export %s has no blocks or no VAE encoder for %d reference(s) of the picture's sides over %d (it has %v)",
			filepath.Base(model), n, div, p.References())
	}
	for _, f := range c.Image {
		r, err := os.Open(f)
		if err != nil {
			return err
		}
		img, _, err := image.Decode(r)
		r.Close()
		if err != nil {
			return fmt.Errorf("%s: %w", f, err)
		}
		t1 := time.Now()
		z, err := p.EncodeImage(ctx, img, div)
		if err != nil {
			return err
		}
		logf("the reference %s in %.2f s", filepath.Base(f), time.Since(t1).Seconds())
		if c.Dump != "" { // the reference's latents, to check the VAE's encoder
			if err := writeF32(filepath.Join(c.Dump, fmt.Sprintf("ref_%d.f32", len(refs))), z); err != nil {
				return err
			}
		}
		refs = append(refs, z)
	}
	if n := len(refs); n > 0 {
		logf("editing: %d reference(s)", n)
	}
	conds, err := p.EncodeAll(ctx, c.Prompts...)
	if err != nil {
		return err
	}
	for i := range c.Prompts {
		t1 := time.Now()
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
		if os.Getenv("HWNPU_KLEIN_DECODE") != "" { // a check of the VAE alone: --noise holds final latents
			sig = sig[:1]
		}
		x, err = p.Generate(ctx, conds[i], refs, x, sig, func(i, n int, d time.Duration) {
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
