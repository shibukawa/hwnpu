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
	"strings"
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

// referenceArg is an --image argument's file and the reference's size as the picture's sides over
// div: FILE or FILE@full (1: the picture's size), FILE@half (2: half its sides). A file whose own
// name ends in @half or @full is read as it is, at the picture's size.
func referenceArg(arg string) (file string, div int, err error) {
	div = 1
	if _, err := os.Stat(arg); err == nil {
		return arg, div, nil
	}
	at := strings.LastIndex(arg, "@")
	if at < 0 {
		return "", 0, fmt.Errorf("--image: %s: no such file", arg)
	}
	switch strings.ToLower(arg[at+1:]) {
	case "half":
		div = 2
	case "full":
		div = 1
	default:
		return "", 0, fmt.Errorf("--image: %s: no such file (a reference's size is FILE@half or FILE@full)", arg)
	}
	file = arg[:at]
	if _, err := os.Stat(file); err != nil {
		return "", 0, fmt.Errorf("--image: %s: no such file", file)
	}
	return file, div, nil
}

// referenceNames names references of the picture's sides over divs as Pipeline.References does:
// "1+2@half".
func referenceNames(divs []int) string {
	full, half := 0, 0
	for _, d := range divs {
		if d > 1 {
			half++
		} else {
			full++
		}
	}
	var name []string
	if full > 0 {
		name = append(name, fmt.Sprint(full))
	}
	if half > 0 {
		name = append(name, fmt.Sprintf("%d@half", half))
	}
	return strings.Join(name, "+")
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
	files, divs := make([]string, len(c.Image)), make([]int, len(c.Image))
	for i, arg := range c.Image {
		if files[i], divs[i], err = referenceArg(arg); err != nil {
			return err
		}
	}
	if len(files) > 0 && !p.HasReferences(divs) {
		return fmt.Errorf("--image: the export %s does not run these references (%s); it runs %s, and the others once tools/amd/iron_flux2klein.py --refs compiled their rows",
			filepath.Base(model), referenceNames(divs), strings.Join(p.References(), ", "))
	}
	for i, f := range files {
		div := divs[i]
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
		size := ""
		if div > 1 {
			size = fmt.Sprintf(" at %dx%d", p.C.Width/div, p.C.Height/div)
		}
		logf("the reference %s%s in %.2f s", filepath.Base(f), size, time.Since(t1).Seconds())
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
