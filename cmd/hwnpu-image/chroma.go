package main

import (
	"cmp"
	"context"
	"fmt"
	"image/png"
	"math/rand/v2"
	"os"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/imagegen/chroma"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
)

// chromaModel is the Chroma export m runs here: on Apple's Neural Engine the Core AI export of
// --dir (with --cpu the Core ML one), downloaded and converted when missing; on AMD's NPU ~/models/NAME.amd-npu2 of tools/amd/iron_chroma.py.
func (c *GenerateCmd) chromaModel(ctx context.Context, m chat.Model) (string, error) {
	format, err := imageFormat(ctx, c.Backend)
	switch {
	case c.CPU: // the CPU interpreter runs Core ML exports
		format = llm.BackendCoreML
	case err == nil && llm.AppleFormat(format):
	case err == nil:
		return "", fmt.Errorf("%s runs on AMD's NPU and Apple's Neural Engine; not yet on %s", m.Name, format)
	default:
		return amdOr(err, m, "tools/amd/iron_chroma.py")
	}
	if c.Size%16 != 0 {
		return "", fmt.Errorf("--size %d: Chroma's pictures are multiples of 16 pixels", c.Size)
	}
	o := chroma.ConvertOptions{Backend: format, Width: c.Size, Height: c.Size, TextFormat: c.TextFormat, DitFormat: c.DitFormat}
	return chroma.Prepare(ctx, chroma.Setup{Dir: filepath.Join(cmp.Or(c.Dir, chat.ModelsDir()), "Chroma"), Repo: m.Repo, Name: m.Name,
		ConvertOptions: o, Log: console.Log(os.Stderr)})
}

// chroma draws the pictures with an export of Chroma: of tools/amd/iron_chroma.py on AMD's NPU,
// or of hwnpu-image convert on Apple's Neural Engine (Core AI's) or, with --cpu, the CPU
// interpreter (Chroma1-Flash: Heun's 8 steps by default, no classifier-free guidance).
func (c *GenerateCmd) chroma(ctx context.Context, model string, outs []string) error {
	lw := console.Log(os.Stderr)
	live := console.Live(lw)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	t0 := time.Now()
	programs := chroma.IsProgramExport(model)
	var dev *hwnpu.Device
	switch {
	case c.CPU:
	case programs:
		// strict, so that a program off the NPU, or one whose placement is not verified, says why;
		// a Core AI export on Core AI
		d, _, err := llm.OpenDevice(ctx, exportDevice(chroma.ExportFormat(model), c.Backend), cacheDirOf(c.Cache),
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
	p, err := chroma.Open(ctx, dev, model, lw)
	if err != nil {
		return err
	}
	defer p.Close()
	p.Logf = logf
	logf("loaded %s (%dx%d) in %s", filepath.Base(model), p.C.Width, p.C.Height, console.Duration(time.Since(t0)))
	console.Running()
	// with several pictures the DiT stays loaded between them (--keep-dit)
	p.KeepDiT = c.KeepDiT && len(c.Prompts) > 1
	sig := p.Sigmas(c.Steps)
	steps := len(sig) - 1
	// the export's defaults: Chroma1-Flash Heun's 8 steps without guidance, Chroma1-HD 40 steps
	// with guidance 3 (model.json)
	smp := chroma.Sampling{Solver: cmp.Or(c.Sampler, p.Model.Sampler, "euler"), Guidance: cmp.Or(c.Guidance, p.Model.Guidance, 1)}
	if c.Negative != "" && smp.Guidance == 1 {
		return fmt.Errorf("--negative: %s runs no classifier-free guidance; give a --guidance over 1 to use it", p.Model.Name)
	}
	if len(c.CFGSigmas) > 0 {
		if len(c.CFGSigmas) != 2 || smp.Guidance == 1 {
			return fmt.Errorf("--cfg-sigmas: give LO,HI, with a --guidance over 1")
		}
		smp.CFGMin, smp.CFGMax = c.CFGSigmas[0], c.CFGSigmas[1]
	} else if cs := p.Model.CFGSigmas; len(cs) == 2 && c.Sampler == "" && c.Steps == 0 {
		smp.CFGMin, smp.CFGMax = cs[0], cs[1] // the export's default with its default sampler and steps
	}
	if smp.Guidance != 1 {
		n := 0
		for _, sg := range sig[:steps] {
			if smp.CFGMax <= 0 || sg >= smp.CFGMin && sg <= smp.CFGMax {
				n++
			}
		}
		logf("%s, classifier-free guidance %g at %d of %d steps", smp.Solver, smp.Guidance, n, steps)
	}
	// every prompt (and the negative one) in one pass of the text encoder: its groups are loaded
	// once for all the pictures, not once a picture (T5's twelve fp16 programs take 11 s to load on
	// the M1; new-model-checklist 7.1, "every prompt at once")
	texts := append(append([]string(nil), c.Prompts...), c.Negative)
	if smp.Guidance == 1 {
		texts = texts[:len(c.Prompts)]
	}
	conds, err := p.EncodeAll(ctx, texts...)
	if err != nil {
		return err
	}
	for i := range c.Prompts {
		t1 := time.Now()
		cond := conds[i]
		var neg *chroma.Cond
		if smp.Guidance != 1 {
			neg = &conds[len(c.Prompts)]
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
