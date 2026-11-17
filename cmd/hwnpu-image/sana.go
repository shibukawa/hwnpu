package main

import (
	"context"
	"fmt"
	"image/png"
	"io"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
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
	// prompts are drawn into outs, the pipeline loaded once: batch of them together (an export
	// of programs: their text and their VAE's stages loaded once for all), else one after another
	prompts, outs []string
	batch         int
	keepDiT       bool
	steps         int
	guidance      float64
	cfgSigmas     []float64 // LO, HI: the steps that run the guidance (nil: the pipeline's)
	seed          uint64
	resident, cpu bool
	dump, cache   string
	// width and height are the picture's, taken from an export of several sizes (Core AI's
	// squares, an IRON export's aspect ratios)
	width, height int
}

// runSana draws the picture with an export of Sana: of tools/amd/iron_sana.py on
// AMD's NPU, or of hwnpu-image convert on Apple's Neural Engine (Core AI's or Core ML's) or, with
// cpu, the CPU interpreter (Core ML's).
func runSana(ctx context.Context, r sanaRun) error {
	lw := console.Log(os.Stderr)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	t0 := time.Now()
	programs := sana.IsProgramExport(r.model)
	var dev *hwnpu.Device
	w, h := 0, 0 // the export's default size
	switch {
	case r.cpu:
	case programs:
		// llm.OpenDevice takes the Neural Engine's lock (design.md 16.132): with another process
		// compiling for it, a placement check of Gemma 2's programs took over 2 minutes. Strict,
		// so that a program off the NPU, or one whose placement is not verified, says why. An
		// export of OpenVINO IR runs on Intel's NPU (or OpenVINO, HWNPU_BACKEND=openvino)
		m, err := sana.ReadProgramManifest(r.model)
		if err != nil {
			return err
		}
		if len(m.Sizes) > 0 {
			w, h = r.width, r.height
		}
		name, opts := exportDevice(m.Backend, ""), []hwnpu.Option{hwnpu.WithPlacement(hwnpu.PlacementStrict)}
		if m.Backend == llm.BackendIntel {
			name, opts = "", nil
			if b := llm.ProgramBackend(strings.TrimSpace(os.Getenv(llm.BackendEnv))); b != llm.BackendIntel {
				name = llm.BackendIntel
			}
		}
		d, _, err := llm.OpenDevice(ctx, name, cacheDirOf(r.cache), opts...)
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
		if len(sana.IronSizes(r.model)) > 1 {
			w, h = r.width, r.height
		}
	}
	p, err := sana.OpenAt(ctx, dev, r.model, w, h, lw)
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
	// and the VAE's stages are loaded for each, or with an export of programs for each batch of
	// them (design.md 16.216)
	p.KeepDiT = r.keepDiT && len(r.prompts) > 1
	batch := 1
	if p.Batched() {
		batch = max(1, r.batch)
	}
	for lo := 0; lo < len(r.prompts); lo += batch {
		hi := min(lo+batch, len(r.prompts))
		t1 := time.Now()
		if err := drawSana(ctx, p, r, lo, hi, steps, guidance, lw, logf); err != nil {
			return err
		}
		for i := lo; i < hi; i++ {
			if hi-lo == 1 {
				fmt.Fprintf(os.Stderr, "%s: %s (%s since the start)\n", r.outs[i], console.Duration(time.Since(t1)), console.Duration(time.Since(t0)))
			} else {
				fmt.Fprintf(os.Stderr, "%s: %s a picture, %d together (%s since the start)\n", r.outs[i],
					console.Duration(time.Since(t1)/time.Duration(hi-lo)), hi-lo, console.Duration(time.Since(t0)))
			}
		}
	}
	return nil
}

// drawSana draws the pictures of the prompts lo to hi-1 into their outs with the loaded pipeline
// p: the text of all of them, each one's steps, then the VAE over all of them.
func drawSana(ctx context.Context, p *sana.Pipeline, r sanaRun, lo, hi, steps int, guidance float64,
	lw io.Writer, logf func(string, ...any)) error {
	live := console.Live(lw)
	conds, negs, err := p.EncodeAll(ctx, r.prompts[lo:hi], r.negative)
	if err != nil {
		return err
	}
	xs := make([][]float32, hi-lo)
	for i := lo; i < hi; i++ {
		var noise []float32
		if r.noise != "" {
			if noise, err = readF32(r.noise); err != nil {
				return err
			}
			if len(noise) != p.Dit.Latent*p.Dit.Tokens() {
				return fmt.Errorf("%s: %d values for latents of %d", r.noise, len(noise), p.Dit.Latent*p.Dit.Tokens())
			}
		} else {
			s := r.seed
			if s == 0 {
				s = rand.Uint64()
			} else {
				s += uint64(i)
			}
			logf("seed %d", s)
			noise = p.Noise(s)
		}
		name := "denoising"
		if hi-lo > 1 {
			name = fmt.Sprintf("denoising %d/%d", i-lo+1, hi-lo)
		}
		bar := console.NewBar(lw, name)
		bar.Set(0, fmt.Sprintf("step 1/%d", steps))
		x, err := p.Generate(ctx, conds[i-lo], negs[i-lo], noise, steps, guidance, func(i, n int, d time.Duration) {
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
		xs[i-lo] = x
	}
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
		return err
	}
	bar.Done("in " + console.Duration(bar.Elapsed()))
	for i, img := range imgs {
		f, err := os.Create(r.outs[lo+i])
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
	}
	return nil
}
