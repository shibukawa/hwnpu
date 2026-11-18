package main

import (
	"context"
	"errors"
	"fmt"
	"image/png"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/imagegen/aspect"
	"github.com/shibukawa/hwnpu/internal/imagegen/flux3klein"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
)

// flux3Model is the FLUX.3 klein export m runs here: on Apple's Neural Engine the Core AI export,
// downloaded and converted when missing (with the DiT's lut8 tables fitted to the activations when
// the exports' directory or a source holds flux3klein.StatsFile); on AMD's NPU the export of
// tools/amd/iron_flux3klein.py of every size of models.yaml npu.amd (one export), from the shared
// layers or made with IRON when missing (flux3klein.PrepareAMD). It returns the export and, on AMD,
// the sizes it draws (nil: --size's own).
func (c *GenerateCmd) flux3Model(ctx context.Context, m chat.Model) (string, []aspect.Aspect, error) {
	format, err := imageFormat(ctx, c.Backend)
	switch {
	case err == nil && format == llm.BackendCoreAI:
	case err == nil:
		return "", nil, fmt.Errorf("%s runs on AMD's NPU and Apple's Neural Engine; not yet on %s", m.Name, format)
	case runtime.GOOS == "darwin" || chat.HereVendor() != "amd":
		return "", nil, err
	default:
		as, err := npuAspects(m, "amd", npuGeneration("amd"))
		if err != nil {
			return "", nil, err
		}
		w, h, err := c.pictureIn(as, 16)
		if err != nil {
			return "", nil, err
		}
		dir, err := flux3klein.PrepareAMD(chat.WithModel(ctx, m), m, aspect.SizesOf(as), w, h, console.Log(os.Stderr))
		return dir, as, err
	}
	o := flux3klein.ConvertOptions{TextFormat: c.TextFormat, DitFormat: c.DitFormat}
	dir, err := flux3klein.Prepare(chat.WithModel(ctx, m), flux3klein.Setup{Dir: filepath.Join(chat.ModelsDir(), "FLUX.3-klein"), Repo: m.Repo,
		Name: m.Name, ConvertOptions: o, Log: console.Log(os.Stderr)})
	return dir, nil, err
}

// flux3SizesHere are FLUX.3 klein's sizes on AMD's NPU (models.yaml npu.amd), for an export given
// by its directory.
func flux3SizesHere() ([]aspect.Aspect, error) {
	m, ok := chat.Lookup("flux-3-image-klein")
	if !ok {
		return nil, errors.New("flux-3-image-klein is not in models.yaml")
	}
	return npuAspects(m, "amd", npuGeneration("amd"))
}

// flux3 draws the pictures with an export of FLUX.3 klein (16 flow-map steps, no guidance), a
// prompt after another: of Convert on Apple's Neural Engine, of tools/amd/iron_flux3klein.py on
// AMD's NPU, whose --size names the aspect ratios of as (nil: flux3SizesHere for an IRON export
// given by its directory).
func (c *GenerateCmd) flux3(ctx context.Context, model string, as []aspect.Aspect, outs []string) error {
	var err error
	if as == nil && flux3klein.IsIronExport(model) {
		if as, err = flux3SizesHere(); err != nil {
			return err
		}
	}
	w, h, err := c.picture(16)
	if as != nil {
		w, h, err = c.pictureIn(as, 16)
	}
	if err != nil {
		return err
	}
	if c.Negative != "" || c.Guidance > 1 {
		return errors.New("FLUX.3 klein is distilled: it runs no classifier-free guidance (--negative, --guidance)")
	}
	if c.Steps != 0 && c.Steps != flux3klein.Steps {
		return fmt.Errorf("FLUX.3 klein is distilled for %d steps (--steps)", flux3klein.Steps)
	}
	if len(c.Image) > 0 {
		return errors.New("--image: FLUX.3 klein edits on hwnpu are not supported yet (text to image only)")
	}
	lw := console.Log(os.Stderr)
	live := console.Live(lw)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	t0 := time.Now()
	var dev *hwnpu.Device
	if flux3klein.IsIronExport(model) {
		if dev, err = hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD)); err != nil {
			return err
		}
		iron.ShareExport(model, logf)
	} else if dev, _, err = llm.OpenDevice(ctx, llm.BackendCoreAI, cacheDirOf(c.Cache), hwnpu.WithPlacement(hwnpu.PlacementStrict)); err != nil {
		return err
	}
	defer dev.Close()
	p, err := flux3klein.Open(ctx, dev, model, w, h, lw)
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
	if !p.Manifest().FittedTables {
		logf("note: the export's DiT tables are spaced evenly (no activation statistics when it was converted): fine detail may change")
	}
	logf("loaded %s (%dx%d) in %s", filepath.Base(model), w, h, console.Duration(time.Since(t0)))
	console.Running()
	sig := flux3klein.Sigmas()
	steps := len(sig) - 1
	for i, prompt := range c.Prompts {
		t1 := time.Now()
		cond, err := p.Encode(ctx, prompt)
		if err != nil {
			return err
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
		progress := func(i, total int, d time.Duration) {
			per := bar.Elapsed() / time.Duration(i+1)
			bar.Set(float64(i+1)/float64(total), fmt.Sprintf("step %d/%d, %.1f s a step, %s left", min(i+2, total), total, per.Seconds(),
				console.Duration(time.Duration(total-i-1)*per)))
			if !live {
				logf("step %d/%d: sigma %.3f -> %.3f, %.1f s", i+1, total, sig[i], sig[i+1], d.Seconds())
			}
		}
		x, err = p.Generate(ctx, cond, x, sig, progress)
		if err != nil {
			bar.Done("failed")
			return err
		}
		bar.Done(fmt.Sprintf("%d steps in %s (%.1f s a step)", steps, console.Duration(bar.Elapsed()), (bar.Elapsed() / time.Duration(steps)).Seconds()))
		if c.Latents != "" {
			if err := writeF32(c.Latents, x); err != nil {
				return err
			}
		}
		bar = console.NewBar(lw, "decoding")
		img, err := p.Decode(ctx, cond, x, func(i, total int, d time.Duration) {
			bar.Set(float64(i+1)/float64(total), fmt.Sprintf("VAE stage %d/%d", min(i+2, total), total))
			if !live {
				logf("vae stage %d: %.2f s", i, d.Seconds())
			}
		})
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
