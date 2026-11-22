package main

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	"image/png"
	"io"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	devenv "github.com/shibukawa/hwnpu/internal/dev"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/imagegen/aspect"
	"github.com/shibukawa/hwnpu/internal/imagegen/flux2klein"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
)

// npuGeneration is the generation of this machine's NPU of vendor as models.yaml names it.
func npuGeneration(vendor string) string { return flux2klein.NPUGeneration(vendor) }

// npuAspects are the picture sizes of FLUX.2 klein's export for vendor's NPU on generation gen.
func npuAspects(m chat.Model, vendor, gen string) ([]aspect.Aspect, error) {
	return flux2klein.NPUAspects(m, vendor, gen)
}

// kleinSizesHere are FLUX.2 klein's sizes on this machine's NPU, for an export given by its
// directory (the model is then the default one, FLUX.2-klein-4B).
func kleinSizesHere() ([]aspect.Aspect, error) {
	m, ok := chat.Lookup("FLUX.2-klein-4B")
	if !ok {
		return nil, errors.New("FLUX.2-klein-4B is not in models.yaml")
	}
	vendor := chat.HereVendor()
	if vendor == "" {
		return nil, errors.New("no NPU here for FLUX.2 klein's sizes")
	}
	return npuAspects(m, vendor, npuGeneration(vendor))
}

// kleinModel is the FLUX.2 klein export m runs here: on Apple's Neural Engine the Core AI export,
// downloaded and converted when missing; on AMD's NPU the export of tools/amd/iron_flux2klein.py
// of every size of models.yaml npu.amd (one export), from the shared layers or made with IRON when
// missing (flux2klein.PrepareAMD). It returns the export and the sizes it draws.
func (c *GenerateCmd) kleinModel(ctx context.Context, m chat.Model) (string, []aspect.Aspect, error) {
	format, ferr := imageFormat(ctx, c.Backend)
	onApple := ferr == nil && format == llm.BackendCoreAI
	vendor := "amd"
	if onApple {
		vendor = "apple"
	}
	as, err := npuAspects(m, vendor, npuGeneration(vendor))
	if err != nil {
		return "", nil, err
	}
	w, h, err := c.pictureIn(as, 16)
	if err != nil {
		return "", nil, err
	}
	switch {
	case onApple:
	case ferr == nil:
		return "", nil, fmt.Errorf("%s runs on AMD's NPU and Apple's Neural Engine; not yet on %s", m.Name, format)
	case runtime.GOOS == "darwin" || chat.HereVendor() != "amd":
		return "", nil, ferr
	default:
		dir, err := flux2klein.PrepareAMD(chat.WithModel(ctx, m), m, aspect.SizesOf(as), w, h, console.Log(os.Stderr))
		return dir, as, err
	}
	if w%16 != 0 || h%16 != 0 {
		return "", nil, fmt.Errorf("--size %s: FLUX.2 klein's pictures are multiples of 16 pixels", c.Size)
	}
	spec := m.NPU["apple"]
	// every size of models.yaml npu.apple in one export (the weights once; design.md 16.280)
	o := flux2klein.ConvertOptions{Width: w, Height: h, Sizes: aspect.SizesOf(as),
		TextFormat: cmp.Or(c.TextFormat, spec.Text), DitFormat: cmp.Or(c.DitFormat, spec.Dit)}
	dir, err := flux2klein.Prepare(chat.WithModel(ctx, m), flux2klein.Setup{Dir: filepath.Join(chat.ModelsDir(), "FLUX.2-klein"), Repo: m.Repo,
		Name: m.Name, ConvertOptions: o, Log: console.Log(os.Stderr)})
	return dir, as, err
}

// kleinEdit is the edit's programs of the Core AI export model (flux2klein.PrepareEdit): of its
// sizes and DiT format, next to it, from the shared layers or converted from the checkpoint.
func kleinEdit(ctx context.Context, model string, log io.Writer) (string, error) {
	m, ok := chat.Lookup("FLUX.2-klein-4B")
	if !ok {
		return "", errors.New("FLUX.2-klein-4B is not in models.yaml")
	}
	man, err := flux2klein.ReadProgramManifest(model)
	if err != nil {
		return "", err
	}
	return flux2klein.PrepareEdit(chat.WithModel(ctx, m), flux2klein.Setup{Dir: filepath.Dir(model), Repo: m.Repo, Name: m.Name, Log: log},
		flux2klein.EditOptions{Sizes: man.SizeList(), DitFormat: man.Formats["dit"]})
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

// klein draws the pictures with an export of FLUX.2 klein 4B (4 Euler steps, no guidance): of
// tools/amd/iron_flux2klein.py on AMD's NPU, or of hwnpu-image convert on Apple's Neural Engine
// (Core AI's).
func (c *GenerateCmd) klein(ctx context.Context, model string, as []aspect.Aspect, outs []string) error {
	if as == nil { // an export given by its directory: the sizes of this machine's NPU
		var err error
		if as, err = kleinSizesHere(); err != nil {
			return err
		}
	}
	pw, ph, err := c.pictureIn(as, 16)
	if err != nil {
		return err
	}
	lw := console.Log(os.Stderr)
	live := console.Live(lw)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	t0 := time.Now()
	if c.Negative != "" || c.Guidance > 1 {
		return errors.New("FLUX.2 klein is distilled: it runs no classifier-free guidance (--negative, --guidance)")
	}
	var dev *hwnpu.Device
	if flux2klein.IsProgramExport(model) {
		// strict, so that a program off the NPU, or one whose placement is not verified, says why
		d, _, err := llm.OpenDevice(ctx, llm.BackendCoreAI, cacheDirOf(c.Cache), hwnpu.WithPlacement(hwnpu.PlacementStrict))
		if err != nil {
			return err
		}
		dev = d
	} else {
		d, err := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
		if err != nil {
			return err
		}
		dev = d
		iron.ShareExport(model, logf)
	}
	defer dev.Close()
	w, h := 0, 0
	if flux2klein.IsProgramExport(model) || len(flux2klein.IronSizes(model)) > 1 {
		w, h = pw, ph // an export of several sizes draws at --size (one of a size alone draws at its own)
	}
	p, err := flux2klein.OpenAt(ctx, dev, model, w, h, lw)
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
	if flux2klein.IsProgramExport(model) && (len(c.Image) > 0 || len(c.RefLatents) > 0) {
		// Apple's Neural Engine edits with programs of their own, converted at the first edit
		dir, err := kleinEdit(ctx, model, lw)
		if err != nil {
			return err
		}
		if err := p.UseEdit(dir); err != nil {
			return err
		}
	}
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
		how := "tools/amd/iron_flux2klein.py --refs compiled their rows"
		if flux2klein.IsProgramExport(model) {
			how = "flux2klein.EditOptions.Refs converted their programs"
		}
		return fmt.Errorf("--image: the export %s does not run these references (%s); it runs %s, and the others once %s",
			filepath.Base(model), referenceNames(divs), strings.Join(p.References(), ", "), how)
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
	// --batch pictures at a time, their steps together (a group of the DiT's blocks runs every
	// picture before the next group's weights are read, or its program loads), then the VAE: on
	// Apple's Neural Engine its stages over all of them, on AMD's NPU a picture after another (a
	// stage hands its output to the next on the device). An edit's pictures share its references
	batch := max(1, c.Batch)
	for lo := 0; lo < len(c.Prompts); lo += batch {
		hi := min(lo+batch, len(c.Prompts))
		n := hi - lo
		t1 := time.Now()
		xs := make([][]float32, n)
		for i := lo; i < hi; i++ {
			if c.Noise != "" {
				if xs[i-lo], err = readF32(c.Noise); err != nil {
					return err
				}
				continue
			}
			s := c.Seed
			if s == 0 {
				s = rand.Uint64()
			} else {
				s += uint64(i)
			}
			logf("seed %d", s)
			xs[i-lo] = p.Noise(s)
		}
		sg := sig
		if devenv.On("klein.decode") { // a check of the VAE alone: --noise holds final latents
			sg = sig[:1]
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
		var took stepTimes
		progress := func(i, total int, d time.Duration) {
			took = append(took, d)
			per := bar.Elapsed() / time.Duration(i+1)
			bar.Set(float64(i+1)/float64(total), fmt.Sprintf("step %d/%d, %s, %s left", min(i+2, total), total, times(per),
				console.Duration(time.Duration(total-i-1)*per)))
			if !live {
				logf("step %d/%d: sigma %.3f -> %.3f, %.1f s", i+1, total, sg[i], sg[i+1], d.Seconds())
			}
		}
		xs, err = p.GenerateAll(ctx, conds[lo:hi], refs, xs, sg, progress)
		if err != nil {
			bar.Done("failed")
			return err
		}
		bar.Done(took.summary(bar.Elapsed(), times))
		if c.Latents != "" {
			for _, x := range xs {
				if err := writeF32(c.Latents, x); err != nil {
					return err
				}
			}
		}
		bar = console.NewBar(lw, "decoding")
		p.Stage = func(i, total int, d time.Duration) {
			bar.Set(float64(i+1)/float64(total), fmt.Sprintf("VAE stage %d/%d", min(i+2, total), total))
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
			out := outs[lo+i]
			f, err := os.Create(out)
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
				fmt.Fprintf(os.Stderr, "%s: %s a picture, %d together (%s since the start)\n", out, console.Duration(time.Since(t1)/time.Duration(n)), n,
					console.Duration(time.Since(t0)))
			} else {
				fmt.Fprintf(os.Stderr, "%s: %s (%s since the start)\n", out, console.Duration(time.Since(t1)), console.Duration(time.Since(t0)))
			}
		}
	}
	return nil
}
