package engine

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/ai/imagegen"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagegen/aspect"
	"github.com/shibukawa/hwnpu/internal/imagegen/flux2klein"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
)

// FLUX.2 klein 4B: 4 Euler steps without guidance, from a prompt and up to four reference
// pictures. AMD's NPU runs the export of tools/amd/iron_flux2klein.py of every size of
// models.yaml's npu.amd (one export, from the shared layers or made with IRON); Apple's Neural
// Engine a Core AI export of every size and edit of npu.apple (one export, the weights once),
// converted from the checkpoint, and the edits of a reference of 1024 in an export beside it
// (flux2klein.DeviceExport: compiled on the device, so prepared for Edit alone).

func prepareKlein(e airole.PrepareEnv, edit bool) (f airole.Files, err error) {
	m, base := e.Model, e.Base
	if dir := first(m.Programs[imagegen.ProgramExport]); dir != "" {
		f.Set("export", dir)
		if edit && flux2klein.IsProgramExport(dir) {
			// the export beside it of the edits of a reference of the picture's size, when it has one
			man, err := flux2klein.ReadProgramManifest(dir)
			if err != nil {
				return f, err
			}
			if man.Device != "" {
				dd := flux2klein.DeviceExport(dir)
				if !llm.FetchExport(chat.WithModel(e.Ctx, base), dd, dd+".done") {
					return f, fmt.Errorf("ai/imagegen: %s draws its edits of a reference of the picture's size with %s beside it, which is missing (hwnpu-image convert makes both)", dir, filepath.Base(dd))
				}
				f.Set("device", dd)
			}
		}
		return f, nil
	}
	if e.NPU.CPU(m) {
		return f, fmt.Errorf("ai/imagegen: %s runs on the NPU alone (not on the CPU)", m.Name)
	}
	vendor := "apple"
	if e.NPU.AMD() {
		vendor = "amd"
	} else if e.NPU.Format != llm.BackendCoreAI {
		return f, fmt.Errorf("ai/imagegen: %s runs on AMD's NPU and Apple's Neural Engine; not yet on %s", m.Name, e.NPU.Format)
	}
	as, err := flux2klein.NPUAspects(m, vendor, flux2klein.NPUGeneration(vendor))
	if err != nil {
		return f, err
	}
	ctx := chat.WithModel(e.Ctx, base)
	var dir string
	if e.NPU.AMD() {
		// one export of every size, here or in the sources, else made with IRON; it edits too
		if dir, err = flux2klein.PrepareAMD(ctx, base, aspect.SizesOf(as), as[0].Width, as[0].Height, e.Log); err != nil {
			return f, err
		}
		f.Set("export", dir)
		return f, nil
	}
	// every size and edit of models.yaml npu.apple in one export (the weights once), as hwnpu-image's
	spec := m.NPU["apple"]
	o, err := flux2klein.AppleOptions(m, flux2klein.NPUGeneration(vendor), as[0].Width, as[0].Height,
		setting(m, imagegen.SettingTextFormat, spec.Text), setting(m, imagegen.SettingDitFormat, spec.Dit))
	if err != nil {
		return f, err
	}
	setup := kleinSetup(base, o, e.Log)
	if dir, err = flux2klein.Prepare(ctx, setup); err != nil {
		return f, err
	}
	f.Set("export", dir)
	if edit {
		dd, err := flux2klein.PrepareDevice(ctx, setup)
		if err != nil {
			return f, err
		}
		if dd != "" {
			f.Set("device", dd)
		}
	}
	return f, nil
}

// kleinSetup is the Setup of FLUX.2 klein's Core AI export of base, in the models' FLUX.2-klein.
func kleinSetup(base chat.Model, o flux2klein.ConvertOptions, log io.Writer) flux2klein.Setup {
	return flux2klein.Setup{Dir: filepath.Join(chat.ModelsDir(), "FLUX.2-klein"), Repo: base.Repo, Name: base.Name, ConvertOptions: o, Log: log}
}

// klein draws with FLUX.2 klein.
type klein struct {
	g   *generator
	dir string
	// p draws pictures of at; another size closes it and opens that size's
	p  *flux2klein.Pipeline
	at image.Point
}

func loadKlein(ctx context.Context, g *generator, f airole.Files, e airole.LoadEnv) (family, error) {
	k := &klein{g: g, dir: f.Path("export")}
	var err error
	g.amd = !flux2klein.IsProgramExport(k.dir)
	var as []aspect.Aspect
	vendor := "apple"
	if g.amd {
		vendor = "amd"
	}
	if as, err = flux2klein.NPUAspects(g.model, vendor, flux2klein.NPUGeneration(vendor)); err != nil {
		return nil, err
	}
	g.parse = func(spec string) (int, int, error) {
		w, h, err := aspect.ParseOrSideIn(spec, as, 16)
		if err == nil && (w%16 != 0 || h%16 != 0) {
			err = fmt.Errorf("--size %s: FLUX.2 klein's pictures are multiples of 16 pixels", spec)
		}
		return w, h, err
	}
	if g.amd {
		// the sizes of the export: every size of models.yaml's npu.amd, or those of one made by hand
		if s := flux2klein.IronSizes(k.dir); len(s) > 0 {
			g.sizes = points(s)
		} else {
			g.sizes = points(aspect.SizesOf(as))
		}
		g.edits = true
		if g.editRefs, err = amdEdits(g.model, as, g.sizes); err != nil {
			return nil, err
		}
		if g.dev, err = openAMD(ctx); err != nil {
			return nil, err
		}
		iron.ShareExport(k.dir, logfOf(e.Log))
	} else {
		// the sizes and the edits the export holds (models.yaml's), those of the export beside it
		// (flux2klein.DeviceExport) only once it is here (Edit)
		man, err := flux2klein.ReadProgramManifest(k.dir)
		if err != nil {
			return nil, err
		}
		g.sizes = points(man.SizeList())
		device := man.Device != "" && llm.ExportDone(filepath.Join(filepath.Dir(k.dir), man.Device)+".done")
		g.editSizes, g.editRefs = []image.Point{}, map[image.Point][][]int{}
		for _, s := range man.Edits {
			pt := image.Pt(s.Width, s.Height)
			for _, st := range s.Steps {
				if st.Device && !device {
					continue
				}
				if len(g.editRefs[pt]) == 0 {
					g.editSizes = append(g.editSizes, pt)
				}
				g.editRefs[pt] = append(g.editRefs[pt], slices.Sorted(slices.Values(st.Refs)))
			}
		}
		g.edits, g.warmAll = len(g.editSizes) > 0, true
		if g.dev, _, err = llm.OpenDevice(ctx, llm.BackendCoreAI, e.Cache, placement(e)); err != nil {
			return nil, err
		}
	}
	// the first size is opened now, so that preparing compiles it and a missing part says so here
	if _, err := k.open(ctx, g.sizes[0]); err != nil {
		return nil, err
	}
	return k, nil
}

// open is the pipeline for pictures of size, opened once and kept until another size is asked.
func (k *klein) open(ctx context.Context, size image.Point) (*flux2klein.Pipeline, error) {
	if k.p != nil && k.at == size {
		return k.p, nil
	}
	if k.p != nil {
		k.p.Close()
		k.p = nil
	}
	w, h := size.X, size.Y
	if !flux2klein.IsProgramExport(k.dir) && len(flux2klein.IronSizes(k.dir)) <= 1 {
		w, h = 0, 0 // an export of a size alone draws at its own
	}
	t0 := time.Now()
	p, err := flux2klein.OpenAt(chat.WithModel(ctx, k.g.model), k.g.dev, k.dir, w, h, k.g.log)
	if err != nil {
		return nil, err
	}
	p.Logf = logfOf(k.g.log)
	k.g.logf("loaded %s (%dx%d) in %.1f s", filepath.Base(k.dir), p.C.Width, p.C.Height, time.Since(t0).Seconds())
	k.p, k.at = p, size
	return p, nil
}

func (k *klein) warm(ctx context.Context, r *run) error { return k.draw(ctx, r) }

func (k *klein) draw(ctx context.Context, r *run) error {
	o, c := r.o, r.checks()
	if o.Negative != "" || o.Guidance > 1 {
		return errors.New("FLUX.2 klein is distilled: it runs no classifier-free guidance (Negative, Guidance)")
	}
	if n := len(o.References) + len(c.RefLatents); n > flux2klein.MaxReferences {
		return fmt.Errorf("ai/imagegen: %d references; %s takes up to %d", n, k.g.name, flux2klein.MaxReferences)
	}
	ctx = chat.WithModel(ctx, k.g.model)
	if k.p == nil || k.at != r.size {
		r.phase(imagegen.PhaseOpen)
	}
	t0 := time.Now()
	p, err := k.open(ctx, r.size)
	if err != nil {
		return err
	}
	r.opened(t0)
	dump, err := r.dumpDir()
	if err != nil {
		return err
	}
	p.Dump = dump
	// the references' sizes: those asked, else those of the export's edit at the size (on Apple's
	// a reference of the picture's size first, the others of half it at 1024 x 1024)
	divs := make([]int, len(o.References))
	asked := false
	for i := range divs {
		divs[i] = 1
		if i < len(o.ReferenceScales) && o.ReferenceScales[i] > 1 {
			divs[i], asked = o.ReferenceScales[i], true
		}
	}
	if pats, ok := k.g.editRefs[r.size]; !asked && len(divs) > 0 && ok {
		i := slices.IndexFunc(pats, func(p []int) bool { return len(p) == len(divs) })
		if i < 0 { // a number the table leaves out (models.yaml npu edits): not drawn, as on Apple's
			if _, _, err := k.g.editFor(r.size.X, r.size.Y, len(divs)); err != nil {
				return err
			}
			return fmt.Errorf("ai/imagegen: %s draws from no %d reference(s) at %d x %d", k.g.name, len(divs), r.size.X, r.size.Y)
		}
		copy(divs, pats[i])
	}
	if len(divs) > 0 && !p.HasReferences(divs) {
		how := "tools/amd/iron_flux2klein.py --refs compiled their rows"
		if flux2klein.IsProgramExport(k.dir) {
			how = "models.yaml npu.apple edits name them (at their sizes)"
		}
		return fmt.Errorf("ai/imagegen: the export %s does not run these references (%s) at %d x %d; it runs %s, and the others once %s",
			filepath.Base(k.dir), referenceNames(divs), p.C.Width, p.C.Height, strings.Join(p.References(), ", "), how)
	}
	// the references' latents: those given, then the pictures' (the VAE's encoder)
	refs := append([][]float32(nil), c.RefLatents...)
	if len(o.References) > 0 {
		r.phase(imagegen.PhaseReferences)
	}
	t1 := time.Now()
	for i, img := range o.References {
		if img == nil {
			return errors.New("ai/imagegen: a nil reference")
		}
		t := time.Now()
		z, err := p.EncodeImage(ctx, img, divs[i])
		if err != nil {
			return err
		}
		size := ""
		if divs[i] > 1 {
			size = fmt.Sprintf(" at %dx%d", p.C.Width/divs[i], p.C.Height/divs[i])
		}
		k.g.logf("reference %d%s in %.2f s", i+1, size, time.Since(t).Seconds())
		if dump != "" { // the reference's latents, to check the VAE's encoder
			if err := writeF32(filepath.Join(dump, fmt.Sprintf("ref_%d.f32", len(refs))), z); err != nil {
				return err
			}
		}
		refs = append(refs, z)
	}
	r.out.References = time.Since(t1)
	if n := len(refs); n > 0 {
		k.g.logf("editing: %d reference(s)", n)
	}
	r.phase(imagegen.PhasePrompt)
	t1 = time.Now()
	conds, err := p.EncodeAll(ctx, r.prompts...)
	if err != nil {
		return err
	}
	r.out.Prompts = time.Since(t1)
	sig := p.Sigmas(o.Steps)
	if c.NoSteps { // a check of the VAE alone: Noise holds final latents
		sig = sig[:1]
	}
	// Batch pictures at a time, their steps together (a group of the DiT's blocks runs every
	// picture before the next group's weights are read, or its program loads), then the VAE: on
	// Apple's Neural Engine its stages over all of them, on AMD's NPU a picture after another (a
	// stage hands its output to the next on the device)
	batch := max(1, o.Batch)
	for lo := 0; lo < len(r.prompts); lo += batch {
		hi := min(lo+batch, len(r.prompts))
		xs := r.batch(lo, hi, len(sig)-1, p.Noise)
		xs, err = p.GenerateAll(ctx, conds[lo:hi], refs, xs, sig, r.stepsOf(sig))
		if err != nil {
			return err
		}
		r.latents(lo, xs...)
		r.phase(imagegen.PhaseDecode)
		p.Stage = r.stage
		t := time.Now()
		imgs, err := p.DecodeAll(ctx, xs...)
		p.Stage = nil
		if err != nil {
			return err
		}
		if err := r.drawn(lo, time.Since(t), imgs...); err != nil {
			return err
		}
	}
	return nil
}

func (k *klein) close() error {
	if k.p == nil {
		return nil
	}
	err := k.p.Close()
	k.p = nil
	return err
}

// amdEdits are the references of the edits of models.yaml's npu.amd edits, by the sizes of the
// export (as the aspects as name them): those of the picture's size ("2") and of half its sides
// ("2@half"), as Apple's export holds them (flux2klein.ParseReferences). The sizes the table
// leaves out draw from any number of the picture's size.
func amdEdits(m chat.Model, as []aspect.Aspect, sizes []image.Point) (map[image.Point][][]int, error) {
	edits := m.NPU["amd"].Edits
	if len(edits) == 0 {
		return nil, nil
	}
	out := map[image.Point][][]int{}
	for _, a := range as {
		names, ok := edits[a.Name]
		pt := image.Pt(a.Width, a.Height)
		if !ok || !slices.Contains(sizes, pt) {
			continue
		}
		for _, n := range names {
			divs, err := flux2klein.ParseReferences(n)
			if err != nil {
				return nil, fmt.Errorf("%s: models.yaml npu amd edits %s: %w", m.Name, a.Name, err)
			}
			out[pt] = append(out[pt], slices.Sorted(slices.Values(divs)))
		}
	}
	return out, nil
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

// writeF32 writes xs as raw float32.
func writeF32(path string, xs []float32) error {
	b := make([]byte, 4*len(xs))
	for i, x := range xs {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return os.WriteFile(path, b, 0o644)
}
