// Package engine runs ai/imagegen's models in this process: FLUX.2 klein, FLUX.3 klein and Anima
// of internal/imagegen. ai/engine imports it; ai/imagegen alone is their client through hwnpud
// (docs/design/hwnpud.md, "Clients without the engine").
package engine

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"image"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/ai/imagegen"
	"github.com/shibukawa/hwnpu/ai/imagegen/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
)

// The families of the models the generator draws with (chat.Model's Family's prefix).
const (
	famKlein = bridge.FamKlein
	famFlux3 = bridge.FamFlux3
	famAnima = bridge.FamAnima
)

func init() {
	for name, edit := range map[string]bool{"image": false, "image-edit": true} {
		r := bridge.Roles[name]
		r.Usable = currentExport
		r.Prepare = prepare(edit)
		r.Load = load(name)
	}
	bridge.ExportFamily = exportFamily
}

// familyOf is the family of the generator that draws with m, "" for none.
func familyOf(m chat.Model) string {
	for _, f := range []string{famKlein, famFlux3, famAnima} {
		if strings.HasPrefix(m.Family, f) {
			return f
		}
	}
	return ""
}

// prepare is the Prepare of a generator's role; edit prepares FLUX.2 klein's edits of a reference
// of the picture's size too on Apple's Neural Engine (flux2klein.PrepareDevice: an export of their
// own, compiled on the device), which imagegen.Generate leaves out.
func prepare(edit bool) func(e airole.PrepareEnv) (airole.Files, error) {
	return func(e airole.PrepareEnv) (airole.Files, error) {
		var f airole.Files
		fam := familyOf(e.Base)
		if dir := first(e.Model.Programs[imagegen.ProgramExport]); dir != "" {
			fam = exportFamily(dir)
			if fam == "" {
				return f, fmt.Errorf("ai/imagegen: %s is no export of a picture model (FLUX.2 klein, FLUX.3 klein, Anima)", dir)
			}
		}
		var err error
		switch fam {
		case famKlein:
			f, err = prepareKlein(e, edit)
		case famFlux3:
			f, err = prepareFlux3(e)
		case famAnima:
			f, err = prepareAnima(e)
		default:
			return f, fmt.Errorf("ai/imagegen: %s is not drawn by the library (FLUX.2 klein, FLUX.3 klein and Anima are)", e.Model.Name)
		}
		f.SetValue("family", fam)
		f.SetValue("base", e.Base.Name)
		if first(e.Model.Programs[imagegen.ProgramExport]) != "" {
			f.SetValue(valueGiven, "true") // the definition's export, by path: currentExport leaves it
		}
		return f, err
	}
}

// load is the Load of the role name: the generator, as the client's imagegen.Generator.
func load(name string) func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
	return func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
		g := &generator{name: e.Model, log: e.Log, family: f.Values["family"]}
		g.model, _ = chat.Lookup(f.Values["base"])
		if g.model.Name == "" {
			g.model = e.Def
		}
		var err error
		switch g.family {
		case famKlein:
			g.p, err = loadKlein(ctx, g, f, e)
		case famFlux3:
			g.p, err = loadFlux3(ctx, g, f, e)
		case famAnima:
			g.p, err = loadAnima(ctx, g, f, e)
		default:
			err = fmt.Errorf("ai/imagegen: %s was prepared for no family the library draws (%q)", e.Model, g.family)
		}
		if err != nil {
			if g.dev != nil {
				g.dev.Close()
			}
			return nil, err
		}
		return bridge.Wrap[name](g), nil
	}
}

func first(s []string) string {
	if len(s) > 0 {
		return s[0]
	}
	return ""
}

// family is what draws a family's pictures on a generator: its pipeline, opened for a size
// at a time.
type family interface {
	// draw draws r's pictures at r.size.
	draw(ctx context.Context, r *run) error
	// warm draws a picture of one step at size, which compiles its programs.
	warm(ctx context.Context, r *run) error
	close() error
}

// generator draws pictures in this process (imagegen.Generator's). It draws one at a time; others
// wait.
type generator struct {
	mu     sync.Mutex
	dev    *hwnpu.Device // nil on the CPU interpreter
	name   string
	model  chat.Model // the model of the definitions it was made from
	family string
	sizes  []image.Point
	// parse is SizeOf's: a size's name as the family reads it
	parse func(spec string) (w, h int, err error)
	log   io.Writer
	// edits is whether it draws from references; warmAll whether Warm draws every size (FLUX.2
	// klein on Apple's), else the first; amd whether it runs on AMD's NPU, whose programs are made
	// at Prepare (Warm does nothing)
	edits, warmAll, amd bool
	// editSizes are the sizes it draws from references at, nil for every one of sizes; editRefs
	// the references of each size's edits, each the picture's sides over 1 or 2 in order
	// (FLUX.2 klein: models.yaml npu.apple and npu.amd edits). A size it lacks (or a nil
	// editRefs) draws from any number of the picture's size (AMD's sizes the table leaves out),
	// and only when it is the nearest to the size asked
	editSizes []image.Point
	editRefs  map[image.Point][][]int
	p         family
}

func (g *generator) Model() string  { return g.name }
func (g *generator) Family() string { return g.family }
func (g *generator) Edits() bool    { return g.edits }

func (g *generator) Sizes() []image.Point { return append([]image.Point(nil), g.sizes...) }

func (g *generator) EditSizes() []image.Point {
	if !g.edits {
		return nil
	}
	if g.editSizes == nil {
		return g.Sizes()
	}
	return append([]image.Point(nil), g.editSizes...)
}

func (g *generator) SizeFor(w, h int) image.Point { return byNearness(g.sizes, w, h)[0] }

// byNearness is sizes from the nearest w x h in shape, then in pixels (as they are for none).
func byNearness(sizes []image.Point, w, h int) []image.Point {
	out := slices.Clone(sizes)
	if w <= 0 || h <= 0 {
		return out
	}
	score := func(s image.Point) float64 {
		shape := math.Abs(math.Log(float64(w)/float64(h)) - math.Log(float64(s.X)/float64(s.Y)))
		area := math.Abs(math.Log(float64(w)*float64(h)) - math.Log(float64(s.X)*float64(s.Y)))
		return shape*100 + area
	}
	slices.SortStableFunc(out, func(a, b image.Point) int { return cmp.Compare(score(a), score(b)) })
	return out
}

// editFor is the size and the references (the picture's sides over each) of an edit from n
// references for a picture of about w x h: the nearest of EditSizes with an edit of n. A size
// editRefs leaves out takes any number of the picture's size, but only as the nearest: an edit
// of a number the table of the nearest sizes lacks (4 at 1024 x 1024) is not drawn at another
// shape, so it fails as on Apple's.
func (g *generator) editFor(w, h, n int) (image.Point, []int, error) {
	sizes := byNearness(g.EditSizes(), w, h)
	for i, s := range sizes {
		refs, ok := g.editRefs[s]
		if !ok {
			if i > 0 {
				continue
			}
			divs := make([]int, n)
			for i := range divs {
				divs[i] = 1
			}
			return s, divs, nil
		}
		for _, r := range refs {
			if len(r) == n {
				return s, r, nil
			}
		}
	}
	var have []string
	for _, s := range sizes {
		for _, r := range g.editRefs[s] {
			have = append(have, fmt.Sprintf("%s at %d x %d", referenceNames(r), s.X, s.Y))
		}
	}
	return image.Point{}, nil, fmt.Errorf("ai/imagegen: %s here draws from no %d reference(s) (it draws from %s)", g.name, n, strings.Join(have, ", "))
}

func (g *generator) SizeOf(spec string) (image.Point, error) {
	w, h, err := g.parse(spec)
	if err != nil {
		return image.Point{}, err
	}
	return image.Pt(w, h), nil
}

func (g *generator) Warm(ctx context.Context) error {
	if g.amd || g.dev == nil {
		return nil
	}
	sizes := g.Sizes()
	if !g.warmAll {
		sizes = sizes[:1]
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	for _, s := range sizes {
		o := imagegen.Options{Width: s.X, Height: s.Y, Steps: 1, Seed: 1}
		if at, _, err := g.editFor(s.X, s.Y, 1); g.edits && err == nil && at == s {
			ref := image.NewRGBA(image.Rect(0, 0, s.X, s.Y))
			for i := range ref.Pix {
				ref.Pix[i] = 128
			}
			o.References = []image.Image{ref}
		}
		if g.log != nil {
			fmt.Fprintf(g.log, "  compiling the programs of %d x %d: a picture of one step\n", s.X, s.Y)
		}
		r := g.newRun([]string{"a gray picture"}, o)
		if err := g.p.warm(ctx, r); err != nil {
			return fmt.Errorf("ai/imagegen: warming %d x %d: %w", s.X, s.Y, err)
		}
	}
	return nil
}

func (g *generator) GenerateAll(ctx context.Context, prompts []string, o imagegen.Options) (*imagegen.Pictures, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if len(o.References) > 0 && !g.edits {
		if g.family == famKlein {
			return nil, fmt.Errorf("ai/imagegen: %s draws from references on Apple's Neural Engine with the edit's programs: use imagegen.Edit in place of imagegen.Generate", g.name)
		}
		return nil, fmt.Errorf("ai/imagegen: %s draws from a prompt alone (no references)", g.name)
	}
	r := g.newRun(prompts, o)
	if err := g.p.draw(ctx, r); err != nil {
		return nil, err
	}
	return r.finish(), nil
}

func (g *generator) Close() error {
	g.mu.Lock()
	defer g.mu.Unlock()
	var err error
	if g.p != nil {
		err = g.p.close()
		g.p = nil
	}
	if g.dev != nil {
		if e := g.dev.Close(); err == nil {
			err = e
		}
		g.dev = nil
	}
	return err
}

// logf writes a line of the generator's log.
func (g *generator) logf(format string, args ...any) {
	if g.log != nil {
		fmt.Fprintf(g.log, format+"\n", args...)
	}
}

// run is a call of GenerateAll: its prompts, its options and what it took.
type run struct {
	g       *generator
	prompts []string
	o       imagegen.Options
	size    image.Point
	out     imagegen.Pictures
	t0      time.Time
	cur     *imagegen.BatchStats
}

// newRun is a run of the prompts at the size o asks (of the first reference, else the first); an
// edit's at the nearest size with an edit of its references (EditSizes).
func (g *generator) newRun(prompts []string, o imagegen.Options) *run {
	w, h := o.Width, o.Height
	if (w <= 0 || h <= 0) && len(o.References) > 0 && o.References[0] != nil {
		b := o.References[0].Bounds()
		w, h = b.Dx(), b.Dy()
	}
	size := g.SizeFor(w, h)
	if len(o.References) > 0 && g.edits {
		// the nearest size with an edit of these many (the family's draw says when there is none)
		if s, _, err := g.editFor(w, h, len(o.References)); err == nil {
			size = s
		}
	}
	r := &run{g: g, prompts: prompts, o: o, size: size, t0: time.Now()}
	r.out.Size = r.size
	r.out.Images = make([]*image.RGBA, len(prompts))
	r.out.Seeds = make([]uint64, len(prompts))
	return r
}

func (r *run) phase(s string) {
	if r.o.Phase != nil {
		r.o.Phase(s)
	}
}

// checks are the run's Checks, the zero value without.
func (r *run) checks() imagegen.Checks {
	if r.o.Checks == nil {
		return imagegen.Checks{}
	}
	return *r.o.Checks
}

// dumpDir is Checks.Dump, made.
func (r *run) dumpDir() (string, error) {
	d := r.checks().Dump
	if d == "" {
		return "", nil
	}
	return d, os.MkdirAll(d, 0o755)
}

// batch begins the batch of prompts lo to hi of steps steps: the noise of each picture (noise of
// its seed, or Checks.Noise).
func (r *run) batch(lo, hi, steps int, noise func(seed uint64) []float32) [][]float32 {
	r.out.Steps = steps
	b := imagegen.Batch{First: lo, N: hi - lo, Steps: steps}
	xs := make([][]float32, hi-lo)
	for i := lo; i < hi; i++ {
		if n := r.checks().Noise; n != nil {
			xs[i-lo] = append([]float32(nil), n...)
			continue
		}
		s := r.o.Seed
		if s == 0 {
			s = rand.Uint64()
		} else {
			s += uint64(i)
		}
		r.out.Seeds[i] = s
		b.Seeds = append(b.Seeds, s)
		xs[i-lo] = noise(s)
	}
	r.out.Batches = append(r.out.Batches, imagegen.BatchStats{First: lo, N: hi - lo})
	r.cur = &r.out.Batches[len(r.out.Batches)-1]
	r.phase(imagegen.PhaseSteps)
	if r.o.OnBatch != nil {
		r.o.OnBatch(b)
	}
	return xs
}

// step is told step i of n done in d, between sigmas from and to.
func (r *run) step(i, n int, from, to float64, d time.Duration) {
	r.cur.Steps = append(r.cur.Steps, d)
	r.cur.Denoise += d
	if r.o.OnStep != nil {
		r.o.OnStep(imagegen.Step{Step: i, Steps: n, From: from, To: to, Took: d})
	}
	if r.o.Progress != nil {
		r.o.Progress(i+1, n)
	}
}

// stepsOf is a pipeline's step function (i of n, d) over sigmas.
func (r *run) stepsOf(sig []float64) func(i, n int, d time.Duration) {
	return func(i, n int, d time.Duration) {
		from, to := 0.0, 0.0
		if i+1 < len(sig) {
			from, to = sig[i], sig[i+1]
		}
		r.step(i, n, from, to, d)
	}
}

// stage is a pipeline's stage function of the VAE's decoder.
func (r *run) stage(i, n int, d time.Duration) {
	r.cur.Stages = append(r.cur.Stages, d)
	if r.o.OnStage != nil {
		r.o.OnStage(imagegen.Stage{Stage: i, Stages: n, Took: d})
	}
}

// latents tells Checks.Latents the final latents of the pictures from lo.
func (r *run) latents(lo int, xs ...[]float32) {
	if fn := r.checks().Latents; fn != nil {
		for i, x := range xs {
			fn(lo+i, x)
		}
	}
}

// drawn keeps the pictures from lo, decoded in d, and tells them.
func (r *run) drawn(lo int, d time.Duration, imgs ...*image.RGBA) error {
	if len(imgs) == 0 {
		return errors.New("ai/imagegen: no picture decoded")
	}
	r.cur.Decode = d
	for i, img := range imgs {
		r.out.Images[lo+i] = img
	}
	if r.o.Progress != nil {
		r.o.Progress(r.out.Steps+1, r.out.Steps)
	}
	if r.o.Drawn != nil {
		for i, img := range imgs {
			r.o.Drawn(lo+i, img)
		}
	}
	return nil
}

// finish is what the run drew.
func (r *run) finish() *imagegen.Pictures {
	r.out.Took = time.Since(r.t0)
	return &r.out
}

// opened adds the time since t0, of opening a size's programs, to Pictures.Open.
func (r *run) opened(t0 time.Time) { r.out.Open += time.Since(t0) }
