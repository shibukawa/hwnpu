package imagegen

import (
	"context"
	"errors"
	"fmt"
	"image"
	"io"
	"math"
	"math/rand/v2"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagegen/aspect"
	"github.com/shibukawa/hwnpu/internal/imagegen/flux2klein"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
)

// Generate is the use of a model that draws pictures for an app: a model, a family or a use of
// the definitions ("image", "FLUX.2-klein-4B"). For now the model is FLUX.2 klein; the others
// (Anima, Chroma) are drawn by hwnpu-image generate alone. On AMD's NPU it also draws from
// reference pictures; on Apple's Neural Engine that takes programs of their own (Edit).
func Generate(name string) ai.Use[*Generator] {
	return ai.Define[*Generator](role, name)
}

// Edit is Generate that draws from reference pictures (Options.References) on every NPU: on
// Apple's Neural Engine it prepares the edit's programs too (the DiT over the picture's and the
// reference's rows, and the VAE's encoder), which Generate leaves out: they are converted from
// the checkpoint (a 16 GB download), and compiled at the first edit.
func Edit(name string) ai.Use[*Generator] {
	return ai.Define[*Generator](editRole, name)
}

// isKlein reports whether m is FLUX.2 klein.
func isKlein(m chat.Model) bool { return strings.HasPrefix(m.Family, "flux2-klein") }

var (
	role     = newRole("image", false)
	editRole = newRole("image-edit", true)
)

// newRole is the role of a generator, with the edit's programs on Apple's Neural Engine when edit.
func newRole(name string, edit bool) *airole.Role {
	return &airole.Role{
		Name:  name,
		Kinds: []chat.Kind{chat.KindImage},
		Prepare: func(e airole.PrepareEnv) (f airole.Files, err error) {
			m := e.Model
			if !isKlein(m) {
				return f, fmt.Errorf("ai/imagegen: %s is not drawn by the library yet (FLUX.2 klein is; hwnpu-image generate draws the others)", m.Name)
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
			ctx := chat.WithModel(e.Ctx, m)
			var dir string
			if e.NPU.AMD() {
				// one export of every size, here or in the sources, else made with IRON; it edits too
				dir, err = flux2klein.PrepareAMD(ctx, m, aspect.SizesOf(as), as[0].Width, as[0].Height, e.Log)
				if err != nil {
					return f, err
				}
				f.Set("export", dir)
				return f, nil
			}
			// the export hwnpu-image generate draws with, of the sizes it holds by default
			spec := m.NPU["apple"]
			setup := flux2klein.Setup{Dir: filepath.Join(chat.ModelsDir(), "FLUX.2-klein"), Repo: m.Repo, Name: m.Name,
				ConvertOptions: flux2klein.ConvertOptions{Width: as[0].Width, Height: as[0].Height, TextFormat: spec.Text, DitFormat: spec.Dit}, Log: e.Log}
			if dir, err = flux2klein.Prepare(ctx, setup); err != nil {
				return f, err
			}
			f.Set("export", dir)
			if edit {
				man, err := flux2klein.ReadProgramManifest(dir)
				if err != nil {
					return f, err
				}
				ed, err := flux2klein.PrepareEdit(ctx, setup, flux2klein.EditOptions{Sizes: man.SizeList(), DitFormat: man.Formats["dit"]})
				if err != nil {
					return f, err
				}
				f.Set("edit", ed)
			}
			return f, nil
		},
		Load: func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
			g := &Generator{name: e.Model, dir: f.Path("export"), edit: f.Path("edit"), log: e.Log}
			g.model, _ = chat.Lookup(e.Model)
			if e.NPU.AMD() {
				as, err := flux2klein.NPUAspects(g.model, "amd", flux2klein.NPUGeneration("amd"))
				if err != nil {
					return nil, err
				}
				for _, a := range as {
					g.sizes = append(g.sizes, image.Pt(a.Width, a.Height))
				}
				g.edits = true
			} else {
				// the sizes the export holds (models.yaml's eight are hwnpu-image's, an export each)
				man, err := flux2klein.ReadProgramManifest(g.dir)
				if err != nil {
					return nil, err
				}
				for _, s := range man.SizeList() {
					g.sizes = append(g.sizes, image.Pt(s[0], s[1]))
				}
				g.edits = g.edit != ""
			}
			var err error
			if e.NPU.AMD() {
				g.dev, err = hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
				if err == nil {
					iron.ShareExport(g.dir, func(format string, args ...any) {
						if e.Log != nil {
							fmt.Fprintf(e.Log, format+"\n", args...)
						}
					})
				}
			} else {
				// strict, so that a program off the NPU, or one whose placement is not verified, says why
				g.dev, _, err = llm.OpenDevice(ctx, e.NPU.Device, e.Cache, hwnpu.WithPlacement(hwnpu.PlacementStrict))
			}
			if err != nil {
				return nil, err
			}
			// the first size is opened now, so that preparing compiles it and a missing part says so here
			if _, err := g.open(ctx, g.sizes[0]); err != nil {
				g.dev.Close()
				return nil, err
			}
			return g, nil
		},
	}
}

// Options are how a picture is drawn.
type Options struct {
	// References are pictures the prompt edits or composes from, up to 4: "image 1", "image 2",
	// ... in the prompt. Each is cropped about its centre to the picture's shape and scaled to
	// its size.
	References []image.Image
	// Width and Height are the picture's size wanted: the nearest of Sizes in shape is drawn.
	// 0: the shape of the first reference, else a square.
	Width, Height int
	// Seed is the noise's seed; 0 takes one from the time.
	Seed uint64
	// Steps are the denoising steps; 0 is the model's own (FLUX.2 klein: 4).
	Steps int
	// Progress, when not nil, is called after each step (step from 1) and once more when the
	// picture is decoded (step == steps + 1).
	Progress func(step, steps int)
}

// Generator draws pictures. It draws one at a time; others wait.
type Generator struct {
	mu    sync.Mutex
	dev   *hwnpu.Device
	dir   string
	name  string
	model chat.Model
	sizes []image.Point
	log   io.Writer
	// edit is the edit's programs on Apple's Neural Engine (Edit), edits whether it draws from
	// references
	edit  string
	edits bool

	// p draws pictures of at; another size closes it and opens that size's
	p  *flux2klein.Pipeline
	at image.Point
}

// Model is the name of the model in the definitions.
func (g *Generator) Model() string { return g.name }

// Sizes are the picture sizes the model draws on this machine's NPU.
func (g *Generator) Sizes() []image.Point { return append([]image.Point(nil), g.sizes...) }

// SizeFor is the size of Sizes nearest w x h in shape; between sizes of one shape, the nearest
// in pixels. A picture of another shape is drawn from by stretching it to this size, or by
// leaving the crop to Generate.
func (g *Generator) SizeFor(w, h int) image.Point {
	if w <= 0 || h <= 0 {
		return g.sizes[0]
	}
	best, score := g.sizes[0], math.Inf(1)
	for _, s := range g.sizes {
		shape := math.Abs(math.Log(float64(w)/float64(h)) - math.Log(float64(s.X)/float64(s.Y)))
		area := math.Abs(math.Log(float64(w)*float64(h)) - math.Log(float64(s.X)*float64(s.Y)))
		if sc := shape*100 + area; sc < score {
			best, score = s, sc
		}
	}
	return best
}

// open is the pipeline for pictures of size, opened once and kept until another size is asked.
func (g *Generator) open(ctx context.Context, size image.Point) (*flux2klein.Pipeline, error) {
	if g.p != nil && g.at == size {
		return g.p, nil
	}
	if g.p != nil {
		g.p.Close()
		g.p = nil
	}
	p, err := flux2klein.OpenAt(chat.WithModel(ctx, g.model), g.dev, g.dir, size.X, size.Y, g.log)
	if err != nil {
		return nil, err
	}
	if g.edit != "" {
		if err := p.UseEdit(g.edit); err != nil {
			p.Close()
			return nil, err
		}
	}
	if g.log != nil {
		p.Logf = func(format string, args ...any) { fmt.Fprintf(g.log, format+"\n", args...) }
	}
	g.p, g.at = p, size
	return p, nil
}

// Generate draws a picture from a prompt, and from the references when there are any (an edit).
func (g *Generator) Generate(ctx context.Context, prompt string, o Options) (*image.RGBA, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if len(o.References) > 0 && !g.edits {
		return nil, fmt.Errorf("ai/imagegen: %s draws from references on Apple's Neural Engine with the edit's programs: use imagegen.Edit in place of imagegen.Generate", g.name)
	}
	if len(o.References) > flux2klein.MaxReferences {
		return nil, fmt.Errorf("ai/imagegen: %d references; %s takes up to %d", len(o.References), g.name, flux2klein.MaxReferences)
	}
	w, h := o.Width, o.Height
	if (w <= 0 || h <= 0) && len(o.References) > 0 {
		b := o.References[0].Bounds()
		w, h = b.Dx(), b.Dy()
	}
	ctx = chat.WithModel(ctx, g.model)
	p, err := g.open(ctx, g.SizeFor(w, h))
	if err != nil {
		return nil, err
	}
	divs := make([]int, len(o.References))
	for i := range divs {
		divs[i] = 1
	}
	if len(divs) > 0 && !p.HasReferences(divs) {
		return nil, fmt.Errorf("ai/imagegen: %s here does not draw from %d reference(s) at %d x %d (it runs %s)",
			g.name, len(divs), p.C.Width, p.C.Height, strings.Join(p.References(), ", "))
	}
	refs := make([][]float32, len(o.References))
	for i, img := range o.References {
		if img == nil {
			return nil, errors.New("ai/imagegen: a nil reference")
		}
		if refs[i], err = p.EncodeImage(ctx, img, 1); err != nil {
			return nil, err
		}
	}
	cond, err := p.Encode(ctx, prompt)
	if err != nil {
		return nil, err
	}
	seed := o.Seed
	if seed == 0 {
		seed = rand.Uint64()
	}
	sig := p.Sigmas(o.Steps)
	steps := len(sig) - 1
	x, err := p.Generate(ctx, cond, refs, p.Noise(seed), sig, func(i, n int, _ time.Duration) {
		if o.Progress != nil {
			o.Progress(i+1, n)
		}
	})
	if err != nil {
		return nil, err
	}
	img, err := p.Decode(ctx, x)
	if err != nil {
		return nil, err
	}
	if o.Progress != nil {
		o.Progress(steps+1, steps)
	}
	return img, nil
}

// Request is what a GenerateNode reads: a prompt, and the pictures it edits or composes from.
type Request struct {
	Prompt     string
	References []image.Image
}

// GenerateNode is the node of a generator in a network or a call (flow, ai.App.From,
// ai.NewCall): requests in, a picture out for each. o's References are not used; a request's
// are.
func GenerateNode(u ai.Use[*Generator], o Options) *ai.ModelNode[Request, image.Image, Options] {
	return ai.NewModelNode(u, o, func(ctx context.Context, g *Generator, in flow.Stream[Request], options func() Options) flow.Stream[image.Image] {
		return flow.Map(in, func(c flow.Chunk[Request]) (image.Image, error) {
			o := options()
			o.References = c.Data.References
			img, err := g.Generate(ctx, c.Data.Prompt, o)
			if err != nil {
				return nil, err
			}
			return img, nil
		})
	})
}

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (g *Generator) Close() error {
	g.mu.Lock()
	defer g.mu.Unlock()
	var err error
	if g.p != nil {
		err = g.p.Close()
		g.p = nil
	}
	if e := g.dev.Close(); err == nil {
		err = e
	}
	return err
}
