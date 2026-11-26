package imagegen

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"image"
	"io"
	"math"
	"slices"
	"strings"
	"sync"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/chat"
)

// Generate is the use of a model that draws pictures for an app: a model, a family or a use of
// the definitions ("image", "FLUX.2-klein-4B", "anima", "flux3-klein"). FLUX.2 klein
// also draws from reference pictures: on AMD's NPU at every size, on Apple's Neural Engine at the
// sizes of models.yaml npu.apple's edits (512 x 512; 1024 x 1024 with Edit, whose programs of a
// reference of 1024 are an export of their own, compiled on the device).
func Generate(name string) ai.Use[*Generator] {
	return ai.Define[*Generator](role, name)
}

// Edit is Generate for an app that draws from reference pictures (Options.References): on Apple's
// Neural Engine it prepares FLUX.2 klein's edits at 1024 x 1024 too (a reference of 1024 and none,
// one or two of 512), an export of their own beside the model's (about 5 GB) that the device
// compiles at the prepare's compile (hours on an M1); Generate draws from references at 512 x 512
// alone there. The other models draw from a prompt alone.
func Edit(name string) ai.Use[*Generator] {
	return ai.Define[*Generator](editRole, name)
}

// The settings and programs a picture model's definition may give (ai.Definition):
//
//	settings: text-format  the text encoder's weight format (Apple's and Intel's conversions)
//	          dit-format   the DiT's weight format
//	          size         the picture's size the export is converted for, where an export holds
//	                       one size (Anima on Intel's NPU): an aspect ratio
//	                       (1:1, 1:1s, 16:9, ...) or a side in pixels; default 1:1
//	          dit          Anima: the diffusion model, a file of split_files/diffusion_models/
//	          group        Anima: the DiT's blocks a program holds
//	          qchunk       Anima: the queries an attention scores at a time (default 512)
//	          resident-dit Anima on AMD's NPU: "true" keeps every group of the DiT's blocks on
//	                       the NPU instead of reading each group's weights before it runs
//	programs: export       an export's directory (Export): of tools/amd/iron_*.py for AMD's NPU,
//	                       or of hwnpu-image convert; run as it is
//
// A model of device cpu (Anima) runs a Core ML export with hwnpu's CPU interpreter
// (slow; for checking without the NPU).
const (
	SettingTextFormat  = "text-format"
	SettingDitFormat   = "dit-format"
	SettingSize        = "size"
	SettingDit         = "dit"
	SettingGroup       = "group"
	SettingQChunk      = "qchunk"
	SettingResidentDiT = "resident-dit"
	ProgramExport      = "export"
)

var (
	role     = newRole("image", false)
	editRole = newRole("image-edit", true)
)

// The families of the models the generator draws with (chat.Model's Family's prefix).
const (
	famKlein = "flux2-klein"
	famFlux3 = "flux3-klein"
	famAnima = "anima"
)

// familyOf is the family of the generator that draws with m, "" for none.
func familyOf(m chat.Model) string {
	for _, f := range []string{famKlein, famFlux3, famAnima} {
		if strings.HasPrefix(m.Family, f) {
			return f
		}
	}
	return ""
}

// newRole is the role of a generator; edit prepares FLUX.2 klein's edits of a reference of the
// picture's size too on Apple's Neural Engine (flux2klein.PrepareDevice: an export of their own,
// compiled on the device), which Generate leaves out.
func newRole(name string, edit bool) *airole.Role {
	return &airole.Role{
		Name:   name,
		Remote: func(h *airpc.Handle) io.Closer { return &Generator{remote: h} },
		Kinds:  []chat.Kind{chat.KindImage},
		// 1: Warm draws a picture of every size at the compile, so that none compiles at its first use
		// 2: Anima, Chroma (since removed) and FLUX.3 klein; FLUX.2 klein's export on Apple's of the eight sizes
		// 3: FLUX.2 klein's export on Apple's of every size and edit in one, and one beside it
		// 4: FLUX.2 klein's export v6 (two single blocks a program): the store's record of v4 is of a
		// layout this hwnpu no longer reads (ai.Store.find does not look at the export's version)
		// 5: an M1 takes a FLUX.2 klein export of its own (models.yaml npu.apple generation_edits:
		// its one reference of a picture of 1024 x 1024 at 512), where the store's record names the
		// other chips', whose prebuilt programs draw noise on it
		Revision: 5,
		Prepare: func(e airole.PrepareEnv) (airole.Files, error) {
			var f airole.Files
			fam := familyOf(e.Base)
			if dir := first(e.Model.Programs[ProgramExport]); dir != "" {
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
			return f, err
		},
		Load: func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
			g := &Generator{name: e.Model, log: e.Log, family: f.Values["family"]}
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
			return g, nil
		},
	}
}

func first(s []string) string {
	if len(s) > 0 {
		return s[0]
	}
	return ""
}

// family is what draws a family's pictures on a Generator: its pipeline, opened for a size
// at a time.
type family interface {
	// draw draws r's pictures at r.size.
	draw(ctx context.Context, r *run) error
	// warm draws a picture of one step at size, which compiles its programs.
	warm(ctx context.Context, r *run) error
	close() error
}

// Generator draws pictures. It draws one at a time; others wait.
type Generator struct {
	remote *airpc.Handle // the model in hwnpud (ai's HWNPU_DAEMON), nil when it is here
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

// Model is the name of the model in the definitions.
func (g *Generator) Model() string {
	if g.remote != nil {
		return airpc.Value[string](g.remote, "Model")
	}
	return g.name
}

// Family is the model's family: flux2-klein, flux3-klein or anima.
func (g *Generator) Family() string {
	if g.remote != nil {
		return airpc.Value[string](g.remote, "Family")
	}
	return g.family
}

// Edits reports whether it draws from reference pictures (Options.References).
func (g *Generator) Edits() bool {
	if g.remote != nil {
		return airpc.Value[bool](g.remote, "Edits")
	}
	return g.edits
}

// Sizes are the picture sizes the model draws on this machine's NPU.
func (g *Generator) Sizes() []image.Point {
	if g.remote != nil {
		return airpc.Value[[]image.Point](g.remote, "Sizes")
	}
	return append([]image.Point(nil), g.sizes...)
}

// EditSizes are the picture sizes it draws from references at on this machine's NPU (none when it
// does not edit).
func (g *Generator) EditSizes() []image.Point {
	if g.remote != nil {
		return airpc.Value[[]image.Point](g.remote, "EditSizes")
	}
	if !g.edits {
		return nil
	}
	if g.editSizes == nil {
		return g.Sizes()
	}
	return append([]image.Point(nil), g.editSizes...)
}

// SizeFor is the size of Sizes nearest w x h in shape; between sizes of one shape, the nearest
// in pixels. A picture of another shape is drawn from by stretching it to this size, or by
// leaving the crop to Generate.
func (g *Generator) SizeFor(w, h int) image.Point {
	if g.remote != nil {
		return airpc.Value[image.Point](g.remote, "SizeFor", w, h)
	}
	return byNearness(g.sizes, w, h)[0]
}

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
func (g *Generator) editFor(w, h, n int) (image.Point, []int, error) {
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

// SizeOf is the size a name of one means for the model, as hwnpu-image's --size takes it: an
// aspect ratio (1:1 for 1024 x 1024, 1:1s for 512 x 512, 2:3, 3:4, 9:16, 3:2, 4:3, 16:9: the
// model's own sizes of them), or a side in pixels for a square. It need not be one of Sizes
// (Generate draws the nearest); an error says what the model takes.
func (g *Generator) SizeOf(spec string) (image.Point, error) {
	if g.remote != nil {
		return airpc.Call1[image.Point](g.remote, context.Background(), "SizeOf", spec)
	}
	w, h, err := g.parse(spec)
	if err != nil {
		return image.Point{}, err
	}
	return image.Pt(w, h), nil
}

// Warm draws a picture of one step at each of Sizes (FLUX.2 klein; the first size alone for the
// others), from a gray reference at the sizes it edits at: its programs are loaded only as a picture needs
// them (the text encoder, the VAE's encoder, the DiT's groups of the step, the VAE's decoder), so
// that the load alone compiles none of them. The compile of "models prepare" (ai.Warmer) calls
// it, so that no picture waits for a compile. On AMD's NPU, whose programs are made at Prepare,
// it does nothing.
func (g *Generator) Warm(ctx context.Context) error {
	if g.remote != nil {
		return airpc.Call0(g.remote, ctx, "Warm")
	}
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
		o := Options{Width: s.X, Height: s.Y, Steps: 1, Seed: 1}
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

// Generate draws a picture from a prompt, and from the references when there are any (an edit).
func (g *Generator) Generate(ctx context.Context, prompt string, o Options) (*image.RGBA, error) {
	p, err := g.GenerateAll(ctx, []string{prompt}, o)
	if err != nil {
		return nil, err
	}
	return p.Images[0], nil
}

// GenerateAll draws a picture for each prompt, the model's programs loaded once for all of them:
// Options.Batch of them at a time where the model draws several together, the i-th of the seed
// Options.Seed + i. It returns them with what drawing them took. An edit's pictures share its
// references.
func (g *Generator) GenerateAll(ctx context.Context, prompts []string, o Options) (*Pictures, error) {
	if len(prompts) == 0 {
		return nil, errors.New("ai/imagegen: no prompt")
	}
	if g.remote != nil {
		// the callbacks of o (Progress, Drawn, ...) are called here as hwnpud draws
		return airpc.Call1[*Pictures](g.remote, ctx, "GenerateAll", prompts, o)
	}
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
	if g.remote != nil {
		return g.remote.Release() // the Runtime in hwnpud unloads it
	}
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
func (g *Generator) logf(format string, args ...any) {
	if g.log != nil {
		fmt.Fprintf(g.log, format+"\n", args...)
	}
}

func init() {
	airole.Register(role)
	airole.Register(editRole)
}
