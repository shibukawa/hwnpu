package imagegen

import (
	"context"
	"errors"
	"fmt"
	"image"
	"io"
	"path/filepath"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/ai/imagegen/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/catalog"
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
//	          dit-group    Anima: the DiT's blocks a program holds
//	          qchunk       Anima: the queries an attention scores at a time (default 512)
//	programs: export       an export's directory (Export): of tools/amd/iron_*.py for AMD's NPU,
//	                       or of hwnpu-image convert; run as it is
//
// A model of device cpu (Anima) runs a Core ML export with hwnpu's CPU interpreter
// (slow; for checking without the NPU).
const (
	SettingTextFormat = "text-format"
	SettingDitFormat  = "dit-format"
	SettingSize       = "size"
	SettingDit        = "dit"
	SettingDitGroup   = "dit-group"
	SettingQChunk     = "qchunk"
	ProgramExport     = "export"
)

var (
	role     = newRole("image")
	editRole = newRole("image-edit")
)

// newRole is the role of a generator; its Prepare, Load and Usable are the engine's
// (ai/imagegen/engine). image-edit prepares FLUX.2 klein's edits of a reference of the picture's
// size too on Apple's Neural Engine (an export of their own, compiled on the device), which
// Generate leaves out.
func newRole(name string) *airole.Role {
	return &airole.Role{
		Name:   name,
		Remote: func(h *airpc.Handle) io.Closer { return &Generator{p: airpc.RemoteProxy[generatorImpl](h)} },
		Kinds:  []catalog.Kind{catalog.KindImage},
		// 1: Warm draws a picture of every size at the compile, so that none compiles at its first use
		// 2: Anima, Chroma (since removed) and FLUX.3 klein; FLUX.2 klein's export on Apple's of the eight sizes
		// 3: FLUX.2 klein's export on Apple's of every size and edit in one, and one beside it
		// 4: FLUX.2 klein's export v6 (two single blocks a program): the store's record of v4 is of a
		// layout this hwnpu no longer reads (ai.Store.find does not look at the export's version)
		// 5: an M1 takes a FLUX.2 klein export of its own (models.yaml npu.apple generation_edits:
		// its one reference of a picture of 1024 x 1024 at 512), where the store's record names the
		// other chips', whose prebuilt programs draw noise on it
		Revision: 5,
	}
}

// Export is the model an export's directory runs as (an ai.Definition the app defines and uses
// by its ID): an export of tools/amd/iron_flux2klein.py, iron_flux3klein.py or iron_anima.py
// for AMD's NPU, or of hwnpu-image convert (Anima) and the library's conversions for Apple's
// Neural Engine or Intel's NPU. It is like the family's model of the definitions
// (FLUX.2-klein-4B, flux-3-image-klein, Anima-Turbo), which names
// it and its license; the export's own files say how it draws. It reads the export with the
// engine (import _ "github.com/shibukawa/hwnpu/ai/engine").
func Export(dir string) (ai.Definition, error) {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return ai.Definition{}, err
	}
	if bridge.ExportFamily == nil {
		return ai.Definition{}, fmt.Errorf("ai/imagegen: reading the export %s takes the engine (import _ \"github.com/shibukawa/hwnpu/ai/engine\")", dir)
	}
	like := map[string]string{bridge.FamKlein: "FLUX.2-klein-4B", bridge.FamFlux3: "flux-3-image-klein", bridge.FamAnima: "Anima-Turbo"}[bridge.ExportFamily(abs)]
	if like == "" {
		return ai.Definition{}, fmt.Errorf("ai/imagegen: %s is no export of a picture model (FLUX.2 klein, FLUX.3 klein, Anima)", dir)
	}
	return ai.Definition{Like: like, Programs: map[string][]string{ProgramExport: {abs}}}, nil
}

// Generator draws pictures. It draws one at a time; others wait.
type Generator struct {
	p airpc.Proxy[generatorImpl]
}

// generatorImpl is a picture model of the engine in this process: Generator's methods but those
// made of them.
type generatorImpl interface {
	Model() string
	Family() string
	Edits() bool
	Sizes() []image.Point
	EditSizes() []image.Point
	SizeFor(w, h int) image.Point
	SizeOf(spec string) (image.Point, error)
	Warm(ctx context.Context) error
	GenerateAll(ctx context.Context, prompts []string, o Options) (*Pictures, error)
	Close() error
}

// Model is the name of the model in the definitions.
func (g *Generator) Model() string { return airpc.ProxyValue(g.p, "Model", generatorImpl.Model) }

// Family is the model's family: flux2-klein, flux3-klein or anima.
func (g *Generator) Family() string { return airpc.ProxyValue(g.p, "Family", generatorImpl.Family) }

// Edits reports whether it draws from reference pictures (Options.References).
func (g *Generator) Edits() bool { return airpc.ProxyValue(g.p, "Edits", generatorImpl.Edits) }

// Sizes are the picture sizes the model draws on this machine's NPU.
func (g *Generator) Sizes() []image.Point {
	return airpc.ProxyValue(g.p, "Sizes", generatorImpl.Sizes)
}

// EditSizes are the picture sizes it draws from references at on this machine's NPU (none when it
// does not edit).
func (g *Generator) EditSizes() []image.Point {
	return airpc.ProxyValue(g.p, "EditSizes", generatorImpl.EditSizes)
}

// SizeFor is the size of Sizes nearest w x h in shape; between sizes of one shape, the nearest
// in pixels. A picture of another shape is drawn from by stretching it to this size, or by
// leaving the crop to Generate.
func (g *Generator) SizeFor(w, h int) image.Point {
	return airpc.ProxyValue(g.p, "SizeFor", func(l generatorImpl) image.Point { return l.SizeFor(w, h) }, w, h)
}

// SizeOf is the size a name of one means for the model, as hwnpu-image's --size takes it: an
// aspect ratio (1:1 for 1024 x 1024, 1:1s for 512 x 512, 2:3, 3:4, 9:16, 3:2, 4:3, 16:9: the
// model's own sizes of them), or a side in pixels for a square. It need not be one of Sizes
// (Generate draws the nearest); an error says what the model takes.
func (g *Generator) SizeOf(spec string) (image.Point, error) {
	return airpc.ProxyCall1(g.p, context.Background(), "SizeOf", func(l generatorImpl) (image.Point, error) {
		return l.SizeOf(spec)
	}, spec)
}

// Warm draws a picture of one step at each of Sizes (FLUX.2 klein; the first size alone for the
// others), from a gray reference at the sizes it edits at: its programs are loaded only as a picture needs
// them (the text encoder, the VAE's encoder, the DiT's groups of the step, the VAE's decoder), so
// that the load alone compiles none of them. The compile of "models prepare" (ai.Warmer) calls
// it, so that no picture waits for a compile. On AMD's NPU, whose programs are made at Prepare,
// it does nothing.
func (g *Generator) Warm(ctx context.Context) error {
	return airpc.ProxyCall0(g.p, ctx, "Warm", generatorImpl.Warm)
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
	// the callbacks of o (Progress, Drawn, ...) are called here as hwnpud draws
	return airpc.ProxyCall1(g.p, ctx, "GenerateAll", func(l generatorImpl) (*Pictures, error) {
		return l.GenerateAll(ctx, prompts, o)
	}, prompts, o)
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
func (g *Generator) Close() error { return g.p.Close() }

func init() {
	wrap := func(l any) io.Closer { return &Generator{p: airpc.LocalProxy(l.(generatorImpl))} }
	for _, r := range []*airole.Role{role, editRole} {
		airole.Register(r)
		bridge.Roles[r.Name] = r
		bridge.Wrap[r.Name] = wrap
	}
}
