package flux2klein

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime/debug"
	"slices"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/imagegen/aspect"
	"github.com/shibukawa/hwnpu/internal/imagegen/flux2klein/mil"
	"github.com/shibukawa/hwnpu/internal/llm"
)

// On Apple's Neural Engine an edit (reference images) runs programs an export of Convert does not
// hold: the DiT's groups over the picture's rows and the references' (more rows than any size of
// text to image) and the VAE's encoder. They are an export of their own (ConvertEdit), next to
// the export of Convert and read with it (Pipeline.UseEdit), so that drawing from text alone
// downloads, converts and compiles none of them: Core AI compiles every function of a program
// together, and the DiT's groups of an edit would double the first run of every picture.

// EditKind is the kind of manifest.json of an export of ConvertEdit.
const EditKind = "flux2klein-edit"

// EditVersion is the layout of the programs ConvertEdit writes; an export of another is
// converted again.
const EditVersion = 1

// EditOptions select what ConvertEdit writes.
type EditOptions struct {
	// Sizes are the picture sizes (width, height) the edits are drawn at: those of the export of
	// Convert it is read with (ProgramManifest.SizeList), or some of them. Default DefaultSizes.
	Sizes [][2]int
	// Refs are the references a step takes, each a list of the picture's sides over 1 (a
	// reference of the picture's size) or 2 (of half its sides). Default one of the picture's size.
	Refs [][]int
	// DitFormat is the DiT's weight format (default lut8), DoubleGroup and SingleGroup its blocks
	// a program holds (ConvertOptions'). The encoder's weights are fp16.
	DitFormat                string
	DoubleGroup, SingleGroup int
	// Only, when not empty, writes these parts alone (doubles, singles, encoder).
	Only map[string]bool
}

func (o *EditOptions) defaults() {
	c := ConvertOptions{DitFormat: o.DitFormat, DoubleGroup: o.DoubleGroup, SingleGroup: o.SingleGroup}
	c.defaults()
	o.DitFormat, o.DoubleGroup, o.SingleGroup = c.DitFormat, c.DoubleGroup, c.SingleGroup
	if len(o.Sizes) == 0 {
		o.Sizes = DefaultSizes
	}
	if len(o.Refs) == 0 {
		o.Refs = [][]int{{1}}
	}
}

// EditStep is the DiT's programs of a step with references of the picture's sides over Refs.
type EditStep struct {
	Refs    []int   `json:"refs"`
	Doubles []Group `json:"doubles"`
	Singles []Group `json:"singles"`
}

// EditSize is the programs of an export of ConvertEdit for pictures of Width x Height: each
// step's, and the encoder's stages of each reference size (by the picture's sides over it).
type EditSize struct {
	Width    int              `json:"width"`
	Height   int              `json:"height"`
	Steps    []EditStep       `json:"steps"`
	Encoders map[int][]string `json:"encoders"`
}

// EditManifest is manifest.json of an export of ConvertEdit. Its programs are relative to its
// directory.
type EditManifest struct {
	Kind    string            `json:"kind"`
	Backend string            `json:"backend"`
	Version int               `json:"version"`
	Formats map[string]string `json:"formats"`
	// Hidden and Latent are the DiT's width and the patches' channels, which the export of
	// Convert read with it must have.
	Hidden int        `json:"hidden"`
	Latent int        `json:"latent"`
	Sizes  []EditSize `json:"sizes"`
}

// At is the programs of m for pictures of w x h, false when it holds none.
func (m *EditManifest) At(w, h int) (EditSize, bool) {
	for _, s := range m.Sizes {
		if s.Width == w && s.Height == h {
			return s, true
		}
	}
	return EditSize{}, false
}

// Step is the step of s with references of the picture's sides over divs (in any order: the
// programs depend on the rows alone, which ConvertEdit lays out in the order of its Refs).
func (s EditSize) Step(divs []int) (EditStep, bool) {
	want := slices.Sorted(slices.Values(divs))
	for _, st := range s.Steps {
		if slices.Equal(slices.Sorted(slices.Values(st.Refs)), want) {
			return st, true
		}
	}
	return EditStep{}, false
}

// ReadEditManifest reads dir/manifest.json of an export of ConvertEdit.
func ReadEditManifest(dir string) (*EditManifest, error) {
	b, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return nil, err
	}
	var m EditManifest
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	if m.Kind != EditKind {
		return nil, fmt.Errorf("flux2klein: %s is not an export of the edit's programs", dir)
	}
	if m.Version != EditVersion {
		return nil, fmt.Errorf("flux2klein: %s is of layout %d, this hwnpu reads %d: convert it again", dir, m.Version, EditVersion)
	}
	return &m, nil
}

// ConvertEdit writes the edit's programs of the FLUX.2 klein repository root into out (Core AI
// programs and manifest.json): for each size and each of o.Refs the DiT's groups over the
// picture's and the references' rows, and the VAE's encoder in stages at each reference size.
// Each program holds a function per size (SizeFunction), the weights once.
func ConvertEdit(root, out string, o EditOptions, log io.Writer) error {
	o.defaults()
	if log == nil {
		log = io.Discard
	}
	base, err := mil.ReadDit(root, o.Sizes[0][0], o.Sizes[0][1])
	if err != nil {
		return err
	}
	vc, err := mil.ReadVae(root)
	if err != nil {
		return err
	}
	w, err := mil.OpenWeights(root, base)
	if err != nil {
		return err
	}
	defer w.Close()
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}
	m := &EditManifest{Kind: EditKind, Backend: llm.BackendCoreAI, Version: EditVersion,
		Formats: map[string]string{"dit": o.DitFormat, "encoder": "fp16"}, Hidden: base.Hidden, Latent: base.Latent}
	for _, sz := range o.Sizes {
		m.Sizes = append(m.Sizes, EditSize{Width: sz[0], Height: sz[1], Encoders: map[int][]string{}})
	}
	jobs := console.NewJobs(log)
	const ext = ".aimodel"
	// sized writes a program of a function for each of sizes (part(w, h)), unless Only leaves its
	// kind out, and returns its name for each size
	sized := func(kind string, sizes [][2]int, part func(w, h int) (*mil.Part, error), format string) ([]string, error) {
		var names []string
		var fns []llm.AIFunction
		name := ""
		for _, sz := range sizes {
			p, err := part(sz[0], sz[1])
			if err != nil {
				return nil, err
			}
			if name == "" {
				name = p.Name + ext
			}
			f := SizeFunction(sz[0], sz[1])
			names = append(names, name+"#"+f)
			fns = append(fns, llm.AIFunction{Name: f, Model: p, Options: llm.Options{Backend: llm.BackendCoreAI, Format: format}})
		}
		if len(o.Only) > 0 && !o.Only[kind] {
			return names, nil
		}
		jobs.Add(name, func() (string, error) {
			if _, err := llm.WriteAIFunctions(filepath.Join(out, name), w, fns); err != nil {
				return "", fmt.Errorf("flux2klein: %s: %w", name, err)
			}
			return fmt.Sprintf("%s, %d sizes", format, len(fns)), nil
		})
		return names, nil
	}
	dit := func(w, h int, refs []int) (mil.Dit, error) {
		c, err := mil.ReadDit(root, w, h)
		if err != nil {
			return c, err
		}
		c.Refs = refs
		return c, c.CheckRefs()
	}
	for _, refs := range o.Refs {
		if len(refs) == 0 || len(refs) > MaxReferences {
			return fmt.Errorf("flux2klein: a step of %d references (1 to %d)", len(refs), MaxReferences)
		}
		steps := make([]EditStep, len(o.Sizes))
		for i := range steps {
			steps[i].Refs = refs
		}
		for _, gr := range mil.Groups(base.Doubles, o.DoubleGroup) {
			names, err := sized("doubles", o.Sizes, func(w, h int) (*mil.Part, error) {
				c, err := dit(w, h, refs)
				return mil.Doubles(c, gr[0], gr[1]), err
			}, o.DitFormat)
			if err != nil {
				return err
			}
			for i := range steps {
				steps[i].Doubles = append(steps[i].Doubles, Group{names[i], gr[0], gr[1]})
			}
		}
		for _, gr := range mil.Groups(base.Singles, o.SingleGroup) {
			names, err := sized("singles", o.Sizes, func(w, h int) (*mil.Part, error) {
				c, err := dit(w, h, refs)
				return mil.Singles(c, gr[0], gr[1]), err
			}, o.DitFormat)
			if err != nil {
				return err
			}
			for i := range steps {
				steps[i].Singles = append(steps[i].Singles, Group{names[i], gr[0], gr[1]})
			}
		}
		for i := range m.Sizes {
			m.Sizes[i].Steps = append(m.Sizes[i].Steps, steps[i])
		}
	}
	// the encoder of each reference size: one program a stage, a function for each size (a
	// reference of half a picture's sides may be another's whole)
	var encSizes [][2]int
	for _, sz := range o.Sizes {
		for _, refs := range o.Refs {
			for _, d := range refs {
				if es := [2]int{sz[0] / d, sz[1] / d}; !slices.Contains(encSizes, es) {
					encSizes = append(encSizes, es)
				}
			}
		}
	}
	n := len(mil.VaeEncStages(vc, encSizes[0][0], encSizes[0][1]))
	encNames := make([][]string, len(encSizes)) // by size, the stages
	for s := 0; s < n; s++ {
		names, err := sized("encoder", encSizes, func(w, h int) (*mil.Part, error) { return mil.VaeEncProgram(vc, w, h, s), nil }, "fp16")
		if err != nil {
			return err
		}
		for i := range encSizes {
			encNames[i] = append(encNames[i], names[i])
		}
	}
	for i, sz := range o.Sizes {
		for _, refs := range o.Refs {
			for _, d := range refs {
				k := slices.Index(encSizes, [2]int{sz[0] / d, sz[1] / d})
				m.Sizes[i].Encoders[d] = encNames[k]
			}
		}
	}
	// a job takes up to about 1.5 GB (a group of the DiT's blocks read as float32)
	if err := jobs.Run(2 << 30); err != nil {
		return err
	}
	b, err := json.MarshalIndent(m, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(out, "manifest.json"), append(b, '\n'), 0o644)
}

// EditExportName is the directory PrepareEdit writes the edit's programs of name and o into.
func EditExportName(name string, o EditOptions) string {
	o.defaults()
	var refs []string
	for _, r := range o.Refs {
		refs = append(refs, mil.RefsName(r))
	}
	return fmt.Sprintf("%s-edit-%s-%s-%s-v%d.%s", strings.ToLower(name), aspect.Tag(o.Sizes), strings.Join(refs, "+"), o.DitFormat, EditVersion, llm.BackendCoreAI)
}

// PrepareEdit makes the edit's programs of s's model ready in s.Dir for Apple's Neural Engine,
// as Prepare makes the export of Convert ready: from the shared layers, else converted from the
// checkpoint (downloaded when missing). It returns their directory.
func PrepareEdit(ctx context.Context, s Setup, o EditOptions) (string, error) {
	o.Only = nil
	o.defaults()
	log := s.Log
	if log == nil {
		log = io.Discard
	}
	out := filepath.Join(s.Dir, EditExportName(s.Name, o))
	if llm.FetchExport(ctx, out, out+".done") {
		return out, nil
	}
	var want []chat.File
	for _, f := range Files {
		want = append(want, chat.File{Repo: s.Repo, Name: f})
	}
	if err := chat.FetchFiles(ctx, want, log); err != nil {
		return "", err
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if err := os.MkdirAll(s.Dir, 0o755); err != nil {
		return "", err
	}
	if err := os.RemoveAll(out); err != nil {
		return "", err
	}
	fmt.Fprintf(log, "converting %s (the edit's programs: %s, DiT %s): a few minutes; the first edit then compiles them for the NPU\n",
		filepath.Base(out), aspect.Tag(o.Sizes), o.DitFormat)
	t0 := time.Now()
	if err := ConvertEdit(chat.RepoDir(s.Repo), out, o, log); err != nil {
		return "", err
	}
	fmt.Fprintf(log, "converted in %s\n", time.Since(t0).Round(time.Second))
	debug.FreeOSMemory()
	if err := cache.WriteDone(out+".done", out); err != nil {
		return "", err
	}
	llm.ShareExport(ctx, out)
	return out, nil
}
