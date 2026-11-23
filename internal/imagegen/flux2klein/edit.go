package flux2klein

import (
	"context"
	"fmt"
	"io"
	"path/filepath"
	"runtime/debug"
	"slices"
	"time"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/llm"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagegen/aspect"
)

// On Apple's Neural Engine an edit (reference images) runs programs text to image does not: the
// DiT's groups over the picture's rows and the references' (more rows than any size of text to
// image) and the VAE's encoder. The export holds them for the edits models.yaml npu names
// (ConvertOptions.Edits): the DiT's as more functions of the same programs, the weights once, so
// that one download and one prebuilt hold both.

// EditStep is the DiT's programs of a step with references of the picture's sides over Refs.
type EditStep struct {
	Refs []int `json:"refs"`
	// Device says the programs are the DeviceExport's (ProgramManifest.Device), compiled on the
	// device: an edit of PrebuiltKeys keys or more.
	Device  bool    `json:"device,omitempty"`
	Doubles []Group `json:"doubles"`
	Singles []Group `json:"singles"`
}

// EditSize is the edits of an export for pictures of Width x Height: each step's programs, and
// the encoder's stages of each reference size (by the picture's sides over it).
type EditSize struct {
	Width    int              `json:"width"`
	Height   int              `json:"height"`
	Steps    []EditStep       `json:"steps"`
	Encoders map[int][]string `json:"encoders"`
}

// Step is the step of s with references of the picture's sides over divs (in any order: the
// programs depend on the rows alone, which Convert lays out in the order of the step's Refs).
func (s EditSize) Step(divs []int) (EditStep, bool) {
	want := slices.Sorted(slices.Values(divs))
	for _, st := range s.Steps {
		if slices.Equal(slices.Sorted(slices.Values(st.Refs)), want) {
			return st, true
		}
	}
	return EditStep{}, false
}

// NeedsDevice reports whether a step with references of the picture's sides over divs runs the
// programs of the DeviceExport, and whether they are here (whole: its .done). On AMD's NPU, and
// for the steps of the export itself, it is false.
func (p *Pipeline) NeedsDevice(divs []int) (needs, here bool) {
	if p.prog == nil || p.prog.edit == nil || len(divs) == 0 {
		return false, false
	}
	st, ok := p.prog.edit.Step(divs)
	if !ok || !st.Device {
		return false, false
	}
	dd := filepath.Join(filepath.Dir(p.prog.dir), p.prog.m.Device)
	return true, llm.ExportDone(dd + ".done")
}

// NPUEdits are the edits of m's export for vendor's NPU on generation gen: models.yaml npu's
// edits, at the sizes NPUAspects gives their aspect ratios, in the order of aspect.All.
func NPUEdits(m chat.Model, vendor, gen string) ([]Edit, error) {
	spec := m.NPU[vendor]
	if len(spec.Edits) == 0 {
		return nil, nil
	}
	sizes := spec.SizesOn(gen)
	var out []Edit
	for _, a := range aspect.All {
		names, ok := spec.Edits[a.Name]
		if !ok {
			continue
		}
		wh := sizes[a.Name]
		e := Edit{Width: wh[0], Height: wh[1]}
		for _, n := range names {
			divs, err := ParseReferences(n)
			if err != nil {
				return nil, fmt.Errorf("%s: models.yaml npu %s edits %s: %w", m.Name, vendor, a.Name, err)
			}
			e.Refs = append(e.Refs, divs)
		}
		out = append(out, e)
	}
	return out, nil
}

// AppleOptions are the ConvertOptions of m's export for Apple's Neural Engine on chip gen, drawing
// w x h first: every size and edit of models.yaml npu.apple in one export (the weights once),
// its weight formats unless text or dit name others.
func AppleOptions(m chat.Model, gen string, w, h int, text, dit string) (ConvertOptions, error) {
	as, err := NPUAspects(m, "apple", gen)
	if err != nil {
		return ConvertOptions{}, err
	}
	edits, err := NPUEdits(m, "apple", gen)
	if err != nil {
		return ConvertOptions{}, err
	}
	spec := m.NPU["apple"]
	if text == "" {
		text = spec.Text
	}
	if dit == "" {
		dit = spec.Dit
	}
	return ConvertOptions{Width: w, Height: h, Sizes: aspect.SizesOf(as), Edits: edits, TextFormat: text, DitFormat: dit}, nil
}

// PrepareDevice makes the DeviceExport of s's export ready beside it (Prepare's): the programs of
// the edits of PrebuiltKeys keys or more, which only such an edit needs. From the shared layers,
// else converted from the checkpoint (downloaded when missing). It returns its directory, "" when
// the export has none. The device compiles its programs at their first load.
func PrepareDevice(ctx context.Context, s Setup) (string, error) {
	o := s.ConvertOptions
	o.Only = nil
	if err := o.FromRecipe(s.Name); err != nil {
		return "", err
	}
	o.defaults()
	log := s.Log
	if log == nil {
		log = io.Discard
	}
	out, err := Prepare(ctx, s)
	if err != nil {
		return "", err
	}
	m, err := ReadProgramManifest(out)
	if err != nil {
		return "", err
	}
	if m.Device == "" {
		return "", nil
	}
	dout := DeviceExport(out)
	if llm.FetchExport(ctx, dout, dout+".done") {
		return dout, nil
	}
	if err := fetchCheckpoint(ctx, s.Repo, dout, log); err != nil {
		return "", err
	}
	fmt.Fprintf(log, "converting %s (the edits of a reference of the picture's size): a minute; their first edit then compiles them on the device\n", filepath.Base(dout))
	t0 := time.Now()
	o.Only = map[string]bool{"device": true}
	if err := Convert(chat.RepoDir(s.Repo), out, o, log); err != nil {
		return "", err
	}
	fmt.Fprintf(log, "converted in %s\n", time.Since(t0).Round(time.Second))
	debug.FreeOSMemory()
	if err := cache.WriteDone(dout+".done", dout); err != nil {
		return "", err
	}
	llm.ShareExport(ctx, dout)
	return dout, nil
}
