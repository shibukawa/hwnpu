package chat

import (
	"cmp"
	"fmt"
	"slices"
	"strings"
)

// Run is how hwnpu-chat runs a chat model when its flags do not say: models.yaml's prefill,
// format, context and paged, the defaults of --prefill, --format, --context and --paged ("" and
// 0: hwnpu-chat's own).
type Run struct {
	Prefill string // PrefillAuto, PrefillOn or PrefillOff
	Format  string // the weight format of its programs (lut8, int8, ...)
	Context int    // tokens of a prompt and its answer together
	Paged   int    // the chunk length to split it at its attention with, -1 never
}

// over is r with the fields o gives.
func (r Run) over(o Run) Run {
	return Run{Prefill: cmp.Or(o.Prefill, r.Prefill), Format: cmp.Or(o.Format, r.Format),
		Context: cmp.Or(o.Context, r.Context), Paged: cmp.Or(o.Paged, r.Paged)}
}

// zero reports whether r gives nothing.
func (r Run) zero() bool { return r == Run{} }

// Variant is a model's settings on a machine of more memory than its Machine (models.yaml's
// variants): what it takes running and how it runs, the fields it gives replacing the model's.
// Qwen3-1.7B reads prompts with a program of its own from 16 GB, where the 3.26 GiB of its two
// programs fit, and with the one that generates on 8 GB.
type Variant struct {
	Memory PerVendor // nil: the model's
	Run    Run
}

// On is the model as a machine of machine GB runs it: with the variant of the largest memory the
// machine has (Variants), or as it is when none is that small.
func (m Model) On(machine int) Model {
	best := 0
	for gb := range m.Variants {
		if gb <= machine && gb > best {
			best = gb
		}
	}
	if best == 0 {
		return m
	}
	v := m.Variants[best]
	if v.Memory != nil {
		m.MemoryBy = m.MemoryBy.over(v.Memory)
		m.Memory = m.MemoryBy.For("")
	}
	m.Run = m.Run.over(v.Run)
	return m
}

// RunEntry is a model's or a variant's fields of Run in a definition file.
type RunEntry struct {
	Prefill *string `yaml:"prefill"`
	Format  *string `yaml:"format"`
	Context *int    `yaml:"context"`
	Paged   *int    `yaml:"paged"`
}

// variantEntry is a variant in a definition file: variants: {16: {memory: 7.6, prefill: auto}}.
type variantEntry struct {
	Memory   *PerVendor `yaml:"memory"`
	RunEntry `yaml:",inline"`
}

// toRun checks a definition's fields of Run.
func toRun(e RunEntry) (Run, error) {
	var r Run
	if e.Prefill != nil {
		if r.Prefill = *e.Prefill; !slices.Contains([]string{PrefillAuto, PrefillOn, PrefillOff}, r.Prefill) {
			return r, fmt.Errorf("prefill %q is not %s, %s or %s", r.Prefill, PrefillAuto, PrefillOn, PrefillOff)
		}
	}
	if e.Format != nil {
		if r.Format = strings.TrimSpace(*e.Format); r.Format == "" {
			return r, fmt.Errorf("format is empty")
		}
	}
	if e.Context != nil {
		if r.Context = *e.Context; r.Context <= 0 {
			return r, fmt.Errorf("context %d is not a positive number of tokens", r.Context)
		}
	}
	if e.Paged != nil {
		if r.Paged = *e.Paged; r.Paged == 0 || r.Paged < -1 {
			return r, fmt.Errorf("paged %d is not a chunk of tokens or -1 (never)", r.Paged)
		}
	}
	return r, nil
}

// toVariants checks a definition's variants: each above the model's machine, giving something,
// and the fields of Run only for a chat model.
func toVariants(m Model, es map[int]*variantEntry) (map[int]Variant, error) {
	out := map[int]Variant{}
	for gb, e := range es {
		switch {
		case e == nil:
			return nil, fmt.Errorf("variant of %d GB gives nothing", gb)
		case m.Machine <= 0:
			return nil, fmt.Errorf("variants need machine, the memory the model itself is for")
		case gb <= m.Machine:
			return nil, fmt.Errorf("variant of %d GB is not above machine (%d GB), which the model itself is for", gb, m.Machine)
		}
		r, err := toRun(e.RunEntry)
		if err != nil {
			return nil, fmt.Errorf("variant of %d GB: %w", gb, err)
		}
		if !r.zero() && m.Kind != KindChat {
			return nil, fmt.Errorf("variant of %d GB: prefill, format, context and paged are for chat models (hwnpu-chat)", gb)
		}
		v := Variant{Run: r}
		if e.Memory != nil {
			v.Memory = *e.Memory
		}
		if v.Memory == nil && r.zero() {
			return nil, fmt.Errorf("variant of %d GB gives nothing", gb)
		}
		out[gb] = v
	}
	return out, nil
}
