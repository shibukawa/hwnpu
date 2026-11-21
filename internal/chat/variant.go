package chat

import (
	"cmp"
	"encoding/hex"
	"fmt"
	"slices"
	"strings"
)

// Run is how hwnpu-chat runs a chat model when its flags do not say: models.yaml's prefill,
// format, context, paged and tier, the defaults of --prefill, --format, --context, --paged and
// --tier ("" and 0: hwnpu-chat's own).
type Run struct {
	Prefill string // PrefillAuto, PrefillOn or PrefillOff
	Format  string // the weight format of its programs (lut8, int8, ...)
	Context int    // tokens of a prompt and its answer together
	Paged   int    // the chunk length to split it at its attention with, -1 never
	// Tier is the caches shorter than the context to run on while a conversation is short, as
	// --tier takes them (Tiers): auto, none or a list. LFM3-1.7B has none on a machine of 8 GB,
	// where the whole program of short conversations would hold its weights a second time
	// (3.1 GB against 1.7), and auto from 16 GB.
	Tier string
	// Codes is the hash of the lookup tables chosen against the activations (GPTQ) that its
	// programs are converted with (Setup.CodesHash): Qwen2.5-Coder's 4 bit tables, which
	// rounding alone leaves far from the model (docs/design/llm-qwen.md).
	Codes string
}

// over is r with the fields o gives.
func (r Run) over(o Run) Run {
	return Run{Prefill: cmp.Or(o.Prefill, r.Prefill), Format: cmp.Or(o.Format, r.Format),
		Context: cmp.Or(o.Context, r.Context), Paged: cmp.Or(o.Paged, r.Paged), Tier: cmp.Or(o.Tier, r.Tier),
		Codes: cmp.Or(o.Codes, r.Codes)}
}

// zero reports whether r gives nothing.
func (r Run) zero() bool { return r == Run{} }

// Variant is a model's settings on a machine of more memory than its Machine (models.yaml's
// variants): what it takes running and how it runs, the fields it gives replacing the model's.
// Qwen3-1.7B reads prompts with a program of its own from 16 GB and with the one that generates
// on 8 GB; LFM3-1.7B has its whole program of short conversations from 16 GB.
type Variant struct {
	Memory PerVendor // nil: the model's
	Run    Run
	// AMDExport replaces the model's amd_export (nil: the model's): gpt-oss-20b keeps 16 of its
	// 32 experts a layer on AMD's NPU on 16 GB, all of them from 24 GB
	AMDExport []string
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
	if v.AMDExport != nil {
		m.AMDExport = v.AMDExport
	}
	return m
}

// RunEntry is a model's or a variant's fields of Run in a definition file.
type RunEntry struct {
	Prefill *string `yaml:"prefill"`
	Format  *string `yaml:"format"`
	Context *int    `yaml:"context"`
	Paged   *int    `yaml:"paged"`
	Tier    *string `yaml:"tier"`
	Codes   *string `yaml:"codes"`
}

// variantEntry is a variant in a definition file: variants: {16: {memory: 2.3, prefill: auto}}.
type variantEntry struct {
	Memory    *PerVendor `yaml:"memory"`
	AMDExport *string    `yaml:"amd_export"`
	RunEntry  `yaml:",inline"`
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
	if e.Codes != nil {
		r.Codes = strings.TrimSpace(*e.Codes)
		if _, err := hex.DecodeString(r.Codes); err != nil || len(r.Codes) != 8 {
			return r, fmt.Errorf("codes %q is not the 8 hex digits of a codes directory's hash", r.Codes)
		}
	}
	if e.Tier != nil {
		if r.Tier = strings.TrimSpace(*e.Tier); r.Tier == "" {
			return r, fmt.Errorf("tier is empty (none for no shorter caches)")
		}
		if _, err := Tiers(r.Tier); err != nil {
			return r, fmt.Errorf("tier %q is not auto, none or a list of caches: %w", r.Tier, err)
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
			return nil, fmt.Errorf("variant of %d GB: prefill, format, context, paged, tier and codes are for chat models (hwnpu-chat)", gb)
		}
		v := Variant{Run: r}
		if e.Memory != nil {
			v.Memory = *e.Memory
		}
		if e.AMDExport != nil {
			switch {
			case len(m.AMDExport) == 0:
				return nil, fmt.Errorf("variant of %d GB: amd_export, but the model has none", gb)
			case len(strings.Fields(*e.AMDExport)) == 0:
				return nil, fmt.Errorf("variant of %d GB: amd_export is empty", gb)
			}
			v.AMDExport = strings.Fields(*e.AMDExport)
		}
		if v.Memory == nil && r.zero() && v.AMDExport == nil {
			return nil, fmt.Errorf("variant of %d GB gives nothing", gb)
		}
		out[gb] = v
	}
	return out, nil
}
