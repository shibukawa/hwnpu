package chat

import (
	"encoding/hex"
	"fmt"
	"slices"
	"strings"
)

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
		if !r.Zero() && m.Kind != KindChat {
			return nil, fmt.Errorf("variant of %d GB: prefill, format, context, paged, tier and codes are for chat models (hwnpu-chat)", gb)
		}
		v := Variant{Run: r}
		if e.Memory != nil {
			v.Memory = *e.Memory
		}
		if v.Memory == nil && r.Zero() {
			return nil, fmt.Errorf("variant of %d GB gives nothing", gb)
		}
		out[gb] = v
	}
	return out, nil
}
