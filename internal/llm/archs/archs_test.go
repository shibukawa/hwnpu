package archs

import (
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
)

// hwnpu-convert makes one flag of each setting name for all the architectures (llm.BindFlags),
// so two architectures may share a name only with the same default.
func TestParamNames(t *testing.T) {
	seen := map[string]string{}
	owner := map[string]string{}
	for _, a := range llm.Archs() {
		for _, p := range a.Params {
			if d, ok := seen[p.Name]; ok && d != p.Default {
				t.Errorf("-%s: default %q for %s, %q for %s", p.Name, d, owner[p.Name], p.Default, a.Name)
			}
			seen[p.Name], owner[p.Name] = p.Default, a.Name
		}
	}
}

// Every architecture has a default weight format that Intel's NPU runs (no lookup tables), and
// one that the Neural Engine runs.
func TestIntelFormats(t *testing.T) {
	for _, a := range llm.Archs() {
		f := a.DefaultFormatFor(nil, "intel")
		if _, _, ok := llm.LinearFormat(f); !ok && f != "fp16" {
			t.Errorf("%s: %q for Intel's NPU", a.Name, f)
		}
		if f := a.DefaultFormatFor(nil, "coreml"); f != a.DefaultFormat(nil) {
			t.Errorf("%s: %q for the Neural Engine, DefaultFormat says %q", a.Name, f, a.DefaultFormat(nil))
		}
	}
	for name, want := range map[string]string{"lfm3": "int4", "gemma4": "int8", "bert": "int8", "plamo-bi": "int8"} {
		if got := llm.Lookup(name).DefaultFormatFor(nil, "intel"); got != want {
			t.Errorf("%s: %s for Intel's NPU, want %s", name, got, want)
		}
	}
}
