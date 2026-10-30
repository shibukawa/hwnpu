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
