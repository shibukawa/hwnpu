package chat

import (
	"path/filepath"
	"testing"
)

// A run takes the prebuilt programs of the chips its model's recipe prebuilds for alone
// (model-recipes.yaml apple.prebuilt and archs): not Qwen3's, measured on the M1 and left to the
// device, nor another chip's before it is checked.
func TestPrebuiltTaken(t *testing.T) {
	dir := t.TempDir()
	for _, tc := range []struct {
		model, arch string
		want        bool
	}{
		{"Qwen3.5-0.8B", "h13g", true},
		{"Qwen3.5-0.8B", "h15g", true},
		{"Qwen3.5-0.8B", "h14g", false},
		{"LFM3-1.7B", "h13g", true},
		{"LFM3-1.7B", "h15g", false},
		{"Qwen3-0.6B", "h13g", false},
		{"plamo-3-translate-2b", "h13g", false},
		{"gemma-4-E2B-it", "h13g", false},
	} {
		m, ok := Lookup(tc.model)
		if !ok {
			t.Fatalf("%s: not a model", tc.model)
		}
		export := filepath.Join(dir, m.DirName(), "x.coreai")
		if got := prebuiltTaken(export, tc.arch); got != tc.want {
			t.Errorf("%s for %s: %v, want %v", tc.model, tc.arch, got, tc.want)
		}
	}
	// an export of no model of the definitions takes them
	if !prebuiltTaken(filepath.Join(dir, "nobody", "x.coreai"), "h13g") {
		t.Error("an export of no model: not taken")
	}
}
