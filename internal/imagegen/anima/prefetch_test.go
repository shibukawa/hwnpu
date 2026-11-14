package anima

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm/iron"
)

// TestVAEPeak checks the VAE's peak follows Decode's rule: a stage with the next beside it only
// where PrefetchLimit lets them load together.
func TestVAEPeak(t *testing.T) {
	v := &VAE{sizes: []int64{100, 300, 250, 50}}
	if got := v.Peak(); got != 550 {
		t.Errorf("no limit: peak %d, want 550 (stages 1 and 2 together)", got)
	}
	v.PrefetchLimit = 400
	if got := v.Peak(); got != 400 {
		t.Errorf("limit 400: peak %d, want 400 (stages 0 and 1)", got)
	}
	v.PrefetchLimit = 200
	if got := v.Peak(); got != 300 {
		t.Errorf("limit 200: peak %d, want 300 (stage 1 alone)", got)
	}
}

// writeGraph writes a graph export's manifest with one bank and one other argument (MiB).
func writeGraph(t *testing.T, dir string, bank, arg int) {
	t.Helper()
	m := map[string]any{
		"version": iron.ManifestVersion, "kind": "graph", "name": filepath.Base(dir),
		"banks": map[string]any{"scratch": map[string]any{"bytes": bank << 20}},
		"graph": map[string]any{
			"elf": "graph.elf", "kernels": []any{map[string]any{"name": "main:sequence"}},
			"arguments": []any{
				map[string]any{"index": 0, "kind": "input", "bytes": arg << 20},
				map[string]any{"index": 1, "kind": "scratch", "bytes": bank << 20},
			},
		},
	}
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), b, 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestPrefetchFits checks the second set of banks is kept only when the stage with it stays
// within the VAE's peak (Qwen-Image's 512² DiT would otherwise set the run's peak, 16.200).
func TestPrefetchFits(t *testing.T) {
	if !iron.Prefetch {
		t.Skip("HWNPU_IRON_PREFETCH=0")
	}
	root := t.TempDir()
	writeGraph(t, filepath.Join(root, "dit_in"), 10, 6)
	writeGraph(t, filepath.Join(root, "dit_blocks"), 400, 20)
	stage := []string{"dit_in", "dit_blocks"}
	// the stage pins 16 + 420 MiB, with the prefetch's bank 836 MiB
	v := &VAE{sizes: []int64{900 << 20}}
	if !PrefetchFits(v, root, stage, "dit_blocks") {
		t.Error("836 MiB within a 900 MiB peak: no prefetch")
	}
	v.sizes = []int64{800 << 20}
	if PrefetchFits(v, root, stage, "dit_blocks") {
		t.Error("836 MiB over an 800 MiB peak: prefetched")
	}
	if PrefetchFits(nil, root, stage, "dit_blocks") || PrefetchFits(v, root, []string{"missing"}, "missing") {
		t.Error("prefetched without a VAE or a manifest")
	}
}
