package anima

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/bonsaiai/internal/llm/iron"
)

// TestVAEPeak checks the VAE's peak follows run's rule: the stages share a buffer a bank kind, as
// large as the largest stage's, and a stage holds its own arguments beside them.
func TestVAEPeak(t *testing.T) {
	v := &VAE{info: []stageInfo{
		{banks: map[string]int{"scratch": 100}, args: 10},
		{banks: map[string]int{"scratch": 300, "weights": 20}, args: 40},
		{banks: map[string]int{"scratch": 250}, args: 60},
	}}
	if got := v.Peak(); got != 300+20+60 {
		t.Errorf("peak %d, want 380 (the largest scratch and weights banks, the largest arguments)", got)
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
// within the VAE's peak (Qwen-Image's 512² DiT would otherwise set the run's peak, docs/design/image-amd.md).
func TestPrefetchFits(t *testing.T) {
	if !iron.Prefetch {
		t.Skip("BONSAIAI_DEV=iron.prefetch=0")
	}
	root := t.TempDir()
	writeGraph(t, filepath.Join(root, "dit_in"), 10, 6)
	writeGraph(t, filepath.Join(root, "dit_blocks"), 1400, 20)
	stage := []string{"dit_in", "dit_blocks"}
	// the stage pins 16 + 1420 MiB, with the prefetch's bank 2836 MiB and the margin 3220: over PrefetchFloor
	v := &VAE{info: []stageInfo{{args: 3300 << 20}}}
	if !PrefetchFits(v, root, stage, "dit_blocks") {
		t.Error("3220 MiB within a 3300 MiB peak: no prefetch")
	}
	v.info = []stageInfo{{args: 3100 << 20}}
	if PrefetchFits(v, root, stage, "dit_blocks") {
		t.Error("3220 MiB over a 3100 MiB peak: prefetched")
	}
	writeGraph(t, filepath.Join(root, "small"), 400, 20)
	if !PrefetchFits(v, root, []string{"small"}, "small") {
		t.Error("820 MiB and the margin under PrefetchFloor: no prefetch")
	}
	if PrefetchFits(nil, root, stage, "dit_blocks") || PrefetchFits(v, root, []string{"missing"}, "missing") {
		t.Error("prefetched without a VAE or a manifest")
	}
}
