package qwen35

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Parts of a real Qwen3.5-MoE (HWNPU_QWEN36_DIR: a checkpoint hwnpu-dev mxfp4 wrote) as Core ML
// packages in HWNPU_QWEN36_OUT, for tools/coreai/aicheck run to compare on the Neural Engine with
// the CPU interpreter: the first part (layer 0 to its router) of one token a call, as hwnpu-chat
// splits it.
func TestWriteMoEParts(t *testing.T) {
	dir, out := os.Getenv("HWNPU_QWEN36_DIR"), os.Getenv("HWNPU_QWEN36_OUT")
	if dir == "" || out == "" {
		t.Skip("HWNPU_QWEN36_DIR, HWNPU_QWEN36_OUT")
	}
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	c, err := ParseMoE(raw)
	if err != nil {
		t.Fatal(err)
	}
	w, err := llm.OpenWeights(filepath.Join(dir, "model.safetensors"), MoEArch)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	for name, o := range map[string]llm.Options{
		"first-q1":   {Seq: 1, Cache: 8192, Paged: 8192, Start: 0, Layers: llm.NoLayers},
		"first-q128": {Seq: 128, Cache: 8192, Paged: 8192, Start: 0, Layers: llm.NoLayers},
	} {
		if _, err := llm.WritePart(filepath.Join(out, name+".mlpackage"), c, w, o, func(*mlpackage.Package) {}); err != nil {
			t.Fatal(err)
		}
	}
}
