package llm

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm/paged"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// RefreshAttention writes the attention program of a Core AI conversion from before the form a
// kv head at a time (paged.Heads) again, in that form, and removes the programs over shorter
// chunks for AddSmaller to write; it leaves alone a conversion in the form already, one kv head
// for all the heads (the grouped form stays), and another backend's. A conversion from before
// paged.Version is written again on every backend.
func TestRefreshAttention(t *testing.T) {
	old := func(t *testing.T, backend string, sh paged.Shape) (string, *Manifest) {
		t.Helper()
		path := filepath.Join(t.TempDir(), "m.parts.json")
		base := filepath.Join(filepath.Dir(path), "m")
		// as a conversion from before the form: the grouped attention, no AttnForm
		var rec *PartRecord
		var err error
		if backend == BackendCoreAI {
			var ins, outs []mlpackage.Feature
			_, ins, outs, err = writeAI(base+".attn.aimodel", nil, func(b *mlpackage.Builder) ([]mlpackage.Feature, []mlpackage.Feature, error) {
				return paged.Build(b, sh, "grouped", 4, 32)
			})
			if err == nil {
				rec = &PartRecord{Path: base + ".attn.aimodel", Inputs: records(ins...), Output: records(outs[0])[0], State: records(outs[1:]...)}
			}
		} else {
			rec, err = writeAttention(base+".attn", backend, sh, "grouped", 4, 32, "test")
		}
		if err != nil {
			t.Fatal(err)
		}
		man := &Manifest{Model: "test", Format: "fp16", Seq: 4, Cache: 64, Paged: 32, Attention: rec, Backend: backend, AttnVersion: paged.Version,
			Parts: []PartRecord{{Path: filepath.Join(filepath.Dir(path), "m.part1")}}}
		if err := WriteManifest(path, man); err != nil {
			t.Fatal(err)
		}
		if err := AddSmaller(path, "grouped", sh.Scale, 8); err != nil {
			t.Fatal(err)
		}
		if man, err = ReadManifest(path); err != nil {
			t.Fatal(err)
		}
		if len(man.Smaller) != 2 {
			t.Fatalf("%d programs over shorter chunks, want 2 (8 and 16)", len(man.Smaller))
		}
		return path, man
	}
	bytecode := func(t *testing.T, r *PartRecord) string {
		t.Helper()
		b, err := os.ReadFile(filepath.Join(r.Path, "main.hash"))
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	several := paged.Shape{Heads: 16, KVHeads: 4, HeadDim: 8}
	t.Run("a conversion from before the form", func(t *testing.T) {
		path, man := old(t, BackendCoreAI, several)
		// AddSmaller wrote the shorter chunks' in the form already; the manifest's own is grouped
		before, small := bytecode(t, man.Attention), man.Smaller[0].Path
		if err := RefreshAttention(path, "grouped", 0); err != nil {
			t.Fatal(err)
		}
		back, err := ReadManifest(path)
		if err != nil {
			t.Fatal(err)
		}
		if back.AttnForm != paged.Heads || bytecode(t, back.Attention) == before {
			t.Errorf("the attention program is not written again: form %q", back.AttnForm)
		}
		if len(back.Smaller) != 0 {
			t.Errorf("%d programs over shorter chunks kept", len(back.Smaller))
		}
		if _, err := os.Stat(small); err == nil {
			t.Errorf("%s is still there", filepath.Base(small))
		}
		if len(back.Attention.Inputs) != len(man.Attention.Inputs) || back.Attention.Path != man.Attention.Path {
			t.Errorf("the attention program changed its inputs or its path")
		}
		// in the form: nothing more
		again := bytecode(t, back.Attention)
		if err := AddSmaller(path, "grouped", 0, 8); err != nil {
			t.Fatal(err)
		}
		if err := RefreshAttention(path, "grouped", 0); err != nil {
			t.Fatal(err)
		}
		if back, err = ReadManifest(path); err != nil {
			t.Fatal(err)
		}
		if bytecode(t, back.Attention) != again || len(back.Smaller) != 2 {
			t.Errorf("a conversion in the form is written again, or lost its shorter chunks (%d)", len(back.Smaller))
		}
	})
	t.Run("one kv head", func(t *testing.T) {
		path, man := old(t, BackendCoreAI, paged.Shape{Heads: 8, KVHeads: 1, HeadDim: 16, Scale: 1})
		before := bytecode(t, man.Attention)
		if err := RefreshAttention(path, "grouped", 1); err != nil {
			t.Fatal(err)
		}
		back, err := ReadManifest(path)
		if err != nil {
			t.Fatal(err)
		}
		if back.AttnForm != "" || bytecode(t, back.Attention) != before || len(back.Smaller) != 2 {
			t.Errorf("one kv head for all the heads: form %q, %d shorter chunks", back.AttnForm, len(back.Smaller))
		}
	})
	t.Run("a conversion from before the version", func(t *testing.T) {
		path, man := old(t, "", several)
		man.AttnVersion = 0
		if err := WriteManifest(path, man); err != nil {
			t.Fatal(err)
		}
		small := man.Smaller[0].Path
		if err := RefreshAttention(path, "grouped", 0); err != nil {
			t.Fatal(err)
		}
		back, err := ReadManifest(path)
		if err != nil {
			t.Fatal(err)
		}
		if back.AttnVersion != paged.Version || back.AttnForm != "" || len(back.Smaller) != 0 {
			t.Errorf("version %d, form %q, %d shorter chunks: want %d, the model's, none", back.AttnVersion, back.AttnForm, len(back.Smaller), paged.Version)
		}
		if _, err := os.Stat(small); err == nil {
			t.Errorf("%s is still there", filepath.Base(small))
		}
	})
	t.Run("Core ML", func(t *testing.T) {
		path, _ := old(t, "", several)
		if err := RefreshAttention(path, "grouped", 0); err != nil {
			t.Fatal(err)
		}
		back, err := ReadManifest(path)
		if err != nil {
			t.Fatal(err)
		}
		if back.AttnForm != "" || len(back.Smaller) != 2 {
			t.Errorf("Core ML's attention changed: form %q, %d shorter chunks", back.AttnForm, len(back.Smaller))
		}
	})
}
