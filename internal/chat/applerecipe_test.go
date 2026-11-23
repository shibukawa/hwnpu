package chat

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/llm"
	shared "github.com/shibukawa/hwnpu/internal/store"
)

// An export of another version than its model's recipe is not taken as done (made again or
// fetched); one of no version (v0-HASH, before the recipes gave one) is of version 1.
func TestStaleExport(t *testing.T) {
	m, ok := Lookup("FLUX.2-klein-4B")
	if !ok {
		t.Fatal("no FLUX.2-klein-4B")
	}
	v := m.Targets[cache.CoreAITarget]
	if v < 1 {
		t.Fatalf("FLUX.2-klein-4B: Targets[coreai] %d", v)
	}
	dir := t.TempDir()
	for _, c := range []struct {
		kind    string
		version int
		done    bool
	}{{CoreAIKind, v, true}, {CoreAIKind, v + 1, false}, {CoreAIKind, 0, v == 1}, {"other", v + 1, true}} {
		export := filepath.Join(dir, "x.coreai")
		os.RemoveAll(export)
		if err := os.MkdirAll(export, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := shared.WriteEntry(export, shared.Entry{Format: shared.EntryFormat, Kind: c.kind, Model: m.Name, Version: c.version}); err != nil {
			t.Fatal(err)
		}
		if err := cache.WriteDone(export+".done", export); err != nil {
			t.Fatal(err)
		}
		if got := llm.ExportDone(export + ".done"); got != c.done {
			t.Errorf("%s of version %d: done %v, want %v", c.kind, c.version, got, c.done)
		}
	}
}
