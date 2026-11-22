package llm

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/cache"
)

// An export made on another machine names the checkpoint in that machine's cache: it is read
// from the same place in this one's.
func TestLocalCheckpoint(t *testing.T) {
	root := t.TempDir()
	cache.SetRoot(root)
	t.Cleanup(func() { cache.SetRoot("") })
	for _, rel := range []string{"original/openai/gpt-oss-20b/model.safetensors.index.json", "derived/Qwen/Qwen3.6-35B-A3B/model.safetensors.index.json"} {
		local := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(local), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(local, []byte("{}"), 0o644); err != nil {
			t.Fatal(err)
		}
		other := filepath.Join("/Users/someone/Library/Caches/hwnpu", rel)
		if got := localCheckpoint(other); got != local {
			t.Errorf("localCheckpoint(%s) = %s, want %s", other, got, local)
		}
		if got := localCheckpoint(local); got != local {
			t.Errorf("localCheckpoint(%s) = %s, want it kept", local, got)
		}
	}
	missing := "/Users/someone/Library/Caches/hwnpu/original/none/model.safetensors"
	if got := localCheckpoint(missing); got != missing {
		t.Errorf("localCheckpoint(%s) = %s, want it kept", missing, got)
	}
}
