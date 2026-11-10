package anima

import (
	"testing"

	"github.com/shibukawa/hwnpu/internal/chat"
)

// TestFilesPinned checks that models.yaml pins the sha256 of every file Prepare downloads for each
// Anima of the catalog, so that a download with flipped bits is caught (chat.Fetch checks them).
func TestFilesPinned(t *testing.T) {
	if err := chat.LoadModels(nil, false, nil); err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, m := range chat.Catalog {
		if m.Kind != chat.KindImage || m.Repo != RepoSplit {
			continue
		}
		n++
		if m.File == "" {
			t.Errorf("%s: no file", m.Name)
		}
		for _, f := range Files(m.File) {
			if _, ok := chat.Pinned(f.Repo, f.Name); !ok {
				t.Errorf("%s: %s of %s is not pinned", m.Name, f.Name, f.Repo)
			}
		}
		if d := m.With["diffusers"]; d != RepoDiffusers {
			t.Errorf("%s: with diffusers %q, Prepare reads %s", m.Name, d, RepoDiffusers)
		}
	}
	if n == 0 {
		t.Fatal("no Anima in the catalog")
	}
}
