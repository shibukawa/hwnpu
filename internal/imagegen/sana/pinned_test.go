package sana

import (
	"testing"

	"github.com/shibukawa/hwnpu/internal/chat"
)

// TestFilesPinned checks that models.yaml pins the sha256 of every file Prepare downloads for each
// Sana of the catalog (hwnpu-image passes the model's repository as Setup.Repo).
func TestFilesPinned(t *testing.T) {
	if err := chat.LoadModels(nil, false, nil); err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, m := range chat.Catalog {
		if m.Family != "sana" {
			continue
		}
		n++
		for _, f := range Files {
			if _, ok := chat.Pinned(m.Repo, f.Name); !ok {
				t.Errorf("%s: %s is not pinned", m.Name, f.Name)
			}
		}
	}
	if n == 0 {
		t.Fatal("no Sana in the catalog")
	}
}
