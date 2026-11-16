package chat

import (
	"path/filepath"
	"testing"
)

// TestModelOfDirByFile takes, of the models of one directory, the one whose file the export is
// named for: Anima-Turbo's export is Anima-Turbo's, not Anima-Aesthetic's (the first by name).
func TestModelOfDirByFile(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "Anima")
	for export, want := range map[string]string{
		"anima-turbo-v1.1-1024x1024-int8-int8-g4-q512-v4.coreai":     "Anima-Turbo",
		"anima-aesthetic-v1.1-1024x1024-int8-int8-g4-q512-v4.coreai": "Anima-Aesthetic",
		"anima-turbo-v1.1-1024x1024-int8-int8-g4-q512-v4.coreai/dit": "Anima-Turbo",
	} {
		m, d, ok := modelOfDir(filepath.Join(dir, filepath.FromSlash(export)))
		if !ok || m.Name != want || d != dir {
			t.Errorf("%s: %q in %q (%v), want %q", export, m.Name, d, ok, want)
		}
	}
}
