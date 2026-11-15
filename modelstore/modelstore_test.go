package modelstore

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/configdir"

	shared "github.com/shibukawa/hwnpu/internal/store"
)

func put(t *testing.T, path string, b []byte) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
}

// lockIn puts files of repo in the layout of a source at root, with their lock.
func lockIn(t *testing.T, root, repo string, files map[string]string) {
	t.Helper()
	l := shared.Lock{Format: shared.LockFormat, Repo: repo, Commit: "0123456789abcdef0123456789abcdef01234567", Files: map[string]shared.LockedFile{}}
	for name, body := range files {
		put(t, filepath.Join(root, "original", filepath.FromSlash(repo), filepath.FromSlash(name)), []byte(body))
		h := sha256.Sum256([]byte(body))
		l.Files[name] = shared.LockedFile{SHA256: hex.EncodeToString(h[:]), Size: int64(len(body))}
	}
	b, _ := json.Marshal(l)
	put(t, filepath.Join(root, "original", filepath.FromSlash(repo), shared.LockName), b)
}

// TestLayers reads a file of a source and one of the bundled models, keeps both in the root
// Options give, and keeps off Hugging Face when told.
func TestLayers(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("a request to Hugging Face: %s", r.URL)
		http.NotFound(w, r)
	}))
	defer srv.Close()
	root, share, app := t.TempDir(), t.TempDir(), t.TempDir()
	lockIn(t, share, "lab/a", map[string]string{"config.json": "from the share"})
	lockIn(t, filepath.Join(app, "models"), "lab/b", map[string]string{"config.json": "bundled"})
	off := false
	cd := configdir.New("", "app")
	cd.ResourcePath = app
	s, err := Open(Options{Root: root, Sources: []string{"file://" + filepath.ToSlash(share)}, HuggingFace: &off, Bundle: &cd})
	if err != nil {
		t.Fatal(err)
	}
	ls := s.Layers()
	if len(ls) != 3 || !ls[0].Writeback || ls[1].Bundled || !ls[2].Bundled {
		t.Fatalf("layers %+v", ls)
	}
	ctx := context.Background()
	for repo, want := range map[string]string{"lab/a": "from the share", "lab/b": "bundled"} {
		p, err := s.File(ctx, repo, "config.json")
		if err != nil {
			t.Fatal(err)
		}
		if b, _ := os.ReadFile(p); string(b) != want || p != filepath.Join(root, "original", filepath.FromSlash(repo), "config.json") || filepath.Dir(p) != s.Dir(repo) {
			t.Errorf("%s: %s holds %q", repo, p, b)
		}
	}
	if _, err := s.File(ctx, "lab/a", "missing.json"); !NotFound(err) {
		t.Errorf("a file no layer has: %v", err)
	}
}

// TestNoRoom fails a download that would put the cache over its limit, and Prune frees it.
func TestNoRoom(t *testing.T) {
	root, share := t.TempDir(), t.TempDir()
	lockIn(t, share, "lab/a", map[string]string{"config.json": "from the share"})
	lockIn(t, share, "lab/b", map[string]string{"config.json": strings.Repeat("x", 400)})
	off := false
	s, err := Open(Options{Root: root, Sources: []string{"file://" + filepath.ToSlash(share)}, HuggingFace: &off, Limits: &Limits{MaxSize: 300}})
	if err != nil {
		t.Fatal(err)
	}
	defer Open(Options{})
	ctx := context.Background()
	if _, err := s.File(ctx, "lab/a", "config.json"); err != nil {
		t.Fatal(err)
	}
	// lab/a, used now, is kept: lab/b does not fit
	if _, err := s.File(ctx, "lab/b", "config.json"); !errors.Is(err, ErrNoRoom) {
		t.Fatalf("over the limit: %v", err)
	}
	u, err := s.Usage()
	if err != nil || len(u.Units) != 1 || u.Units[0].Name != "original/lab/a" {
		t.Fatalf("usage %+v, %v", u, err)
	}
	if units, err := s.PlanPrune(PruneOptions{MaxSize: 1, Keep: -1}); err != nil || len(units) != 0 {
		t.Errorf("a unit this process uses is pruned: %v, %v", units, err)
	}
}
