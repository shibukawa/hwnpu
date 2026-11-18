package flux3klein

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/store"
)

// The activation statistics come from Setup.Dir first, else from a source's StatsRepo (a
// repository of no commit) when the file has statsSum; one that differs, or none, is not taken.
func TestStatsFromSources(t *testing.T) {
	t.Setenv(cache.RootEnv, t.TempDir())
	nas := t.TempDir()
	repo := filepath.Join(nas, filepath.FromSlash(store.OriginalKey(StatsRepo, "")))
	os.MkdirAll(repo, 0o755)
	body := []byte("stats of the DiT")
	if err := os.WriteFile(filepath.Join(repo, StatsFile), body, 0o644); err != nil {
		t.Fatal(err)
	}
	sum, err := store.FileSHA256(filepath.Join(repo, StatsFile))
	if err != nil {
		t.Fatal(err)
	}
	lock := `{"format": 1, "repo": "` + StatsRepo + `", "files": {"` + StatsFile + `": {"sha256": "` + sum + `", "size": 16}}}`
	os.WriteFile(filepath.Join(repo, store.LockName), []byte(lock), 0o644)
	store.SetSources([]store.Source{{Store: store.Dir{Name: "nas", Root: nas}}}, false)
	t.Cleanup(func() { store.SetSources(nil, true) })
	ctx := context.Background()

	defer func(s string) { statsSum = s }(statsSum)
	statsSum = sum
	p := stats(ctx, t.TempDir(), nil)
	if b, _ := os.ReadFile(p); !bytes.Equal(b, body) {
		t.Fatalf("stats from the source: %q (%q)", p, b)
	}
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, StatsFile), []byte("mine"), 0o644)
	if p := stats(ctx, dir, nil); p != filepath.Join(dir, StatsFile) {
		t.Errorf("Setup.Dir's file is not taken first: %q", p)
	}
	statsSum = strings.Repeat("0", 64)
	var log bytes.Buffer
	if p := stats(ctx, t.TempDir(), &log); p != "" || !strings.Contains(log.String(), "not taken") {
		t.Errorf("a file of another SHA-256 is taken: %q (%s)", p, log.String())
	}
	store.SetSources(nil, false)
	os.RemoveAll(cache.Root())
	if p := stats(ctx, t.TempDir(), nil); p != "" {
		t.Errorf("with no source: %q", p)
	}
}
