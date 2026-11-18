package store

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/cache"
)

// counted is a folder store that counts the puts of each key.
type counted struct {
	Dir
	mu   sync.Mutex
	puts map[string]int
}

func (c *counted) Put(ctx context.Context, key, src string) error {
	c.mu.Lock()
	c.puts[key]++
	c.mu.Unlock()
	return c.Dir.Put(ctx, key, src)
}

func (c *counted) n(key string) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.puts[key]
}

// TestEarly puts the files of a conversion while it is made: PutEntry then puts only what
// changed, and removes what the conversion removed; what was there before is not put early.
func TestEarly(t *testing.T) {
	t.Setenv(cache.RootEnv, t.TempDir())
	settle, poll := EarlySettle, EarlyPoll
	EarlySettle, EarlyPoll = 100*time.Millisecond, 20*time.Millisecond
	t.Cleanup(func() { EarlySettle, EarlyPoll = settle, poll })
	ctx := context.Background()
	root := t.TempDir()
	st := &counted{Dir: Dir{Name: "nas", Root: root}, puts: map[string]int{}}
	s := Source{Store: st, Writeback: true}
	dir := filepath.Join(t.TempDir(), "x.coreai")
	write(t, filepath.Join(dir, "stale.bin"), "from a conversion cut off")
	old := time.Now().Add(-time.Hour)
	os.Chtimes(filepath.Join(dir, "stale.bin"), old, old)
	rel := EntryPath("x", "any", "x.coreai", "v0-12345678")

	StartEarly(s, rel, dir)
	write(t, filepath.Join(dir, "a", "main.mlirb"), "aaaa")
	write(t, filepath.Join(dir, "b.bin"), "b1")
	write(t, filepath.Join(dir, "gone.bin"), "removed by the conversion")
	write(t, filepath.Join(dir, "w.tmp"), "a temporary file")
	key := func(f string) string { return rel + "/" + f }
	waitFor(t, func() bool { return st.n(key("a/main.mlirb")) == 1 && st.n(key("b.bin")) == 1 && st.n(key("gone.bin")) == 1 })
	if st.n(key("stale.bin")) != 0 || st.n(key("w.tmp")) != 0 {
		t.Errorf("put early: %v", st.puts)
	}
	// the folder has no entry.json yet: a marker keeps Clean off it, even with its files put
	// longer ago than WritingStale (a conversion that compiles for long)
	if !hasMarker(files(t, root), rel+"/") {
		t.Errorf("no marker while files are put early: %v", files(t, root))
	}
	long := time.Now().Add(-2 * WritingStale)
	for _, f := range files(t, root) {
		if !isMarker(f) {
			os.Chtimes(filepath.Join(root, filepath.FromSlash(f)), long, long)
		}
	}
	if c, err := PlanClean(ctx, s); err != nil || len(c.Folders) != 0 {
		t.Errorf("Clean takes a folder being made: %+v %v", c.Folders, err)
	}
	write(t, filepath.Join(dir, "b.bin"), "b2, changed after")
	os.Remove(filepath.Join(dir, "gone.bin"))
	os.Remove(filepath.Join(dir, "w.tmp"))
	os.Remove(filepath.Join(dir, "stale.bin"))
	write(t, filepath.Join(dir, "c.bin"), "the last, written just before the end")
	describeAt(t, dir, rel, map[string]string{"x": "1"})
	if err := PutEntry(ctx, s, dir); err != nil {
		t.Fatal(err)
	}
	if hasMarker(files(t, root), rel+"/") {
		t.Errorf("the marker is left: %v", files(t, root))
	}
	if n := st.n(key("a/main.mlirb")); n != 1 {
		t.Errorf("a/main.mlirb put %d times", n)
	}
	if st.n(key("b.bin")) < 2 || st.n(key("c.bin")) != 1 {
		t.Errorf("puts %v", st.puts)
	}
	got := files(t, root)
	want := []string{key("a/main.mlirb"), key("b.bin"), key("c.bin"), key(EntryFile)}
	if len(got) != len(want) {
		t.Errorf("files %v, want %v", got, want)
	}
	for _, w := range want {
		if !slicesHas(got, w) {
			t.Errorf("no %s in %v", w, got)
		}
	}
	if b, _ := os.ReadFile(filepath.Join(root, filepath.FromSlash(key("b.bin")))); string(b) != "b2, changed after" {
		t.Errorf("b.bin %q", b)
	}
	if _, err := FetchEntry(ctx, []Source{s}, rel, func(Entry) bool { return true }, filepath.Join(t.TempDir(), "y"), nil); err != nil {
		t.Errorf("FetchEntry: %v", err)
	}

	// a source with an entry there already gets nothing before PutEntry
	dir2 := filepath.Join(t.TempDir(), "x.coreai")
	StartEarly(s, rel, dir2)
	write(t, filepath.Join(dir2, "a", "main.mlirb"), "other")
	time.Sleep(4 * EarlySettle)
	if n := st.n(key("a/main.mlirb")); n != 1 {
		t.Errorf("put over an entry there: %d", n)
	}
}

func waitFor(t *testing.T, ok func() bool) {
	t.Helper()
	for end := time.Now().Add(5 * time.Second); time.Now().Before(end); time.Sleep(10 * time.Millisecond) {
		if ok() {
			return
		}
	}
	t.Fatal("timed out")
}
