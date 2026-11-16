package store

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/cache"
)

// TestClean removes what writebacks cut off left, the same in a folder, on WebDAV and in S3, and
// leaves what is whole, what is being written and what is not hwnpu's.
func TestClean(t *testing.T) {
	t.Setenv(cache.RootEnv, t.TempDir())
	_, s3 := fakeS3(t, "pre")
	for _, c := range []struct {
		name string
		st   Store
	}{
		{"folder", Dir{Name: "share", Root: t.TempDir()}},
		{"webdav", openDAV(t, newDAV(t))},
		{"s3", s3},
	} {
		t.Run(c.name, func(t *testing.T) { testClean(t, Source{Store: c.st, Writeback: true}) })
	}
}

func testClean(t *testing.T, s Source) {
	ctx := context.Background()
	defer func(d time.Duration) { WritingStale = d }(WritingStale)
	WritingStale = 2 * time.Second // what was written before the pause below is old (WebDAV's times are to the second)
	put := func(key, body string) {
		t.Helper()
		if err := s.Store.Write(ctx, key, []byte(body)); err != nil {
			t.Fatal(err)
		}
	}
	mark := func(key string, seen time.Time) {
		b, _ := json.Marshal(writingMark{Host: "h", PID: 1, Started: seen, Seen: seen})
		put(key, string(b))
	}
	old := time.Now().Add(-2 * WritingStale)
	// an entry that is whole, with a temporary file of an upload cut off beside it
	exp := filepath.Join(t.TempDir(), "e")
	write(t, filepath.Join(exp, "w.bin"), "weights")
	whole := describeAt(t, exp, "converted/m/t/m.x/v0-aaaaaaaa", map[string]string{"x": "1"})
	if err := PutEntry(ctx, s, exp); err != nil {
		t.Fatal(err)
	}
	put(whole.Path+"/.put-0123456789abcdef", "half a bank")
	// a writeback cut off (no entry.json): an old marker, files and a temporary one
	cut := "converted/m/t/m.x/v0-bbbbbbbb"
	mark(cut+"/"+WritingPrefix+"1", old)
	put(cut+"/w.bin", "some")
	put(cut+"/.put-fedcba9876543210", "more")
	// one no marker is in at all (a hwnpu before markers)
	bare := "converted/n/t/n.x/v1"
	put(bare+"/sub/a.bin", "a")
	busy := "converted/m/t/m.x/v0-cccccccc"
	put(busy+"/w.bin", "going")
	// a repository: the lock names a.bin
	sum := strings.Repeat("ab", 32)
	if err := WriteLock(ctx, s, Lock{Repo: "lab/r", Commit: strings.Repeat("1", 40), Files: map[string]LockedFile{"a.bin": {SHA256: sum, Size: 1}}}); err != nil {
		t.Fatal(err)
	}
	put("original/lab/r/a.bin", "a")
	put("original/lab/r/dir/b.bin", "not named")
	put("original/lab/r/.put-0000000000000000", "half")
	// one with no lock, and one being written
	put("original/lab/nolock/c.bin", "c")
	put("original/lab/w/d.bin", "d")
	// not hwnpu's
	put(".DS_Store", "x")
	put("notes/readme.txt", "x")
	time.Sleep(WritingStale + 1100*time.Millisecond)
	// being written now: marked, or (a hwnpu before the markers) a file changed just now, where
	// the store says when (S3's fake does not)
	mark(busy+"/"+WritingPrefix+"2", time.Now())
	mark("original/lab/w/"+WritingPrefix+"3", time.Now())
	unmarked := "converted/o/t/o.x/v1"
	put(unmarked+"/.put-1111111111111111", "an upload going on")
	_, fake := s.Store.(*S3)
	times := !fake

	c, err := PlanClean(ctx, s)
	if err != nil {
		t.Fatal(err)
	}
	got := map[string][]string{}
	whole2 := map[string]bool{}
	for _, f := range c.Folders {
		got[f.Path] = f.Files
		whole2[f.Path] = f.Whole
	}
	want := map[string][]string{
		whole.Path:            {whole.Path + "/.put-0123456789abcdef"},
		cut:                   {cut + "/.put-fedcba9876543210", cut + "/" + WritingPrefix + "1", cut + "/w.bin"},
		bare:                  {bare + "/sub/a.bin"},
		"original/lab/r":      {"original/lab/r/.put-0000000000000000", "original/lab/r/dir/b.bin"},
		"original/lab/nolock": {"original/lab/nolock/c.bin"},
	}
	for d, w := range want {
		g := slices.Sorted(slices.Values(got[d]))
		if !slices.Equal(g, w) {
			t.Errorf("%s: %v, want %v", d, g, w)
		}
	}
	if !times {
		want[unmarked] = []string{unmarked + "/.put-1111111111111111"}
	}
	if len(got) != len(want) {
		t.Errorf("folders %v", got)
	}
	if !whole2[cut] || !whole2[bare] || !whole2["original/lab/nolock"] || whole2[whole.Path] || whole2["original/lab/r"] {
		t.Errorf("whole: %v", whole2)
	}
	bytes := int64(len("half a bank") + len("some") + len("more") + len("a") + len("half") + len("not named") + len("c"))
	bytes += int64(len(must(s.Store.Read(ctx, cut+"/"+WritingPrefix+"1"))))
	if !times {
		bytes += int64(len("an upload going on"))
	}
	if c.Bytes() != bytes {
		t.Errorf("bytes %d, want %d", c.Bytes(), bytes)
	}

	// a writeback starts in the cut-off folder before Apply: it is left
	mark(cut+"/"+WritingPrefix+"4", time.Now())
	n, err := c.Apply(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if left := len(want[cut]); n != len(keysOfPlan(c))-left {
		t.Errorf("removed %d", n)
	}
	keys := keysOf(must(listAll(ctx, s.Store, "")))
	for _, k := range []string{cut + "/w.bin", busy + "/w.bin", "original/lab/w/d.bin", whole.Path + "/" + EntryFile, whole.Path + "/w.bin",
		"original/lab/r/a.bin", "original/lab/r/" + LockName, ".DS_Store", "notes/readme.txt"} {
		if !slices.Contains(keys, k) {
			t.Errorf("%s was removed", k)
		}
	}
	for _, k := range []string{whole.Path + "/.put-0123456789abcdef", bare + "/sub/a.bin", "original/lab/r/dir/b.bin",
		"original/lab/r/.put-0000000000000000", "original/lab/nolock/c.bin"} {
		if slices.Contains(keys, k) {
			t.Errorf("%s is left", k)
		}
	}
	if times && !slices.Contains(keys, unmarked+"/.put-1111111111111111") {
		t.Error("an upload going on with no marker was removed")
	}
	if d, ok := s.Store.(Dir); ok {
		if _, err := os.Stat(filepath.Join(d.Root, "converted", "n")); err == nil {
			t.Error("the emptied folders are left")
		}
	}
	if w, ok := s.Store.(*WebDAV); ok {
		if rs, err := w.propfind(ctx, "converted/n/", "0"); err == nil {
			t.Errorf("the emptied folders are left: %v", rs)
		}
	}
	// the entry that is whole is read as before
	if _, err := FetchEntry(ctx, []Source{s}, whole.Path, func(e Entry) bool { return e.Key == whole.Key }, filepath.Join(t.TempDir(), "x"), nil); err != nil {
		t.Errorf("the whole entry after clean: %v", err)
	}
}

func keysOfPlan(c Clean) []string {
	var keys []string
	for _, f := range c.Folders {
		keys = append(keys, f.Files...)
	}
	return keys
}

func must[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

// TestMarkWriting marks the folder of a writeback while it goes on, refreshes the marker and
// removes it at the end; PutEntry leaves none.
func TestMarkWriting(t *testing.T) {
	t.Setenv(cache.RootEnv, t.TempDir())
	ctx := context.Background()
	root := t.TempDir()
	s := Source{Store: Dir{Name: "share", Root: root}, Writeback: true}
	defer func(r time.Duration) { WritingRefresh = r }(WritingRefresh)
	WritingRefresh = 20 * time.Millisecond
	done, err := MarkWriting(ctx, s, "converted/m/t/m.x/v1")
	if err != nil {
		t.Fatal(err)
	}
	markers, _ := filepath.Glob(filepath.Join(root, "converted", "m", "t", "m.x", "v1", WritingPrefix+"*"))
	if len(markers) != 1 {
		t.Fatalf("markers %v", markers)
	}
	var first, later writingMark
	json.Unmarshal(must(os.ReadFile(markers[0])), &first)
	time.Sleep(100 * time.Millisecond)
	json.Unmarshal(must(os.ReadFile(markers[0])), &later)
	if !later.Seen.After(first.Seen) || later.Started != first.Started {
		t.Errorf("not refreshed: %+v then %+v", first, later)
	}
	if busy, _ := writing(ctx, s, []string{strings.TrimPrefix(filepath.ToSlash(markers[0]), filepath.ToSlash(root)+"/")}, time.Now()); !busy {
		t.Error("a fresh marker is not taken as a writeback going on")
	}
	done()
	done()
	if _, err := os.Stat(markers[0]); err == nil {
		t.Error("the marker is left")
	}
	exp := filepath.Join(t.TempDir(), "e")
	write(t, filepath.Join(exp, "w.bin"), "weights")
	describeAt(t, exp, "converted/m/t/m.x/v2", nil)
	if err := PutEntry(ctx, s, exp); err != nil {
		t.Fatal(err)
	}
	for _, k := range keysOf(must(listAll(ctx, s.Store, ""))) {
		if isMarker(k) || isTemp(k) {
			t.Errorf("%s is left after PutEntry", k)
		}
	}
}
