package store

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/store/s3fake"
)

func write(t *testing.T, p, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func fakeS3(t *testing.T, prefix string) (*s3fake.Server, *S3) {
	t.Helper()
	srv := s3fake.New(t)
	t.Setenv("AWS_ACCESS_KEY_ID", "id")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "secret")
	t.Setenv("AWS_PROFILE", "")
	st, err := OpenS3(S3Config{Name: "lab", Bucket: "models", Prefix: prefix, Endpoint: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	return srv, st
}

// TestStores puts, reads, lists and deletes objects in a folder and in S3 alike.
func TestStores(t *testing.T) {
	ctx := context.Background()
	srv, s3 := fakeS3(t, "cache")
	for _, st := range []Store{Dir{Name: "share", Root: t.TempDir()}, s3} {
		t.Run(st.String(), func(t *testing.T) {
			src := filepath.Join(t.TempDir(), "f")
			write(t, src, "hello")
			if err := st.Put(ctx, "a/b/f", src); err != nil {
				t.Fatal(err)
			}
			if err := st.Write(ctx, "a/c.json", []byte(`{}`)); err != nil {
				t.Fatal(err)
			}
			if n, ok, err := st.Has(ctx, "a/b/f"); err != nil || !ok || n != 5 {
				t.Errorf("Has: %d %v %v", n, ok, err)
			}
			if _, ok, err := st.Has(ctx, "a/none"); err != nil || ok {
				t.Errorf("Has of none: %v %v", ok, err)
			}
			dst := filepath.Join(t.TempDir(), "x", "f")
			if n, err := st.Get(ctx, "a/b/f", dst, nil); err != nil || n != 5 {
				t.Fatalf("Get: %d %v", n, err)
			}
			if b, _ := os.ReadFile(dst); string(b) != "hello" {
				t.Errorf("got %q", b)
			}
			if _, err := st.Get(ctx, "a/none", dst+"2", nil); !isNotFound(err) {
				t.Errorf("Get of none: %v", err)
			}
			keys, err := st.List(ctx, "a/")
			if err != nil || strings.Join(keys, " ") != "a/b/f a/c.json" {
				t.Errorf("List: %v %v", keys, err)
			}
			if err := st.Delete(ctx, "a/b/f"); err != nil {
				t.Fatal(err)
			}
			if _, ok, _ := st.Has(ctx, "a/b/f"); ok {
				t.Error("deleted, still there")
			}
			if _, err := st.Read(ctx, "../x"); err == nil {
				t.Error("a key out of the store was taken")
			}
		})
	}
	if _, ok := srv.Get("models", "cache/a/c.json"); !ok {
		t.Error("the prefix was not used")
	}
	if srv.Signed == 0 {
		t.Error("nothing was signed")
	}
}

func isNotFound(err error) bool {
	return err != nil && strings.Contains(err.Error(), ErrNotFound.Error())
}

// TestS3Large puts a file above the part size in parts, and goes on with a download cut off.
func TestS3Large(t *testing.T) {
	old := partSize
	partSize = 1 << 10
	t.Cleanup(func() { partSize = old })
	ctx := context.Background()
	srv, st := fakeS3(t, "")
	big := bytes.Repeat([]byte("0123456789abcdef"), 200) // 3200 bytes: 4 parts
	src := filepath.Join(t.TempDir(), "big")
	write(t, src, string(big))
	if err := st.Put(ctx, "big", src); err != nil {
		t.Fatal(err)
	}
	if b, _ := srv.Get("models", "big"); !bytes.Equal(b, big) {
		t.Fatalf("the parts made %d bytes", len(b))
	}
	if srv.Requests["POST"] != 2 {
		t.Errorf("%d POST (create and complete)", srv.Requests["POST"])
	}
	dst := filepath.Join(t.TempDir(), "big")
	write(t, dst+".part", string(big[:1000]))
	gets := srv.Requests["GET"]
	if _, err := st.Get(ctx, "big", dst, nil); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(dst); !bytes.Equal(b, big) {
		t.Errorf("went on to %d bytes", len(b))
	}
	if srv.Requests["GET"]-gets != 1 {
		t.Errorf("%d GETs to go on", srv.Requests["GET"]-gets)
	}
}

// TestS3Unsigned reads a public bucket without keys.
func TestS3Unsigned(t *testing.T) {
	srv := s3fake.New(t)
	t.Setenv("AWS_ACCESS_KEY_ID", "")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "")
	t.Setenv("AWS_PROFILE", "none")
	t.Setenv("AWS_SHARED_CREDENTIALS_FILE", filepath.Join(t.TempDir(), "none"))
	if _, err := OpenS3(S3Config{Name: "lab", Bucket: "models", Endpoint: srv.URL}); err == nil || !strings.Contains(err.Error(), "no keys") {
		t.Errorf("no keys: %v", err)
	}
	srv.Put("models", "p/f", []byte("public"))
	st, err := OpenS3(S3Config{Name: "lab", Bucket: "models", Prefix: "p", Endpoint: srv.URL, NoSign: true})
	if err != nil {
		t.Fatal(err)
	}
	b, err := st.Read(context.Background(), "f")
	if err != nil || string(b) != "public" {
		t.Errorf("%q %v", b, err)
	}
	if srv.Signed != 0 {
		t.Error("signed")
	}
	if err := st.Write(context.Background(), "g", nil); err == nil {
		t.Error("a public bucket was written")
	}
}

// TestProfile reads the static keys, the region and the endpoint of a profile.
func TestProfile(t *testing.T) {
	d := t.TempDir()
	write(t, filepath.Join(d, "credentials"), "[default]\naws_access_key_id = D\naws_secret_access_key = DS\n[rustfs]\naws_access_key_id = R\naws_secret_access_key = RS\n")
	write(t, filepath.Join(d, "config"), "[profile rustfs]\nregion = lab-1\ns3 =\n  endpoint_url = http://rustfs:9000\n  addressing_style = path\n")
	t.Setenv("AWS_SHARED_CREDENTIALS_FILE", filepath.Join(d, "credentials"))
	t.Setenv("AWS_CONFIG_FILE", filepath.Join(d, "config"))
	t.Setenv("AWS_PROFILE", "")
	t.Setenv("AWS_ACCESS_KEY_ID", "")
	p := readProfile("rustfs")
	if p.creds.AccessKeyID != "R" || p.creds.SecretAccessKey != "RS" || p.region != "lab-1" || p.endpoint != "http://rustfs:9000" {
		t.Errorf("rustfs: %+v", p)
	}
	if p := readProfile(""); p.creds.AccessKeyID != "D" {
		t.Errorf("default: %+v", p)
	}
	t.Setenv("AWS_ACCESS_KEY_ID", "E")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "ES")
	if p := readProfile(""); p.creds.AccessKeyID != "E" {
		t.Errorf("the environment's keys: %+v", p)
	}
}

// describe gives dir an entry.json at rel.
func describeAt(t *testing.T, dir, rel string, recipe map[string]string) Entry {
	t.Helper()
	e, err := Describe(dir, Entry{Kind: "iron-lm", Scope: ScopePortable, Recipe: recipe, Path: rel})
	if err != nil {
		t.Fatal(err)
	}
	return e
}

// TestEntries puts an export in one source and makes it from there, and checks it.
func TestEntries(t *testing.T) {
	t.Setenv(cache.RootEnv, t.TempDir())
	ctx := context.Background()
	exp := filepath.Join(t.TempDir(), "m.amd-npu2")
	write(t, filepath.Join(exp, "manifest.json"), `{"version":1}`)
	write(t, filepath.Join(exp, "bank.bin"), "weights")
	write(t, filepath.Join(exp, "check", "x.bin"), "weights")
	recipe := map[string]string{"model": "m", "commit": "abc"}
	rel := EntryPath("m", "amd-xdna2", "m.amd-npu2", VersionDir(0, false, Fingerprint(recipe)))
	if !strings.HasPrefix(rel, "converted/m/amd-xdna2/m.amd-npu2/v0-") || len(path.Base(rel)) != 11 {
		t.Fatalf("path %s", rel)
	}
	e := describeAt(t, exp, rel, recipe)
	if e.Key != Fingerprint(recipe) || len(e.Files) != 3 {
		t.Fatalf("%+v", e)
	}
	if err := Verify(exp, true); err != nil {
		t.Fatal(err)
	}
	near, far := Source{Store: Dir{Name: "near", Root: t.TempDir()}}, Source{Store: Dir{Name: "far", Root: t.TempDir()}, Writeback: true}
	ss := []Source{near, far}
	if err := PutEntry(ctx, far, exp); err != nil {
		t.Fatal(err)
	}
	if b, err := os.ReadFile(filepath.Join(far.Store.(Dir).Root, filepath.FromSlash(rel), "check", "x.bin")); err != nil || string(b) != "weights" {
		t.Fatalf("in the source: %q %v", b, err)
	}
	byKey := func(k string) func(Entry) bool { return func(e Entry) bool { return e.Key == k } }
	out := filepath.Join(t.TempDir(), "m.amd-npu2")
	i, err := FetchEntry(ctx, ss, rel, byKey(e.Key), out, nil)
	if err != nil || i != 1 {
		t.Fatalf("%d %v", i, err)
	}
	if err := Verify(out, true); err != nil {
		t.Fatal(err)
	}
	if _, err := FetchEntry(ctx, ss, rel, byKey("other"), out+"2", nil); err != ErrNotFound {
		t.Errorf("another key: %v", err)
	}
	if es, err := ListEntries(ctx, far); err != nil || len(es) != 1 || es[0] != rel {
		t.Errorf("ListEntries: %v %v", es, err)
	}
	// a download cut off before left a file of the right size with other bytes (a reboot before
	// they were on the disk): it is downloaded again
	cut := filepath.Join(t.TempDir(), "m.amd-npu2")
	write(t, filepath.Join(cut+".part", "bank.bin"), "\x00\x00\x00\x00\x00\x00\x00")
	write(t, filepath.Join(cut+".part", "manifest.json"), `{"version":1}`)
	if _, err := FetchEntry(ctx, ss, rel, byKey(e.Key), cut, nil); err != nil {
		t.Fatal(err)
	}
	if err := Verify(cut, true); err != nil {
		t.Errorf("a file of the right size and other bytes was kept: %v", err)
	}
	// a file that changed in the source is found out
	write(t, filepath.Join(far.Store.(Dir).Root, filepath.FromSlash(rel), "bank.bin"), "tampered")
	if _, err := FetchEntry(ctx, ss, rel, byKey(e.Key), out+"4", nil); err == nil {
		t.Error("a changed file was taken")
	}
}

// TestOriginals puts files of a repository in a source with its lock and reads them back by
// commit and SHA-256; a file of another commit is not put.
func TestOriginals(t *testing.T) {
	t.Setenv(cache.RootEnv, t.TempDir())
	ctx := context.Background()
	s := Source{Store: Dir{Name: "nas", Root: t.TempDir()}, Writeback: true}
	ck := filepath.Join(t.TempDir(), "config.json")
	write(t, ck, `{"a":1}`)
	sum, _ := FileSHA256(ck)
	c1, c2 := strings.Repeat("a", 40), strings.Repeat("b", 40)
	if err := PutOriginal(ctx, s, "lab/m", c1, "config.json", ck, ""); err != nil {
		t.Fatal(err)
	}
	if err := PutOriginal(ctx, s, "lab/m", c1, "sub/w.bin", ck, sum); err != nil {
		t.Fatal(err)
	}
	if err := PutOriginal(ctx, s, "lab/m", c2, "other.json", ck, ""); !errors.Is(err, ErrOtherCommit) {
		t.Errorf("another commit: %v", err)
	}
	root := s.Store.(Dir).Root
	// the file of another commit is not put over the lock's (the put goes before the lock's)
	ck2 := filepath.Join(t.TempDir(), "config.json")
	write(t, ck2, `{"b":2}`)
	if err := PutOriginal(ctx, s, "lab/m", c2, "config.json", ck2, ""); !errors.Is(err, ErrOtherCommit) {
		t.Errorf("another commit's config.json: %v", err)
	}
	if b, _ := os.ReadFile(filepath.Join(root, "original", "lab", "m", "config.json")); string(b) != `{"a":1}` {
		t.Errorf("config.json of the lock's commit put over: %q", b)
	}
	if b, _ := os.ReadFile(filepath.Join(root, "original", "lab", "m", "sub", "w.bin")); string(b) != `{"a":1}` {
		t.Errorf("not a plain folder: %q", b)
	}
	clearLocks()
	l, err := ReadLock(ctx, s, "lab/m")
	if err != nil || l.Commit != c1 || len(l.Files) != 2 || l.Files["config.json"].SHA256 != sum || l.Files["config.json"].Size != 7 {
		t.Fatalf("lock: %+v %v", l, err)
	}
	dst := filepath.Join(t.TempDir(), "config.json")
	if i, got, err := FetchOriginal(ctx, []Source{s}, "lab/m", c1, "config.json", "", dst, nil); err != nil || i != 0 || got.Commit != c1 {
		t.Errorf("by commit: %d %v %v", i, got, err)
	}
	os.Remove(dst)
	if _, _, err := FetchOriginal(ctx, []Source{s}, "lab/m", "", "config.json", sum, dst, nil); err != nil {
		t.Errorf("by SHA-256: %v", err)
	}
	os.Remove(dst)
	if _, _, err := FetchOriginal(ctx, []Source{s}, "lab/m", c2, "config.json", "", dst, nil); err != ErrNotFound {
		t.Errorf("another commit: %v", err)
	}
	write(t, filepath.Join(root, "original", "lab", "m", "unlocked.bin"), "x")
	if _, _, err := FetchOriginal(ctx, []Source{s}, "lab/m", "", "unlocked.bin", "", dst, nil); err != ErrNotFound {
		t.Errorf("a file the lock does not name: %v", err)
	}
}

// TestQueue puts an entry in a source in the background and leaves work that fails for later.
func TestQueue(t *testing.T) {
	t.Setenv(cache.RootEnv, t.TempDir())
	ctx := context.Background()
	exp := filepath.Join(t.TempDir(), "e")
	write(t, filepath.Join(exp, "a"), "a")
	describeAt(t, exp, "converted/m/t/e/v0-12345678", map[string]string{"x": "1"})
	s := Source{Store: Dir{Name: "share", Root: t.TempDir()}, Writeback: true}
	SetSources([]Source{s}, true)
	t.Cleanup(func() { SetSources(nil, true) })
	PutLater(s, exp)
	PutLater(Source{Store: Dir{Name: "gone", Root: "/nonexistent"}}, exp) // no such source now: kept
	Flush(ctx)
	if es, _ := ListEntries(ctx, s); len(es) != 1 {
		t.Errorf("entries %v", es)
	}
	left, _ := filepath.Glob(filepath.Join(cache.Sub("work", "queue"), "*.json"))
	if len(left) != 1 {
		t.Errorf("%d jobs left", len(left))
	}
	if b, _ := os.ReadFile(cache.Sub("work", "queue.log")); !strings.Contains(string(b), "no source") {
		t.Errorf("log: %s", b)
	}
}

// TestWait does at the end of a command the work a process before left in the queue, and says
// nothing when there is none.
func TestWait(t *testing.T) {
	t.Setenv(cache.RootEnv, t.TempDir())
	var out strings.Builder
	Wait(&out)
	if out.Len() != 0 {
		t.Errorf("an empty queue: %q", out.String())
	}
	exp := filepath.Join(t.TempDir(), "e")
	write(t, filepath.Join(exp, "a"), "a")
	describeAt(t, exp, "converted/m/t/e/v0-12345678", map[string]string{"x": "1"})
	s := Source{Store: Dir{Name: "share", Root: t.TempDir()}, Writeback: true}
	SetSources([]Source{s}, true)
	t.Cleanup(func() { SetSources(nil, true) })
	j := job{Op: "entry", Source: s.String(), Dir: exp} // left by a process before, not started here
	b, _ := json.Marshal(j)
	write(t, filepath.Join(queueDir(), j.id()+".json"), string(b))
	Wait(&out)
	if es, _ := ListEntries(context.Background(), s); len(es) != 1 {
		t.Errorf("entries %v", es)
	}
	if !strings.Contains(out.String(), "writing 1 back") {
		t.Errorf("said %q", out.String())
	}
}

// TestOriginalsSideBySide puts the files of one repository at once: the lock names them all.
func TestOriginalsSideBySide(t *testing.T) {
	t.Setenv(cache.RootEnv, t.TempDir())
	ctx := context.Background()
	s := Source{Store: Dir{Name: "nas", Root: t.TempDir()}, Writeback: true}
	c := strings.Repeat("a", 40)
	const n = 12
	err := each(ctx, n, n, func(ctx context.Context, i int) error {
		f := filepath.Join(t.TempDir(), "f")
		write(t, f, strings.Repeat("x", i+1))
		return PutOriginal(ctx, s, "lab/m", c, fmt.Sprintf("f%02d.bin", i), f, "")
	})
	if err != nil {
		t.Fatal(err)
	}
	clearLocks()
	if l, err := ReadLock(ctx, s, "lab/m"); err != nil || len(l.Files) != n {
		t.Errorf("lock names %d of %d files: %v", len(l.Files), n, err)
	}
}

func TestGC(t *testing.T) {
	t.Setenv(cache.RootEnv, t.TempDir())
	ctx := context.Background()
	s := Source{Store: Dir{Name: "t", Root: t.TempDir()}}
	mk := func(model, body string) (Entry, string) {
		d := t.TempDir()
		os.WriteFile(filepath.Join(d, "own"), []byte(body), 0o644)
		r := map[string]string{"model": model}
		rel := EntryPath(model, "t", model+".x", VersionDir(0, false, Fingerprint(r)))
		e, err := Describe(d, Entry{Kind: "k", Model: model, Recipe: r, Path: rel})
		if err != nil {
			t.Fatal(err)
		}
		if err := PutEntry(ctx, s, d); err != nil {
			t.Fatal(err)
		}
		return e, rel
	}
	a, ra := mk("a", "a's own")
	b, rb := mk("b", "b's own")
	root := s.Store.(Dir).Root
	half := filepath.Join(root, "converted", "c", "t", "c.x", "v1", "half")
	write(t, half, "a writeback cut off")
	long := time.Now().Add(-2 * WritingStale) // not one being written
	os.Chtimes(half, long, long)
	g, err := PlanGC(ctx, s, func(e Entry) bool { return e.Model == "a" }, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(g.Entries) != 1 || g.Entries[0].Key != a.Key || len(g.Orphans) != 0 || g.Bytes != int64(len("a's own")) {
		t.Fatalf("plan: %+v", g)
	}
	if err := g.Apply(ctx, s); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(root, "converted", "a")); err == nil {
		t.Error("a's folders are left")
	}
	dst := filepath.Join(t.TempDir(), "b")
	if _, err := FetchEntry(ctx, []Source{s}, rb, func(e Entry) bool { return e.Key == b.Key }, dst, nil); err != nil {
		t.Fatalf("b after gc: %v", err)
	}
	if _, err := FetchEntry(ctx, []Source{s}, ra, func(Entry) bool { return true }, dst+"2", nil); !errors.Is(err, ErrNotFound) {
		t.Fatalf("a after gc: %v", err)
	}
	g, _ = PlanGC(ctx, s, func(Entry) bool { return false }, true)
	if len(g.Entries) != 0 || len(g.Orphans) != 1 || g.Orphans[0] != "converted/c/t/c.x/v1/half" {
		t.Fatalf("orphans: %+v", g)
	}
}

// TestMigrate moves the layout before (blobs, upstream, entries) to original/ and converted/.
func TestMigrate(t *testing.T) {
	t.Setenv(cache.RootEnv, t.TempDir())
	ctx := context.Background()
	root := t.TempDir()
	blob := func(body string) string {
		p := filepath.Join(t.TempDir(), "b")
		write(t, p, body)
		sum, _ := FileSHA256(p)
		write(t, filepath.Join(root, "blobs", "sha256", sum[:2], sum), body)
		return sum
	}
	c1, c2 := strings.Repeat("1", 40), strings.Repeat("2", 40)
	for _, f := range []struct{ commit, name, body string }{{c1, "config.json", "{}"}, {c1, "w/model.bin", "weights"}, {c2, "config.json", "{2}"}} {
		sum := blob(f.body)
		write(t, filepath.Join(root, "upstream", "lab", "m@"+f.commit, filepath.FromSlash(f.name)+".json"), fmt.Sprintf(`{"sha256":%q,"size":%d}`, sum, len(f.body)))
	}
	sum := blob("program")
	e := Entry{Format: EntryFormat, Key: strings.Repeat("k", 64), Kind: "coreai", Model: "M", Target: "coreai",
		Recipe: map[string]string{"export": "M.coreai"}, Files: []File{{Path: "main.mlirb", Size: 7, SHA256: sum}}}
	b, _ := json.Marshal(e)
	write(t, filepath.Join(root, "entries", "coreai", e.Key+".json"), string(b))
	place := func(e Entry) (string, bool) {
		return EntryPath(e.Model, e.Target, e.Recipe["export"], VersionDir(0, false, e.Key)), e.Recipe["export"] != ""
	}
	r, err := MigrateDir(ctx, root, place, true, nil)
	if err != nil {
		t.Fatal(err)
	}
	if r.Repos != 1 || r.Files != 2 || r.Entries != 1 || len(r.Skipped) != 1 {
		t.Fatalf("%+v", r)
	}
	s := Source{Store: Dir{Name: "nas", Root: root}}
	clearLocks()
	if l, err := ReadLock(ctx, s, "lab/m"); err != nil || l.Commit != c1 || len(l.Files) != 2 {
		t.Errorf("lock %+v %v", l, err)
	}
	if b, _ := os.ReadFile(filepath.Join(root, "original", "lab", "m", "w", "model.bin")); string(b) != "weights" {
		t.Errorf("model.bin %q", b)
	}
	rel := "converted/M/coreai/M.coreai/v0-kkkkkkkk"
	if got, err := ReadEntryAt(ctx, s, rel); err != nil || got.Path != rel {
		t.Errorf("entry %+v %v", got, err)
	}
	if _, err := os.Stat(filepath.Join(root, "entries")); err == nil {
		t.Error("entries/ is left")
	}
	if _, err := os.Stat(filepath.Join(root, "upstream", "lab", "m@"+c2)); err != nil {
		t.Error("the other commit was removed")
	}
}

// TestParseSource reads the URLs of the sources and their order in the environment.
func TestParseSource(t *testing.T) {
	for _, c := range []struct{ url, want, err string }{
		{"file:///mnt/llm/hwnpu?writeback=true", "folder /mnt/llm/hwnpu", ""},
		{"file:///mnt/x?writeback=nope", "", "not true or false"},
		{"file://relative/x", "", "share of Windows"},
		{"https://me:secret@nas:5006/hwnpu", "https://nas:5006/hwnpu/, read only", ""},
		{"http://nas/dav?writeback=1", "webdav http://nas/dav/", ""},
		{"s3://bucket/pre?endpoint=http://h:9100&path_style=true&no_sign=true", "s3://bucket/pre/ at http://h:9100", ""},
		{"s3:///x", "", "no bucket"},
		{"https://nas/x?color=red", "", "color is not a setting"},
		{"ftp://nas/x", "", "not a file://"},
	} {
		s, err := ParseSource("V", c.url)
		if c.err != "" {
			if err == nil || !strings.Contains(err.Error(), c.err) {
				t.Errorf("%s: %v, want %q", c.url, err, c.err)
			}
			continue
		}
		if err != nil || !strings.Contains(s.String(), c.want) || strings.Contains(s.String(), "secret") {
			t.Errorf("%s: %v %v, want %q", c.url, s, err, c.want)
		}
	}
	t.Setenv(SourceEnv+"_10", "file:///ten")
	t.Setenv(SourceEnv+"_2", "file:///two?writeback=true")
	t.Setenv(SourceEnv, "file:///first")
	t.Setenv(SourceEnv+"S", "file:///another-variable")
	t.Setenv(HuggingFaceEnv, "false")
	ss, hf, err := FromEnv()
	if err != nil || hf || len(ss) != 3 {
		t.Fatalf("%v %v %v", ss, hf, err)
	}
	var got []string
	for _, s := range ss {
		got = append(got, s.Store.(Dir).Root)
	}
	if strings.Join(got, " ") != "/first /two /ten" || !ss[1].Writeback || ss[0].Writeback || ss[2].From != SourceEnv+"_10" {
		t.Errorf("order %v", got)
	}
	t.Setenv(SourceEnv+"_x", "file:///x")
	if _, _, err := FromEnv(); err == nil {
		t.Error("a source variable without a number")
	}
}
