package chat

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/shibukawa/bonsaiai/internal/cache"
	"github.com/shibukawa/bonsaiai/internal/llm"
	"github.com/shibukawa/bonsaiai/internal/recipe"
	shared "github.com/shibukawa/bonsaiai/internal/store"
)

const testCommit = "0123456789abcdef0123456789abcdef01234567"

// fakeHub serves files of repositories at main (and at testCommit, main's), says their commit and (for files of names
// ending in .safetensors, as LFS's) their SHA-256, and counts the requests by method. gated
// repositories say so on the API.
func fakeHub(t *testing.T, files map[string]string, gated ...string) map[string]int {
	t.Helper()
	var mu sync.Mutex
	counts := map[string]int{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		counts[r.Method]++
		mu.Unlock()
		p := strings.TrimPrefix(r.URL.Path, "/")
		if repo, ok := strings.CutPrefix(p, "api/models/"); ok {
			g := "false"
			for _, x := range gated {
				if x == repo {
					g = `"auto"`
				}
			}
			fmt.Fprintf(w, `{"gated": %s}`, g)
			return
		}
		body, ok := files[strings.Replace(p, "/resolve/"+testCommit+"/", "/resolve/main/", 1)]
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("X-Repo-Commit", testCommit)
		if strings.HasSuffix(p, ".safetensors") {
			w.Header().Set("X-Linked-Etag", `"`+digest(body)+`"`)
		}
		if r.Method == http.MethodGet {
			fmt.Fprint(w, body)
		}
	}))
	t.Cleanup(srv.Close)
	old := hubURL
	hubURL = srv.URL + "/"
	t.Cleanup(func() { hubURL = old })
	return counts
}

// TestWriteback checks the forward proxy: a file downloaded from the hub goes in the source
// written back to, with its lock, and is read from there next with no request to the hub; a
// gated repository's files go too.
func TestWriteback(t *testing.T) {
	t.Setenv(cache.RootEnv, t.TempDir())
	counts := fakeHub(t, map[string]string{
		"lab/m/resolve/main/config.json":       `{"a":1}`,
		"lab/m/resolve/main/model.safetensors": "the weights",
		"lab/g/resolve/main/config.json":       `{"g":1}`,
	}, "lab/g")
	share := t.TempDir()
	if _, err := withSources(t, []string{"file://" + filepath.ToSlash(share) + "?writeback=true"}); err != nil {
		t.Fatal(err)
	}
	if ss := shared.Sources(); len(ss) != 1 || !ss[0].Writeback {
		t.Fatalf("sources %v", ss)
	}
	ctx := context.Background()
	a := map[string]string{}
	for _, f := range []string{"config.json", "model.safetensors"} {
		a[f] = fetched(t, "lab/m", f)
	}
	fetched(t, "lab/g", "config.json")
	// they went as they were downloaded (a tee), not from the queue after
	if b, _ := os.ReadFile(cache.Sub("work", "queue.log")); strings.Count(string(b), "as it was downloaded") != 3 {
		t.Errorf("not teed: %s", b)
	}
	shared.Flush(ctx)
	src := shared.Sources()[0]
	l, err := shared.ReadLock(ctx, src, "lab/m")
	if err != nil || l.Commit != testCommit || len(l.Files) != 2 || l.Files["model.safetensors"].SHA256 != digest("the weights") {
		t.Fatalf("lock %+v %v", l, err)
	}
	if read(t, filepath.Join(share, "original", "lab", "m", "model.safetensors")) != "the weights" {
		t.Error("not a plain folder")
	}
	if _, err := shared.ReadLock(ctx, src, "lab/g"); err != nil {
		t.Errorf("the gated repository: %v", err)
	}
	// another machine
	t.Setenv(cache.RootEnv, t.TempDir())
	if _, err := withSources(t, []string{"file://" + filepath.ToSlash(share) + "?writeback=true"}); err != nil {
		t.Fatal(err)
	}
	clearCommits()
	before := counts["GET"] + counts["HEAD"]
	for _, f := range []string{"config.json", "model.safetensors"} {
		if fetched(t, "lab/m", f) != a[f] {
			t.Errorf("%s differs", f)
		}
	}
	if d := counts["GET"] + counts["HEAD"] - before; d != 0 {
		t.Errorf("%d requests to the hub for files the source has", d)
	}
}

// TestSharedExport makes an export once, puts it in the source written back to, and makes it
// from there on another machine (another root) without building: v0-HASH with no version (other
// tools: another one), vN with a version (other tools: the same), vN-HASH in development.
func TestSharedExport(t *testing.T) {
	ctx := context.Background()
	share := t.TempDir()
	build := func(out string, n *int) func() error {
		return func() error {
			*n++
			if err := os.MkdirAll(out, 0o755); err != nil {
				return err
			}
			return os.WriteFile(filepath.Join(out, "manifest.json"), []byte(`{"version":1}`), 0o644)
		}
	}
	for _, c := range []struct {
		name, defs, dir string
		sameAfterTools  bool
	}{
		{"no version", "", "v0-", false},
		{"version", "    targets: {amd-xdna2: {version: 3}}\n", "v3", true},
		{"develop", "    targets: {amd-xdna2: {version: 3}}\n    develop: true\n", "v3-", false},
		// the version of its recipe (model-recipes.yaml models.LFM3-1.7B.amd.version)
		{"recipe", "", lfm3RecipeDir(t), true},
	} {
		t.Run(c.name, func(t *testing.T) {
			share := filepath.Join(share, strings.ReplaceAll(c.name, " ", "-"))
			// a model of no recipe but for "recipe"
			name, defs := "x-lm", "  x-lm:\n    from: hf:a/x-lm\n    kind: chat\n"+c.defs+"    machine: 8\n"
			if c.name == "recipe" {
				name, defs = "LFM3-1.7B", "  LFM3-1.7B:\n    machine: 8\n"
			}
			if _, err := withSources(t, []string{"file://" + filepath.ToSlash(share) + "?writeback=true"}, "version: 1\nmodels:\n"+defs); err != nil {
				t.Fatal(err)
			}
			recipe := map[string]string{"model": name, "target": cache.AMDTarget, "commit": testCommit, "tools": "t1"}
			t.Setenv(cache.RootEnv, t.TempDir())
			out := filepath.Join(t.TempDir(), name+".amd-npu2")
			built := 0
			if err := SharedExport(ctx, "iron-lm", recipe, out, nil, build(out, &built)); err != nil {
				t.Fatal(err)
			}
			shared.Flush(ctx)
			if built != 1 {
				t.Fatalf("built %d", built)
			}
			es, _ := shared.ListEntries(ctx, shared.Sources()[0])
			if len(es) != 1 || !strings.HasPrefix(es[0], "converted/"+name+"/amd-xdna2/"+name+".amd-npu2/"+c.dir) || (c.dir == "v3" || c.name == "recipe") && !strings.HasSuffix(es[0], "/"+c.dir) {
				t.Fatalf("entries %v", es)
			}
			t.Setenv(cache.RootEnv, t.TempDir()) // another machine
			out2 := filepath.Join(t.TempDir(), name+".amd-npu2")
			if err := SharedExport(ctx, "iron-lm", recipe, out2, nil, build(out2, &built)); err != nil {
				t.Fatal(err)
			}
			if built != 1 || read(t, filepath.Join(out2, "manifest.json")) != `{"version":1}` {
				t.Errorf("built %d", built)
			}
			if _, err := os.Stat(out2 + ".done"); err != nil {
				t.Error("no .done")
			}
			recipe["tools"] = "t2"
			out3 := filepath.Join(t.TempDir(), name+".amd-npu2")
			if err := SharedExport(ctx, "iron-lm", recipe, out3, nil, build(out3, &built)); err != nil {
				t.Fatal(err)
			}
			if want := map[bool]int{true: 1, false: 2}[c.sameAfterTools]; built != want {
				t.Errorf("built %d with other tools, want %d", built, want)
			}
			recipe["commit"] = strings.Repeat("f", 40)
			out4 := filepath.Join(t.TempDir(), name+".amd-npu2")
			before := built
			if err := SharedExport(ctx, "iron-lm", recipe, out4, nil, build(out4, &built)); err != nil {
				t.Fatal(err)
			}
			if built != before+1 {
				t.Error("an export of another commit was taken")
			}
		})
	}
}

// TestSharedExportImported takes an export made by hand that bonsaiai-dev sources import put in the
// source (ImportedRecipe) when the source has none of the recipe's, without building it or
// putting it again; with neither, it builds and puts the recipe's.
func TestSharedExportImported(t *testing.T) {
	ctx := context.Background()
	share := t.TempDir()
	if _, err := withSources(t, []string{"file://" + filepath.ToSlash(share) + "?writeback=true"}, "version: 1\nmodels:\n  LFM3-1.7B:\n    machine: 8\n"); err != nil {
		t.Fatal(err)
	}
	m := modelNamed("LFM3-1.7B")
	name := "lfm3-1.7b.amd-npu2"
	// made by hand and imported
	hand := filepath.Join(t.TempDir(), name)
	if err := os.MkdirAll(hand, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(hand, "manifest.json"), []byte(`{"by":"hand"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	e := PlaceEntry(ImportKind, m, cache.AMDTarget, name, ImportedRecipe(m, cache.AMDTarget, name))
	if _, err := shared.Describe(hand, e); err != nil {
		t.Fatal(err)
	}
	if err := shared.PutEntry(ctx, shared.Sources()[0], hand); err != nil {
		t.Fatal(err)
	}
	recipe := map[string]string{"model": m.Name, "target": cache.AMDTarget, "tool": "iron_x.py", "tools": "t1"}
	imported := []SharedRecipe{{Kind: ImportKind, Recipe: ImportedRecipe(m, cache.AMDTarget, name)}}
	built := 0
	build := func(out string) func() error {
		return func() error {
			built++
			if err := os.MkdirAll(out, 0o755); err != nil {
				return err
			}
			return os.WriteFile(filepath.Join(out, "manifest.json"), []byte(`{"by":"build"}`), 0o644)
		}
	}
	t.Setenv(cache.RootEnv, t.TempDir()) // another machine
	out := filepath.Join(t.TempDir(), name)
	if err := SharedExportOr(ctx, "iron-x", recipe, imported, out, nil, build(out)); err != nil {
		t.Fatal(err)
	}
	shared.Flush(ctx)
	if built != 0 || read(t, filepath.Join(out, "manifest.json")) != `{"by":"hand"}` {
		t.Fatalf("built %d, %s", built, read(t, filepath.Join(out, "manifest.json")))
	}
	if _, err := os.Stat(out + ".done"); err != nil {
		t.Error("no .done")
	}
	if es, _ := shared.ListEntries(ctx, shared.Sources()[0]); len(es) != 1 {
		t.Errorf("the imported export was put again: %v", es)
	}
	// nor began to be put as it was fetched (the early puts mark the folder they put in)
	filepath.WalkDir(share, func(p string, d os.DirEntry, err error) error {
		if err == nil && strings.HasPrefix(d.Name(), shared.WritingPrefix) {
			t.Errorf("puts began for a fetched export: %s", p)
		}
		return nil
	})
	// a second run here: the export it fetched is taken
	if err := SharedExportOr(ctx, "iron-x", recipe, imported, out, nil, build(out)); err != nil || built != 0 {
		t.Fatalf("built %d: %v", built, err)
	}
	// another name: neither is there, so it is built and put as the recipe's
	other := "lfm3-1.7b-512.amd-npu2"
	out2 := filepath.Join(t.TempDir(), other)
	imported2 := []SharedRecipe{{Kind: ImportKind, Recipe: ImportedRecipe(m, cache.AMDTarget, other)}}
	if err := SharedExportOr(ctx, "iron-x", recipe, imported2, out2, nil, build(out2)); err != nil {
		t.Fatal(err)
	}
	shared.Flush(ctx)
	es, _ := shared.ListEntries(ctx, shared.Sources()[0])
	if built != 1 || len(es) != 2 {
		t.Fatalf("built %d, entries %v", built, es)
	}
	rel := es[0]
	if !strings.Contains(rel, "/"+other+"/") {
		rel = es[1]
	}
	if got, err := shared.ReadEntryAt(ctx, shared.Sources()[0], rel); err != nil || got.Kind != "iron-x" {
		t.Errorf("the built export's entry %s: %+v %v", rel, got, err)
	}
}

// TestFetchFiles downloads the files of a repository store.FetchStreams at a time, each whole.
func TestFetchFiles(t *testing.T) {
	t.Setenv(cache.RootEnv, t.TempDir())
	if _, err := withSources(t, nil); err != nil {
		t.Fatal(err)
	}
	var mu sync.Mutex
	now, most := 0, 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p := strings.TrimPrefix(r.URL.Path, "/")
		if strings.HasPrefix(p, "api/") {
			fmt.Fprint(w, `{"gated": false}`)
			return
		}
		w.Header().Set("X-Repo-Commit", testCommit)
		if r.Method != http.MethodGet {
			return
		}
		mu.Lock()
		now++
		most = max(most, now)
		mu.Unlock()
		time.Sleep(50 * time.Millisecond)
		fmt.Fprint(w, "body of "+path.Base(p))
		mu.Lock()
		now--
		mu.Unlock()
	}))
	t.Cleanup(srv.Close)
	old := hubURL
	hubURL = srv.URL + "/"
	t.Cleanup(func() { hubURL = old })
	var files []File
	for i := 0; i < 7; i++ {
		files = append(files, File{Repo: "lab/m", Name: fmt.Sprintf("shard-%d.bin", i)})
	}
	if err := FetchFiles(context.Background(), files, nil); err != nil {
		t.Fatal(err)
	}
	for _, f := range files {
		if got := read(t, f.Path()); got != "body of "+f.Name {
			t.Errorf("%s: %q", f.Name, got)
		}
	}
	if most != shared.FetchStreams || most < 2 {
		t.Errorf("%d downloads at once, want %d", most, shared.FetchStreams)
	}
	// an error stops the rest and is returned
	if err := FetchFiles(context.Background(), []File{{Repo: "lab/m", Name: "../x"}}, nil); err == nil {
		t.Error("no error for a name outside the repository")
	}
}

// An export put in a source before its model had a version (v0-HASH) is taken when the version
// is written (the same key: the same bytes), without building it, and goes in the source at
// the version.
func TestSharedExportVersionWritten(t *testing.T) {
	ctx := context.Background()
	share := t.TempDir()
	recipe := map[string]string{"model": "x-lm", "target": cache.AMDTarget, "commit": testCommit, "tools": "t1"}
	built := 0
	build := func(out string) func() error {
		return func() error {
			built++
			os.MkdirAll(out, 0o755)
			return os.WriteFile(filepath.Join(out, "manifest.json"), []byte(`{"version":1}`), 0o644)
		}
	}
	src := []string{"file://" + filepath.ToSlash(share) + "?writeback=true"}
	if _, err := withSources(t, src, "version: 1\nmodels:\n  x-lm:\n    from: hf:a/x-lm\n    kind: chat\n    machine: 8\n"); err != nil {
		t.Fatal(err)
	}
	t.Setenv(cache.RootEnv, t.TempDir())
	out := filepath.Join(t.TempDir(), "x-lm.amd-npu2")
	if err := SharedExport(ctx, "iron-lm", recipe, out, nil, build(out)); err != nil {
		t.Fatal(err)
	}
	shared.Flush(ctx)
	if _, err := withSources(t, src, "version: 1\nmodels:\n  x-lm:\n    from: hf:a/x-lm\n    kind: chat\n    targets: {amd-xdna2: {version: 2}}\n    machine: 8\n"); err != nil {
		t.Fatal(err)
	}
	t.Setenv(cache.RootEnv, t.TempDir()) // another machine
	out2 := filepath.Join(t.TempDir(), "x-lm.amd-npu2")
	if err := SharedExport(ctx, "iron-lm", recipe, out2, nil, build(out2)); err != nil {
		t.Fatal(err)
	}
	shared.Flush(ctx)
	if built != 1 || read(t, filepath.Join(out2, "manifest.json")) != `{"version":1}` {
		t.Fatalf("built %d", built)
	}
	if e, err := shared.ReadEntry(out2); err != nil || e.Version != 2 || !strings.HasSuffix(e.Path, "/v2") {
		t.Errorf("here: %+v %v", e, err)
	}
	es, _ := shared.ListEntries(ctx, shared.Sources()[0])
	if !slices.ContainsFunc(es, func(p string) bool { return strings.HasSuffix(p, "x-lm.amd-npu2/v2") }) {
		t.Errorf("not put at the version: %v", es)
	}
	// taken here again without the sources
	if err := SharedExport(ctx, "iron-lm", recipe, out2, nil, build(out2)); err != nil || built != 1 {
		t.Errorf("built %d: %v", built, err)
	}
}

// TestSharedExportFetchOnly: in a ctx that converts nothing, an export no source has is not
// built (llm.NotFetchedError), and one a source has is still taken.
func TestSharedExportFetchOnly(t *testing.T) {
	share := t.TempDir()
	if _, err := withSources(t, []string{"file://" + filepath.ToSlash(share) + "?writeback=true"}, "version: 1\nmodels:\n  LFM3-1.7B:\n    machine: 8\n"); err != nil {
		t.Fatal(err)
	}
	recipe := map[string]string{"model": "LFM3-1.7B", "target": cache.AMDTarget, "tool": "iron_x.py", "tools": "t1"}
	built := 0
	build := func(out string) func() error {
		return func() error {
			built++
			if err := os.MkdirAll(out, 0o755); err != nil {
				return err
			}
			return os.WriteFile(filepath.Join(out, "manifest.json"), []byte("{}"), 0o644)
		}
	}
	ctx := llm.WithFetchOnly(context.Background())
	out := filepath.Join(t.TempDir(), "x.amd-npu2")
	var nf *llm.NotFetchedError
	if err := SharedExportOr(ctx, "iron-x", recipe, nil, out, nil, build(out)); !errors.As(err, &nf) || built != 0 {
		t.Fatalf("fetch only: %v, built %d", err, built)
	}
	// made and put by another machine: taken
	if err := SharedExportOr(context.Background(), "iron-x", recipe, nil, out, nil, build(out)); err != nil {
		t.Fatal(err)
	}
	shared.Flush(context.Background())
	t.Setenv(cache.RootEnv, t.TempDir())
	again := filepath.Join(t.TempDir(), "x.amd-npu2")
	if err := SharedExportOr(ctx, "iron-x", recipe, nil, again, nil, build(again)); err != nil || built != 1 {
		t.Fatalf("from the source: %v, built %d", err, built)
	}
}

// TestSharedExportPinned: an export models.yaml pins (exports: its folder and digest) is taken
// from that folder whatever the recipe now (other tools, or no recipe at all), only with its
// digest; with another digest it is not, and what is built then goes beside it as v0-HASH, the
// pinned folder left as it was.
func TestSharedExportPinned(t *testing.T) {
	ctx := context.Background()
	share := t.TempDir()
	src := []string{"file://" + filepath.ToSlash(share) + "?writeback=true"}
	name := "LFM3-1.7B.amd-npu2"
	built := 0
	build := func(out, body string) func() error {
		return func() error {
			built++
			os.MkdirAll(out, 0o755)
			return os.WriteFile(filepath.Join(out, "manifest.json"), []byte(body), 0o644)
		}
	}
	defs := "version: 1\nmodels:\n  LFM3-1.7B:\n    machine: 8\n"
	if _, err := withSources(t, src, defs); err != nil {
		t.Fatal(err)
	}
	recipe := func(tools string) map[string]string {
		return map[string]string{"model": "LFM3-1.7B", "target": cache.AMDTarget, "commit": testCommit, "tools": tools}
	}
	t.Setenv(cache.RootEnv, t.TempDir())
	out := filepath.Join(t.TempDir(), name)
	if err := SharedExport(ctx, "iron-lm", recipe("t1"), out, nil, build(out, `{"made":"t1"}`)); err != nil {
		t.Fatal(err)
	}
	shared.Flush(ctx)
	es, _ := shared.ListEntries(ctx, shared.Sources()[0])
	if len(es) != 1 {
		t.Fatalf("entries %v", es)
	}
	put, err := shared.ReadEntryAt(ctx, shared.Sources()[0], es[0])
	if err != nil {
		t.Fatal(err)
	}
	pin := Pin{At: path.Base(es[0]), Digest: put.Digest()}
	pinned := func(p Pin) string {
		return defs + "    exports:\n      amd-xdna2:\n        " + name + ": " + p.String() + "\n"
	}
	if _, err := withSources(t, src, pinned(pin)); err != nil {
		t.Fatal(err)
	}
	if got, ok := PinOf(modelNamed("LFM3-1.7B"), cache.AMDTarget, name); !ok || got != pin {
		t.Fatalf("pin read as %+v %v", got, ok)
	}
	// another machine, other tools: the pinned export, not built
	t.Setenv(cache.RootEnv, t.TempDir())
	out2 := filepath.Join(t.TempDir(), name)
	if err := SharedExport(ctx, "iron-lm", recipe("t2"), out2, nil, build(out2, `{"made":"t2"}`)); err != nil {
		t.Fatal(err)
	}
	if built != 1 || read(t, filepath.Join(out2, "manifest.json")) != `{"made":"t1"}` {
		t.Fatalf("built %d, %s", built, read(t, filepath.Join(out2, "manifest.json")))
	}
	// no recipe (the commit not known), only its name: still the pinned one
	t.Setenv(cache.RootEnv, t.TempDir())
	out3 := filepath.Join(t.TempDir(), name)
	m := modelNamed("LFM3-1.7B")
	imported := []SharedRecipe{{Kind: ImportKind, Recipe: ImportedRecipe(m, cache.AMDTarget, name)}}
	if err := SharedExportOr(ctx, "iron-lm", nil, imported, out3, nil, build(out3, `{"made":"nil"}`)); err != nil {
		t.Fatal(err)
	}
	if built != 1 || read(t, filepath.Join(out3, "manifest.json")) != `{"made":"t1"}` {
		t.Fatalf("no recipe: built %d", built)
	}
	// another digest pinned: not taken; built, and put as v0-HASH of its recipe
	bad := pin
	bad.Digest = shared.DigestPrefix + strings.Repeat("0", 64)
	if _, err := withSources(t, src, pinned(bad)); err != nil {
		t.Fatal(err)
	}
	t.Setenv(cache.RootEnv, t.TempDir())
	out4 := filepath.Join(t.TempDir(), name)
	if err := SharedExport(ctx, "iron-lm", recipe("t2"), out4, nil, build(out4, `{"made":"t2"}`)); err != nil {
		t.Fatal(err)
	}
	shared.Flush(ctx)
	if built != 2 {
		t.Fatalf("built %d with a digest no source has", built)
	}
	if again, err := shared.ReadEntryAt(ctx, shared.Sources()[0], es[0]); err != nil || again.Digest() != pin.Digest {
		t.Errorf("the pinned folder changed: %v", err)
	}
	es2, _ := shared.ListEntries(ctx, shared.Sources()[0])
	if len(es2) != 2 || !slices.ContainsFunc(es2, func(p string) bool { return p != es[0] && strings.Contains(path.Base(p), "v0-") }) {
		t.Errorf("entries %v", es2)
	}
	// the export made here is taken here again (its own recipe)
	if err := SharedExport(ctx, "iron-lm", recipe("t2"), out4, nil, build(out4, `{"made":"t2"}`)); err != nil || built != 2 {
		t.Errorf("built %d: %v", built, err)
	}
}

func TestParsePin(t *testing.T) {
	d := strings.Repeat("ab", 32)
	if p, err := ParsePin("v0-7bfc0b02 sha256:" + d); err != nil || p.At != "v0-7bfc0b02" || p.Digest != "sha256:"+d {
		t.Errorf("%+v %v", p, err)
	}
	for _, bad := range []string{"", "v0-7bfc0b02", "v0-7bfc0b02 " + d, "../x sha256:" + d, "v1 sha256:12", "v1 md5:" + d} {
		if _, err := ParsePin(bad); err == nil {
			t.Errorf("%q: no error", bad)
		}
	}
}

func TestFormatTaken(t *testing.T) {
	e := shared.Entry{Target: cache.AMDTarget, Recipe: map[string]string{}}
	if !FormatTaken(e) {
		t.Error("an AMD export without format refused")
	}
	e.Recipe["format"] = fmt.Sprint(AMDExportFormat + 1)
	if FormatTaken(e) {
		t.Error("an AMD export of a later format taken")
	}
	if !FormatTaken(shared.Entry{Target: CoreAITarget, Recipe: map[string]string{"format": "9"}}) {
		t.Error("a Core AI export refused")
	}
}

// TestFindExportPinned: an export found by name whose entry.json is not of the pinned bytes is
// passed by (cache.ExportStale), one of them is found, and one made by hand (no entry.json) too.
func TestFindExportPinned(t *testing.T) {
	t.Setenv(cache.RootEnv, t.TempDir())
	name := "LFM3-1.7B.amd-npu2-q4"
	dir := filepath.Join(cache.Entries(cache.AMDTarget), name)
	os.MkdirAll(dir, 0o755)
	os.WriteFile(filepath.Join(dir, "manifest.json"), []byte(`{}`), 0o644)
	have := func(d string) bool { _, err := os.Stat(filepath.Join(d, "manifest.json")); return err == nil }
	if cache.FindExport(name, have) != dir {
		t.Fatal("an export made by hand not found")
	}
	e, err := shared.Describe(dir, shared.Entry{Kind: "iron-lm", Model: "LFM3-1.7B", Target: cache.AMDTarget, Recipe: map[string]string{"a": "b"}})
	if err != nil {
		t.Fatal(err)
	}
	defs := func(digest string) string {
		return "version: 1\nmodels:\n  LFM3-1.7B:\n    machine: 8\n    exports:\n      amd-xdna2:\n        " + name + ": v0-12345678 " + digest + "\n"
	}
	if _, err := withSources(t, nil, defs(e.Digest())); err != nil {
		t.Fatal(err)
	}
	if cache.FindExport(name, have) != dir {
		t.Error("the pinned export not found")
	}
	if _, err := withSources(t, nil, defs(shared.DigestPrefix+strings.Repeat("0", 64))); err != nil {
		t.Fatal(err)
	}
	if d := cache.FindExport(name, have); d != "" {
		t.Errorf("an export of other bytes than the pin's found: %s", d)
	}
}

// lfm3RecipeDir is the sources' folder of LFM3-1.7B's export for AMD's NPU: vN, its recipe's
// version.
func lfm3RecipeDir(t *testing.T) string {
	f, err := recipe.Builtin()
	if err != nil {
		t.Fatal(err)
	}
	m, _ := f.AMDOf("LFM3-1.7B")
	return fmt.Sprintf("v%d", m.Version)
}
