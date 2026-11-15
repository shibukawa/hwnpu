package chat

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/shibukawa/hwnpu/internal/cache"
	shared "github.com/shibukawa/hwnpu/internal/store"
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
	a := t.TempDir()
	for _, f := range []string{"config.json", "model.safetensors"} {
		if err := fetch(ctx, "lab/m", f, filepath.Join(a, f), nil); err != nil {
			t.Fatal(err)
		}
	}
	if err := fetch(ctx, "lab/g", "config.json", filepath.Join(a, "g.json"), nil); err != nil {
		t.Fatal(err)
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
	before := counts["GET"] + counts["HEAD"]
	b := t.TempDir()
	for _, f := range []string{"config.json", "model.safetensors"} {
		if err := fetch(ctx, "lab/m", f, filepath.Join(b, f), nil); err != nil {
			t.Fatal(err)
		}
		if read(t, filepath.Join(b, f)) != read(t, filepath.Join(a, f)) {
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
	} {
		t.Run(c.name, func(t *testing.T) {
			share := filepath.Join(share, strings.ReplaceAll(c.name, " ", "-"))
			if _, err := withSources(t, []string{"file://" + filepath.ToSlash(share) + "?writeback=true"}, "version: 1\nmodels:\n  LFM3-1.7B:\n"+c.defs+"    machine: 8\n"); err != nil {
				t.Fatal(err)
			}
			recipe := map[string]string{"model": "LFM3-1.7B", "target": cache.AMDTarget, "commit": testCommit, "tools": "t1"}
			t.Setenv(cache.RootEnv, t.TempDir())
			out := filepath.Join(t.TempDir(), "LFM3-1.7B.amd-npu2")
			built := 0
			if err := SharedExport(ctx, "iron-lm", recipe, out, nil, build(out, &built)); err != nil {
				t.Fatal(err)
			}
			shared.Flush(ctx)
			if built != 1 {
				t.Fatalf("built %d", built)
			}
			es, _ := shared.ListEntries(ctx, shared.Sources()[0])
			if len(es) != 1 || !strings.HasPrefix(es[0], "converted/LFM3-1.7B/amd-xdna2/LFM3-1.7B.amd-npu2/"+c.dir) || c.dir == "v3" && !strings.HasSuffix(es[0], "/v3") {
				t.Fatalf("entries %v", es)
			}
			t.Setenv(cache.RootEnv, t.TempDir()) // another machine
			out2 := filepath.Join(t.TempDir(), "LFM3-1.7B.amd-npu2")
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
			out3 := filepath.Join(t.TempDir(), "LFM3-1.7B.amd-npu2")
			if err := SharedExport(ctx, "iron-lm", recipe, out3, nil, build(out3, &built)); err != nil {
				t.Fatal(err)
			}
			if want := map[bool]int{true: 1, false: 2}[c.sameAfterTools]; built != want {
				t.Errorf("built %d with other tools, want %d", built, want)
			}
			recipe["commit"] = strings.Repeat("f", 40)
			out4 := filepath.Join(t.TempDir(), "LFM3-1.7B.amd-npu2")
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
