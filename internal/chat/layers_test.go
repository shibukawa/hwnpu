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

// fakeHub serves files of repositories at main, says their commit and (for files of names
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
		body, ok := files[p]
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

// TestLayerFill checks the forward proxy: a file downloaded from the hub goes in a layer that
// is filled, and is read from there next (the hub only asked what it is); a gated repository's
// files are not filled.
func TestLayerFill(t *testing.T) {
	t.Setenv(cache.RootEnv, t.TempDir())
	counts := fakeHub(t, map[string]string{
		"lab/m/resolve/main/config.json":       `{"a":1}`,
		"lab/m/resolve/main/model.safetensors": "the weights",
		"lab/g/resolve/main/config.json":       `{"g":1}`,
	}, "lab/g")
	share := t.TempDir()
	if _, err := userFiles(t, fmt.Sprintf("version: 1\nsources:\n  share: {type: path, path: %q}\ncache:\n  layers:\n    - {use: share, fill: true}\n", share)); err != nil {
		t.Fatal(err)
	}
	if ls := shared.Layers(); len(ls) != 1 || !ls[0].Fill {
		t.Fatalf("layers %v", ls)
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
	idx := filepath.Join(share, "upstream", "lab", "m@"+testCommit, "config.json.json")
	if _, err := os.Stat(idx); err != nil {
		t.Fatalf("not filled: %v", err)
	}
	if _, err := os.Stat(filepath.Join(share, "upstream", "lab", "g@"+testCommit)); err == nil {
		t.Error("a gated repository's file was filled")
	}
	gets := counts["GET"]
	b := t.TempDir()
	for _, f := range []string{"config.json", "model.safetensors"} {
		if err := fetch(ctx, "lab/m", f, filepath.Join(b, f), nil); err != nil {
			t.Fatal(err)
		}
		if read(t, filepath.Join(b, f)) != read(t, filepath.Join(a, f)) {
			t.Errorf("%s differs", f)
		}
	}
	if d := counts["GET"] - gets; d != 0 {
		t.Errorf("%d downloads from the hub for files the layer has", d)
	}
}

// TestLayersEnv reads HWNPU_CACHE_LAYERS in place of the definitions' layers.
func TestLayersEnv(t *testing.T) {
	share := t.TempDir()
	t.Setenv(LayersEnv, share+"?fill=1&scopes=upstream+portable,s3://b/p?no_sign=1&endpoint=http://h:9000")
	t.Setenv(LayersWriteEnv, "")
	ls, err := buildLayers(nil, "", nil, func(string, string, ...any) {})
	if err != nil {
		t.Fatal(err)
	}
	if len(ls) != 2 || !ls[0].Fill || ls[0].Write || len(ls[0].Scopes) != 2 || !strings.Contains(ls[1].String(), "s3://b/p/ at http://h:9000") {
		t.Errorf("%+v", ls)
	}
	t.Setenv(LayersWriteEnv, "0")
	if ls, _ := buildLayers(nil, "", nil, func(string, string, ...any) {}); ls[0].Fill {
		t.Error("HWNPU_CACHE_WRITE=0 left fill")
	}
	t.Setenv(LayersEnv, "off")
	if ls, _ := buildLayers([]layerEntry{{Use: "x"}}, "f", nil, nil); len(ls) != 0 {
		t.Errorf("off: %v", ls)
	}
	t.Setenv(LayersEnv, share+"?scopes=nothing")
	if _, err := buildLayers(nil, "", nil, nil); err == nil {
		t.Error("an unknown scope")
	}
}

// TestSharedExport makes an export once, puts it in the layer that is written, and makes it
// from there on another machine (another root) without building.
func TestSharedExport(t *testing.T) {
	ctx := context.Background()
	share := t.TempDir()
	layer := shared.Layer{Store: shared.Dir{Name: "share", Root: share}, Write: true}
	shared.SetLayers([]shared.Layer{layer})
	t.Cleanup(func() { shared.SetLayers(nil) })
	recipe := map[string]string{"model": "m", "commit": testCommit, "tools": "t1"}
	build := func(out string, n *int) func() error {
		return func() error {
			*n++
			if err := os.MkdirAll(out, 0o755); err != nil {
				return err
			}
			return os.WriteFile(filepath.Join(out, "manifest.json"), []byte(`{"version":1}`), 0o644)
		}
	}
	t.Setenv(cache.RootEnv, t.TempDir())
	out := filepath.Join(t.TempDir(), "m.amd-npu2")
	built := 0
	if err := SharedExport(ctx, "iron-lm", recipe, out, nil, build(out, &built)); err != nil {
		t.Fatal(err)
	}
	shared.Flush(ctx)
	if built != 1 {
		t.Fatalf("built %d", built)
	}
	if es, _ := shared.ListEntries(ctx, layer); len(es) != 1 {
		t.Fatalf("entries %v", es)
	}
	t.Setenv(cache.RootEnv, t.TempDir()) // another machine
	out2 := filepath.Join(t.TempDir(), "m.amd-npu2")
	if err := SharedExport(ctx, "iron-lm", recipe, out2, nil, build(out2, &built)); err != nil {
		t.Fatal(err)
	}
	if built != 1 || read(t, filepath.Join(out2, "manifest.json")) != `{"version":1}` {
		t.Errorf("built %d", built)
	}
	if _, err := os.Stat(out2 + ".done"); err != nil {
		t.Error("no .done")
	}
	// other tools: another key, made here
	recipe["tools"] = "t2"
	out3 := filepath.Join(t.TempDir(), "m.amd-npu2")
	if err := SharedExport(ctx, "iron-lm", recipe, out3, nil, build(out3, &built)); err != nil {
		t.Fatal(err)
	}
	if built != 2 {
		t.Errorf("built %d with other tools", built)
	}
}
