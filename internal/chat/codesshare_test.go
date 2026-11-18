package chat

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/cache"
	shared "github.com/shibukawa/hwnpu/internal/store"
)

// TestCodesShared puts the codes a conversion read here in a shared layer and makes their
// directory again from it on a machine that never had them, so that a conversion with the
// definitions' codes (models.yaml's codes) can be done again after the conversions in the
// sources are no longer taken (ailower.FormatVersion). Codes of another hash are not taken.
func TestCodesShared(t *testing.T) {
	t.Setenv(cache.RootEnv, t.TempDir())
	sharedLog = testLog{t}
	l := shared.Source{Store: shared.Dir{Name: "nas", Root: t.TempDir()}, Writeback: true}
	shared.SetSources([]shared.Source{l}, false)
	t.Cleanup(func() { shared.SetSources(nil, true) })
	ctx := WithModel(context.Background(), Model{Name: "Qwen2.5-Coder-7B-Instruct", Repo: "Qwen/Qwen2.5-Coder-7B-Instruct@c03e", SHA256: map[string]string{"model.safetensors": "01"}})

	dir := filepath.Join(t.TempDir(), "Qwen2.5-Coder-7B-Instruct")
	src := filepath.Join(t.TempDir(), "codes")
	os.MkdirAll(src, 0o755)
	os.WriteFile(filepath.Join(src, "index.json"), []byte(`{"model.layers.0.mlp.down_proj.weight": 1}`), 0o644)
	os.WriteFile(filepath.Join(src, "k0.idx"), []byte("indices"), 0o644)
	os.WriteFile(filepath.Join(src, "k0.tab"), []byte("tables"), 0o644)
	hash, err := codesHash(src)
	if err != nil {
		t.Fatal(err)
	}
	putCodes(ctx, dir, hash, src)
	shared.Flush(ctx)
	es, err := shared.ListEntries(ctx, l)
	if err != nil || len(es) != 1 || !strings.HasPrefix(es[0], "converted/Qwen2.5-Coder-7B-Instruct/any/codes-"+hash+"/") {
		t.Fatalf("entries in the layer: %v, %v", es, err)
	}
	if _, err := os.Stat(filepath.Join(src, shared.EntryFile)); err == nil {
		t.Error("the codes given were written in")
	}

	dst := CodesDir(dir, hash)
	if !fetchCodes(ctx, dir, hash, dst) {
		t.Fatal("not made from the layer")
	}
	for f, want := range map[string]string{"k0.idx": "indices", "k0.tab": "tables"} {
		if b, err := os.ReadFile(filepath.Join(dst, f)); err != nil || string(b) != want {
			t.Errorf("%s: %q, %v", f, b, err)
		}
	}
	if h, err := codesHash(dst); err != nil || h != hash {
		t.Errorf("the index made: %s, %v", h, err)
	}
	if _, err := os.Stat(dst + ".shared"); err == nil {
		t.Error("the download's directory is left")
	}
	if fetchCodes(ctx, dir, "00000000", CodesDir(dir, "00000000")) {
		t.Error("codes of another hash were taken")
	}
}
