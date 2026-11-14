package chat

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/llm"
	shared "github.com/shibukawa/hwnpu/internal/store"
	"github.com/shibukawa/hwnpu/internal/store/s3fake"
)

// TestCoreAIShared puts a Core AI export made here in a shared layer, a folder (a NAS) and an S3
// bucket alike, and makes it again from the layer once it is gone here: the same files, its mark
// written. An export of another name, or of a model pinned to other files, is not taken.
func TestCoreAIShared(t *testing.T) {
	srv := s3fake.New(t)
	t.Setenv("AWS_ACCESS_KEY_ID", "id")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "secret")
	t.Setenv("AWS_PROFILE", "")
	s3, err := shared.OpenS3(shared.S3Config{Name: "lab", Bucket: "models", Prefix: "hwnpu", Endpoint: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	for _, st := range []shared.Store{shared.Dir{Name: "nas", Root: t.TempDir()}, s3} {
		t.Run(st.String(), func(t *testing.T) {
			t.Setenv(cache.RootEnv, t.TempDir())
			sharedLog = testLog{t}
			l := shared.Layer{Store: st, Write: true}
			shared.SetLayers([]shared.Layer{l})
			t.Cleanup(func() { shared.SetLayers(nil) })
			m := Model{Name: "swiftf0", Repo: "lars76/swift-f0", SHA256: map[string]string{"model.onnx": "ab12"}}
			ctx := WithModel(context.Background(), m)

			dir := filepath.Join(t.TempDir(), "swift-f0")
			export, done := llm.CoreAIExport(dir, "swiftf0-r3-fp16-t1896")
			files := map[string]string{"model.aimodel/main.mlirb": "bytecode", "model.aimodel/main.hash": "hash"}
			if err := llm.StartExport(export); err != nil {
				t.Fatal(err)
			}
			for f, body := range files {
				os.MkdirAll(filepath.Dir(filepath.Join(export, f)), 0o755)
				os.WriteFile(filepath.Join(export, f), []byte(body), 0o644)
			}
			if err := llm.FinishExport(export, done); err != nil {
				t.Fatal(err)
			}
			llm.ShareExport(ctx, export)
			shared.Flush(ctx)
			if es, err := shared.ListEntries(ctx, l); err != nil || len(es) != 1 {
				t.Fatalf("entries in the layer: %v, %v", es, err)
			}

			os.RemoveAll(export)
			os.Remove(done)
			if !llm.FetchExport(ctx, export, done) {
				t.Fatal("not made from the layer")
			}
			for f, body := range files {
				if b, err := os.ReadFile(filepath.Join(export, f)); err != nil || string(b) != body {
					t.Errorf("%s: %q, %v", f, b, err)
				}
			}
			if !llm.ExportDone(done) {
				t.Error("no mark after it was made from the layer")
			}

			other, otherDone := llm.CoreAIExport(dir, "swiftf0-r4-fp16-t1896")
			if llm.FetchExport(ctx, other, otherDone) {
				t.Error("an export of another name was taken")
			}
			m.SHA256 = map[string]string{"model.onnx": "cd34"}
			os.RemoveAll(export)
			os.Remove(done)
			if llm.FetchExport(WithModel(context.Background(), m), export, done) {
				t.Error("an export of other files was taken")
			}
		})
	}
}

// TestCoreAIRecipe names the export in the key with the export it is in (a size of depth's), and
// finds the model by its directory when the context names none.
func TestCoreAIRecipe(t *testing.T) {
	m := Model{Name: "da3-small", Repo: "depth-anything/DA3-SMALL", SHA256: map[string]string{"model.safetensors": "ef"}}
	ctx := WithModel(context.Background(), m)
	if _, r := coreAIRecipe(ctx, "/m/DA3-SMALL/da3-small-v1.coreai/518x392"); r == nil || r["export"] != "da3-small-v1.coreai/518x392" {
		t.Errorf("a size: %v", r)
	}
	if _, r := coreAIRecipe(ctx, "/m/DA3-SMALL/da3-small-v1.coreml"); r != nil {
		t.Errorf("a Core ML export has a key: %v", r)
	}
	saved := Catalog
	t.Cleanup(func() { Catalog = saved })
	Catalog = []Model{m}
	if got, r := coreAIRecipe(context.Background(), "/m/DA3-SMALL/x.coreai"); r == nil || got.Name != "da3-small" {
		t.Errorf("by the directory: %v %v", got.Name, r)
	}
}

type testLog struct{ t *testing.T }

func (l testLog) Write(p []byte) (int, error) { l.t.Log(string(p)); return len(p), nil }

// TestTablesShared puts the tables written here in a shared layer and makes the file again from
// it, so that a machine that runs converted programs needs no checkpoint.
func TestTablesShared(t *testing.T) {
	t.Setenv(cache.RootEnv, t.TempDir())
	sharedLog = testLog{t}
	l := shared.Layer{Store: shared.Dir{Name: "nas", Root: t.TempDir()}, Write: true}
	shared.SetLayers([]shared.Layer{l})
	t.Cleanup(func() { shared.SetLayers(nil) })
	ctx := WithModel(context.Background(), Model{Name: "whisper", Repo: "openai/whisper-large-v3-turbo", SHA256: map[string]string{"model.safetensors": "01"}})
	a := &llm.Arch{Name: "whisper"}
	out := filepath.Join(t.TempDir(), "whisper-large-v3-turbo", TablesFile)
	os.MkdirAll(filepath.Dir(out), 0o755)
	os.WriteFile(out, []byte("tables"), 0o644)
	m, r := tablesRecipe(ctx, out, a, []string{"model.decoder.embed_tokens.weight"})
	putTables(ctx, m, r, out)
	shared.Flush(ctx)
	os.Remove(out)
	if !fetchTables(ctx, r, out) {
		t.Fatal("not made from the layer")
	}
	if b, err := os.ReadFile(out); err != nil || string(b) != "tables" {
		t.Errorf("%q, %v", b, err)
	}
	if _, err := os.Stat(out + ".shared"); err == nil {
		t.Error("the download's directory is left")
	}
	if _, r2 := tablesRecipe(ctx, out, a, []string{"other"}); fetchTables(ctx, r2, out+"2") {
		t.Error("tables of other names were taken")
	}
}
