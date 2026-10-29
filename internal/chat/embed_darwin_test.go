package chat

import (
	"cmp"
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/shibukawa/hwnpu/internal/backend/coreml"
)

// Compares the embeddings of PLaMo-Embedding-1B on the Neural Engine with those of the Hugging
// Face implementation written by tools/ref/embed_golden.py: the same tokens, and the same
// direction. Needs HWNPU_PLAMO_DIR with config.json, tokenizer.model and model.safetensors (the
// programs are converted there when missing) and HWNPU_PLAMO_GOLDEN (default
// HWNPU_PLAMO_DIR/embed_golden.json).
func TestPlamoEmbeddingGolden(t *testing.T) {
	dir := os.Getenv("HWNPU_PLAMO_DIR")
	if dir == "" {
		t.Skip("HWNPU_PLAMO_DIR not set")
	}
	ctx := context.Background()
	if infos, err := (coreml.Provider{}).Enumerate(ctx); err != nil || len(infos) == 0 {
		t.Skip("no Neural Engine")
	}
	path := os.Getenv("HWNPU_PLAMO_GOLDEN")
	if path == "" {
		path = filepath.Join(dir, "embed_golden.json")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	type item struct {
		Text      string    `json:"text"`
		IDs       []int32   `json:"ids"`
		Embedding []float32 `json:"embedding"`
	}
	var golden struct{ Queries, Documents []item }
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatal(err)
	}
	m, _ := Lookup("plamo-embedding-1b")
	c, err := PrepareEmbedder(ctx, EmbedSetup{Dir: dir, Repo: m.Repo, Seqs: DefaultEmbedSeqs, Log: os.Stderr})
	if err != nil {
		t.Fatal(err)
	}
	c.Log = os.Stderr
	if d, err := os.UserCacheDir(); err == nil {
		c.CacheDir = filepath.Join(d, "hwnpu")
	}
	e, err := LoadEmbedder(ctx, c)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()

	embed := func(items []item, query bool) [][]float32 {
		var out [][]float32
		for _, g := range items {
			if ids, _ := e.Tokens(g.Text, query); !slices.Equal(ids, g.IDs) {
				t.Errorf("%q: tokens %v, the reference has %v", g.Text, ids, g.IDs)
			}
			r, err := e.Embed(ctx, g.Text, query, false)
			if err != nil {
				t.Fatal(err)
			}
			for _, v := range r.Vector {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatalf("%q: the embedding has %v (fp16 overflow?)", g.Text, v)
				}
			}
			cos := Cosine(r.Vector, g.Embedding)
			t.Logf("cosine %.4f with the reference (%d tokens in %.0f ms): %q", cos, r.Tokens, r.Time.Seconds()*1000, g.Text)
			if cos < 0.98 {
				t.Errorf("%q: cosine %.4f with the reference, want 0.98 or more", g.Text, cos)
			}
			out = append(out, r.Vector)
		}
		return out
	}
	qs, ds := embed(golden.Queries, true), embed(golden.Documents, false)

	// each query ranks the documents as the reference does
	for i, q := range golden.Queries {
		rank := func(v func(j int) []float32, qv []float32) []int {
			order := make([]int, len(ds))
			for j := range order {
				order[j] = j
			}
			slices.SortStableFunc(order, func(a, b int) int {
				return cmp.Compare(Cosine(qv, v(b)), Cosine(qv, v(a)))
			})
			return order
		}
		got := rank(func(j int) []float32 { return ds[j] }, qs[i])
		want := rank(func(j int) []float32 { return golden.Documents[j].Embedding }, q.Embedding)
		if got[0] != want[0] {
			t.Errorf("%q: the closest document is %d, the reference's %d (orders %v and %v)", q.Text, got[0], want[0], got, want)
		}
	}
}
