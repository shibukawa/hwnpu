package chat

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/qwen35"
)

// tinyQwen35 is a small Qwen3.5 text model: Gated DeltaNet and full attention layers.
const tinyQwen35 = `{"model_type": "qwen3_5_text", "hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 4,
	"num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 48, "rms_norm_eps": 1e-6,
	"tie_word_embeddings": true, "max_position_embeddings": 96,
	"layer_types": ["linear_attention", "full_attention", "linear_attention", "full_attention"],
	"linear_conv_kernel_dim": 4, "linear_key_head_dim": 4, "linear_num_key_heads": 8,
	"linear_num_value_heads": 16, "linear_value_head_dim": 4,
	"rope_parameters": {"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.5}}`

// The engine reuses earlier prompts with Qwen3.5 split at its attention (the Gated DeltaNet's
// recurrent states put aside and brought back with the sequences) and still gives the logits of
// reading the prompt from the start.
func TestEngineReuseQwen35(t *testing.T) {
	cfg, err := qwen35.ParseConfig([]byte(tinyQwen35))
	if err != nil {
		t.Fatal(err)
	}
	w := llm.NewSynthetic(cfg)
	const cache = 40
	sess := pagedSessionOf(t, context.Background(), nil, qwen35.Arch, cfg, w, cache, 8, 1, 4)
	if sess.Paged() != 8 {
		t.Fatalf("the session is not split at its attention: %d", sess.Paged())
	}
	// the reference holds the Gated DeltaNet's state in fp16 from step to step: reading a+x 4
	// tokens a step it is 1.7e-3 from the program without a cache, the engine 5.9e-4
	checkReuse(t, cfg, w, sess, cache, 3e-3)
}

// Prepare downloads a Qwen3.5 checkpoint in two files by its index and converts it split at its
// attention with the longest context it was trained for.
func TestPrepareQwen35(t *testing.T) {
	cfg, err := qwen35.ParseConfig([]byte(tinyQwen35))
	if err != nil {
		t.Fatal(err)
	}
	w := llm.NewSynthetic(cfg)
	names := slices.Sorted(maps.Keys(w))
	half := map[string]llm.Synthetic{"m-1.safetensors": {}, "m-2.safetensors": {}}
	index := map[string]string{}
	for i, n := range names {
		f := fmt.Sprintf("m-%d.safetensors", 1+i%2)
		half[f][n] = w[n]
		index[n] = f
	}
	idx, _ := json.Marshal(map[string]any{"metadata": map[string]any{}, "weight_map": index})
	files := map[string][]byte{
		"/org/qwen/resolve/main/config.json":                  []byte(tinyQwen35),
		"/org/qwen/resolve/main/tokenizer.json":               []byte(imageTokenizer),
		"/org/qwen/resolve/main/model.safetensors.index.json": idx,
		"/org/qwen/resolve/main/m-1.safetensors":              safetensorsOf(t, half["m-1.safetensors"]),
		"/org/qwen/resolve/main/m-2.safetensors":              safetensorsOf(t, half["m-2.safetensors"]),
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, ok := files[r.URL.Path]
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Write(b)
	}))
	defer srv.Close()
	defer func(u string) { hubURL = u }(hubURL)
	hubURL = srv.URL + "/"
	dir := filepath.Join(t.TempDir(), "qwen")
	c, err := Prepare(context.Background(), Setup{Dir: dir, Repo: "org/qwen", Draft: 8, Backend: "coreml", Format: "fp16", Log: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Packages) != 2 {
		t.Fatalf("packages %q", c.Packages)
	}
	// the program that reads prompts takes a chunk per call, and computes the logits after the
	// last token alone (llm.Options.LastRow); the one that checks drafts also outputs what each
	// row adds to the Gated DeltaNet's state (-sr, state-rows)
	for i, name := range []string{"sr-c96-q8", "c96-q96-last"} {
		p, q := c.Packages[i], []int{8, 96}[i]
		if want := fmt.Sprintf("qwen35-r1-fp16-p96-%s.parts.json", name); filepath.Base(p) != want {
			t.Errorf("package %s, want %s", filepath.Base(p), want)
		}
		man, err := llm.ReadManifest(p)
		if err != nil {
			t.Fatal(err)
		}
		// cut at the full attention layers 1 and 3
		if man.Paged != 96 || man.Cache != 96 || man.Seq != q || man.Attention == nil || len(man.Parts) != 3 {
			t.Errorf("%s: paged %d, cache %d, seq %d, %d parts", p, man.Paged, man.Cache, man.Seq, len(man.Parts))
		}
	}
	if !Downloaded(dir) || filepath.Base(c.Weights) != TablesFile {
		t.Errorf("downloaded %v, weights %s", Downloaded(dir), c.Weights)
	}
	// the tables are the checkpoint's rows: the same as read from the checkpoint
	a, model, err := llm.LoadConfig(filepath.Join(dir, "config.json"), "")
	if err != nil {
		t.Fatal(err)
	}
	ck, err := llm.OpenWeights(filepath.Join(dir, "model.safetensors"), a)
	if err != nil {
		t.Fatal(err)
	}
	tb, err := llm.OpenWeights(c.Weights, a)
	if err != nil {
		t.Fatal(err)
	}
	for _, tok := range []int32{0, 7, 47} {
		x, err := llm.TokenRow(ck, model, tok)
		if err != nil {
			t.Fatal(err)
		}
		y, err := llm.TokenRow(tb, model, tok)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(x, y) {
			t.Fatalf("token %d: the tables' row differs from the checkpoint's", tok)
		}
	}
	ck.Close()
	tb.Close()
	// with the checkpoint gone and the repository not serving it, the converted model is ready
	// all the same: nothing is downloaded or converted again
	for _, f := range []string{"model.safetensors.index.json", "m-1.safetensors", "m-2.safetensors"} {
		os.Remove(filepath.Join(dir, f))
		delete(files, "/org/qwen/resolve/main/"+f)
	}
	c2, err := Prepare(context.Background(), Setup{Dir: dir, Repo: "org/qwen", Draft: 8, Backend: "coreml", Format: "fp16", Log: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(c2.Packages, c.Packages) || c2.Weights != c.Weights {
		t.Errorf("second Prepare: packages %q, weights %s", c2.Packages, c2.Weights)
	}
}
