package chat

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestLookup(t *testing.T) {
	for _, name := range []string{"LFM3-1.7B", "lfm3-1.7b", "LiquidAI/LFM3-1.7B"} {
		if m, ok := Lookup(name); !ok || m.Name != "LFM3-1.7B" || m.Kind != KindChat {
			t.Errorf("%s: %+v, %v", name, m, ok)
		}
	}
	if m, ok := Lookup("PLaMo-Embedding-1B"); !ok || m.Kind != KindEmbed {
		t.Errorf("plamo: %+v, %v", m, ok)
	}
	if _, ok := Lookup("LFM3"); ok {
		t.Error("a part of a name found a model")
	}
	seen := map[string]bool{}
	for _, m := range Catalog {
		if seen[strings.ToLower(m.Name)] || m.Kind == "" || m.Repo == "" || m.About == "" {
			t.Errorf("catalog entry %+v: repeated or incomplete", m)
		}
		seen[strings.ToLower(m.Name)] = true
	}
}

// A repository outside the catalog is a chat or an embedding model by its config.json.
func TestResolve(t *testing.T) {
	configs := map[string]string{
		"/org/my-lfm3/resolve/main/config.json":  `{"model_type": "lfm3", ` + tinyLFM3[1:],
		"/org/my-plamo/resolve/main/config.json": tinyPlamo,
		"/org/my-bert/resolve/main/config.json":  `{"model_type": "bert", "architectures": ["BertModel"], "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 4, "intermediate_size": 64, "vocab_size": 100}`,
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, ok := configs[r.URL.Path]
		if !ok {
			http.NotFound(w, r)
			return
		}
		io.WriteString(w, c)
	}))
	defer srv.Close()
	defer func(u string) { hubURL = u }(hubURL)
	hubURL = srv.URL + "/"
	root := t.TempDir()
	ctx := context.Background()
	for _, c := range []struct {
		name string
		kind Kind
	}{
		{"LFM3-1.7B", KindChat}, // the catalog's, nothing downloaded
		{"org/my-lfm3", KindChat},
		{"org/my-plamo", KindEmbed},
	} {
		m, err := Resolve(ctx, c.name, root, io.Discard)
		if err != nil || m.Kind != c.kind {
			t.Errorf("%s: %+v, %v; want a %s model", c.name, m, err, c.kind)
		}
	}
	if m, _ := Resolve(ctx, "org/my-plamo", root, io.Discard); m.Name != "my-plamo" || m.Repo != "org/my-plamo" {
		t.Errorf("org/my-plamo: %+v", m)
	}
	for _, name := range []string{"org/my-bert", "org/none", "no-such-model"} {
		if m, err := Resolve(ctx, name, root, io.Discard); err == nil {
			t.Errorf("%s: %+v, want an error", name, m)
		}
	}
}
