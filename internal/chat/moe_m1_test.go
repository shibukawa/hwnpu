package chat

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
)

// A mixture of experts (llm.Routed) is refused on an M1 before anything past config.json
// downloads: its experts' program needs gather_nd, which an M1's Neural Engine does not take.
func TestPrepareMoEOnM1(t *testing.T) {
	cacheHere(t)
	defer func(f func() string) { chipName = f }(chipName)
	chipName = func() string { return "M1" }
	config := `{"model_type": "lfm2_moe", "hidden_size": 64, "intermediate_size": 128,
		"moe_intermediate_size": 32, "num_hidden_layers": 2, "num_dense_layers": 1,
		"num_experts": 4, "num_experts_per_tok": 2, "num_attention_heads": 4,
		"num_key_value_heads": 2, "vocab_size": 256, "norm_eps": 1e-5, "conv_L_cache": 3,
		"layer_types": ["conv", "full_attention"], "max_position_embeddings": 256}`
	var asked []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		asked = append(asked, r.URL.Path)
		if r.URL.Path != "/org/moe/resolve/main/config.json" {
			http.NotFound(w, r)
			return
		}
		io.WriteString(w, config)
	}))
	defer srv.Close()
	defer func(u string) { hubURL = u }(hubURL)
	hubURL = srv.URL + "/"
	_, err := Prepare(context.Background(), Setup{Dir: filepath.Join(t.TempDir(), "moe"), Repo: "org/moe", Backend: "coreai", Log: io.Discard})
	if err == nil || !strings.Contains(err.Error(), "M2 or later") {
		t.Fatalf("Prepare on an M1: %v", err)
	}
	if len(asked) != 1 {
		t.Errorf("asked for %q, want config.json alone", asked)
	}
}
