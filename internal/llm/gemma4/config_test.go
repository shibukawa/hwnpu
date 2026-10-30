package gemma4

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
)

// The config.json of google/gemma-4-E2B-it parses to what modeling_gemma4.py builds from it.
func TestE2BConfig(t *testing.T) {
	raw, err := os.ReadFile("testdata/gemma-4-E2B-it.config.json")
	if err != nil {
		t.Fatal(err)
	}
	a, err := llm.Detect(raw)
	if err != nil || a != Arch {
		t.Fatalf("detected %v, %v", a, err)
	}
	c, err := ParseConfig(raw)
	if err != nil {
		t.Fatal(err)
	}
	if c.Prefix != "model.language_model." || c.NumLayers != 35 || c.HiddenSize != 1536 || c.PerLayerDim != 256 {
		t.Fatalf("parsed %+v", c)
	}
	// Layers 15 to 34 reuse the keys and values of layer 13 (sliding) and 14 (full), and have
	// the double wide MLP.
	if c.firstShared() != 15 || c.source("sliding_attention") != 13 || c.source("full_attention") != 14 {
		t.Errorf("first shared layer %d, sources %d and %d", c.firstShared(), c.source("sliding_attention"), c.source("full_attention"))
	}
	if c.mlpWidth(14) != 6144 || c.mlpWidth(15) != 12288 {
		t.Errorf("MLP widths %d and %d", c.mlpWidth(14), c.mlpWidth(15))
	}
	// Proportional RoPE: 64 of the 256 frequencies of a full layer's head of 512 turn.
	inv := c.RopeInvFreq("full_attention")
	turning := 0
	for _, f := range inv {
		if f != 0 {
			turning++
		}
	}
	if len(inv) != 256 || turning != 64 || inv[1] != 1/math.Pow(1e6, 2.0/512) {
		t.Errorf("full attention frequencies: %d, %d turning, second %g", len(inv), turning, inv[1])
	}
	ts := c.Tensors()
	for name, want := range map[string][]int{
		"model.language_model.embed_tokens_per_layer.weight":     {262144, 8960},
		"model.language_model.layers.4.self_attn.q_proj.weight":  {4096, 1536},
		"model.language_model.layers.4.self_attn.k_proj.weight":  {512, 1536},
		"model.language_model.layers.34.mlp.down_proj.weight":    {1536, 12288},
		"model.language_model.per_layer_model_projection.weight": {8960, 1536},
		"model.language_model.layers.0.layer_scalar":             {1},
	} {
		if got := ts[name]; len(got) != len(want) || got[0] != want[0] || got[len(got)-1] != want[len(want)-1] {
			t.Errorf("%s: %v, want %v", name, got, want)
		}
	}
	if _, ok := ts["model.language_model.layers.15.self_attn.k_proj.weight"]; ok {
		t.Error("a shared layer reads its own k_proj")
	}
	// The per-layer embeddings stay on the host; the projection before layer 0 is in part 1.
	opts := llm.Options{Format: "lut4", Group: 16}
	pre, _, post := llm.EstimateBytes(c, opts)
	if pre < 6e6 || pre > 8e6 || post > 3e8 {
		t.Errorf("before the layers %d bytes, after %d", pre, post)
	}
	target, _ := llm.LookupTarget("m3")
	plan, err := llm.PlanSplit(c, opts, int64(llm.SplitMargin*float64(target.MaxBytes)), 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("lut4 for the M3: %v", plan)
}

// Settings the conversion does not handle are errors, not ignored.
func TestParseConfigRejects(t *testing.T) {
	for _, c := range []struct{ change, want string }{
		{`"enable_moe_block": false`, `"enable_moe_block": true`},
		{`"attention_k_eq_v": false`, `"attention_k_eq_v": true`},
		{`"attention_bias": false`, `"attention_bias": true`},
		{`"use_bidirectional_attention": null`, `"use_bidirectional_attention": "vision"`},
		{`"tie_word_embeddings": true`, `"tie_word_embeddings": false`},
		{`"rope_type": "proportional"`, `"rope_type": "llama3"`},
		{`"rope_theta": 10000.0, "rope_type": "default"`, `"rope_theta": 10000.0, "rope_type": "default", "partial_rotary_factor": 0.5`},
		{`"num_kv_shared_layers": 3`, `"num_kv_shared_layers": 6`},
		{`"hidden_activation": "gelu_pytorch_tanh"`, `"hidden_activation": "swish"`},
		{`"vocab_size_per_layer_input": 48`, `"vocab_size_per_layer_input": 32`},
		{`["sliding_attention", "sliding_attention", "full_attention"`, `["chunked_attention", "sliding_attention", "full_attention"`},
		{`"model_type": "gemma4", "text_config"`, `"model_type": "gemma3", "text_config"`},
		// layers 3 to 5 share keys and values, but no full layer comes before them
		{`"sliding_attention", "sliding_attention", "full_attention", "sliding_attention"`, `"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention"`},
	} {
		raw := strings.Replace(tiny, c.change, c.want, 1)
		if raw == tiny {
			t.Fatalf("%q is not in the config", c.change)
		}
		if _, err := ParseConfig([]byte(raw)); err == nil {
			t.Errorf("%s: no error", c.want)
		} else {
			t.Logf("%s: %v", c.want, err)
		}
	}
	// A null hidden_size_per_layer_input turns the per-layer embeddings off.
	c, err := ParseConfig([]byte(strings.Replace(tiny, `"hidden_size_per_layer_input": 16`, `"hidden_size_per_layer_input": null`, 1)))
	if err != nil || c.PerLayerDim != 0 || c.TokenInputs() != nil {
		t.Errorf("null per-layer embeddings: %v, %v", c, err)
	}
}
