package lfm2moe

import (
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
)

// LFM2-8B-A1B's config.json, its layer types shortened to 6 layers.
const cfg8b = `{"architectures":["Lfm2MoeForCausalLM"],"model_type":"lfm2_moe","hidden_size":2048,"intermediate_size":7168,
"moe_intermediate_size":1792,"num_hidden_layers":6,"num_dense_layers":2,"num_experts":32,"num_experts_per_tok":4,
"norm_topk_prob":true,"routed_scaling_factor":1.0,"use_expert_bias":true,"num_attention_heads":32,"num_key_value_heads":8,
"vocab_size":65536,"norm_eps":1e-5,"conv_L_cache":3,"conv_bias":false,"rope_theta":1000000.0,"max_position_embeddings":128000,
"layer_types":["conv","conv","full_attention","conv","conv","full_attention"]}`

func TestPlanRouted(t *testing.T) {
	c, err := ParseConfig([]byte(cfg8b))
	if err != nil {
		t.Fatal(err)
	}
	if !c.TieWordEmbed || c.HeadDim != 64 {
		t.Errorf("tied %v, head_dim %d: want tied and 64", c.TieWordEmbed, c.HeadDim)
	}
	plan, err := llm.PlanRouted(c, llm.Options{Format: "lut8"}, 1<<30)
	if err != nil {
		t.Fatal(err)
	}
	// a part ends where each mixture of experts layer begins
	want := [][2]int{{0, 2}, {2, 1}, {3, 1}, {4, 1}, {5, 1}}
	if len(plan) != len(want) {
		t.Fatalf("plan %v, want starts and layers %v", plan, want)
	}
	for i, p := range plan {
		if p.Start != want[i][0] || p.Layers != want[i][1] {
			t.Errorf("part %d: %d+%d, want %v", i, p.Start, p.Layers, want[i])
		}
	}
}
