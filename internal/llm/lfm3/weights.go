package lfm3

import (
	"fmt"
	"math"
	"strings"

	"github.com/shibukawa/hwnpu/internal/llm"
)

func pow(a, b float64) float64 { return math.Pow(a, b) }

// OpenWeights opens a safetensors or GGUF checkpoint of LFM3 by its extension.
func OpenWeights(path string) (llm.Weights, error) { return llm.OpenWeights(path, Arch) }

// ggufName maps a Hugging Face tensor name to llama.cpp's.
func ggufName(name string) string {
	switch name {
	case "model.embed_tokens.weight", "lm_head.weight":
		return "token_embd.weight"
	case "model.embedding_norm.weight":
		return "output_norm.weight"
	}
	var layer int
	var rest string
	if _, err := fmt.Sscanf(name, "model.layers.%d.%s", &layer, &rest); err != nil {
		return ""
	}
	m := map[string]string{
		"operator_norm.weight":       "attn_norm.weight",
		"post_operator_norm.weight":  "post_attention_norm.weight",
		"ffn_norm.weight":            "ffn_norm.weight",
		"post_ffn_norm.weight":       "post_ffw_norm.weight",
		"feed_forward.w1.weight":     "ffn_gate.weight",
		"feed_forward.w3.weight":     "ffn_up.weight",
		"feed_forward.w2.weight":     "ffn_down.weight",
		"conv.conv.weight":           "shortconv.conv.weight",
		"conv.conv.bias":             "shortconv.conv.bias",
		"conv.in_proj.weight":        "shortconv.in_proj.weight",
		"conv.in_proj.bias":          "shortconv.in_proj.bias",
		"conv.out_proj.weight":       "shortconv.out_proj.weight",
		"conv.out_proj.bias":         "shortconv.out_proj.bias",
		"self_attn.q_proj.weight":    "attn_q.weight",
		"self_attn.k_proj.weight":    "attn_k.weight",
		"self_attn.v_proj.weight":    "attn_v.weight",
		"self_attn.o_proj.weight":    "attn_output.weight",
		"self_attn.gate_proj.weight": "attn_gate.weight",
		"self_attn.q_norm.weight":    "attn_q_norm.weight",
		"self_attn.k_norm.weight":    "attn_k_norm.weight",
	}
	g, ok := m[rest]
	if !ok {
		return ""
	}
	return fmt.Sprintf("blk.%d.%s", layer, g)
}

// ggufShape restores the torch Conv1d layout (channels, 1, kernel) of the short convolution,
// which llama.cpp stores as (channels, kernel).
func ggufShape(name string, shape []int) []int {
	if strings.HasSuffix(name, "conv.conv.weight") && len(shape) == 2 {
		return []int{shape[0], 1, shape[1]}
	}
	return shape
}
