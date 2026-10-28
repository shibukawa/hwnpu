package lfm3

import (
	"fmt"
	"math"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu/internal/gguf"
	"github.com/shibukawa/hwnpu/internal/safetensors"
)

func pow(a, b float64) float64 { return math.Pow(a, b) }

// Weights supplies tensors by their Hugging Face name (model.layers.3.self_attn.q_proj.weight) as
// float32 in row major order, whatever the checkpoint's own naming and storage format.
type Weights interface {
	Get(name string) ([]float32, []int, error)
	// Rows reads rows [begin, end) of a matrix, for embedding lookups without loading the table.
	Rows(name string, begin, end int) ([]float32, error)
	// Source describes the checkpoint for the model metadata.
	Source() string
	Close() error
}

// Embed gathers the embedding rows of the given tokens as fp32 (tokens x hidden).
func Embed(w Weights, hidden int, tokens []int32) ([]float32, error) {
	out := make([]float32, 0, len(tokens)*hidden)
	for _, t := range tokens {
		row, err := w.Rows("model.embed_tokens.weight", int(t), int(t)+1)
		if err != nil {
			return nil, err
		}
		out = append(out, row...)
	}
	return out, nil
}

// OpenWeights opens a safetensors or GGUF checkpoint by its extension.
func OpenWeights(path string) (Weights, error) {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".safetensors":
		f, err := safetensors.Open(path)
		if err != nil {
			return nil, err
		}
		return stWeights{f: f, path: path}, nil
	case ".gguf":
		f, err := gguf.Open(path)
		if err != nil {
			return nil, err
		}
		if a := f.String("general.architecture"); a != "lfm3" {
			f.Close()
			return nil, fmt.Errorf("lfm3: %s is a %q model", path, a)
		}
		return ggufWeights{f: f, path: path}, nil
	}
	return nil, fmt.Errorf("lfm3: %s: expected a .safetensors or .gguf file", path)
}

type stWeights struct {
	f    *safetensors.File
	path string
}

func (w stWeights) Get(name string) ([]float32, []int, error) {
	t, ok := w.f.Lookup(name)
	if !ok {
		return nil, nil, fmt.Errorf("lfm3: %s has no tensor %s", w.path, name)
	}
	v, err := w.f.Float32(name)
	return v, t.Shape, err
}

func (w stWeights) Rows(name string, begin, end int) ([]float32, error) {
	t, ok := w.f.Lookup(name)
	if !ok || len(t.Shape) != 2 {
		return nil, fmt.Errorf("lfm3: %s has no matrix %s", w.path, name)
	}
	return w.f.Float32Range(name, begin*t.Shape[1], end*t.Shape[1])
}

func (w stWeights) Source() string { return "safetensors " + filepath.Base(w.path) }
func (w stWeights) Close() error   { return w.f.Close() }

type ggufWeights struct {
	f    *gguf.File
	path string
}

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

func (w ggufWeights) Get(name string) ([]float32, []int, error) {
	gn := ggufName(name)
	if gn == "" {
		return nil, nil, fmt.Errorf("lfm3: no GGUF name for %s", name)
	}
	t, ok := w.f.Lookup(gn)
	if !ok {
		return nil, nil, fmt.Errorf("lfm3: %s has no tensor %s (%s)", w.path, gn, name)
	}
	v, err := w.f.Float32(gn)
	shape := t.Shape
	if strings.HasSuffix(name, "conv.conv.weight") && len(shape) == 2 {
		shape = []int{shape[0], 1, shape[1]} // torch Conv1d layout (channels, 1, kernel)
	}
	return v, shape, err
}

func (w ggufWeights) Rows(name string, begin, end int) ([]float32, error) {
	gn := ggufName(name)
	t, ok := w.f.Lookup(gn)
	if gn == "" || !ok || len(t.Shape) != 2 {
		return nil, fmt.Errorf("lfm3: %s has no matrix %s", w.path, name)
	}
	return w.f.Float32Range(gn, begin*t.Shape[1], end*t.Shape[1])
}

func (w ggufWeights) Source() string {
	return fmt.Sprintf("gguf %s (%s)", filepath.Base(w.path), w.f.String("general.name"))
}
func (w ggufWeights) Close() error { return w.f.Close() }
