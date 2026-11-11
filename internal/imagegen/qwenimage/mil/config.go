// Package mil describes Qwen-Image-2.1 (Qwen/Qwen-Image-2.1-Turbo) as MIL programs, which hwnpu
// writes as Core ML packages for Apple's Neural Engine (llm.WritePart): the parts of
// tools/amd/iron_qwenimage.py's graphs. Qwen3-VL-8B's language model in groups of layers (the
// last with the DiT's txt_in), the DiT's text pass (once a prompt: its blocks over the text rows at
// t = 0, giving each block's keys and values), the DiT's blocks over the image rows in groups, and
// the VAE's decoder in stages. The timestep's embedding and the shared modulation are the host's
// (one row, computed in float32: new-model-checklist 3.3).
//
// They run in fp16. The residual streams of the text encoder and of the DiT are held divided by
// TextScale and DitScale (the embedding rows, the output projections and the gates scaled down),
// and every norm first divides each row by its largest magnitude (norms do not change with the
// scale of their input), so that no square overflows.
package mil

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu/internal/imagegen/milpart"
)

// Part is one program of Qwen-Image (milpart.Part).
type Part = milpart.Part

type graph = milpart.Graph

func part(name string, body func(g *graph)) *Part {
	return &Part{Name: name, Model: "qwenimage", Body: body}
}

// VAEScale is the VAE's downsampling: a latent pixel is 16 x 16 of the picture's.
const VAEScale = 16

// TextLen is the prompt's positions the text encoder runs on: the template (about 25 tokens, the
// system message's 14 dropped) and the prompt. TextRows is the DiT's text rows: the rows after the
// system message, zeros past them, which the image's attention masks.
const (
	TextLen  = 512
	TextRows = TextLen
)

// Dit is QwenImage21Transformer2DModel's shape (transformer/config.json) at one picture size.
type Dit struct {
	Width, Height int
	Hidden        int // heads x head_dim
	Heads         int
	HeadDim       int
	Layers        int
	MLP           int
	Latent        int
	Axes          [3]int // RoPE's dimensions of the frame, the height and the width
	Theta         float64
	Eps           float64
}

// GW and GH are the latents' columns and rows.
func (c Dit) GW() int { return c.Width / VAEScale }

// GH is the latents' rows.
func (c Dit) GH() int { return c.Height / VAEScale }

// Tokens is the image's rows.
func (c Dit) Tokens() int { return c.GW() * c.GH() }

// ReadDit reads root/transformer/config.json for a picture of w x h pixels.
func ReadDit(root string, w, h int) (Dit, error) {
	b, err := os.ReadFile(filepath.Join(root, "transformer", "config.json"))
	if err != nil {
		return Dit{}, err
	}
	var cfg struct {
		Class    string  `json:"_class_name"`
		HeadDim  int     `json:"attention_head_dim"`
		Heads    int     `json:"num_attention_heads"`
		Layers   int     `json:"num_layers"`
		Context  int     `json:"context_in_dim"`
		MLPRatio int     `json:"mlp_ratio"`
		Axes     []int   `json:"axes_dims_rope"`
		In       int     `json:"in_channels"`
		Out      int     `json:"out_channels"`
		Patch    int     `json:"patch_size"`
		Eps      float64 `json:"eps"`
		Causal   bool    `json:"causal_condition"`
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		return Dit{}, fmt.Errorf("qwenimage: transformer/config.json: %w", err)
	}
	E := cfg.Heads * cfg.HeadDim
	switch {
	case cfg.Class != "QwenImage21Transformer2DModel":
		return Dit{}, fmt.Errorf("qwenimage: the transformer is %s, not Qwen-Image-2.1's", cfg.Class)
	case cfg.Patch > 1 || cfg.In != cfg.Out || len(cfg.Axes) != 3 || cfg.Axes[0]+cfg.Axes[1]+cfg.Axes[2] != cfg.HeadDim:
		return Dit{}, fmt.Errorf("qwenimage: patches of %d, %d to %d channels, RoPE %v: not supported", cfg.Patch, cfg.In, cfg.Out, cfg.Axes)
	case !cfg.Causal:
		return Dit{}, fmt.Errorf("qwenimage: the text's keys and values are held once a prompt, which needs causal_condition")
	case cfg.Context != E:
		return Dit{}, fmt.Errorf("qwenimage: text of %d for a DiT of %d", cfg.Context, E)
	case w <= 0 || h <= 0 || w%(2*VAEScale) != 0 || h%(2*VAEScale) != 0:
		return Dit{}, fmt.Errorf("qwenimage: %dx%d is not in multiples of %d", w, h, 2*VAEScale)
	}
	return Dit{Width: w, Height: h, Hidden: E, Heads: cfg.Heads, HeadDim: cfg.HeadDim, Layers: cfg.Layers,
		MLP: cfg.MLPRatio * E, Latent: cfg.In, Axes: [3]int{cfg.Axes[0], cfg.Axes[1], cfg.Axes[2]}, Theta: 10000, Eps: cfg.Eps}, nil
}

// Text is Qwen3-VL-8B's language model's shape (text_encoder/config.json's text_config).
type Text struct {
	Hidden, Layers, Heads, KVHeads, HeadDim, MLP, Vocab int
	Theta, Eps                                          float64
}

// ReadText reads text/text_encoder/config.json.
func ReadText(text string) (Text, error) {
	b, err := os.ReadFile(filepath.Join(text, "text_encoder", "config.json"))
	if err != nil {
		return Text{}, err
	}
	var cfg struct {
		Type string `json:"model_type"`
		Text struct {
			Hidden  int     `json:"hidden_size"`
			Layers  int     `json:"num_hidden_layers"`
			Heads   int     `json:"num_attention_heads"`
			KV      int     `json:"num_key_value_heads"`
			HeadDim int     `json:"head_dim"`
			MLP     int     `json:"intermediate_size"`
			Vocab   int     `json:"vocab_size"`
			Theta   float64 `json:"rope_theta"`
			Eps     float64 `json:"rms_norm_eps"`
			Act     string  `json:"hidden_act"`
			Bias    bool    `json:"attention_bias"`
		} `json:"text_config"`
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		return Text{}, fmt.Errorf("qwenimage: text_encoder/config.json: %w", err)
	}
	t := cfg.Text
	if cfg.Type != "qwen3_vl" || t.Act != "silu" || t.Bias {
		return Text{}, fmt.Errorf("qwenimage: the text encoder is %s (%s), not Qwen3-VL's", cfg.Type, t.Act)
	}
	return Text{Hidden: t.Hidden, Layers: t.Layers, Heads: t.Heads, KVHeads: t.KV, HeadDim: t.HeadDim, MLP: t.MLP,
		Vocab: t.Vocab, Theta: t.Theta, Eps: t.Eps}, nil
}

// Groups are the groups of count layers (or blocks) of n, [first, count], the last shorter.
func Groups(n, count int) [][2]int {
	var out [][2]int
	for first := 0; first < n; first += count {
		out = append(out, [2]int{first, min(count, n-first)})
	}
	return out
}
