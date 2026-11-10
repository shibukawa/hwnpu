package mil

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu/internal/imagegen/milpart"
)

// Part is one program of Sana (milpart.Part).
type Part = milpart.Part

type graph = milpart.Graph

func part(name string, body func(g *graph)) *Part {
	return &Part{Name: name, Model: "sana", Body: body}
}

// Dit is SanaTransformer2DModel's shape (transformer/config.json) at one picture size.
type Dit struct {
	Width, Height int
	Hidden        int // heads x 32
	Heads         int
	Layers        int
	CrossHeads    int
	CrossHeadDim  int
	MLP           int // the GLUMBConv's hidden channels
	Caption       int // the text's width (Gemma 2's hidden, 2304)
	Latent        int // DC-AE's channels (32)
	QKNorm        bool
}

// VAEScale is DC-AE's downsampling: a latent pixel is 32 x 32 of the picture's.
const VAEScale = 32

// TextRows is the caption's rows the DiT takes (the pipelines' max_sequence_length).
const TextRows = 300

// GW and GH are the latents' columns and rows.
func (c Dit) GW() int { return c.Width / VAEScale }

// GH is the latents' rows.
func (c Dit) GH() int { return c.Height / VAEScale }

// Tokens is the latent pixels.
func (c Dit) Tokens() int { return c.GW() * c.GH() }

// ReadDit reads root/transformer/config.json for a picture of w x h pixels.
func ReadDit(root string, w, h int) (Dit, error) {
	b, err := os.ReadFile(filepath.Join(root, "transformer", "config.json"))
	if err != nil {
		return Dit{}, err
	}
	var cfg struct {
		Heads        int      `json:"num_attention_heads"`
		HeadDim      int      `json:"attention_head_dim"`
		Layers       int      `json:"num_layers"`
		CrossHeads   int      `json:"num_cross_attention_heads"`
		CrossHeadDim int      `json:"cross_attention_head_dim"`
		MLPRatio     float64  `json:"mlp_ratio"`
		Caption      int      `json:"caption_channels"`
		In           int      `json:"in_channels"`
		Patch        int      `json:"patch_size"`
		Interp       *float64 `json:"interpolation_scale"`
		Guidance     bool     `json:"guidance_embeds"`
		QKNorm       *string  `json:"qk_norm"`
		AttnBias     bool     `json:"attention_bias"`
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		return Dit{}, fmt.Errorf("sana: transformer/config.json: %w", err)
	}
	switch {
	case cfg.Guidance:
		return Dit{}, ErrSprint
	case cfg.HeadDim != 32 || cfg.Patch > 1:
		return Dit{}, fmt.Errorf("sana: heads of %d and patches of %d (32 and 1 supported)", cfg.HeadDim, cfg.Patch)
	case cfg.Interp != nil:
		return Dit{}, fmt.Errorf("sana: a positional embedding is not supported")
	case cfg.AttnBias:
		return Dit{}, fmt.Errorf("sana: the self-attention's biases are not supported")
	case cfg.QKNorm != nil && *cfg.QKNorm != "rms_norm_across_heads":
		return Dit{}, fmt.Errorf("sana: qk_norm %q is not supported", *cfg.QKNorm)
	case w <= 0 || h <= 0 || w%VAEScale != 0 || h%VAEScale != 0:
		return Dit{}, fmt.Errorf("sana: %dx%d is not in multiples of %d", w, h, VAEScale)
	}
	E := cfg.Heads * cfg.HeadDim
	return Dit{Width: w, Height: h, Hidden: E, Heads: cfg.Heads, Layers: cfg.Layers, CrossHeads: cfg.CrossHeads,
		CrossHeadDim: cfg.CrossHeadDim, MLP: int(cfg.MLPRatio * float64(E)), Caption: cfg.Caption, Latent: cfg.In,
		QKNorm: cfg.QKNorm != nil}, nil
}

// ErrSprint is the error of a Sana-Sprint checkpoint (a DiT with the guidance embedded, SCM
// steps), which hwnpu no longer runs: its pictures were far from Sana's (design.md 16.148).
var ErrSprint = errors.New("sana: Sana-Sprint is not supported (its pictures were far below Sana's); use Sana_600M_1024px_diffusers or SANA1.5_1.6B_1024px_diffusers")

// Gemma is Gemma 2's shape (text_encoder/config.json).
type Gemma struct {
	Hidden, Layers, Heads, KVHeads, HeadDim, MLP int
	Theta, Eps, QueryScalar                      float64
	Vocab                                        int
}

// TextLen is the prompt's rows Gemma 2 runs on: the instruction (about 210 tokens) and up to
// 298 of the prompt's (sana.Pipeline.Tokens).
const TextLen = 512

// ReadGemma reads text/text_encoder/config.json.
func ReadGemma(text string) (Gemma, error) {
	b, err := os.ReadFile(filepath.Join(text, "text_encoder", "config.json"))
	if err != nil {
		return Gemma{}, err
	}
	var cfg struct {
		Type    string  `json:"model_type"`
		Hidden  int     `json:"hidden_size"`
		Layers  int     `json:"num_hidden_layers"`
		Heads   int     `json:"num_attention_heads"`
		KV      int     `json:"num_key_value_heads"`
		HeadDim int     `json:"head_dim"`
		MLP     int     `json:"intermediate_size"`
		Theta   float64 `json:"rope_theta"`
		Eps     float64 `json:"rms_norm_eps"`
		QS      float64 `json:"query_pre_attn_scalar"`
		Vocab   int     `json:"vocab_size"`
		Act     string  `json:"hidden_activation"`
		Window  int     `json:"sliding_window"`
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		return Gemma{}, fmt.Errorf("sana: text_encoder/config.json: %w", err)
	}
	if cfg.Type != "gemma2" || cfg.Act != "gelu_pytorch_tanh" || cfg.Window < TextLen {
		return Gemma{}, fmt.Errorf("sana: text_encoder is %s (%s, window %d), not Gemma 2", cfg.Type, cfg.Act, cfg.Window)
	}
	return Gemma{Hidden: cfg.Hidden, Layers: cfg.Layers, Heads: cfg.Heads, KVHeads: cfg.KV, HeadDim: cfg.HeadDim, MLP: cfg.MLP,
		Theta: cfg.Theta, Eps: cfg.Eps, QueryScalar: cfg.QS, Vocab: cfg.Vocab}, nil
}
