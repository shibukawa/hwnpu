package mil

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Dit is ChromaTransformer2DModel's shape (transformer/config.json) at one picture size.
type Dit struct {
	Width, Height int
	Hidden        int // heads x head_dim (3072)
	Heads         int
	HeadDim       int
	MLP           int // 4 hidden
	Latent        int // a 2 x 2 patch of the VAE's 16 channels (64)
	Context       int // T5's width (4096)
	Doubles       int
	Singles       int
	Axes          []int // the rotary embedding's dimensions per axis (16, 56, 56)
	ApproxIn      int   // the approximator's input (64)
	ApproxHidden  int
	ApproxLayers  int
	// TextRows is the text's rows the DiT takes (T5's).
	TextRows int
}

// GH and GW are the patches' rows and columns (16 x 16 pixels a patch).
func (c Dit) GH() int { return c.Height / 16 }

// GW is the patches' columns.
func (c Dit) GW() int { return c.Width / 16 }

// Tokens is the image's rows.
func (c Dit) Tokens() int { return c.GH() * c.GW() }

// Rows is the DiT's rows: the image's, then the text's.
func (c Dit) Rows() int { return c.Tokens() + c.TextRows }

// Mods is the approximator's rows: 3 a single block, 6 + 6 a double block, 2 the final norm.
func (c Dit) Mods() int { return 3*c.Singles + 12*c.Doubles + 2 }

// SingleMod is the first modulation row of single block i (shift, scale, gate).
func (c Dit) SingleMod(i int) int { return 3 * i }

// DoubleMod is the first modulation row of double block i's image stream (text: its text stream):
// shift, scale, gate of the attention, then of the MLP.
func (c Dit) DoubleMod(i int, text bool) int {
	r := 3*c.Singles + 6*i
	if text {
		r += 6 * c.Doubles
	}
	return r
}

// TextRows is T5's rows (ChromaPipeline's max_sequence_length).
const TextRows = 512

// ReadDit reads root/transformer/config.json for a picture of w x h pixels.
func ReadDit(root string, w, h int) (Dit, error) {
	b, err := os.ReadFile(filepath.Join(root, "transformer", "config.json"))
	if err != nil {
		return Dit{}, err
	}
	var cfg struct {
		Class        string `json:"_class_name"`
		HeadDim      int    `json:"attention_head_dim"`
		Heads        int    `json:"num_attention_heads"`
		Layers       int    `json:"num_layers"`
		Singles      int    `json:"num_single_layers"`
		In           int    `json:"in_channels"`
		Out          *int   `json:"out_channels"`
		Patch        int    `json:"patch_size"`
		Joint        int    `json:"joint_attention_dim"`
		Axes         []int  `json:"axes_dims_rope"`
		ApproxIn     int    `json:"approximator_num_channels"`
		ApproxHidden int    `json:"approximator_hidden_dim"`
		ApproxLayers int    `json:"approximator_layers"`
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		return Dit{}, fmt.Errorf("chroma: transformer/config.json: %w", err)
	}
	sum := 0
	for _, a := range cfg.Axes {
		sum += a
	}
	switch {
	case cfg.Class != "ChromaTransformer2DModel":
		return Dit{}, fmt.Errorf("chroma: the transformer is %s, not ChromaTransformer2DModel", cfg.Class)
	case cfg.Patch != 1 || cfg.In != 64 || (cfg.Out != nil && *cfg.Out != cfg.In):
		return Dit{}, fmt.Errorf("chroma: patches of %d, %d channels (1 and 64 supported)", cfg.Patch, cfg.In)
	case len(cfg.Axes) != 3 || sum != cfg.HeadDim || cfg.HeadDim%2 != 0:
		return Dit{}, fmt.Errorf("chroma: rotary axes %v for heads of %d", cfg.Axes, cfg.HeadDim)
	case w <= 0 || h <= 0 || w%16 != 0 || h%16 != 0:
		return Dit{}, fmt.Errorf("chroma: %dx%d is not in multiples of 16", w, h)
	}
	E := cfg.Heads * cfg.HeadDim
	return Dit{Width: w, Height: h, Hidden: E, Heads: cfg.Heads, HeadDim: cfg.HeadDim, MLP: 4 * E, Latent: cfg.In,
		Context: cfg.Joint, Doubles: cfg.Layers, Singles: cfg.Singles, Axes: cfg.Axes, ApproxIn: cfg.ApproxIn,
		ApproxHidden: cfg.ApproxHidden, ApproxLayers: cfg.ApproxLayers, TextRows: TextRows}, nil
}

// T5 is T5-XXL's encoder's shape (text_encoder/config.json).
type T5 struct {
	Hidden, Layers, Heads, HeadDim, MLP int
	Buckets, MaxDistance                int
	Eps                                 float64
	Vocab                               int
}

// ReadT5 reads text/text_encoder/config.json.
func ReadT5(text string) (T5, error) {
	b, err := os.ReadFile(filepath.Join(text, "text_encoder", "config.json"))
	if err != nil {
		return T5{}, err
	}
	var cfg struct {
		Type     string  `json:"model_type"`
		DModel   int     `json:"d_model"`
		DKV      int     `json:"d_kv"`
		DFF      int     `json:"d_ff"`
		Layers   int     `json:"num_layers"`
		Heads    int     `json:"num_heads"`
		Buckets  int     `json:"relative_attention_num_buckets"`
		MaxDist  int     `json:"relative_attention_max_distance"`
		Eps      float64 `json:"layer_norm_epsilon"`
		Proj     string  `json:"feed_forward_proj"`
		Vocab    int     `json:"vocab_size"`
		DenseAct string  `json:"dense_act_fn"`
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		return T5{}, fmt.Errorf("chroma: text_encoder/config.json: %w", err)
	}
	if cfg.Type != "t5" || cfg.Proj != "gated-gelu" || cfg.DenseAct != "gelu_new" {
		return T5{}, fmt.Errorf("chroma: text_encoder is %s (%s, %s), not T5 v1.1", cfg.Type, cfg.Proj, cfg.DenseAct)
	}
	return T5{Hidden: cfg.DModel, Layers: cfg.Layers, Heads: cfg.Heads, HeadDim: cfg.DKV, MLP: cfg.DFF,
		Buckets: cfg.Buckets, MaxDistance: cfg.MaxDist, Eps: cfg.Eps, Vocab: cfg.Vocab}, nil
}

// Vae is the FLUX VAE's shape (vae/config.json).
type Vae struct {
	Channels []int // block_out_channels (128, 256, 512, 512)
	Layers   int   // layers_per_block (2: the decoder's up blocks have 3)
	Groups   int
	Latent   int
	Scaling  float64
	Shift    float64
}

// ReadVae reads vae/config.json of dir.
func ReadVae(dir string) (Vae, error) {
	b, err := os.ReadFile(filepath.Join(dir, "vae", "config.json"))
	if err != nil {
		return Vae{}, err
	}
	var cfg struct {
		Class    string  `json:"_class_name"`
		Channels []int   `json:"block_out_channels"`
		Layers   int     `json:"layers_per_block"`
		Groups   int     `json:"norm_num_groups"`
		Latent   int     `json:"latent_channels"`
		Scaling  float64 `json:"scaling_factor"`
		Shift    float64 `json:"shift_factor"`
		Attn     *bool   `json:"mid_block_add_attention"`
		PostQ    *bool   `json:"use_post_quant_conv"`
		Act      string  `json:"act_fn"`
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		return Vae{}, fmt.Errorf("chroma: vae/config.json: %w", err)
	}
	if cfg.Class != "AutoencoderKL" || fmt.Sprint(cfg.Channels) != "[128 256 512 512]" || cfg.Layers != 2 || cfg.Groups != 32 ||
		cfg.Latent != 16 || cfg.Act != "silu" || (cfg.Attn != nil && !*cfg.Attn) || (cfg.PostQ != nil && *cfg.PostQ) {
		return Vae{}, fmt.Errorf("chroma: the VAE (%s %v, %d latents) is not FLUX's", cfg.Class, cfg.Channels, cfg.Latent)
	}
	return Vae{Channels: cfg.Channels, Layers: cfg.Layers, Groups: cfg.Groups, Latent: cfg.Latent, Scaling: cfg.Scaling, Shift: cfg.Shift}, nil
}
