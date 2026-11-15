// Package mil describes FLUX.2 [klein] 4B (black-forest-labs/FLUX.2-klein-4B) as MIL programs,
// which hwnpu writes as Core AI programs for Apple's Neural Engine (llm.WritePart): the parts of
// tools/amd/iron_flux2klein.py's graphs. Qwen3-4B's first 27 layers in groups (the DiT reads the
// hidden states after layers 9, 18 and 27), context_embedder of the three side by side, the DiT's
// double-stream and single-stream blocks in groups (the first embedding the latents' patches, the
// last ending with norm_out and proj_out), and the FLUX.2 VAE's decoder in stages. The timestep's
// embedding and the shared modulations are the host's (a few rows a step, in float32:
// new-model-checklist 3.3).
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

// Part is one program of FLUX.2 klein (milpart.Part).
type Part = milpart.Part

type graph = milpart.Graph

func part(name string, body func(g *graph)) *Part {
	return &Part{Name: name, Model: "flux2klein", Body: body}
}

// TextRows is the text's rows: the prompt in Qwen3's chat template, padded to the pipeline's
// max_sequence_length.
const TextRows = 512

// TextTaps are the hidden states the DiT reads: after these many of Qwen3's layers, side by side.
var TextTaps = []int{9, 18, 27}

// Dit is Flux2Transformer2DModel's shape (transformer/config.json) at one picture size.
type Dit struct {
	Width, Height int
	Hidden        int // heads x head_dim (3072)
	Heads         int
	HeadDim       int
	MLP           int // mlp_ratio x hidden (9216)
	Latent        int // a 2 x 2 patch of the VAE's 32 channels (128)
	Context       int // the text's width: three hidden states of Qwen3 (7680)
	Doubles       int
	Singles       int
	Axes          []int // the rotary embedding's dimensions per axis (32, 32, 32, 32)
	Theta         float64
	Eps           float64
	TimeIn        int // the timestep's sinusoid (256)
	// TextRows is the text's rows the DiT takes.
	TextRows int
}

// GH is the patches' rows (16 x 16 pixels a patch).
func (c Dit) GH() int { return c.Height / 16 }

// GW is the patches' columns.
func (c Dit) GW() int { return c.Width / 16 }

// Tokens is the image's rows.
func (c Dit) Tokens() int { return c.GH() * c.GW() }

// Rows is the DiT's rows: the image's, then the text's.
func (c Dit) Rows() int { return c.Tokens() + c.TextRows }

// The modulation rows the host gives the programs (the timestep's, shared by the blocks): the
// double blocks' image stream (shift, scale, gate of the attention, then of the MLP), their text
// stream's, the single blocks' (shift, scale, gate) and norm_out's (scale, shift).
const (
	ModImage  = 0
	ModText   = 6
	ModSingle = 12
	ModOut    = 15
	ModRows   = 17
)

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
		Singles  int     `json:"num_single_layers"`
		In       int     `json:"in_channels"`
		Out      *int    `json:"out_channels"`
		Patch    int     `json:"patch_size"`
		Joint    int     `json:"joint_attention_dim"`
		Axes     []int   `json:"axes_dims_rope"`
		MLPRatio float64 `json:"mlp_ratio"`
		Theta    float64 `json:"rope_theta"`
		Eps      float64 `json:"eps"`
		Guidance bool    `json:"guidance_embeds"`
		TimeIn   int     `json:"timestep_guidance_channels"`
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		return Dit{}, fmt.Errorf("flux2klein: transformer/config.json: %w", err)
	}
	sum := 0
	for _, a := range cfg.Axes {
		sum += a
	}
	E := cfg.Heads * cfg.HeadDim
	switch {
	case cfg.Class != "Flux2Transformer2DModel":
		return Dit{}, fmt.Errorf("flux2klein: the transformer is %s, not Flux2Transformer2DModel", cfg.Class)
	case cfg.Patch != 1 || (cfg.Out != nil && *cfg.Out != cfg.In):
		return Dit{}, fmt.Errorf("flux2klein: patches of %d, %d channels (1 supported)", cfg.Patch, cfg.In)
	case len(cfg.Axes) != 4 || sum != cfg.HeadDim || cfg.HeadDim%2 != 0:
		return Dit{}, fmt.Errorf("flux2klein: rotary axes %v for heads of %d", cfg.Axes, cfg.HeadDim)
	case cfg.Guidance:
		return Dit{}, fmt.Errorf("flux2klein: a guidance embedding (the undistilled FLUX.2): not supported")
	case cfg.Joint != len(TextTaps)*2560:
		return Dit{}, fmt.Errorf("flux2klein: text of %d (Qwen3-4B's three hidden states are %d)", cfg.Joint, len(TextTaps)*2560)
	case w <= 0 || h <= 0 || w%16 != 0 || h%16 != 0:
		return Dit{}, fmt.Errorf("flux2klein: %dx%d is not in multiples of 16", w, h)
	}
	return Dit{Width: w, Height: h, Hidden: E, Heads: cfg.Heads, HeadDim: cfg.HeadDim, MLP: int(cfg.MLPRatio * float64(E)),
		Latent: cfg.In, Context: cfg.Joint, Doubles: cfg.Layers, Singles: cfg.Singles, Axes: cfg.Axes, Theta: cfg.Theta,
		Eps: cfg.Eps, TimeIn: cfg.TimeIn, TextRows: TextRows}, nil
}

// Text is Qwen3-4B's shape (text_encoder/config.json).
type Text struct {
	Hidden, Layers, Heads, KVHeads, HeadDim, MLP, Vocab int
	Theta, Eps                                          float64
}

// ReadText reads root/text_encoder/config.json.
func ReadText(root string) (Text, error) {
	b, err := os.ReadFile(filepath.Join(root, "text_encoder", "config.json"))
	if err != nil {
		return Text{}, err
	}
	var cfg struct {
		Type    string  `json:"model_type"`
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
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		return Text{}, fmt.Errorf("flux2klein: text_encoder/config.json: %w", err)
	}
	if cfg.Type != "qwen3" || cfg.Act != "silu" || cfg.Bias || cfg.Layers < TextTaps[len(TextTaps)-1] {
		return Text{}, fmt.Errorf("flux2klein: the text encoder is %s (%s, %d layers), not Qwen3's", cfg.Type, cfg.Act, cfg.Layers)
	}
	return Text{Hidden: cfg.Hidden, Layers: cfg.Layers, Heads: cfg.Heads, KVHeads: cfg.KV, HeadDim: cfg.HeadDim, MLP: cfg.MLP,
		Vocab: cfg.Vocab, Theta: cfg.Theta, Eps: cfg.Eps}, nil
}

// TextGroups are the programs of Qwen3's layers the DiT reads, [first, count]: per layers each,
// none across a tap (TextTaps), so that each tap is a program's output.
func TextGroups(per int) [][2]int {
	var out [][2]int
	first := 0
	for _, tap := range TextTaps {
		for first < tap {
			n := min(per, tap-first)
			out = append(out, [2]int{first, n})
			first += n
		}
	}
	return out
}

// Groups splits n blocks into programs of at most per each, as even as they go (20 by 6: 5, 5,
// 5, 5).
func Groups(n, per int) [][2]int {
	k := (n + per - 1) / per
	var out [][2]int
	first := 0
	for i := 0; i < k; i++ {
		c := n / k
		if i < n%k {
			c++
		}
		out = append(out, [2]int{first, c})
		first += c
	}
	return out
}

// Vae is the FLUX.2 VAE's shape (vae/config.json).
type Vae struct {
	Channels []int // block_out_channels (128, 256, 512, 512)
	Layers   int   // layers_per_block (2: the decoder's up blocks have 3)
	Groups   int
	Latent   int     // 32
	BNEps    float64 // the batch norm of the patched latents
}

// ReadVae reads root/vae/config.json.
func ReadVae(root string) (Vae, error) {
	b, err := os.ReadFile(filepath.Join(root, "vae", "config.json"))
	if err != nil {
		return Vae{}, err
	}
	var cfg struct {
		Class    string  `json:"_class_name"`
		Channels []int   `json:"block_out_channels"`
		Layers   int     `json:"layers_per_block"`
		Groups   int     `json:"norm_num_groups"`
		Latent   int     `json:"latent_channels"`
		Attn     *bool   `json:"mid_block_add_attention"`
		PostQ    *bool   `json:"use_post_quant_conv"`
		Act      string  `json:"act_fn"`
		BNEps    float64 `json:"batch_norm_eps"`
		Patch    []int   `json:"patch_size"`
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		return Vae{}, fmt.Errorf("flux2klein: vae/config.json: %w", err)
	}
	if cfg.Class != "AutoencoderKLFlux2" || fmt.Sprint(cfg.Channels) != "[128 256 512 512]" || cfg.Layers != 2 || cfg.Groups != 32 ||
		cfg.Act != "silu" || (cfg.Attn != nil && !*cfg.Attn) || cfg.PostQ == nil || !*cfg.PostQ || fmt.Sprint(cfg.Patch) != "[2 2]" {
		return Vae{}, fmt.Errorf("flux2klein: the VAE (%s %v, %d latents) is not FLUX.2's", cfg.Class, cfg.Channels, cfg.Latent)
	}
	return Vae{Channels: cfg.Channels, Layers: cfg.Layers, Groups: cfg.Groups, Latent: cfg.Latent, BNEps: cfg.BNEps}, nil
}
