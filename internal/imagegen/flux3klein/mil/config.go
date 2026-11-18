// Package mil describes FLUX.3 [klein] (black-forest-labs/flux-3-image-klein) as MIL programs,
// which hwnpu writes as Core AI programs for Apple's Neural Engine (llm.WritePart). Qwen3-VL's
// language model up to the last hidden state the DiT reads (after layers 4, 8, ..., 32) in groups,
// context_embedder of the eight side by side, the DiT's text and image mode blocks (each stream
// attending to itself), its single blocks over [text, image] in groups (the last ending with
// norm_out and proj_out), and the FLUX.2 VAE's decoder in stages (FLUX.2 klein's, which is
// Chroma's). The timesteps' embeddings and the modulations are the host's (14 rows a step, in
// float32: new-model-checklist 3.3).
//
// They run in fp16, the residual streams held divided by TextScale and DitScale and every norm
// first dividing each row by its largest magnitude, as FLUX.2 klein's (that package's doc).
package mil

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	kmil "github.com/shibukawa/hwnpu/internal/imagegen/flux2klein/mil"
	"github.com/shibukawa/hwnpu/internal/imagegen/milpart"
)

// Part is one program of FLUX.3 klein (milpart.Part).
type Part = milpart.Part

type graph = milpart.Graph

func part(name string, body func(g *graph)) *Part {
	return &Part{Name: name, Model: "flux3klein", Body: body}
}

// TextPad is what the pipeline pads the prompt to a multiple of: the text's rows are the prompt's
// tokens in Qwen3-VL's chat template rounded up to it (pad_multiple of config.json).
const TextPad = 80

// TextTaps are the hidden states the DiT reads: after these many of Qwen3-VL's layers, side by
// side (text_encoder_out_layers).
var TextTaps = []int{4, 8, 12, 16, 20, 24, 28, 32}

// Dit is Flux3Transformer2DModel's shape (transformer/config.json) at one picture size and one
// text length.
type Dit struct {
	Width, Height int
	Hidden        int // heads x head_dim (3072)
	Heads         int
	HeadDim       int
	MLP           int // mlp_ratio x hidden (9216)
	Latent        int // a 2 x 2 patch of the VAE's 32 channels (128)
	Context       int // the text's width: eight hidden states of Qwen3-VL (20480)
	Modes         int // the mode blocks of each stream (5)
	Singles       int // the single blocks (28)
	Axes          []int
	Theta         float64
	Eps           float64
	TimeIn        int // the timesteps' sinusoid (256)
	// TextRows is the text's rows the DiT takes: a multiple of TextPad.
	TextRows int
}

// ImageRopeScale scales the image's row and column position ids (the model was trained with them
// halved: Flux3PrepareLatentsStep's image_rope_scale).
const ImageRopeScale = 0.5

// GH is the patches' rows (16 x 16 pixels a patch).
func (c Dit) GH() int { return c.Height / 16 }

// GW is the patches' columns.
func (c Dit) GW() int { return c.Width / 16 }

// Tokens is the image's rows.
func (c Dit) Tokens() int { return c.GH() * c.GW() }

// Rows is the single blocks' rows: the text's, then the image's.
func (c Dit) Rows() int { return c.TextRows + c.Tokens() }

// The modulation rows the host gives the programs, a step's (each stream's are the same for all
// its rows): the text mode blocks' (shift, scale, gate), the image mode blocks', the single
// blocks' text and image ones, and norm_out's (scale, shift).
const (
	ModTextEarly   = 0
	ModImageEarly  = 3
	ModTextSingle  = 6
	ModImageSingle = 9
	ModOut         = 12
	ModRows        = 14
)

// ReadDit reads root/transformer/config.json for a picture of w x h pixels and text of rows rows.
func ReadDit(root string, w, h, rows int) (Dit, error) {
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
		CondIn   int     `json:"cond_in_channels"`
		Joint    int     `json:"joint_attention_dim"`
		Axes     []int   `json:"axes_dims_rope"`
		MLPRatio float64 `json:"mlp_ratio"`
		Theta    float64 `json:"rope_theta"`
		Eps      float64 `json:"eps"`
		TimeIn   int     `json:"timestep_channels"`
		FlowMap  bool    `json:"flow_map"`
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		return Dit{}, fmt.Errorf("flux3klein: transformer/config.json: %w", err)
	}
	sum := 0
	for _, a := range cfg.Axes {
		sum += a
	}
	E := cfg.Heads * cfg.HeadDim
	switch {
	case cfg.Class != "Flux3Transformer2DModel":
		return Dit{}, fmt.Errorf("flux3klein: the transformer is %s, not Flux3Transformer2DModel", cfg.Class)
	case !cfg.FlowMap:
		return Dit{}, fmt.Errorf("flux3klein: the transformer is not a flow map (the undistilled FLUX.3, with guidance): not supported")
	case len(cfg.Axes) != 4 || sum != cfg.HeadDim || cfg.HeadDim%2 != 0:
		return Dit{}, fmt.Errorf("flux3klein: rotary axes %v for heads of %d", cfg.Axes, cfg.HeadDim)
	case cfg.Joint != len(TextTaps)*2560:
		return Dit{}, fmt.Errorf("flux3klein: text of %d (Qwen3-VL's eight hidden states are %d)", cfg.Joint, len(TextTaps)*2560)
	case w <= 0 || h <= 0 || w%16 != 0 || h%16 != 0:
		return Dit{}, fmt.Errorf("flux3klein: %dx%d is not in multiples of 16", w, h)
	case rows <= 0 || rows%TextPad != 0:
		return Dit{}, fmt.Errorf("flux3klein: text of %d rows (a multiple of %d)", rows, TextPad)
	}
	timeIn := cfg.TimeIn
	if timeIn == 0 {
		timeIn = 256
	}
	return Dit{Width: w, Height: h, Hidden: E, Heads: cfg.Heads, HeadDim: cfg.HeadDim, MLP: int(cfg.MLPRatio * float64(E)),
		Latent: cfg.In, Context: cfg.Joint, Modes: cfg.Layers, Singles: cfg.Singles, Axes: cfg.Axes, Theta: cfg.Theta,
		Eps: cfg.Eps, TimeIn: timeIn, TextRows: rows}, nil
}

// Text is Qwen3-VL's language model's shape (text_encoder/config.json's text_config).
type Text struct {
	Hidden, Layers, Heads, KVHeads, HeadDim, MLP, Vocab int
	Theta, Eps                                          float64
}

// ReadText reads root/text_encoder/config.json: a Qwen3-VL whose language model is read up to the
// last tap. Its multimodal rotary embedding (mrope) turns text alone as a plain one: the three
// position axes are each token's index.
func ReadText(root string) (Text, error) {
	b, err := os.ReadFile(filepath.Join(root, "text_encoder", "config.json"))
	if err != nil {
		return Text{}, err
	}
	var cfg struct {
		Type string `json:"model_type"`
		Text struct {
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
		} `json:"text_config"`
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		return Text{}, fmt.Errorf("flux3klein: text_encoder/config.json: %w", err)
	}
	t := cfg.Text
	if cfg.Type != "qwen3_vl" || t.Type != "qwen3_vl_text" || t.Act != "silu" || t.Bias || t.Layers < TextTaps[len(TextTaps)-1] {
		return Text{}, fmt.Errorf("flux3klein: the text encoder is %s/%s (%s, %d layers), not Qwen3-VL's", cfg.Type, t.Type, t.Act, t.Layers)
	}
	return Text{Hidden: t.Hidden, Layers: t.Layers, Heads: t.Heads, KVHeads: t.KV, HeadDim: t.HeadDim, MLP: t.MLP,
		Vocab: t.Vocab, Theta: t.Theta, Eps: t.Eps}, nil
}

// TextGroups are the programs of Qwen3-VL's layers the DiT reads, [first, count]: per layers each,
// none across a tap, so that each tap is a program's output.
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

// Groups splits n blocks into programs of at most per each, as even as they go.
func Groups(n, per int) [][2]int { return kmil.Groups(n, per) }

// Vae is the FLUX.2 VAE's shape: FLUX.3 klein's VAE is AutoencoderKLFlux2.
type Vae = kmil.Vae

// ReadVae reads root/vae/config.json.
func ReadVae(root string) (Vae, error) { return kmil.ReadVae(root) }
