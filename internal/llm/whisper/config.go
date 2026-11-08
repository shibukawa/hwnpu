// Package whisper converts OpenAI's Whisper (Hugging Face `whisper`, WhisperForConditionalGeneration:
// whisper-large-v3-turbo, kotoba-whisper and the like) into Core ML programs for the Neural
// Engine: the audio encoder (architecture whisper-encoder, chosen with -arch) and the text decoder
// (whisper).
//
// The structure follows transformers' modeling_whisper.py. The encoder reads 30 seconds of log
// mel frames (3000 frames of 10 ms): two convolutions of kernel 3 with GELU, the second of stride
// 2, give 1500 positions, to which the sinusoidal position embeddings are added; then pre norm
// layers of bidirectional attention and a GELU MLP, and a final LayerNorm. The decoder is the
// same pre norm stack with learned position embeddings, causal self attention and cross
// attention over the encoder's output between them, its output layer the token embeddings.
package whisper

import (
	"encoding/json"
	"fmt"
	"slices"
)

// Config is the subset of config.json the conversion needs.
type Config struct {
	ModelType        string   `json:"model_type"`
	Architectures    []string `json:"architectures"`
	DModel           int      `json:"d_model"`
	EncoderLayers    int      `json:"encoder_layers"`
	EncoderHeads     int      `json:"encoder_attention_heads"`
	EncoderFFN       int      `json:"encoder_ffn_dim"`
	DecoderLayers    int      `json:"decoder_layers"`
	DecoderHeads     int      `json:"decoder_attention_heads"`
	DecoderFFN       int      `json:"decoder_ffn_dim"`
	MelBins          int      `json:"num_mel_bins"`
	SourcePositions  int      `json:"max_source_positions"`
	TargetPositions  int      `json:"max_target_positions"`
	VocabSize        int      `json:"vocab_size"`
	Activation       string   `json:"activation_function"`
	ScaleEmbedding   bool     `json:"scale_embedding"`
	WeightedLayerSum bool     `json:"use_weighted_layer_sum"`
	// Prefix begins the tensor names: "model." in WhisperForConditionalGeneration and
	// WhisperModel checkpoints.
	Prefix string `json:"-"`
}

// layerNormEps is nn.LayerNorm's default, which every LayerNorm of the model uses.
const layerNormEps = 1e-5

// ParseConfig parses config.json and rejects settings the conversion does not handle.
func ParseConfig(raw []byte) (*Config, error) {
	c := Config{Activation: "gelu", Prefix: "model."}
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("whisper: %w", err)
	}
	if c.ModelType != "whisper" {
		return nil, fmt.Errorf("whisper: model_type %q is not whisper", c.ModelType)
	}
	for _, a := range c.Architectures {
		if !slices.Contains([]string{"WhisperForConditionalGeneration", "WhisperModel"}, a) {
			// WhisperForAudioClassification and WhisperForCausalLM name their tensors otherwise
			return nil, fmt.Errorf("whisper: architecture %s is not supported (WhisperForConditionalGeneration or WhisperModel)", a)
		}
	}
	switch {
	case c.DModel <= 0 || c.EncoderHeads <= 0 || c.DModel%c.EncoderHeads != 0:
		return nil, fmt.Errorf("whisper: d_model %d over %d encoder heads", c.DModel, c.EncoderHeads)
	case c.DecoderHeads <= 0 || c.DModel%c.DecoderHeads != 0:
		return nil, fmt.Errorf("whisper: d_model %d over %d decoder heads", c.DModel, c.DecoderHeads)
	case c.EncoderLayers <= 0 || c.DecoderLayers <= 0 || c.EncoderFFN <= 0 || c.DecoderFFN <= 0:
		return nil, fmt.Errorf("whisper: %d encoder and %d decoder layers, MLPs of %d and %d", c.EncoderLayers, c.DecoderLayers, c.EncoderFFN, c.DecoderFFN)
	case c.MelBins <= 0 || c.SourcePositions <= 0 || c.TargetPositions <= 0 || c.VocabSize <= 0:
		return nil, fmt.Errorf("whisper: %d mel bins, %d source and %d target positions, vocabulary %d", c.MelBins, c.SourcePositions, c.TargetPositions, c.VocabSize)
	case c.WeightedLayerSum:
		return nil, fmt.Errorf("whisper: use_weighted_layer_sum (an audio classifier's) is not supported")
	}
	switch c.Activation {
	case "gelu", "gelu_new", "gelu_pytorch_tanh", "relu":
	default:
		return nil, fmt.Errorf("whisper: activation_function %q is not supported", c.Activation)
	}
	return &c, nil
}

// Frames is the encoder's input, in mel frames: twice its positions, the second convolution's
// stride (3000 for 30 seconds).
func (c *Config) Frames() int { return 2 * c.SourcePositions }

func (c *Config) encoder() string { return c.Prefix + "encoder." }
func (c *Config) decoder() string { return c.Prefix + "decoder." }

func (c *Config) encoderLayer(i int) string { return fmt.Sprintf("%slayers.%d.", c.encoder(), i) }
func (c *Config) decoderLayer(i int) string { return fmt.Sprintf("%slayers.%d.", c.decoder(), i) }

// attentionTensors are those of a WhisperAttention of width d: k_proj has no bias.
func attentionTensors(t map[string][]int, p string, d int) {
	for _, n := range []string{"q_proj", "k_proj", "v_proj", "out_proj"} {
		t[p+n+".weight"] = []int{d, d}
		if n != "k_proj" {
			t[p+n+".bias"] = []int{d}
		}
	}
}

// norm adds a LayerNorm's tensors.
func norm(t map[string][]int, p string, d int) {
	t[p+".weight"] = []int{d}
	t[p+".bias"] = []int{d}
}
