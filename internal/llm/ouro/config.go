// Package ouro converts ByteDance's Ouro looped language models (Hugging Face `ouro`,
// OuroForCausalLM in the checkpoints' remote code modeling_ouro.py: ByteDance/Ouro-1.4B,
// Ouro-2.6B and their -Thinking versions) into Core ML programs for the Neural Engine and
// OpenVINO IR for Intel's NPU, from the bf16 safetensors checkpoint.
//
// Ouro is a Llama-style decoder whose whole stack of layers runs several times on each token
// (total_ut_steps, 4: "universal transformer" steps), the same weights each time. A layer is
// sandwiched: RMS norms before and after the attention and before and after the SwiGLU MLP, the
// outputs of the second norms added to the residual; attention is plain multi-head attention with
// RoPE over the whole context, no norms over the heads and no biases. After each pass the final
// norm (model.norm) is applied, and that normed state is what the next pass starts from; the
// logits are lm_head of the last pass's. Each pass keeps keys and values of its own for every
// layer (UniversalTransformerCache's slot pass × layers + layer).
//
// hwnpu treats the passes as layers of their own (Layers is layers × passes, layer v the weights
// of layer v mod layers), so that the caches, the parts and the checking layer by layer work as
// for any decoder; a program holds the weights of a layer once however many passes it runs.
//
// The early exit gate (model.early_exit_gate) gives each pass a probability of being the last;
// with early_exit_threshold 1.0, the checkpoints' value, transformers takes the last pass, as
// hwnpu does: the gate is not read.
package ouro

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// Config is the subset of config.json the conversion needs. Missing fields take the defaults of
// OuroConfig (configuration_ouro.py).
type Config struct {
	HiddenSize       int     `json:"hidden_size"`
	IntermediateSize int     `json:"intermediate_size"`
	NumLayers        int     `json:"num_hidden_layers"`
	NumHeads         int     `json:"num_attention_heads"`
	NumKVHeads       int     `json:"num_key_value_heads"`
	HeadDim          int     `json:"head_dim"`
	VocabSize        int     `json:"vocab_size"`
	NormEps          float64 `json:"rms_norm_eps"`
	RopeTheta        float64 `json:"rope_theta"`
	MaxPositions     int     `json:"max_position_embeddings"`
	TieWordEmbed     bool    `json:"tie_word_embeddings"`
	// Passes is total_ut_steps: how many times the layers run on each token.
	Passes int `json:"total_ut_steps"`

	// Settings the conversion rejects unless they are off.
	ModelType          string          `json:"model_type"`
	HiddenAct          string          `json:"hidden_act"`
	UseSlidingWindow   bool            `json:"use_sliding_window"`
	LayerTypes         []string        `json:"layer_types"`
	RopeScaling        json.RawMessage `json:"rope_scaling"`
	EarlyExitThreshold *float64        `json:"early_exit_threshold"`
	EarlyExitStep      *int            `json:"early_exit_step"`
}

const maxShapeDimension = 1<<31 - 1

// LoadConfig reads config.json.
func LoadConfig(path string) (*Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	c, err := ParseConfig(raw)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return c, nil
}

// ParseConfig parses config.json and rejects settings the conversion does not handle.
func ParseConfig(raw []byte) (*Config, error) {
	c := Config{
		HiddenSize: 4096, IntermediateSize: 22016, NumLayers: 32, NumHeads: 32, NumKVHeads: 32,
		VocabSize: 151936, NormEps: 1e-6, RopeTheta: 10000, MaxPositions: 32768, HiddenAct: "silu", Passes: 4,
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("ouro: %w", err)
	}
	if c.HeadDim == 0 && c.NumHeads > 0 {
		c.HeadDim = c.HiddenSize / c.NumHeads
	}
	if err := c.check(); err != nil {
		return nil, fmt.Errorf("ouro: %w", err)
	}
	return &c, nil
}

func (c *Config) check() error {
	switch {
	case c.ModelType != "ouro":
		return fmt.Errorf("model_type %q is not ouro", c.ModelType)
	case c.NumHeads <= 0 || c.HeadDim <= 0:
		return fmt.Errorf("num_attention_heads %d and head_dim %d must be positive", c.NumHeads, c.HeadDim)
	case c.NumHeads > maxShapeDimension || c.HeadDim > maxShapeDimension || int64(c.NumHeads)*int64(c.HeadDim) > maxShapeDimension:
		return fmt.Errorf("attention dimensions %d × %d exceed the supported shape size", c.NumHeads, c.HeadDim)
	case c.HiddenAct != "silu":
		return fmt.Errorf("hidden_act %q is not supported (the MLP is SwiGLU)", c.HiddenAct)
	case c.UseSlidingWindow:
		return fmt.Errorf("use_sliding_window is not supported")
	case len(c.RopeScaling) > 0 && string(c.RopeScaling) != "null":
		return fmt.Errorf("rope_scaling %s is not supported", c.RopeScaling)
	case c.NumKVHeads <= 0 || c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("%d heads do not share %d key and value heads evenly", c.NumHeads, c.NumKVHeads)
	case c.HeadDim%2 != 0:
		return fmt.Errorf("head_dim %d is odd", c.HeadDim)
	case c.RopeTheta <= 0:
		return fmt.Errorf("rope_theta %g", c.RopeTheta)
	case c.Passes < 1:
		return fmt.Errorf("total_ut_steps %d", c.Passes)
	case c.NumLayers < 1:
		return fmt.Errorf("num_hidden_layers %d", c.NumLayers)
	// an exit before the last pass picks, per token, a pass by the gates: not built
	case c.EarlyExitThreshold != nil && *c.EarlyExitThreshold < 1:
		return fmt.Errorf("early_exit_threshold %g is not supported (only the last pass, 1.0)", *c.EarlyExitThreshold)
	case c.EarlyExitStep != nil:
		return fmt.Errorf("early_exit_step %d is not supported (only the last pass)", *c.EarlyExitStep)
	}
	for i, t := range c.LayerTypes {
		if t != "full_attention" {
			return fmt.Errorf("layer %d is %q: only full_attention is supported", i, t)
		}
	}
	return nil
}

// head names the output layer: lm_head, or the embeddings when they are tied.
func (c *Config) head() string {
	if c.TieWordEmbed {
		return c.Embedding()
	}
	return "lm_head.weight"
}

// RopeInvFreq returns the head_dim/2 inverse frequencies of the RoPE (OuroRotaryEmbedding's
// inv_freq, the default rope type).
func (c *Config) RopeInvFreq() []float64 {
	hd := c.HeadDim
	inv := make([]float64, hd/2)
	for j := range inv {
		inv[j] = 1 / math.Pow(c.RopeTheta, float64(2*j)/float64(hd))
	}
	return inv
}

// scoreScale multiplies the attention scores: 1/sqrt(head_dim).
func (c *Config) scoreScale() float64 { return 1 / math.Sqrt(float64(c.HeadDim)) }
