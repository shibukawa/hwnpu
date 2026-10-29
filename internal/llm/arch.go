package llm

import (
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"strings"

	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Arch describes one supported architecture.
type Arch struct {
	// Name identifies the architecture: the GGUF general.architecture and the -arch flag.
	Name string
	// Title is the human readable name for the model description.
	Title string
	// ModelTypes and Architectures are the config.json model_type and architectures values
	// that select this architecture.
	ModelTypes    []string
	Architectures []string
	// Params are the architecture's own build settings, read from Options.Params.
	Params []Param
	// LoadConfig parses config.json and rejects settings the conversion does not handle.
	LoadConfig func(raw []byte) (Model, error)
	// GGUFName maps a Hugging Face tensor name to llama.cpp's ("" when there is none).
	GGUFName func(name string) string
	// GGUFShape, if set, corrects the shape of a GGUF tensor to the Hugging Face layout.
	GGUFShape func(name string, shape []int) []int
}

// Param is an architecture specific build setting.
type Param struct {
	Name, Default, Usage string
}

// Param returns the value of the architecture's setting name in o, or its default.
func (a *Arch) Param(o Options, name string) string {
	if v, ok := o.Params[name]; ok {
		return v
	}
	for _, p := range a.Params {
		if p.Name == name {
			return p.Default
		}
	}
	return ""
}

// Model is a parsed config.json of a supported architecture.
type Model interface {
	// Hidden is the hidden size, the width of the program's input.
	Hidden() int
	// Layers is the number of layers (decoder or encoder).
	Layers() int
	// Output is the width of the model's output per token, and whether it is logits (the
	// vocabulary size) rather than hidden states (an encoder's last hidden state).
	Output() (width int, logits bool)
	// Embedding names the token embedding table, which the host reads.
	Embedding() string
	// Tensors lists the tensors the conversion reads, by Hugging Face name, with their shapes.
	Tensors() map[string][]int
	// Build writes the program into b. The first input is x, the fp16 token embeddings (1, seq,
	// hidden), looked up on the host because Core ML runs gather on the CPU; whatever else the
	// model does before its first layer (position embeddings, an embedding norm) is in the
	// program when Start is 0. Further inputs, such as an encoder's attention mask, follow. The
	// output is y: the model's output, or a hidden state for a truncated model.
	Build(b *mlpackage.Builder, w Weights, opts Options) (ins []mlpackage.Feature, out mlpackage.Feature, err error)
}

// Cached is a causal model that can be built to run one step at a time with its state held by
// the host (Options.Cache). The program then has inputs and outputs named by what the host does
// with them, besides x and y:
//
//   - kv_NAME (1, heads, L, head_dim) with output kv_NAME_new (1, heads, Seq, head_dim): the
//     keys or values of the step's tokens, which the host writes at their positions in kv_NAME.
//     L is Cache, or less for a layer that only sees a window of the sequence: such a cache is a
//     ring, written at position mod L, and StepInputs masks it accordingly. A cache longer than
//     CacheChunkOver is in chunks instead, kv_NAME_c0, kv_NAME_c1, … (CacheChunks), holding the
//     positions in order, with the one output kv_NAME_new
//   - win_NAME (1, W, width) with output win_NAME_new (1, W+Seq, width): a window of the last W
//     rows before the step followed by the step's rows; the host keeps the W rows ending at the
//     last real token (a short convolution's state)
//   - any other input is given by StepInputs (masks, rotary tables)
//
// A step takes up to Seq tokens at positions pos.. ; the rest of x is padding, which only the
// padded positions themselves see.
type Cached interface {
	Model
	BuildCached(b *mlpackage.Builder, w Weights, opts Options) (ins, outs []mlpackage.Feature, err error)
	// StepInputs returns the values of the other inputs for a step of valid tokens from pos, of
	// a program built with Seq q and Cache cache.
	StepInputs(pos, valid, q, cache int) map[string][]float32
}

var archs []*Arch

// Register adds an architecture to the whitelist. It is called from the architecture's init.
func Register(a *Arch) {
	if Lookup(a.Name) != nil {
		panic("llm: architecture " + a.Name + " registered twice")
	}
	archs = append(archs, a)
	slices.SortFunc(archs, func(x, y *Arch) int { return strings.Compare(x.Name, y.Name) })
}

// Archs returns the registered architectures by name.
func Archs() []*Arch { return archs }

// Lookup returns the architecture with the given name, or nil.
func Lookup(name string) *Arch {
	for _, a := range archs {
		if a.Name == name {
			return a
		}
	}
	return nil
}

func names() string {
	var n []string
	for _, a := range archs {
		n = append(n, a.Name)
	}
	return strings.Join(n, ", ")
}

// Detect picks the architecture of a config.json by its model_type, then its architectures.
func Detect(raw []byte) (*Arch, error) {
	var c struct {
		ModelType     string   `json:"model_type"`
		Architectures []string `json:"architectures"`
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, err
	}
	for _, a := range archs {
		if slices.Contains(a.ModelTypes, c.ModelType) {
			return a, nil
		}
	}
	for _, a := range archs {
		for _, name := range c.Architectures {
			if slices.Contains(a.Architectures, name) {
				return a, nil
			}
		}
	}
	return nil, fmt.Errorf("llm: model_type %q (architectures %v) is not supported; supported: %s", c.ModelType, c.Architectures, names())
}

// LoadConfig reads config.json. arch names the architecture, or "" to detect it.
func LoadConfig(path, arch string) (*Arch, Model, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	var a *Arch
	if arch != "" {
		if a = Lookup(arch); a == nil {
			return nil, nil, fmt.Errorf("llm: architecture %q is not supported; supported: %s", arch, names())
		}
	} else if a, err = Detect(raw); err != nil {
		return nil, nil, fmt.Errorf("%s: %w", path, err)
	}
	m, err := a.LoadConfig(raw)
	if err != nil {
		return nil, nil, fmt.Errorf("%s: %w", path, err)
	}
	return a, m, nil
}
