package llm

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"

	"github.com/shibukawa/hwnpu/internal/llm/paged"
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
	// Format is the weight format the converters use when not told (lut4 when empty).
	Format string
	// IntelFormat is the weight format for Intel's NPU when not told (when empty, IntelFormat of
	// the Neural Engine's format: int4 for lut4, int8 for lut6 and lut8).
	IntelFormat string
	// LoadConfig parses config.json and rejects settings the conversion does not handle.
	LoadConfig func(raw []byte) (Model, error)
	// GGUFName maps a Hugging Face tensor name to llama.cpp's ("" when there is none).
	GGUFName func(name string) string
	// GGUFShape, if set, corrects the shape of a GGUF tensor to the Hugging Face layout.
	GGUFShape func(name string, shape []int) []int
	// LastRow reports that the architecture builds Options.LastRow.
	LastRow bool
	// Revision counts the changes to the programs the architecture builds that call for
	// converting a model again, the weights and options the same: hwnpu-chat names its
	// conversions by it (Rev), so that it does not load one of an earlier revision. 1: silu
	// written with exp and real_div (blocks.SiLU, 2026-11-07).
	Revision int
}

// Rev is "-rN" for the architecture's Revision N, "" for none: for the names of the programs
// converted from it.
func (a *Arch) Rev() string {
	if a.Revision == 0 {
		return ""
	}
	return fmt.Sprintf("-r%d", a.Revision)
}

// Param is an architecture specific build setting.
type Param struct {
	Name, Default, Usage string
}

// Formatter is a Model whose weight format, when none is given, depends on the checkpoint rather
// than the architecture alone; an empty Format takes the architecture's.
type Formatter interface {
	Format() string
}

// DefaultFormat is the weight format to use for m when none is given.
func (a *Arch) DefaultFormat(m Model) string {
	if f, ok := m.(Formatter); ok && f.Format() != "" {
		return f.Format()
	}
	if a.Format == "" {
		return "lut4"
	}
	return a.Format
}

// DefaultFormatFor is DefaultFormat for a backend (Options.Backend).
func (a *Arch) DefaultFormatFor(m Model, backend string) string {
	if backend != "intel" {
		return a.DefaultFormat(m)
	}
	if a.IntelFormat != "" {
		return a.IntelFormat
	}
	return IntelFormat(a.DefaultFormat(m))
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
//   - st_NAME with output st_NAME_new of the same shape: a state the step replaces as a whole
//     (a Gated DeltaNet's recurrent state), as it is after the step's valid tokens: the padding
//     after them must leave it unchanged. It cannot give the state after fewer tokens than the
//     step's, so the host runs a step again for a beginning of its tokens (speculative decoding)
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

// StateReplayer is a Cached model whose program can also output, for a state replaced as a whole
// (st_NAME), st_NAME_rows: what each of the step's rows adds to it. From that, the host brings
// the state before a step to the one after the step's first n tokens, which the program cannot
// give, instead of running those tokens again (speculative decoding keeps a beginning of a
// step's tokens).
type StateReplayer interface {
	// ReplayState writes into after the state st_NAME (name) after the first n rows of a step,
	// from before, the state before the step, and rows, the program's st_NAME_rows of shape
	// rowShape; all fp16.
	ReplayState(name string, before, rows []byte, rowShape []int, n int, after []byte) error
}

// CacheInput reports whether a program input holds state the host keeps between steps (Cached):
// a KV cache (kv_), a window (win_) or a state replaced as a whole (st_). The programs of one
// model that take different numbers of tokens per step share these buffers.
func CacheInput(name string) bool {
	return strings.HasPrefix(name, "kv_") || HeldInput(name)
}

// HeldInput reports whether a program input is state that a step overwrites as a whole rather
// than by position, a window (win_) or a recurrent state (st_): a point of the sequence to go back
// to keeps a copy of it.
func HeldInput(name string) bool {
	return strings.HasPrefix(name, "win_") || strings.HasPrefix(name, "st_")
}

// Paged is a Cached model that can be built split at its attention (Options.Paged): the
// attention of PagedLayers runs between the parts, on a cache the host holds in chunks of
// Options.Paged tokens (internal/llm/paged), so that a step costs in proportion to the tokens in
// the cache rather than to its capacity. The parts are cut at those layers (PlanPaged):
//
//   - a part that ends where such a layer N begins computes the layer's queries, keys and values
//     and stops before the attention: it outputs attn_q_N (1, heads, Seq, head_dim), RoPE
//     applied, and kv_k_N_new and kv_v_N_new (1, kv_heads, Seq, head_dim), and its y is the
//     hidden state entering the layer. Anything else the rest of the layer needs from it is an
//     output that the next part takes by name (internal/llm/chain)
//   - the next part starts at layer N with its attention's output, the input attn_o_N (1,
//     heads, Seq, head_dim), and x, the hidden state entering the layer; or attn_ot_N, the same
//     with the tokens first (1, Seq, heads, head_dim), which spares the part a transpose
//     (PLaMo 3's, whose transposed input Core ML put on the CPU)
//   - a layer that attends over the keys and values of an earlier one of PagedLayers (PagedKV,
//     Gemma 4's KV sharing) outputs attn_q_N only: the host attends over that layer's cache
//
// The other layers keep their state in the program as Cached describes. Layer 0 cannot be one of
// PagedLayers.
type Paged interface {
	Cached
	// PagedLayers are the layers whose attention the host runs, in order.
	PagedLayers() []int
	// PagedShape is their attention's heads, kv heads, head size and score scale.
	PagedShape() paged.Shape
	// PagedKV is the layer of PagedLayers whose cache layer attends over: layer itself, or the
	// earlier layer whose keys and values it reuses.
	PagedKV(layer int) int
	// PagedStepInputs is StepInputs for a program built with Options.Paged: without the inputs
	// of PagedLayers' attention, which the host runs (their masks would span the whole cache).
	PagedStepInputs(pos, valid, q, cache int) map[string][]float32
	// MaxContext is the longest sequence the model was trained for (max_position_embeddings).
	MaxContext() int
}

// CacheWriters is a Paged model whose layers past the first CacheLayers write no cache: they
// reuse the keys and values of earlier layers (Gemma 4's shared layers), so a program that only
// reads tokens into the caches (Options.KVOnly) stops before them.
type CacheWriters interface {
	CacheLayers() int
}

// OutputSubset is a model whose program, built with some options, gives the logits of some tokens
// alone: OutputIDs are their ids in the order of the logits, nil for the whole vocabulary. The
// conversion writes them into the manifest (Manifest.OutputIDs), where the host finds which token
// a logit is of (the drafts of Qwen3.5's multi-token prediction layer among the frequent tokens).
type OutputSubset interface {
	OutputIDs(opts Options) ([]int32, error)
}

// KVOnlyLayers is how many layers a program that only reads tokens into the caches
// (Options.KVOnly) builds, the last of them up to its keys and values, and whether the model has
// such a program: the CacheLayers of a model whose later layers write no cache (CacheWriters), or
// all the layers of a Paged model whose last layer's attention the host runs (Qwen3.5, PLaMo 3):
// that layer's output and the output layer feed only the logits, which such a program does not
// give.
func KVOnlyLayers(m Model) (int, bool) {
	if cw, ok := m.(CacheWriters); ok {
		return cw.CacheLayers(), true
	}
	pm, ok := m.(Paged)
	if !ok {
		return 0, false
	}
	n := m.Layers()
	return n, n > 0 && slices.Contains(pm.PagedLayers(), n-1)
}

// LogitSoftcap is a model whose program may leave the soft cap of its logits to the host
// (Gemma 4's -logits host): HostSoftcap is the cap c of a program built with opts, whose logits y
// the host turns into c*tanh(y/c), and 0 when the program applies it. The conversion writes it
// into the parts' metadata (MetaSoftcap), where the host finds it.
type LogitSoftcap interface {
	HostSoftcap(opts Options) float64
}

// HostSoftcap is the soft cap the host applies to the logits of a program with the metadata meta
// (ReadMeta): MetaSoftcap, 0 for none.
func HostSoftcap(meta map[string]string) float64 {
	c, err := strconv.ParseFloat(meta[MetaSoftcap], 64)
	if err != nil || c <= 0 {
		return 0
	}
	return c
}

// Softcap applies the soft cap c to the logits in place: c*tanh(y/c). c 0 leaves them.
func Softcap(logits []float32, c float64) {
	if c <= 0 {
		return
	}
	for i, y := range logits {
		logits[i] = float32(c * math.Tanh(float64(y)/c))
	}
}

// TokenInput is a program input that the host fills per token like x, from a table of the
// checkpoint that stays on the host: the row of Table for each token, Width values, as an input
// of shape (1, seq, Width).
type TokenInput struct {
	Input, Table string
	Width        int
}

// TokenInputs is a model whose program takes inputs looked up per token besides x (Gemma 4's
// per-layer embeddings). Their tables are not part of the program.
type TokenInputs interface {
	TokenInputs() []TokenInput
}

// Prologue is a model with tensors outside its layers that the program uses before layer 0
// (Gemma 4's per-layer input projection). Without it, those of a causal LM count as coming after
// the last layer when the model is split.
type Prologue interface {
	BeforeLayers(name string) bool
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
