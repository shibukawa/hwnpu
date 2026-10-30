// Package llm holds what the language model converters share: the weight sources, the build
// options and the list of supported architectures.
//
// Supported models are a whitelist, as in llama.cpp: each architecture is a Go package that
// writes the model's structure by hand (internal/llm/lfm3 for example) and registers itself with
// Register. internal/llm/archs imports them all, so its import list is the whitelist.
package llm

import (
	"fmt"
	"strconv"
	"strings"
)

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

// Embed gathers the rows of a model's token embedding table for the given tokens as fp32
// (tokens x hidden): the program's input x.
func Embed(w Weights, m Model, tokens []int32) ([]float32, error) {
	out := make([]float32, 0, len(tokens)*m.Hidden())
	for _, t := range tokens {
		row, err := w.Rows(m.Embedding(), int(t), int(t)+1)
		if err != nil {
			return nil, err
		}
		out = append(out, row...)
	}
	return out, nil
}

// TokenRow gathers what the host looks up for one token: the row of the embedding table (x's),
// followed by the token's row of each of the model's TokenInputs, in order. decode.Session.Step
// takes the tokens of a step as such rows.
func TokenRow(w Weights, m Model, token int32) ([]float32, error) {
	row, err := w.Rows(m.Embedding(), int(token), int(token)+1)
	if err != nil {
		return nil, err
	}
	if l, ok := m.(TokenInputs); ok {
		row = append([]float32(nil), row...)
		for _, lk := range l.TokenInputs() {
			v, err := w.Rows(lk.Table, int(token), int(token)+1)
			if err != nil {
				return nil, err
			}
			if len(v) != lk.Width {
				return nil, fmt.Errorf("llm: %s has rows of %d values, the input %s takes %d", lk.Table, len(v), lk.Input, lk.Width)
			}
			row = append(row, v...)
		}
	}
	return row, nil
}

// TokenInputRows gathers the rows of a lookup input for the tokens (tokens x Width).
func TokenInputRows(w Weights, lk TokenInput, tokens []int32) ([]float32, error) {
	out := make([]float32, 0, len(tokens)*lk.Width)
	for _, t := range tokens {
		v, err := w.Rows(lk.Table, int(t), int(t)+1)
		if err != nil {
			return nil, err
		}
		out = append(out, v...)
	}
	return out, nil
}

// FormatOf is the weight format of the named tensor: fp16 if KeepFP16 names it, else the last
// FormatFor rule that matches, else Format.
func (o Options) FormatOf(name string) string {
	for _, k := range o.KeepFP16 {
		if strings.Contains(name, k) {
			return "fp16"
		}
	}
	f := o.Format
	for _, r := range o.FormatFor {
		if sub, form, ok := strings.Cut(r, "="); ok && strings.Contains(name, sub) {
			f = form
		}
	}
	return f
}

// LinearFormat parses the name of a symmetric linear weight format: int4 or int8 with one scale
// per row, or int4-gN / int8-gN with one per N columns (block 0 for whole rows).
func LinearFormat(f string) (bits, block int, ok bool) {
	name, g, grouped := strings.Cut(f, "-g")
	switch name {
	case "int4":
		bits = 4
	case "int8":
		bits = 8
	default:
		return 0, 0, false
	}
	if grouped {
		n, err := strconv.Atoi(g)
		if err != nil || n <= 0 {
			return 0, 0, false
		}
		block = n
	}
	return bits, block, true
}

// Options select what to build. They are common to all architectures; settings that only one
// architecture has go in Params.
type Options struct {
	// Seq is the fixed sequence length of the program: with Cache, the tokens one call takes.
	Seq int
	// Cache, when not zero, builds the program for a causal model (Cached) to run with a KV cache
	// of this many tokens held by the host: it takes the earlier tokens' keys and values and
	// the other state it needs as inputs, and outputs the new ones (see Cached).
	Cache int
	// Layers truncates the model to its first Layers layers (0 means all); with a truncated
	// model the output is the hidden state instead of the logits, for checking layer by layer.
	Layers int
	// Start skips the first Start layers: the input is then the hidden state entering layer
	// Start, and Layers counts from there. Only for checking single layers; Start equal to the
	// number of layers builds the final norm and the output layer alone.
	Start int
	// Hidden makes the output the final hidden state (after the last norm) rather than logits.
	Hidden bool
	// NoHead makes the output the hidden state after the last built layer, before the final norm,
	// even when that is the model's last layer. For checking the last layer on its own.
	NoHead bool
	// Format is "lut4" (4 bit palettized per 16 rows; lut6 and lut8 have larger tables), "int4"
	// or "int8" (symmetric, one scale per row; "int4-g128" one per 128 columns: LinearFormat) or
	// "fp16". The Neural Engine reads lutN fastest, Intel's NPU intN (it cannot run lutN).
	Format string
	// Group is the number of rows sharing a lookup table (16 is what the Neural Engine reads fast).
	Group int
	// ColBlock splits each table's rows into blocks of this many columns (0: whole rows).
	ColBlock int
	// KeepFP16 lists substrings of tensor names to keep in fp16 regardless of Format, such as
	// "embed_tokens" for the tied embedding and output layer.
	KeepFP16 []string
	// FormatFor gives some tensors another format than Format: "SUBSTRING=FORMAT" entries, such
	// as "mlp=lut8"; the last that matches a tensor's name wins, and KeepFP16 over all.
	FormatFor []string
	// Quant selects the table fitting for lut4: "uniform" (default), "kmeans" or "kmeans-minmax";
	// for intN, "mse" (default) searches the clipping point with the smallest squared error and
	// "uniform" scales each block by its largest weight.
	Quant string
	// Backend is the runtime the program is written for: "coreml" (the default, an .mlpackage)
	// or "intel" (OpenVINO IR for Intel's NPU, internal/ovlower).
	Backend string
	// Params holds the architecture's own settings by Param name; missing ones take the default.
	Params map[string]string
}
