// Package llm holds what the language model converters share: the weight sources, the build
// options and the list of supported architectures.
//
// Supported models are a whitelist, as in llama.cpp: each architecture is a Go package that
// writes the model's structure by hand (internal/llm/lfm3 for example) and registers itself with
// Register. internal/llm/archs imports them all, so its import list is the whitelist.
package llm

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

// Embed gathers the embedding rows of the given tokens as fp32 (tokens x hidden).
func Embed(w Weights, hidden int, tokens []int32) ([]float32, error) {
	out := make([]float32, 0, len(tokens)*hidden)
	for _, t := range tokens {
		row, err := w.Rows("model.embed_tokens.weight", int(t), int(t)+1)
		if err != nil {
			return nil, err
		}
		out = append(out, row...)
	}
	return out, nil
}

// Options select what to build. They are common to all architectures; settings that only one
// architecture has go in Params.
type Options struct {
	// Seq is the fixed sequence length of the program.
	Seq int
	// Layers truncates the model to its first Layers layers (0 means all); with a truncated
	// model the output is the hidden state instead of the logits, for checking layer by layer.
	Layers int
	// Start skips the first Start layers: the input is then the hidden state entering layer
	// Start, and Layers counts from there. Only for checking single layers.
	Start int
	// Hidden makes the output the final hidden state (after the last norm) rather than logits.
	Hidden bool
	// Format is "lut4" (4 bit palettized, k-means per 16 rows) or "fp16".
	Format string
	// Group is the number of rows sharing a lookup table (16 is what the Neural Engine reads fast).
	Group int
	// ColBlock splits each table's rows into blocks of this many columns (0: whole rows).
	ColBlock int
	// KeepFP16 lists substrings of tensor names to keep in fp16 regardless of Format, such as
	// "embed_tokens" for the tied embedding and output layer.
	KeepFP16 []string
	// Quant selects the table fitting for lut4: "uniform" (default), "kmeans" or "kmeans-minmax".
	Quant string
	// Params holds the architecture's own settings by Param name; missing ones take the default.
	Params map[string]string
}
