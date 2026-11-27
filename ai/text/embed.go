package text

import (
	"context"
	"io"
	"time"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/text/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/catalog"
	"github.com/shibukawa/hwnpu/internal/chatwire"
)

// Embed is the use of an embedding model (PLaMo-Embedding) for an app: a model, a family or a
// use of the definitions ("embedding", "plamo-embedding-1b").
func Embed(name string) ai.Use[*Embedder] {
	return ai.Define[*Embedder](embedRole, name)
}

// The settings and programs an embedding model's definition may give (ai.Definition):
//
//	settings: seqs      the sequence lengths the programs are converted for, comma separated
//	                    (128,1024): a text runs on the shortest that holds it
//	programs: packages  the programs converted by hwnpu-dev convert, with different -seq; with
//	                    config and weights (default: the checkpoint's)
//
// On AMD's NPU a model runs from its export of tools/amd/iron_plamo_embed.py.
// embedRole's Prepare and Load are the engine's (ai/text/engine).
var embedRole = &airole.Role{
	Remote: func(h *airpc.Handle) io.Closer { return &Embedder{remote: h} },
	Name:   "embed",
	Kinds:  []catalog.Kind{catalog.KindEmbed},
}

// EmbedOptions are how a text is embedded.
type EmbedOptions struct {
	// Query embeds the text as a search query (with the model's instruction for one), not as a
	// text searched.
	Query bool
	// Truncate cuts a text over MaxTokens there instead of failing.
	Truncate bool
}

// Embedding is a text's embedding.
type Embedding struct {
	// Vector is of length 1: the dot product of two is their cosine similarity.
	Vector []float32
	// Tokens are the tokens read, with the model's own (a query's instruction); Truncated says
	// the text was cut to MaxTokens.
	Tokens    int
	Truncated bool
	Took      time.Duration
}

var (
	// ErrTooLong is returned for a text over MaxTokens without EmbedOptions.Truncate.
	ErrTooLong = chatwire.ErrTooLong
	// ErrEmpty is returned for a text with no tokens of its own.
	ErrEmpty = chatwire.ErrEmpty
)

// Embedder turns texts into vectors for search. It runs one text at a time; others wait.
type Embedder struct {
	remote *airpc.Handle // the model in hwnpud (ai's HWNPU_DAEMON), nil when it is here
	local  embedderImpl  // the engine's (ai/text/engine), when it is here
}

// embedderImpl is an embedding model of the engine in this process.
type embedderImpl interface {
	Model() string
	Name() string
	Dim() int
	MaxTokens() int
	Embed(ctx context.Context, text string, o EmbedOptions) (Embedding, error)
	Close() error
}

// Model is the name of the model in the definitions, Name the one it calls itself (the API's).
func (e *Embedder) Model() string {
	if e.remote != nil {
		return airpc.Value[string](e.remote, "Model")
	}
	return e.local.Model()
}
func (e *Embedder) Name() string {
	if e.remote != nil {
		return airpc.Value[string](e.remote, "Name")
	}
	return e.local.Name()
}

// Dim is the length of the vectors, MaxTokens the most tokens of a text.
func (e *Embedder) Dim() int {
	if e.remote != nil {
		return airpc.Value[int](e.remote, "Dim")
	}
	return e.local.Dim()
}
func (e *Embedder) MaxTokens() int {
	if e.remote != nil {
		return airpc.Value[int](e.remote, "MaxTokens")
	}
	return e.local.MaxTokens()
}

// Embed is the embedding of a text.
func (e *Embedder) Embed(ctx context.Context, text string, o EmbedOptions) (Embedding, error) {
	if e.remote != nil {
		return airpc.Call1[Embedding](e.remote, ctx, "Embed", text, o)
	}
	return e.local.Embed(ctx, text, o)
}

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (e *Embedder) Close() error {
	if e.remote != nil {
		return e.remote.Release() // the Runtime in hwnpud unloads it
	}
	return e.local.Close()
}

// Cosine is the cosine similarity of two vectors.
func Cosine(a, b []float32) float64 { return chatwire.Cosine(a, b) }

func init() {
	airole.Register(embedRole)
	bridge.Roles[embedRole.Name] = embedRole
	bridge.Wrap[embedRole.Name] = func(l any) io.Closer { return &Embedder{local: l.(embedderImpl)} }
}
