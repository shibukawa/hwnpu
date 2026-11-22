package text

import (
	"cmp"
	"context"
	"fmt"
	"io"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
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
var embedRole = &airole.Role{
	Name:  "embed",
	Kinds: []chat.Kind{chat.KindEmbed},
	Prepare: func(e airole.PrepareEnv) (f airole.Files, err error) {
		m, ps := e.Model, e.Model.Programs
		if pk := ps["packages"]; len(pk) > 0 {
			f.Set("packages", pk...)
			f.Set("config", cmp.Or(first(ps["config"]), filepath.Join(chat.RepoDir(m.Source()), "config.json")))
			f.Set("weights", cmp.Or(first(ps["weights"]), chat.WeightsIn(m)))
			f.SetValue("name", e.Base.Name)
			f.SetValue("backend", e.NPU.Device)
			return f, nil
		}
		seqs, err := ints(e.Model.Settings["seqs"], chat.DefaultEmbedSeqs)
		if err != nil {
			return f, err
		}
		c, err := chat.PrepareEmbedder(e.Ctx, chat.EmbedSetup{Dir: e.Dir, Repo: m.Source(), Name: e.Base.Name, Seqs: seqs, Backend: e.NPU.Device, Log: e.Log})
		if err != nil {
			return f, err
		}
		f.Set("packages", c.Packages...)
		f.Set("config", c.ModelConfig)
		f.Set("weights", c.Weights)
		f.Set("tokenizer", c.Tokenizer)
		f.SetValue("name", cmp.Or(c.Name, e.Base.Name))
		f.SetValue("backend", c.Backend)
		return f, nil
	},
	Load: func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
		em, err := chat.LoadEmbedder(ctx, chat.EmbedConfig{Packages: f.Paths["packages"], ModelConfig: f.Path("config"), Weights: f.Path("weights"),
			Tokenizer: f.Path("tokenizer"), Name: f.Values["name"], Backend: f.Values["backend"], CacheDir: e.Cache, Log: e.Log})
		if err != nil {
			return nil, err
		}
		return &Embedder{e: em, model: e.Model}, nil
	},
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
	ErrTooLong = chat.ErrTooLong
	// ErrEmpty is returned for a text with no tokens of its own.
	ErrEmpty = chat.ErrEmpty
)

// Embedder turns texts into vectors for search. It runs one text at a time; others wait.
type Embedder struct {
	e     *chat.Embedder
	model string
}

// Model is the name of the model in the definitions, Name the one it calls itself (the API's).
func (e *Embedder) Model() string { return e.model }
func (e *Embedder) Name() string  { return e.e.Name }

// Dim is the length of the vectors, MaxTokens the most tokens of a text.
func (e *Embedder) Dim() int       { return e.e.Dim() }
func (e *Embedder) MaxTokens() int { return e.e.MaxTokens() }

// Embed is the embedding of a text.
func (e *Embedder) Embed(ctx context.Context, text string, o EmbedOptions) (Embedding, error) {
	r, err := e.e.Embed(ctx, text, o.Query, o.Truncate)
	return Embedding{Vector: r.Vector, Tokens: r.Tokens, Truncated: r.Truncated, Took: r.Time}, err
}

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (e *Embedder) Close() error { return e.e.Close() }

// Cosine is the cosine similarity of two vectors.
func Cosine(a, b []float32) float64 { return chat.Cosine(a, b) }

// ints reads a comma separated list of positive numbers; def for "".
func ints(s string, def []int) ([]int, error) {
	if strings.TrimSpace(s) == "" {
		return def, nil
	}
	var out []int
	for _, f := range strings.Split(s, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(f))
		if err != nil || n <= 0 {
			return nil, fmt.Errorf("text: %q is not a positive number", f)
		}
		out = append(out, n)
	}
	return out, nil
}
