package chat

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/chain"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// DefaultEmbedSeqs are the sequence lengths PrepareEmbedder converts the embedding model for: a
// text runs on the shortest that holds it, padded, since the cost follows the length.
var DefaultEmbedSeqs = []int{128, 1024}

// QueryPrefix is the instruction PLaMo-Embedding puts before a search query (encode_query). Its
// tokens are read but left out of the mean.
const QueryPrefix = "次の文章に対して、関連する文章を検索してください: "

// EmbedSetup says where the embedding model is and how to get it.
type EmbedSetup struct {
	Dir  string // holds config.json, tokenizer.model, model.safetensors and the converted programs
	Repo string // Hugging Face repository the missing checkpoint files come from
	Seqs []int  // sequence lengths of the programs
	Log  io.Writer
}

// PrepareEmbedder makes the embedding model ready in Dir like Prepare: it downloads the missing
// checkpoint files and converts one program per sequence length that is missing. It returns the
// EmbedConfig that loads them.
func PrepareEmbedder(ctx context.Context, s EmbedSetup) (EmbedConfig, error) {
	if err := os.MkdirAll(s.Dir, 0o755); err != nil {
		return EmbedConfig{}, err
	}
	for _, f := range []string{"config.json", "tokenizer.model", "model.safetensors"} {
		if err := fetch(ctx, s.Repo, f, filepath.Join(s.Dir, f), s.Log); err != nil {
			return EmbedConfig{}, err
		}
	}
	c := EmbedConfig{ModelConfig: filepath.Join(s.Dir, "config.json"), Weights: filepath.Join(s.Dir, "model.safetensors"), Name: filepath.Base(s.Repo)}
	for _, seq := range s.Seqs {
		out := filepath.Join(s.Dir, fmt.Sprintf("embed-s%d.mlpackage", seq))
		path, err := convert(ctx, c.ModelConfig, c.Weights, out, seq, 0, 0, s.Log)
		if err != nil {
			return EmbedConfig{}, err
		}
		c.Packages = append(c.Packages, path)
	}
	return c, nil
}

// EmbedConfig says where the embedding model is.
type EmbedConfig struct {
	// Packages are the model converted by hwnpu-convert (an .mlpackage or an out.parts.json),
	// each with another sequence length (-seq).
	Packages []string
	// ModelConfig is config.json, Weights the checkpoint the token embeddings are read from,
	// Tokenizer tokenizer.model (default: next to config.json).
	ModelConfig, Weights, Tokenizer string
	// CacheDir is the compile cache ("" for none).
	CacheDir string
	// Log receives what loading does (nil for none).
	Log io.Writer
	// Name is the model's name in the API (default: the first package's).
	Name string
}

// Embedder turns texts into sentence embeddings with PLaMo-Embedding, one text at a time: the
// last hidden state of the text's tokens, averaged and scaled to length 1, so that the dot
// product of two embeddings is their cosine similarity.
type Embedder struct {
	Tok  *tokenizer.SentencePiece
	Name string // the model's name

	dev    *hwnpu.Device
	chains []*chain.Chain
	w      llm.Weights
	model  llm.Model
	bos    int32
	prefix int // tokens of QueryPrefix

	mu    sync.Mutex
	progs []embedProg // shortest first
}

type embedProg struct {
	seq int
	p   decode.Program
}

// LoadEmbedder loads the embedding model.
func LoadEmbedder(ctx context.Context, c EmbedConfig) (*Embedder, error) {
	if len(c.Packages) == 0 {
		return nil, errors.New("chat: no embedding package")
	}
	logf := func(format string, args ...any) {
		if c.Log != nil {
			fmt.Fprintf(c.Log, format+"\n", args...)
		}
	}
	a, model, err := llm.LoadConfig(c.ModelConfig, "")
	if err != nil {
		return nil, err
	}
	if a.Name != "plamo-bi" {
		return nil, fmt.Errorf("chat: %s is not an embedding model hwnpu-chat runs (only PLaMo-Embedding)", a.Title)
	}
	if c.Tokenizer == "" {
		c.Tokenizer = filepath.Join(filepath.Dir(c.ModelConfig), "tokenizer.model")
	}
	tok, err := tokenizer.LoadSentencePiece(c.Tokenizer)
	if err != nil {
		return nil, err
	}
	e, err := newEmbedder(tok, model)
	if err != nil {
		return nil, err
	}
	e.Name = c.Name
	if e.Name == "" {
		e.Name = strings.TrimSuffix(strings.TrimSuffix(filepath.Base(c.Packages[0]), ".parts.json"), ".mlpackage")
	}
	if e.w, err = llm.OpenWeights(c.Weights, a); err != nil {
		return nil, err
	}
	opts := []hwnpu.Option{hwnpu.WithBackend("coreml"), hwnpu.WithPlacement(hwnpu.PlacementPrefer)}
	if c.CacheDir != "" {
		opts = append(opts, hwnpu.WithCacheDir(c.CacheDir))
	}
	if e.dev, err = hwnpu.Open(ctx, opts...); err != nil {
		e.Close()
		return nil, err
	}
	for _, pkg := range c.Packages {
		t0 := time.Now()
		ch, err := chain.Open(ctx, e.dev, pkg)
		if err != nil {
			e.Close()
			return nil, err
		}
		e.chains = append(e.chains, ch)
		where := "all on the NPU"
		if !ch.OnNPU() {
			where = "NOT all on the NPU"
		}
		logf("%s: %d parts, %s, loaded in %.1fs", filepath.Base(pkg), len(ch.Parts()), where, time.Since(t0).Seconds())
		if err := e.add(ch.Program(context.Background())); err != nil {
			e.Close()
			return nil, fmt.Errorf("chat: %s: %w", pkg, err)
		}
	}
	logf("embeddings of %d dimensions, up to %d tokens", e.Dim(), e.MaxTokens())
	return e, nil
}

func newEmbedder(tok *tokenizer.SentencePiece, model llm.Model) (*Embedder, error) {
	bos, ok := tok.ID("<|startoftext|>")
	if !ok {
		return nil, errors.New("chat: the tokenizer has no <|startoftext|>")
	}
	return &Embedder{Tok: tok, model: model, bos: bos, prefix: len(tok.Encode(QueryPrefix))}, nil
}

// add checks a program (x (1, seq, hidden) and mask (1, seq) to y (1, seq, hidden)) and adds it.
func (e *Embedder) add(p decode.Program) error {
	d := e.model.Hidden()
	x, mask, y := p.InShape["x"], p.InShape["mask"], p.OutShape["y"]
	if len(x) != 3 || x[0] != 1 || x[2] != d {
		return fmt.Errorf("no input x (1, seq, %d)", d)
	}
	seq := x[1]
	if !slices.Equal(mask, []int{1, seq}) {
		return fmt.Errorf("no input mask (1, %d)", seq)
	}
	if !slices.Equal(y, []int{1, seq, d}) {
		return fmt.Errorf("no output y (1, %d, %d): not converted as an encoder", seq, d)
	}
	for _, o := range e.progs {
		if o.seq == seq {
			return fmt.Errorf("a program of %d tokens is there already", seq)
		}
	}
	e.progs = append(e.progs, embedProg{seq, p})
	slices.SortFunc(e.progs, func(a, b embedProg) int { return a.seq - b.seq })
	return nil
}

// Close releases the model.
func (e *Embedder) Close() error {
	for i := len(e.chains) - 1; i >= 0; i-- {
		e.chains[i].Close()
	}
	e.chains = nil
	if e.dev != nil {
		e.dev.Close()
		e.dev = nil
	}
	if e.w != nil {
		e.w.Close()
		e.w = nil
	}
	return nil
}

// Dim is the length of an embedding.
func (e *Embedder) Dim() int { return e.model.Hidden() }

// MaxTokens is the most tokens a text can have, with the BOS and a query's instruction.
func (e *Embedder) MaxTokens() int { return e.progs[len(e.progs)-1].seq }

// Tokens returns the tokens the model reads for a text, and how many of them at the start are
// left out of the mean: the BOS, and for a query the instruction QueryPrefix. A document is
// what is searched, and also any text that is not a search query.
func (e *Embedder) Tokens(text string, query bool) (ids []int32, skip int) {
	if query {
		return append([]int32{e.bos}, e.Tok.Encode(QueryPrefix+text)...), 1 + e.prefix
	}
	return append([]int32{e.bos}, e.Tok.Encode(text)...), 1
}

// Embedding is the embedding of one text.
type Embedding struct {
	Vector    []float32 // length 1
	Tokens    int       // tokens read, with the BOS and a query's instruction
	Truncated bool      // the text was cut to MaxTokens
	Time      time.Duration
}

var (
	// ErrTooLong is returned for a text over MaxTokens when it is not to be truncated.
	ErrTooLong = errors.New("chat: the text is longer than the embedding model reads")
	// ErrEmpty is returned for a text with no tokens of its own.
	ErrEmpty = errors.New("chat: empty text")
)

// Embed returns the embedding of text, as a search query or as a document (see Tokens). A text
// over MaxTokens is cut there when truncate is set, and an error otherwise.
func (e *Embedder) Embed(ctx context.Context, text string, query, truncate bool) (Embedding, error) {
	ids, skip := e.Tokens(text, query)
	var r Embedding
	if n := e.MaxTokens(); len(ids) > n {
		if !truncate {
			return r, fmt.Errorf("%w: %d tokens, at most %d", ErrTooLong, len(ids), n)
		}
		ids, r.Truncated = ids[:n], true
	}
	if len(ids) <= skip {
		return r, ErrEmpty
	}
	if err := ctx.Err(); err != nil {
		return r, err
	}
	t0 := time.Now()
	v, err := e.run(ids, skip)
	r.Vector, r.Tokens, r.Time = v, len(ids), time.Since(t0)
	return r, err
}

// run reads ids on the shortest program that holds them, padded and masked, and returns the
// mean of the last hidden state over ids[skip:], scaled to length 1.
func (e *Embedder) run(ids []int32, skip int) ([]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	i := slices.IndexFunc(e.progs, func(p embedProg) bool { return p.seq >= len(ids) })
	if i < 0 {
		return nil, fmt.Errorf("%w: %d tokens, at most %d", ErrTooLong, len(ids), e.MaxTokens())
	}
	p, seq, d := e.progs[i].p, e.progs[i].seq, e.Dim()
	x, err := llm.Embed(e.w, e.model, ids)
	if err != nil {
		return nil, err
	}
	x = append(x, make([]float32, (seq-len(ids))*d)...)
	mask := make([]float32, seq)
	for j := range ids {
		mask[j] = 1
	}
	hwnpu.EncodeF16(p.In["x"], x)
	hwnpu.EncodeF16(p.In["mask"], mask)
	if err := p.Run(); err != nil {
		return nil, err
	}
	y := make([]float32, len(ids)*d)
	hwnpu.DecodeF16(y, p.Out["y"][:2*len(y)])
	return meanPool(y, d, skip), nil
}

// meanPool averages the rows of h (rows x d) from skip on and scales the mean to length 1.
func meanPool(h []float32, d, skip int) []float32 {
	sum := make([]float64, d)
	n := len(h)/d - skip
	for r := skip; r < len(h)/d; r++ {
		for j, v := range h[r*d : (r+1)*d] {
			sum[j] += float64(v)
		}
	}
	var norm float64
	for j := range sum {
		sum[j] /= float64(n)
		norm += sum[j] * sum[j]
	}
	norm = math.Sqrt(norm)
	out := make([]float32, d)
	for j, v := range sum {
		if norm > 0 {
			out[j] = float32(v / norm)
		}
	}
	return out
}

// Cosine is the cosine similarity of two embeddings.
func Cosine(a, b []float32) float64 {
	var ab, aa, bb float64
	for i := range a {
		ab += float64(a[i]) * float64(b[i])
		aa += float64(a[i]) * float64(a[i])
		bb += float64(b[i]) * float64(b[i])
	}
	if aa == 0 || bb == 0 {
		return 0
	}
	return ab / math.Sqrt(aa*bb)
}
