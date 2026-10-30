package chat

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/audio"
	"github.com/shibukawa/hwnpu/internal/llm"
	_ "github.com/shibukawa/hwnpu/internal/llm/archs"
	"github.com/shibukawa/hwnpu/internal/llm/chain"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
	"github.com/shibukawa/hwnpu/internal/vision"
)

// Config says where the model is.
type Config struct {
	// Packages are the model converted with a KV cache (hwnpu-convert -cache), each an
	// .mlpackage or an out.parts.json, all with the same cache length and different tokens per
	// call (-seq): one of 1 to generate and one of about 64 to read prompts fast. They share
	// one cache.
	Packages []string
	// ModelConfig is config.json, Weights the checkpoint the token embeddings are read from,
	// Tokenizer tokenizer.json (default: next to config.json).
	ModelConfig, Weights, Tokenizer string
	// CacheDir is the compile cache ("" for none).
	CacheDir string
	// Vision is the LFM3-VL vision encoder converted by hwnpu-convert -arch lfm3-vision, a
	// package per image size (internal/vision), for a model that reads images (LFM3-VL); none to
	// take no images. Its config.json and image processor settings are next to ModelConfig.
	Vision []string
	// Audio is the LFM3-Audio (or LFM3-Omni) audio encoder converted by hwnpu-convert -arch
	// lfm3-audio-encoder, a package per clip length (internal/audio); none to take no audio.
	Audio []string
	// Log receives what loading does (nil for none).
	Log io.Writer
	// Name is the model's name in the API (default: the first package's).
	Name string
}

// Engine runs a model with a KV cache, one request at a time. A prompt that begins like an
// earlier one reads only the rest: the session goes back to the longest point it has marked
// inside the shared beginning (the end of each prompt read). Besides the sequence in the cache,
// it keeps a few earlier ones as copies of the whole cache, so that requests of several
// conversations in turn (an agent's, and the title or the summary its client asks for on the
// side) do not read each other's prompts again.
type Engine struct {
	Tok   *tokenizer.Tokenizer
	Stops map[int32]bool // tokens that end an answer
	Name  string         // the model's name, from the package

	dev    *hwnpu.Device
	chains []*chain.Chain
	w      llm.Weights
	model  llm.Model
	emb    map[int32][]float32
	vis    *vision.Encoder
	aud    *audio.Encoder
	images []encoded // encoded images and clips, the most recently used last
	nextID int32     // the token id of the next one's first row (they count down from -1)

	mu    sync.Mutex
	sess  *decode.Session
	cur   seq     // the sequence in the cache
	saved []saved // put aside, the most recently used last
}

// encoded is an encoded image or clip: its rows of embeddings stand in prompts as the token ids
// first, first-1, ...; the same one keeps its ids, so that a prompt with it reuses the cache.
type encoded struct {
	key   [32]byte // of the kind and the data
	first int32
	rows  [][]float32
}

// maxImages is how many encoded images and clips are kept.
var maxImages = 16

// seq is a sequence of tokens in the cache.
type seq struct {
	fed   []int32   // the tokens in the cache
	last  []float32 // the logits after fed
	marks []mark
}

type mark struct {
	m      *decode.Mark
	logits []float32
}

type saved struct {
	seq
	all *decode.Mark // the whole cache
}

var (
	maxMarks = 8
	maxSaved = 2
	minSaved = 256 // shorter sequences are cheap to read again
)

// reach is how far into prompt the sequence can be brought without reading, and the mark to go
// back to for it (nil for the sequence as it is, or for none at all when 0).
func (q *seq) reach(prompt []int32) (int, *mark) {
	n := 0
	for n < len(q.fed) && n < len(prompt) && q.fed[n] == prompt[n] {
		n++
	}
	if n == len(q.fed) && (n < len(prompt) || q.last != nil) {
		return n, nil
	}
	var best *mark
	for i := range q.marks {
		m := &q.marks[i]
		if p := m.m.Pos(); p <= n && (p < len(prompt) || m.logits != nil) && (best == nil || p > best.m.Pos()) {
			best = m
		}
	}
	if best == nil {
		return 0, nil
	}
	return best.m.Pos(), best
}

// Load loads the model.
func Load(ctx context.Context, c Config) (*Engine, error) {
	if len(c.Packages) == 0 {
		return nil, errors.New("chat: no package")
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
	cm, ok := model.(llm.Cached)
	if !ok {
		return nil, fmt.Errorf("chat: %s cannot run with a KV cache", a.Title)
	}
	if c.Tokenizer == "" {
		c.Tokenizer = filepath.Join(filepath.Dir(c.ModelConfig), "tokenizer.json")
	}
	e := &Engine{Stops: map[int32]bool{}, model: model, emb: map[int32][]float32{}, nextID: -1}
	e.Name = c.Name
	if e.Name == "" {
		e.Name = strings.TrimSuffix(strings.TrimSuffix(filepath.Base(c.Packages[0]), ".parts.json"), ".mlpackage")
	}
	if e.Tok, err = tokenizer.Load(c.Tokenizer); err != nil {
		return nil, err
	}
	for _, t := range []string{"<|im_end|>", "<|endoftext|>"} {
		if id, ok := e.Tok.ID(t); ok {
			e.Stops[id] = true
		}
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
	for i, pkg := range c.Packages {
		t0 := time.Now()
		var ch *chain.Chain
		if i == 0 {
			ch, err = chain.Open(ctx, e.dev, pkg)
		} else {
			ch, err = e.chains[0].Share(ctx, e.dev, pkg)
		}
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
		p := ch.Program(context.Background())
		if i == 0 {
			e.sess, err = decode.New(p, cm)
		} else {
			err = e.sess.Add(p)
		}
		if err != nil {
			e.Close()
			return nil, fmt.Errorf("chat: %s: %w", pkg, err)
		}
	}
	logf("cache %d tokens, up to %d tokens per call", e.sess.Cache(), e.sess.Q())
	if len(c.Vision) > 0 {
		t0 := time.Now()
		if e.vis, err = vision.Load(ctx, e.dev, filepath.Dir(c.ModelConfig), c.Vision); err != nil {
			e.Close()
			return nil, err
		}
		where := "all on the NPU"
		if !e.vis.OnNPU() {
			where = "NOT all on the NPU"
		}
		logf("vision encoder for images of %v, %s, loaded in %.1fs", e.vis.Sizes(), where, time.Since(t0).Seconds())
	}
	if len(c.Audio) > 0 {
		t0 := time.Now()
		if e.aud, err = audio.Load(ctx, e.dev, filepath.Dir(c.ModelConfig), c.Audio); err != nil {
			e.Close()
			return nil, err
		}
		where := "all on the NPU"
		if !e.aud.OnNPU() {
			where = "NOT all on the NPU"
		}
		logf("audio encoder for clips of up to %.0f s, %s, loaded in %.1fs", e.aud.MaxSeconds(), where, time.Since(t0).Seconds())
	}
	return e, nil
}

// Close releases the model.
func (e *Engine) Close() error {
	if e.vis != nil {
		e.vis.Close()
		e.vis = nil
	}
	if e.aud != nil {
		e.aud.Close()
		e.aud = nil
	}
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

// Context is the most tokens a prompt and its answer can have together.
func (e *Engine) Context() int { return e.sess.Cache() }

// Request is one generation.
type Request struct {
	Prompt    []int32
	MaxTokens int // 0: until the cache is full
	Sampling
	Seed uint64 // 0: random
}

// Stats says what a generation did.
type Stats struct {
	Prompt, Reused, Generated int
	// Finish is "stop" (the model ended the answer or the caller stopped it) or "length".
	Finish string
	// Looped says that the generation stopped because the model repeated itself.
	Looped          bool
	Prefill, Decode time.Duration
}

// ErrContext is returned for a prompt that does not fit in the cache.
var ErrContext = errors.New("chat: the prompt does not fit in the context")

// Generate reads the prompt and generates tokens until a stop token, MaxTokens or a full
// cache. After each token, emit gets all the tokens generated so far (without the stop
// token); it returns true to stop there.
func (e *Engine) Generate(ctx context.Context, r Request, emit func(gen []int32) (stop bool, err error)) (st Stats, err error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	st.Prompt = len(r.Prompt)
	if len(r.Prompt) == 0 {
		return st, errors.New("chat: empty prompt")
	}
	if len(r.Prompt) >= e.sess.Cache() {
		return st, fmt.Errorf("%w: %d tokens, the cache holds %d", ErrContext, len(r.Prompt), e.sess.Cache())
	}
	t0 := time.Now()
	logits, err := e.prefill(ctx, r.Prompt, &st)
	if err != nil {
		return st, err
	}
	st.Prefill = time.Since(t0)
	seed := r.Seed
	if seed == 0 {
		seed = uint64(time.Now().UnixNano())
	}
	rng := rand.New(rand.NewPCG(seed, 0))
	var gen []int32
	t1 := time.Now()
	defer func() { st.Decode = time.Since(t1) }()
	for {
		next := r.pick(slices.Clone(logits), gen, rng)
		if e.Stops[next] {
			st.Finish = "stop"
			return st, nil
		}
		gen = append(gen, next)
		st.Generated = len(gen)
		stop, err := emit(gen)
		if err != nil {
			return st, err
		}
		if stop {
			st.Finish = "stop"
			return st, nil
		}
		if looping(gen) {
			st.Finish, st.Looped = "stop", true
			return st, nil
		}
		if r.MaxTokens > 0 && len(gen) >= r.MaxTokens || e.sess.Pos() >= e.sess.Cache() {
			st.Finish = "length"
			return st, nil
		}
		if err := ctx.Err(); err != nil {
			return st, err
		}
		if logits, err = e.step([]int32{next}); err != nil {
			return st, err
		}
	}
}

// prefill brings the cache to the prompt and returns the logits after it.
func (e *Engine) prefill(ctx context.Context, prompt []int32, st *Stats) ([]float32, error) {
	n, m := e.cur.reach(prompt)
	from := -1
	for i := range e.saved {
		if k, _ := e.saved[i].reach(prompt); k > n {
			n, from = k, i
		}
	}
	if from >= 0 {
		// swap the sequence in the cache for the saved one
		sv := e.saved[from]
		e.saved = slices.Delete(e.saved, from, from+1)
		e.putAside()
		e.sess.Rewind(sv.all)
		e.cur = sv.seq
		n, m = e.cur.reach(prompt)
	} else if n < len(e.cur.fed)/2 {
		e.putAside()
	}
	switch {
	case m != nil:
		e.sess.Rewind(m.m)
		p := m.m.Pos()
		e.cur.fed, e.cur.last = e.cur.fed[:p], m.logits
		e.cur.marks = slices.DeleteFunc(e.cur.marks, func(m mark) bool { return m.m.Pos() > p })
	case n < len(e.cur.fed):
		e.sess.Reset()
		e.cur = seq{}
	}
	st.Reused = len(e.cur.fed)
	for len(e.cur.fed) < len(prompt) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		k := min(len(prompt)-len(e.cur.fed), e.sess.Q())
		if _, err := e.step(prompt[len(e.cur.fed) : len(e.cur.fed)+k]); err != nil {
			return nil, err
		}
	}
	e.cur.marks = append(e.cur.marks, mark{e.sess.Mark(), e.cur.last})
	if len(e.cur.marks) > maxMarks {
		e.cur.marks = e.cur.marks[1:]
	}
	return e.cur.last, nil
}

// putAside saves a copy of the sequence in the cache, when it is long enough to be worth it.
func (e *Engine) putAside() {
	if len(e.cur.fed) < minSaved {
		return
	}
	q := seq{fed: slices.Clone(e.cur.fed), last: e.cur.last, marks: slices.Clone(e.cur.marks)}
	e.saved = append(e.saved, saved{q, e.sess.Save()})
	if len(e.saved) > maxSaved {
		e.saved = e.saved[1:]
	}
}

// Images reports whether the engine reads images.
func (e *Engine) Images() bool { return e.vis != nil }

// Audio reports whether the engine reads audio.
func (e *Engine) Audio() bool { return e.aud != nil }

// ErrImage is returned for an image or a clip the engine cannot read.
var ErrImage = errors.New("chat: media")

// Tokens turns a rendered prompt into tokens. Each MediaMarker in it is the next of media: an
// image is <|vision_start|>, a token per row of its embeddings, <|vision_end|>; a clip is its
// rows alone for LFM3-Audio and between the same tokens for LFM3-Omni.
func (e *Engine) Tokens(ctx context.Context, prompt string, media []Media) ([]int32, error) {
	pieces := strings.Split(prompt, MediaMarker)
	if len(pieces)-1 != len(media) {
		return nil, fmt.Errorf("%w: %d images and clips for %d places in the prompt (they go in user messages)", ErrImage, len(media), len(pieces)-1)
	}
	var ids []int32
	for i, piece := range pieces {
		ids = append(ids, e.Tok.Encode(piece)...)
		if i == len(media) {
			break
		}
		e.mu.Lock()
		enc, err := e.encode(ctx, media[i])
		e.mu.Unlock()
		if err != nil {
			return nil, err
		}
		wrap := !media[i].Audio || e.aud.Omni()
		start, ok1 := e.Tok.ID(vision.Start)
		end, ok2 := e.Tok.ID(vision.End)
		if wrap && (!ok1 || !ok2) {
			return nil, fmt.Errorf("%w: the tokenizer has no %s and %s", ErrImage, vision.Start, vision.End)
		}
		if wrap {
			ids = append(ids, start)
		}
		for r := range enc.rows {
			ids = append(ids, enc.first-int32(r))
		}
		if wrap {
			ids = append(ids, end)
		}
	}
	return ids, nil
}

// encode returns the encoded image or clip, encoding it unless it is kept.
func (e *Engine) encode(ctx context.Context, m Media) (*encoded, error) {
	kind := "image:"
	if m.Audio {
		kind = "audio:"
	}
	key := sha256.Sum256([]byte(kind + m.Data))
	for i, im := range e.images {
		if im.key == key {
			e.images = append(slices.Delete(e.images, i, i+1), im)
			return &e.images[len(e.images)-1], nil
		}
	}
	var flat []float32
	var n int
	switch {
	case m.Audio && e.aud == nil:
		return nil, fmt.Errorf("%w: this model was loaded without an audio encoder", ErrImage)
	case m.Audio:
		clip, err := audio.Decode(m.Data)
		if err != nil {
			return nil, fmt.Errorf("%w: %v", ErrImage, err)
		}
		if flat, n, err = e.aud.Encode(ctx, clip); errors.Is(err, audio.ErrTooLong) {
			return nil, fmt.Errorf("%w: %v", ErrImage, err)
		} else if err != nil {
			return nil, err
		}
	case e.vis == nil:
		return nil, fmt.Errorf("%w: this model was loaded without a vision encoder", ErrImage)
	default:
		pic, err := vision.DecodeDataURL(m.Data)
		if err != nil {
			return nil, fmt.Errorf("%w: %v", ErrImage, err)
		}
		if flat, n, err = e.vis.Encode(ctx, pic); err != nil {
			return nil, err
		}
	}
	d := e.model.Hidden()
	im := encoded{key: key, first: e.nextID}
	for r := range n {
		im.rows = append(im.rows, flat[r*d:(r+1)*d])
	}
	e.nextID -= int32(n)
	if len(e.images) >= maxImages {
		e.images = e.images[1:]
	}
	e.images = append(e.images, im)
	return &e.images[len(e.images)-1], nil
}

// imageRow returns the embedding row of an image token (a negative id).
func (e *Engine) imageRow(t int32) ([]float32, bool) {
	for _, im := range e.images {
		if r := int(im.first - t); r >= 0 && r < len(im.rows) {
			return im.rows[r], true
		}
	}
	return nil, false
}

// step runs the tokens and records them as fed.
func (e *Engine) step(toks []int32) ([]float32, error) {
	embs := make([][]float32, len(toks))
	for i, t := range toks {
		if t < 0 {
			v, ok := e.imageRow(t)
			if !ok {
				return nil, fmt.Errorf("%w: an image of the prompt is no longer kept; send the request again", ErrImage)
			}
			embs[i] = v
			continue
		}
		v, ok := e.emb[t]
		if !ok {
			var err error
			if v, err = llm.Embed(e.w, e.model, []int32{t}); err != nil {
				return nil, err
			}
			if len(e.emb) >= 1<<15 {
				clear(e.emb)
			}
			e.emb[t] = v
		}
		embs[i] = v
	}
	logits, err := e.sess.Step(embs)
	if err != nil {
		// the cache may be half written
		e.sess.Reset()
		e.cur = seq{}
		return nil, err
	}
	e.cur.fed = append(e.cur.fed, toks...)
	e.cur.last = logits
	return logits, nil
}

// looping reports whether the last loopSpan tokens are one pattern of at most loopPeriod tokens
// over and over: a small model that has fallen into that does not come out of it.
func looping(gen []int32) bool {
	if len(gen) < loopSpan {
		return false
	}
	tail := gen[len(gen)-loopSpan:]
next:
	for p := 1; p <= loopPeriod; p++ {
		for i := p; i < len(tail); i++ {
			if tail[i] != tail[i-p] {
				continue next
			}
		}
		return true
	}
	return false
}

const (
	loopSpan   = 64
	loopPeriod = 16
)
