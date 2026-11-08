package chat

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"image"
	"io"
	"math/rand/v2"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/audio"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	_ "github.com/shibukawa/hwnpu/internal/llm/archs"
	"github.com/shibukawa/hwnpu/internal/llm/chain"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/mem"
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
	// Tiers are sets of packages like Packages, one set per cache length, shortest first and
	// Packages last: a conversation runs on the shortest cache that holds it and moves up as it
	// grows, since a step costs in proportion to the cache (Setup.Tiers). Empty: Packages alone.
	Tiers [][]string
	// MTP is the program of the checkpoint's multi-token prediction layer (Setup.MTP), which
	// guesses the drafts of speculative decoding from the model's final hidden states: the
	// Packages must output them (hidden-out). Empty: drafts from the prompt alone.
	MTP []string
	// ModelConfig is config.json, Weights the checkpoint the token embeddings are read from,
	// Tokenizer tokenizer.json (default: next to config.json).
	ModelConfig, Weights, Tokenizer string
	// CacheDir is the compile cache ("" for none).
	CacheDir string
	// PromptDir keeps prompts read on disk for later runs ("" for none): a prompt that begins
	// like an earlier one (an agent's system message and tools) comes back from there instead
	// of being read again after a restart (store).
	PromptDir string
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
	// Backend picks the device (llm.FindDevice): a device backend or the program format of the
	// packages; empty for HWNPU_BACKEND or the NPU found.
	Backend string
	// IRON is a model compiled by amd/IRON for AMD's NPU (tools/amd/iron_export_lm.py), in
	// place of Packages: its manifest, ELFs and arena. Tokenizer defaults to the
	// tokenizer.json in it.
	IRON string
	// Idle, when above zero, releases an IRON model's device memory after this long without a
	// request, and loads it again for the next one (idle.go).
	Idle time.Duration
	// Dense reads the whole cache of a model split at its attention on every call, instead of
	// the two-level attention (decode.Sparse) past the length where it costs fewer calls, about
	// 48K tokens with chunks of 8192.
	Dense bool
}

// programName is a model's name from the path of its program: the file name without the
// extension of a package, an IR or a list of parts.
func programName(path string) string {
	name := filepath.Base(path)
	for _, ext := range []string{".parts.json", ".mlpackage", ".xml"} {
		name = strings.TrimSuffix(name, ext)
	}
	return name
}

// Engine runs a model with a KV cache, one request at a time. A prompt that begins like an
// earlier one reads only the rest: the session goes back to the longest point it has marked
// inside the shared beginning (the end of each prompt read). Besides the sequence in the cache,
// it keeps a few earlier ones as copies of the whole cache, so that requests of several
// conversations in turn (an agent's, and the title or the summary its client asks for on the
// side) do not read each other's prompts again.
type Engine struct {
	Tok      *tokenizer.Tokenizer
	Template Template       // the model's chat format
	Stops    map[int32]bool // tokens that end an answer
	Name     string         // the model's name, from the package
	// NoThink makes the model answer without thinking unless a request asks for thinking
	// (reasoning_effort, enable_thinking): Model.NoThink when -think was not given.
	NoThink bool

	// turnStart is the token that starts a turn of the chat format, where prefill also marks the
	// sequence when markTurns (a model whose recurrent states go back only to a mark: the next
	// turn renders the last assistant turn anew, Qwen3.6 without its thinking, so it shares the
	// prompt only up to that turn's start). -1: none.
	turnStart int32
	turnEnd   int32 // the newline that ends a turn's header, or -1
	markTurns bool

	dev    *hwnpu.Device
	tiers  []tier // by cache length, shortest first
	ti     int    // the tier the session runs on
	w      llm.Weights
	model  llm.Model
	emb    map[int32][]float32 // tokens' rows (llm.TokenRow) read so far
	vis    *vision.Encoder
	aud    *audio.Encoder
	images []encoded // encoded images and clips, the most recently used last
	nextID int32     // the token id of the next one's first row (they count down from -1)

	// drafter guesses the next tokens of a sequence, up to k (speculative decoding); nil for
	// none. Load sets it to Lookup when the model has a program to check drafts with (Draft).
	drafter func(seq []int32, k int) []int32
	mtp     *mtpDrafter // the multi-token prediction layer (Config.MTP); nil for none
	kinds   []tokenKind // Tok's tokens, for repeats; set by the first generation

	ir    *iron.Model // the model of an IRON export (Config.IRON); nil for none
	store *store      // prompts kept on disk (Config.PromptDir); nil for none
	log   io.Writer   // Config.Log

	// idle is how long without a request before an IRON model's device memory is released
	// (Config.Idle, idle.go); lastUsed is the last request's end, idleTimer the clock.
	idle      time.Duration
	lastUsed  time.Time
	idleTimer *time.Timer
	devName   string // the device's name, kept while the device is closed (idle.go)

	mu     sync.Mutex
	closed bool // by Close: the programs and the weights are gone
	sess   session
	cur    seq     // the sequence in the cache
	saved  []saved // put aside, the most recently used last
}

func (e *Engine) logf(format string, args ...any) {
	if e.log != nil {
		fmt.Fprintf(e.log, format+"\n", args...)
	}
}

// encoded is an encoded image or clip: its rows of embeddings stand in prompts as the token ids
// first, first-1, ...; the same one keeps its ids, so that a prompt with it reuses the cache.
type encoded struct {
	key   [32]byte // of the kind and the data
	first int32
	rows  [][]float32
	// An IRON model's towers run on the device when a prompt chunk reaches the media (ironSession):
	// n soft tokens of the decoded image or clip, and no rows.
	n     int
	audio bool
	img   image.Image
	clip  []float32
}

// count is the soft tokens of the image or clip.
func (im *encoded) count() int {
	if im.rows != nil {
		return len(im.rows)
	}
	return im.n
}

// maxImages is how many encoded images and clips are kept.
var maxImages = 16

// seq is a sequence of tokens in the cache.
type seq struct {
	fed   []int32   // the tokens in the cache
	last  []float32 // the logits after fed
	marks []mark
	base  *entry // the store's entry the sequence came from or extended last; nil for none
}

type mark struct {
	m      *decode.Mark
	logits []float32
}

type saved struct {
	seq
	all  *decode.Mark // the whole cache
	tier int          // the tier the cache was taken on
}

// tier is one cache length's programs: the one that generates and the one that reads prompts,
// sharing their cache buffers.
type tier struct {
	chains []*chain.Chain
	progs  []decode.Program
	cache  int
	sparse decode.Sparse // the two-level attention of its sessions, a Bucket of 0 for none
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
	if c.IRON != "" {
		return loadIRON(ctx, c)
	}
	if len(c.Packages) == 0 {
		return nil, errors.New("chat: no package")
	}
	logf := func(format string, args ...any) {
		if c.Log != nil {
			fmt.Fprintf(c.Log, format+"\n", args...)
		}
	}
	name := c.Name
	if name == "" {
		name = programName(c.Packages[0])
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
		c.Tokenizer = filepath.Join(filepath.Dir(c.ModelConfig), TokenizerFiles(a.Name)[0])
	}
	e := &Engine{Stops: map[int32]bool{}, model: model, emb: map[int32][]float32{}, nextID: -1, log: c.Log}
	e.Name = name
	if e.Template, err = TemplateFor(a.Name); err != nil {
		return nil, err
	}
	if e.Tok, err = tokenizer.Load(c.Tokenizer); err != nil {
		return nil, err
	}
	for _, t := range e.Template.Stops() {
		if id, ok := e.Tok.ID(t); ok {
			e.Stops[id] = true
		}
	}
	if e.w, err = llm.OpenWeights(c.Weights, a); err != nil {
		return nil, err
	}
	if e.dev, _, err = llm.OpenDevice(ctx, c.Backend, c.CacheDir); err != nil {
		e.Close()
		return nil, err
	}
	logf("loading %s onto %s: %d programs, up to a minute or two (%s)", name, llm.DeviceName(e.dev), len(c.Packages), compileNote(e.dev))
	wired0, wired := mem.NPUWired()
	tiers := c.Tiers
	if len(tiers) == 0 {
		tiers = [][]string{c.Packages}
	}
	total, loaded := 0, 0
	for _, pkgs := range tiers {
		total += len(pkgs)
	}
	for _, pkgs := range tiers {
		var t tier
		for i, pkg := range pkgs {
			loaded++
			bar := console.NewBar(c.Log, fmt.Sprintf("  [%d/%d] %s", loaded, total, filepath.Base(pkg)))
			var ch *chain.Chain
			if i == 0 {
				ch, err = chain.Open(ctx, e.dev, pkg)
			} else {
				ch, err = t.chains[0].Share(ctx, e.dev, pkg)
			}
			if err != nil {
				bar.Done("failed")
				t.close()
				e.Close()
				return nil, err
			}
			t.chains = append(t.chains, ch)
			t.progs = append(t.progs, ch.Program(context.Background()))
			where := "all on the NPU"
			if !ch.OnNPU() {
				where = "NOT all on the NPU"
			}
			parts := "1 part"
			if n := len(ch.Parts()); n > 1 {
				parts = fmt.Sprintf("%d parts", n)
			}
			bar.Done(fmt.Sprintf("%s, %s, %s", console.Duration(bar.Elapsed()), parts, where))
		}
		s, err := t.session(cm)
		if err != nil {
			t.close()
			e.Close()
			return nil, err
		}
		if !c.Dense && t.chains[0].Paged() > 0 && t.chains[0].HasMass() {
			if sp := decode.DefaultSparse(); s.SetSparse(sp) != nil {
				logf("no two-level attention: %v", s.SetSparse(sp))
			} else {
				t.sparse = sp
			}
		}
		t.cache = s.Cache()
		s.Close()
		if n := len(e.tiers); n > 0 && t.cache <= e.tiers[n-1].cache {
			t.close()
			e.Close()
			return nil, fmt.Errorf("chat: the tiers' caches must grow: %d after %d", t.cache, e.tiers[n-1].cache)
		}
		e.tiers = append(e.tiers, t)
	}
	if e.sess, err = e.tiers[0].session(cm); err != nil {
		e.Close()
		return nil, err
	}
	paged := ""
	if ds, ok := e.sess.(*decode.Session); ok && ds.Paged() > 0 {
		paged = e.fitMemory(ds, mem.Physical(), logf)
	}
	if c.PromptDir != "" {
		last := e.tiers[len(e.tiers)-1]
		var sizes []int64
		for _, ch := range last.chains {
			for _, p := range ch.Parts() {
				sizes = append(sizes, p.WeightBytes)
			}
		}
		dir := filepath.Join(c.PromptDir, storeKey(name, tiers[len(tiers)-1], sizes))
		if e.store, err = openStore(dir); err != nil {
			logf("prompt store: %v (prompts will not be kept)", err)
			e.store = nil
		} else {
			logf("prompt store: %s, %d prompts kept", dir, len(e.store.entries))
		}
	}
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
	if w, ok := mem.NPUWired(); wired && ok && w > wired0 {
		// a program Core ML has just compiled comes onto the Neural Engine at its first call
		// (an M3: 0.27 of 2.43 GiB after the load, all of it after the first question)
		var weights int64
		for _, t := range e.tiers {
			for _, ch := range t.chains {
				for _, p := range ch.Parts() {
					weights += p.WeightBytes
				}
			}
		}
		held := w - wired0
		if held < weights/2 {
			logf("the Neural Engine holds %.2f GiB for them so far: programs compiled just now come onto it at their first call (%.2f GiB of weights in all), in kernel memory of its driver, not in this process's footprint", float64(held)/(1<<30), float64(weights)/(1<<30))
		} else {
			logf("the Neural Engine holds %.2f GiB for them: kernel memory of its driver, in neither this process's footprint nor aned's", float64(held)/(1<<30))
		}
	}
	caches := ""
	if len(e.tiers) > 1 {
		var cs []string
		for _, t := range e.tiers {
			cs = append(cs, fmt.Sprint(t.cache))
		}
		caches = " in tiers of " + strings.Join(cs, ", ")
	}
	if d := e.Draft(); d > 0 {
		e.drafter = LookupDraft
		how := ""
		if len(c.MTP) > 0 {
			t0 := time.Now()
			if e.mtp, err = loadMTP(ctx, e, c); err != nil {
				e.Close()
				return nil, err
			}
			e.drafter = e.mtp.draft
			how = fmt.Sprintf(" (the multi-token prediction layer's %d at most, or the prompt's; loaded in %.1fs)", e.mtp.k, time.Since(t0).Seconds())
		}
		logf("ready: context %d tokens%s%s, up to %d tokens per call, drafts of %d checked per call%s", e.Context(), caches, paged, e.sess.Q(), d, how)
	} else {
		logf("ready: context %d tokens%s%s, up to %d tokens per call", e.Context(), caches, paged, e.sess.Q())
	}
	pruneCache(c.CacheDir, logf) // after loading, which marks this model's entries used
	return e, nil
}

// memoryShare is the part of the machine's memory the cache of a model split at its attention
// may grow to: the rest is the programs' (1 to 3 GiB), the system's and the user's.
const memoryShare = 4

// fitMemory keeps the cache of a session split at its attention (with the summaries of the
// two-level attention) within a quarter of the machine's memory (physical, in bytes; 0 when not known): the cache grows with the conversation,
// so the context is the model's longest unless the cache at that length would not fit. It
// returns what the ready line says of it.
func (e *Engine) fitMemory(ds *decode.Session, physical int64, logf func(string, ...any)) string {
	row := int64(ds.RowBytes())
	if sp := ds.Sparse(); sp.Bucket > 0 {
		row += row / int64(sp.Bucket) // the two-level attention's summaries
	}
	full := row * int64(ds.Cache())
	if budget := physical / memoryShare; physical > 0 && full > budget {
		n := int(budget/row) / ds.Paged() * ds.Paged()
		if n > 0 && n < ds.Cache() && ds.Limit(n) == nil {
			logf("the context is %d tokens, not %d: its cache would take %.2f GiB, over a quarter of the %.0f GiB of memory", n, e.tiers[len(e.tiers)-1].cache, float64(full)/(1<<30), float64(physical)/(1<<30))
			e.tiers[len(e.tiers)-1].cache = n
			full = row * int64(n)
		}
	}
	two := ""
	if sp := ds.Sparse(); sp.Bucket > 0 {
		two = fmt.Sprintf("; past about %dK tokens, %d keys exactly and the rest as means of %d", sparseFrom(ds.Paged(), sp)/1024, sp.Exact*ds.Paged(), sp.Bucket)
	}
	return fmt.Sprintf(", its cache growing with the conversation by %.0f KiB a token to %.2f GiB (attention in chunks of %d%s)", float64(row)/1024, float64(full)/(1<<30), ds.Paged(), two)
}

// sparseFrom is about the length past which the two-level attention runs for a call of one
// token: where its calls (twice the chunks of summaries, one for the summaries read exactly and
// the chunks read exactly) are fewer than the cache's chunks.
func sparseFrom(chunk int, sp decode.Sparse) int {
	for n := chunk; ; n += chunk {
		summaries := (n/sp.Bucket + chunk - 1) / chunk
		if 2*summaries+1+sp.Exact < n/chunk {
			return n
		}
	}
}

// compileNote says how long loading takes on the device the first time.
func compileNote(dev *hwnpu.Device) string {
	switch dev.Info().Backend {
	case "coreml":
		return "Core ML compiles them the first time"
	case "levelzero":
		return "the driver compiles them the first time, which can take several minutes for a large model; the compile cache keeps them"
	}
	return "they are compiled first"
}

// DeviceName is the NPU the model runs on, as logs call it (llm.DeviceName).
func (e *Engine) DeviceName() string {
	if e.dev == nil {
		return e.devName
	}
	return llm.DeviceName(e.dev)
}

// session starts an empty sequence on the tier's programs.
func (t *tier) session(cm llm.Cached) (*decode.Session, error) {
	s, err := decode.New(t.progs[0], cm)
	if err != nil {
		return nil, fmt.Errorf("chat: %s: %w", filepath.Base(t.chains[0].Parts()[0].Path), err)
	}
	for i, p := range t.progs[1:] {
		if err := s.Add(p); err != nil {
			return nil, fmt.Errorf("chat: %s: %w", filepath.Base(t.chains[i+1].Parts()[0].Path), err)
		}
	}
	if t.sparse.Bucket > 0 {
		if err := s.SetSparse(t.sparse); err != nil {
			s.Close()
			return nil, fmt.Errorf("chat: %s: %w", filepath.Base(t.chains[0].Parts()[0].Path), err)
		}
	}
	return s, nil
}

func (t *tier) close() {
	for i := len(t.chains) - 1; i >= 0; i-- {
		t.chains[i].Close()
	}
	t.chains = nil
}

// tierFor is the shortest tier whose cache holds a prompt of n tokens and something after it,
// or the last.
func (e *Engine) tierFor(n int) int {
	for i, t := range e.tiers {
		if n < t.cache {
			return i
		}
	}
	return len(e.tiers) - 1
}

// use starts an empty sequence on tier i (the session's own tier: Reset).
func (e *Engine) use(i int) error {
	if i == e.ti || len(e.tiers) == 0 {
		e.sess.Reset()
		return nil
	}
	cm := e.model.(llm.Cached)
	s, err := e.tiers[i].session(cm)
	if err != nil {
		return err
	}
	closeSession(e.sess)
	e.sess, e.ti = s, i
	return nil
}

// closeSession frees what a session holds (the chunks of a cache split at its attention).
func closeSession(s session) {
	if ds, ok := s.(*decode.Session); ok && ds != nil {
		ds.Close()
	}
}

// room makes sure the cache can take n more tokens, moving the sequence up to a longer tier
// when it cannot (decode.Session.Migrate copies the rows), and says whether it can.
func (e *Engine) room(n int) (bool, error) {
	for e.sess.Pos()+n > e.sess.Cache() && e.ti+1 < len(e.tiers) {
		t := e.tiers[e.ti+1]
		if err := e.sess.Migrate(t.progs[0]); err != nil {
			return false, err
		}
		for _, p := range t.progs[1:] {
			if err := e.sess.Add(p); err != nil {
				return false, err
			}
		}
		e.ti++
	}
	return e.sess.Pos()+n <= e.sess.Cache(), nil
}

// Draft is how many tokens of drafts a generation checks in one call (speculative decoding):
// one fewer than the tokens per call of the program that generates, the one of the fewest, when
// it takes more than 1 (Setup.Draft); 0 for none. Generation stays on that program whether or
// not there are drafts: the Neural Engine takes about 100 ms to switch between programs.
func (e *Engine) Draft() int {
	return e.sess.Qs()[0] - 1
}

// LookupDraft guesses the next tokens of seq from seq itself (prompt lookup decoding): the tokens
// that followed the latest earlier occurrence of its last few tokens (4 down to 1), up to k. An
// agent copies what it read into what it writes, so its answers repeat the prompt in long
// stretches. Guessing from 1 token alone is right less often, but checking a guess costs nothing
// more on the program that generates (design.md 16.37: 7% more tokens per call).
func LookupDraft(seq []int32, k int) []int32 {
	const maxN, minN = 4, 1
	for n := maxN; n >= minN; n-- {
		if len(seq) <= n {
			continue
		}
		key := seq[len(seq)-n:]
	next:
		for i := len(seq) - n - 1; i >= 0; i-- {
			for j := range n {
				if seq[i+j] != key[j] {
					continue next
				}
			}
			return seq[i+n : min(i+n+k, len(seq))]
		}
	}
	return nil
}

// ErrClosed is returned for a request to a model that Close released.
var ErrClosed = errors.New("chat: the model is closed")

// Close releases the model. It waits for the step that a generation is running, and the
// generations after it fail with ErrClosed.
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.closed = true
	if e.idleTimer != nil {
		e.idleTimer.Stop()
		e.idleTimer = nil
	}
	if e.mtp != nil {
		e.mtp.close()
		e.mtp = nil
	}
	if e.vis != nil {
		e.vis.Close()
		e.vis = nil
	}
	if e.aud != nil {
		e.aud.Close()
		e.aud = nil
	}
	if e.sess != nil {
		closeSession(e.sess)
	}
	for i := len(e.tiers) - 1; i >= 0; i-- {
		e.tiers[i].close()
	}
	e.tiers = nil
	if e.ir != nil {
		e.ir.Close()
		e.ir = nil
	}
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
func (e *Engine) Context() int {
	if len(e.tiers) == 0 { // a session given directly (tests)
		return e.sess.Cache()
	}
	return e.tiers[len(e.tiers)-1].cache
}

// Request is one generation.
type Request struct {
	Prompt    []int32
	MaxTokens int // 0: until the cache is full
	Sampling
	Seed uint64 // 0: random
	// Think bounds the model's thinking (ThinkLimit; the zero value does not).
	Think ThinkLimit
}

// ThinkLimit is a budget for the model's thinking: when it has generated Budget tokens inside
// its thinking (between Open and Close, or since the prompt when Inside) without closing it, End
// is fed as if generated, so that the answer begins. A small model may think for thousands of
// tokens, and an agent waits for the answer meanwhile.
type ThinkLimit struct {
	Budget      int   // most tokens of thinking (0: no limit)
	Open, Close int32 // the tokens that open and close the thinking
	End         []int32
	Inside      bool // the prompt ends inside the thinking
}

// ThinkLimit bounds the thinking of a request to budget tokens (0, or a model without thinking
// tokens: none), with the template's tokens; inside says the prompt ends inside the thinking
// (Template.Render).
func (e *Engine) ThinkLimit(budget int, inside bool) ThinkLimit {
	o, c, ok := e.thinkTokens()
	if budget <= 0 || !ok {
		return ThinkLimit{}
	}
	_, _, end := e.Template.Think()
	return ThinkLimit{Budget: budget, Open: o, Close: c, End: e.Tok.Encode(end), Inside: inside}
}

// thinkTokens returns the tokens that open and close the thinking, if the model has them.
func (e *Engine) thinkTokens() (open, close int32, ok bool) {
	if e.Tok == nil || e.Template == nil {
		return 0, 0, false
	}
	o, c, _ := e.Template.Think()
	open, ok1 := e.Tok.ID(o)
	close, ok2 := e.Tok.ID(c)
	return open, close, ok1 && ok2
}

// Stats says what a generation did.
type Stats struct {
	Prompt, Reused, Generated int
	// Restored is how many of the reused tokens came back from the prompt store on disk.
	Restored int
	// Finish is "stop" (the model ended the answer or the caller stopped it) or "length".
	Finish string
	// Looped says that the generation stopped because the model repeated itself.
	Looped bool
	// ThinkCut says that the thinking reached its budget (Request.Think) and was ended;
	// ThinkLooped, that it was ended before, going round in circles (repeats).
	ThinkCut, ThinkLooped bool
	// Drafted and Accepted count the tokens guessed ahead (speculative decoding) and those the
	// model confirmed; Calls the runs of the model while generating.
	Drafted, Accepted, Calls int
	// Cache is the length of the cache the generation ended on (the tier, Config.Tiers).
	Cache           int
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
	if e.closed {
		return st, ErrClosed
	}
	if e.ir != nil && e.ir.Released() {
		if err := e.reload(ctx); err != nil {
			return st, err
		}
	}
	defer e.touch()
	st.Prompt = len(r.Prompt)
	if len(r.Prompt) == 0 {
		return st, errors.New("chat: empty prompt")
	}
	if len(r.Prompt) >= e.Context() {
		return st, fmt.Errorf("%w: %d tokens, the cache holds %d", ErrContext, len(r.Prompt), e.Context())
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
	defer func() { st.Decode, st.Cache = time.Since(t1), e.sess.Cache() }()
	think := r.Think
	inside, thought := think.Budget > 0 && think.Inside, 0
	if e.kinds == nil && e.Tok != nil {
		e.kinds = tokenKinds(e.Tok)
	}
	rep, circling := newRepeats(e.kinds), false
	opens, closes, marks := e.thinkTokens()
	// take takes a generated token into the answer: it is in the cache already, or is next to
	// go in. It says whether the generation is over.
	take := func(tok int32) (over bool, err error) {
		if e.Stops[tok] {
			st.Finish = "stop"
			return true, nil
		}
		gen = append(gen, tok)
		st.Generated = len(gen)
		if think.Budget > 0 {
			switch {
			case inside && tok == think.Close:
				inside = false
			case !inside && tok == think.Open:
				inside, thought = true, 0
			case inside:
				thought++
			}
		}
		if marks && (tok == opens || tok == closes) {
			// the thinking and the answer count their repeats apart
			rep.reset()
			circling = false
		}
		stop, err := emit(gen)
		if err != nil {
			return true, err
		}
		if stop {
			st.Finish = "stop"
			return true, nil
		}
		if looped := rep.add(tok); looped || looping(gen) {
			if !inside {
				st.Finish, st.Looped = "stop", true
				return true, nil
			}
			// the thinking ends as at its budget, and the answer begins
			circling = true
		}
		if r.MaxTokens > 0 && len(gen) >= r.MaxTokens {
			st.Finish = "length"
			return true, nil
		}
		if ok, err := e.room(1); err != nil {
			return true, err
		} else if !ok {
			st.Finish = "length"
			return true, nil
		}
		return false, ctx.Err()
	}
	k := e.Draft()
	if e.drafter == nil {
		k = 0
	}
	next := r.pick(slices.Clone(logits), gen, rng)
	for {
		// next is generated and not in the cache yet
		if over, err := take(next); over || err != nil {
			return st, err
		}
		toks := []int32{next}
		var drafts []int32
		switch {
		case inside && (thought >= think.Budget || circling):
			// end the thinking: End goes into the sequence as if the model had generated it
			if ok, err := e.room(1 + len(think.End)); err != nil {
				return st, err
			} else if !ok {
				st.Finish = "length"
				return st, nil
			}
			toks = append(toks, think.End...)
			gen = append(gen, think.End...)
			st.Generated = len(gen)
			inside, st.ThinkCut, st.ThinkLooped = false, !circling, circling
			circling = false
			rep.reset()
			if stop, err := emit(gen); err != nil {
				return st, err
			} else if stop {
				st.Finish = "stop"
				return st, nil
			}
		case k > 0:
			// guess what follows, as far as the answer, the cache and the thinking allow
			room := e.sess.Cache() - e.sess.Pos() - 1
			if r.MaxTokens > 0 {
				room = min(room, r.MaxTokens-len(gen))
			}
			if inside {
				room = min(room, think.Budget-thought)
			}
			if room > 0 {
				drafts = e.drafter(append(slices.Clone(e.cur.fed), next), min(k, room))
			}
		}
		if len(drafts) == 0 {
			for len(toks) > 0 {
				n := min(len(toks), e.sess.Q())
				if logits, err = e.step(toks[:n]); err != nil {
					return st, err
				}
				st.Calls++
				toks = toks[n:]
			}
			next = r.pick(slices.Clone(logits), gen, rng)
			continue
		}
		// one call checks the drafts: each is kept when the sampling picks it from the logits
		// before it, and the token picked where they part (or after the last) is the next
		st.Drafted += len(drafts)
		var picked int32
		kept, err := e.speculate(append(toks, drafts...), func(row func(i int) []float32) int {
			seen := gen
			for i, d := range drafts {
				picked = r.pick(row(i), seen, rng)
				if picked != d {
					return i + 1
				}
				seen = append(seen, d)
			}
			picked = r.pick(row(len(drafts)), seen, rng)
			return len(drafts) + 1
		})
		if err != nil {
			return st, err
		}
		st.Calls++
		st.Accepted += kept - 1
		for _, d := range drafts[:kept-1] {
			if over, err := take(d); over || err != nil {
				return st, err
			}
		}
		next = picked
	}
}

// logitsQ is, when the tier's program that reads prompts gives no logits (llm.Options.KVOnly),
// the most tokens a call of those that do reads; otherwise 0.
func (e *Engine) logitsQ() int {
	q, none := 0, false
	if e.ti < len(e.tiers) {
		for _, p := range e.tiers[e.ti].progs {
			if p.NoLogits {
				none = true
			} else if n := p.InShape["x"]; len(n) == 3 {
				q = max(q, n[1])
			}
		}
	}
	if !none {
		return 0
	}
	return q
}

// prefill brings the cache to the prompt and returns the logits after it.
func (e *Engine) prefill(ctx context.Context, prompt []int32, st *Stats) ([]float32, error) {
	// how far what is in memory brings the prompt: the sequence in the cache (n, going back
	// to the mark m when not nil) or a sequence put aside (from)
	var n, from int
	var m *mark
	plan := func() {
		n, m = e.cur.reach(prompt)
		from = -1
		for i := range e.saved {
			if k, _ := e.saved[i].reach(prompt); k > n {
				n, from = k, i
			}
		}
	}
	plan()
	// Images' and clips' ids are numbered anew in each process, so the store matches only the
	// tokens before the first of them, and keeps no sequence holding one.
	plain := prompt
	if i := slices.IndexFunc(prompt, func(t int32) bool { return t < 0 }); i >= 0 {
		plain = prompt[:i]
	}
	for e.store != nil && n < len(plain) {
		// the store brings the prompt further than what is in memory: start on its rows; an
		// entry that cannot be read is dropped, and the next best is tried
		se, sp := e.store.find(plain)
		if se == nil || sp <= n {
			break
		}
		if ok, err := e.restore(se, sp, prompt, st); errors.Is(err, errNoLogits) {
			break
		} else if err != nil {
			return nil, err
		} else if ok {
			n, m, from = sp, nil, -2
			break
		}
		plan() // the session may have been emptied
	}
	if from >= 0 {
		// swap the sequence in the cache for the saved one
		sv := e.saved[from]
		e.saved = slices.Delete(e.saved, from, from+1)
		e.putAside()
		if err := e.use(sv.tier); err != nil {
			return nil, err
		}
		e.sess.Rewind(sv.all)
		e.cur = sv.seq
		n, m = e.cur.reach(prompt)
	} else if from == -1 && n < len(e.cur.fed)/2 {
		e.putAside()
	}
	switch {
	case from == -2:
		// restored from the store
	case m != nil:
		e.sess.Rewind(m.m)
		p := m.m.Pos()
		e.cur.fed, e.cur.last = e.cur.fed[:p], m.logits
		e.cur.marks = slices.DeleteFunc(e.cur.marks, func(m mark) bool { return m.m.Pos() > p })
	case n < len(e.cur.fed) || len(e.cur.fed) == 0:
		// start over, on the shortest tier that holds the prompt
		if err := e.use(e.tierFor(len(prompt))); err != nil {
			return nil, err
		}
		e.cur = seq{}
	}
	st.Reused = len(e.cur.fed)
	var kept []storedMark // marks for the store, every markEvery tokens of the reading
	cut := -1             // the last turn's start: a mark there too (markTurns)
	if e.markTurns && e.turnStart >= 0 {
		for i := len(prompt) - 1; i >= 0 && cut < 0; i-- {
			if prompt[i] == e.turnStart {
				cut = i
			}
		}
		// past the turn's header line (<|im_start|>assistant\n, which the next turn renders
		// alike): fewer tokens after the mark, read as decode steps
		for i := cut + 1; cut >= 0 && e.turnEnd >= 0 && i < len(prompt) && i <= cut+4; i++ {
			if prompt[i] == e.turnEnd {
				cut = i + 1
				break
			}
		}
	}
	for len(e.cur.fed) < len(prompt) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		rest := len(prompt) - len(e.cur.fed)
		k := min(rest, e.sess.Q())
		if g := e.logitsQ(); g > 0 && rest > g {
			// the program that reads prompts gives no logits: the last tokens go through the one
			// that generates, which does
			k = min(rest-g, e.sess.Q())
		}
		if before := len(e.cur.fed); cut > before && cut < before+k {
			k = cut - before
		}
		if ok, err := e.room(k); err != nil {
			return nil, err
		} else if !ok {
			return nil, fmt.Errorf("%w: %d tokens, the cache holds %d", ErrContext, len(prompt), e.sess.Cache())
		}
		before := len(e.cur.fed)
		if _, err := e.step(prompt[before : before+k]); err != nil {
			return nil, err
		}
		if len(e.cur.fed) == cut && cut < len(prompt) {
			e.cur.marks = append(e.cur.marks, mark{e.sess.Mark(), e.cur.last})
			if len(e.cur.marks) > maxMarks {
				e.cur.marks = e.cur.marks[1:]
			}
		}
		if e.store != nil && len(e.cur.fed) < len(prompt) && len(e.cur.fed)/markEvery > before/markEvery {
			m := e.sess.Mark()
			kept = append(kept, storedMark{m.Pos(), m.Bufs(), e.cur.last})
		}
	}
	m = &mark{e.sess.Mark(), e.cur.last}
	e.cur.marks = append(e.cur.marks, *m)
	if len(e.cur.marks) > maxMarks {
		e.cur.marks = e.cur.marks[1:]
	}
	if e.store != nil && len(e.cur.fed) >= minStored && !slices.ContainsFunc(e.cur.fed, func(t int32) bool { return t < 0 }) {
		kept = append(kept, storedMark{m.m.Pos(), m.m.Bufs(), m.logits})
		if se, err := e.store.put(e.cur.fed, e.sess.(rowSource), kept, e.cur.base); err != nil {
			e.logf("prompt store: %v", err)
		} else {
			e.cur.base = se
		}
	}
	return e.cur.last, nil
}

// errNoLogits is restore's refusal of a mark that ends the prompt but holds no logits after it.
var errNoLogits = errors.New("chat: the stored mark has no logits")

// restore starts the sequence from the store's entry at the mark at pos, on the shortest tier
// that holds the prompt, and says whether it did: the entry is dropped when its files cannot be
// read.
func (e *Engine) restore(se *entry, pos int, prompt []int32, st *Stats) (bool, error) {
	rows, m, logits, err := e.store.load(se, pos)
	if err != nil {
		e.logf("prompt store: %v (dropped)", err)
		e.store.remove(se)
		return false, nil
	}
	if logits == nil && pos >= len(prompt) {
		// a mark inside a reading by a program that gives no logits (llm.Options.KVOnly)
		return false, errNoLogits
	}
	if n, _ := e.cur.reach(prompt); n < len(e.cur.fed)/2 {
		e.putAside()
	}
	if err := e.use(e.tierFor(len(prompt))); err != nil {
		return false, err
	}
	if err := e.sess.Resume(rows, m); err != nil {
		e.logf("prompt store: %s: %v (dropped)", se.dir, err)
		e.store.remove(se)
		e.sess.Reset()
		e.cur = seq{}
		return false, nil
	}
	e.cur = seq{fed: slices.Clone(se.Tokens[:pos]), last: logits, marks: []mark{{m, logits}}, base: se}
	st.Restored = pos
	return true, nil
}

// putAside saves a copy of the sequence in the cache, when it is long enough to be worth it.
func (e *Engine) putAside() {
	if len(e.cur.fed) < minSaved {
		return
	}
	all := e.sess.Save()
	if all == nil {
		return
	}
	q := seq{fed: slices.Clone(e.cur.fed), last: e.cur.last, marks: slices.Clone(e.cur.marks)}
	e.saved = append(e.saved, saved{q, all, e.ti})
	if len(e.saved) > maxSaved {
		e.saved = e.saved[1:]
	}
}

// Images reports whether the engine reads images.
func (e *Engine) Images() bool { return e.vis != nil || e.ir != nil && e.ir.HasVision() }

// Audio reports whether the engine reads audio.
func (e *Engine) Audio() bool { return e.aud != nil || e.ir != nil && e.ir.HasAudio() }

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
		var start, end int32
		var wrap bool
		if mm := e.irMultimodal(); mm != nil {
			// Gemma 4's processor: <|image> or <|audio>, the soft tokens, <image|> or <audio|>
			start, end, wrap = mm.BOI, mm.EOI, true
			if media[i].Audio {
				start, end = mm.BOA, mm.EOA
			}
			wrap = start >= 0 // LFM3-Audio's clip is its placeholders alone
		} else {
			wrap = !media[i].Audio || e.aud.Omni()
			var ok1, ok2 bool
			start, ok1 = e.Tok.ID(vision.Start)
			end, ok2 = e.Tok.ID(vision.End)
			if wrap && (!ok1 || !ok2) {
				return nil, fmt.Errorf("%w: the tokenizer has no %s and %s", ErrImage, vision.Start, vision.End)
			}
		}
		if wrap {
			ids = append(ids, start)
		}
		for r := range enc.count() {
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
	if e.closed {
		return nil, ErrClosed
	}
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
	if e.ir != nil {
		return e.encodeIRON(m, key)
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

// encodeIRON decodes an image or a clip for an IRON model's towers and counts its soft tokens.
func (e *Engine) encodeIRON(m Media, key [32]byte) (*encoded, error) {
	im := encoded{key: key, first: e.nextID, audio: m.Audio}
	var err error
	if m.Audio {
		if im.clip, err = audio.Decode(m.Data); err != nil {
			return nil, fmt.Errorf("%w: %v", ErrImage, err)
		}
		if _, im.n, err = e.ir.AudioTokens(len(im.clip)); err != nil {
			return nil, fmt.Errorf("%w: %v", ErrImage, err)
		}
	} else {
		if im.img, err = vision.DecodeDataURL(m.Data); err != nil {
			return nil, fmt.Errorf("%w: %v", ErrImage, err)
		}
		b := im.img.Bounds()
		if _, _, im.n, err = e.ir.ImageSize(b.Dx(), b.Dy()); err != nil {
			return nil, fmt.Errorf("%w: %v", ErrImage, err)
		}
	}
	e.nextID -= int32(im.n)
	if len(e.images) >= maxImages {
		e.images = e.images[1:]
	}
	e.images = append(e.images, im)
	return &e.images[len(e.images)-1], nil
}

// media finds the image or clip of a negative id and the id's place in its run.
func (e *Engine) media(t int32) (*encoded, int, bool) {
	for i := range e.images {
		im := &e.images[i]
		if r := int(im.first - t); r >= 0 && r < im.count() {
			return im, r, true
		}
	}
	return nil, 0, false
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
	var logits []float32
	var err error
	if ts, ok := e.sess.(tokenSession); ok {
		logits, err = ts.StepTokens(toks) // images' and clips' ids too: the session runs the towers
	} else {
		var embs [][]float32
		if embs, err = e.rows(toks); err != nil {
			return nil, err
		}
		logits, err = e.sess.Step(embs)
	}
	if err != nil {
		// the cache may be half written
		e.sess.Reset()
		e.cur = seq{}
		return nil, err
	}
	pos0 := len(e.cur.fed)
	e.cur.fed = append(e.cur.fed, toks...)
	e.cur.last = logits
	e.recordHidden(pos0)
	return logits, nil
}

// recordHidden gives the multi-token prediction layer the model's final hidden states of the
// tokens the last step kept, from pos0 on.
func (e *Engine) recordHidden(pos0 int) {
	if e.mtp == nil {
		return
	}
	var rows [][]float32
	if ds, ok := e.sess.(*decode.Session); ok {
		rows = ds.HiddenRows()
	}
	e.mtp.record(e.cur.fed, pos0, rows)
}

// rows gathers the tokens' rows: the embeddings of the tokens and of the images (negative ids).
func (e *Engine) rows(toks []int32) ([][]float32, error) {
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
			if v, err = llm.TokenRow(e.w, e.model, t); err != nil {
				return nil, err
			}
			// at most 32768 tokens of LFM3 (2048 values), fewer of a model with wider rows
			if len(e.emb)*len(v) >= 1<<26 {
				clear(e.emb)
			}
			e.emb[t] = v
		}
		embs[i] = v
	}
	return embs, nil
}

// speculate runs the tokens like step, keeping only as many as accept says
// (decode.Session.Speculate), and records those as fed. The kept tokens are a prefix.
func (e *Engine) speculate(toks []int32, accept func(row func(i int) []float32) int) (int, error) {
	embs, err := e.rows(toks)
	if err != nil {
		return 0, err
	}
	logits, kept, err := e.sess.Speculate(embs, accept)
	if err != nil {
		// the cache may be half written
		e.sess.Reset()
		e.cur = seq{}
		return 0, err
	}
	pos0 := len(e.cur.fed)
	e.cur.fed = append(e.cur.fed, toks[:kept]...)
	e.cur.last = logits
	e.recordHidden(pos0)
	return kept, nil
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
