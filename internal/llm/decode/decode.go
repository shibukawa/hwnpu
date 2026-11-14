// Package decode runs a causal model built with a KV cache (llm.Cached, Options.Cache) one step
// at a time. The cache lives in the program's own input buffers: after each step the new keys,
// values and convolution windows the program outputs are written into the inputs they belong to,
// so the next step reads them where they are. The buffers are plain fp16 bytes, so the same code
// drives Core ML buffers (internal/llm/chain) and the CPU interpreter in tests.
//
// A session may drive several programs of the same model that take different numbers of tokens
// per step and share the cache buffers (Add): a long step to read a prompt, a short one to
// generate. It can go back to an earlier point of the sequence (Mark, Rewind), so that a prompt
// sharing a beginning with the last one only reads the rest.
package decode

import (
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"
	"time"
	"unsafe"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/paged"
)

// Program is the loaded program: its fp16 input and output buffers by name and their shapes.
type Program struct {
	In, Out           map[string][]byte
	InShape, OutShape map[string][]int
	// Run computes the outputs from the inputs. Before it, the inputs have been written; after
	// it, the outputs are read. Sync hooks for device buffers belong in Run.
	Run func() error

	// A program split at its attention (llm.Paged) has RunPart instead of Run: RunPart(i) runs
	// part i, its inputs written and its outputs read as Run's. Between part i and i+1 the session
	// attends for layer Attend[i]: it writes the step's keys and values (kv_k_N_new, kv_v_N_new)
	// into its own cache, attends attn_q_N over it with Attn and writes the result to the input
	// attn_o_N. A layer that reuses an earlier layer's keys and values (llm.Paged.PagedKV) attends
	// over that layer's cache. Context is the longest sequence that cache holds.
	RunPart func(i int) error
	Attend  []int
	Attn    *Attention
	// Smaller are attention programs over shorter chunks than Attn's, shortest first, each
	// chunk dividing the next's and Attn's. The session holds its cache in the shortest chunk
	// that covers the sequence and moves the rows into the next as the sequence passes it: a call
	// costs a fixed part and a part in proportion to its chunk (on an M3's Neural Engine, with
	// LFM3-1.7B, 0.50 ms a layer over 1024 positions, 0.63 over 2048, 0.93 over 4096, 1.49 over
	// 8192), so one chunk that just covers the sequence costs less than one longer and than
	// several shorter. The two-level attention runs on Attn alone.
	Smaller []*Attention
	Context int
	// NoLogits marks a program that only reads tokens into the caches (llm.Options.KVOnly): its
	// y is not logits, and a step on it returns none. Its last part computes the keys and values
	// of layers it does not attend for, which the session writes into its cache.
	NoLogits bool
	// Meta is the metadata of the program's parts (llm.ReadMeta), from which the session learns
	// whether the host applies the logits' soft cap (llm.HostSoftcap).
	Meta map[string]string
}

// Buf is a buffer the attention program can be bound to: a chunk of keys or values of the
// session's cache, or a mask.
type Buf interface {
	Bytes() []byte
	Close() error
}

// Attention is the program without weights that attends over one chunk of a long cache the
// session holds (internal/llm/paged), for the tokens per step of the Program it belongs to.
type Attention struct {
	// Chunk is the positions of a chunk.
	Chunk int
	// Q is the program's input q, and O, M and L its outputs (fp16, internal/llm/paged).
	Q, O, M, L []byte
	// NewBuf allocates a buffer of n bytes of zeros that Run can take, in the memory of the
	// device: the chunks of every program of the session come from the first program's.
	NewBuf func(n int) (Buf, error)
	// Run attends Q over the keys k and values v (1, kv_heads, Chunk, head_dim) with the mask
	// (1, 1, tokens, Chunk), writing O, M and L.
	Run func(k, v, mask Buf) error
	// Mass, when not nil, weighs chunks of summaries for the two-level attention (Sparse).
	Mass *Mass
}

// Session is the state of one sequence.
type Session struct {
	progs   []step // by tokens per step, shortest first
	m       llm.Cached
	kv      []kvCache
	t       int // cache length
	hidden  int
	vocab   int
	pos     int
	lookups []llm.TokenInput // the model's inputs looked up per token besides x
	last    []float32        // h after the last step, if the program outputs it
	// lastRows is h of each kept token of the last step, when the program outputs it per row
	lastRows [][]float32

	// of a program split at its attention: the chunks of each long cache the session holds,
	// by the cache's output (kv_NAME_new), allocated as the sequence reaches them
	chunks map[string][]Buf
	chunk  int     // the positions of those chunks now (Program.Smaller)
	sparse *sparse // the two-level attention (SetSparse), nil when off

	softcap float64 // the soft cap the host applies to the program's logits, 0 for none
	raw     bool    // return the logits before it (SetRawLogits)
}

// kvCache is one cache of keys or values: an input kv_NAME, or the chunks kv_NAME_cJ that hold
// its positions in order (llm.CacheChunks), or chunks the session holds itself (paged).
type kvCache struct {
	out      string   // the step's rows, kv_NAME_new
	parts    []string // the inputs
	lens     []int
	total    int
	long     bool // holds every position of the sequence (not a ring)
	chunk    int  // of a cache the session holds: the positions of a chunk; 0 otherwise
	heads, d int
}

// kvBase gives the cache an input belongs to, kv_NAME for kv_NAME and its chunks kv_NAME_cJ, and
// the chunk's number (-1 for a cache in one piece).
func kvBase(name string) (string, int) {
	if i := strings.LastIndex(name, "_c"); i > 0 {
		if j, err := strconv.Atoi(name[i+2:]); err == nil {
			return name[:i], j
		}
	}
	return name, -1
}

// newOutput is the output that gives the step's rows of a cache input.
func newOutput(name string) string {
	if strings.HasPrefix(name, "kv_") {
		base, _ := kvBase(name)
		return base + "_new"
	}
	return name + "_new"
}

type step struct {
	p Program
	q int
	// of a program split at its attention: the masks of the step's chunks and the merge
	merge *paged.Merge
	zero  Buf          // a mask that hides nothing, for the chunks before the step
	chunk int          // the positions of the masks' chunks (setChunk)
	masks map[int]Buf  // by chunk, those the step's tokens fall in
	spare []Buf        // allocated masks not in use this step
	o     [3][]float32 // the attention's outputs o, m, l, decoded
	res   []float32    // the merged attention
	sp    *sparseStep  // of the two-level attention, made on its first call
}

// New checks the program's inputs and outputs and starts an empty sequence.
func New(p Program, m llm.Cached) (*Session, error) {
	x, ok := p.InShape["x"]
	if !ok || len(x) != 3 {
		return nil, fmt.Errorf("decode: no input x (1, q, hidden)")
	}
	y, ok := p.OutShape["y"]
	if !ok || len(y) < 3 || len(y) > 4 || y[1] != rows(p) || p.NoLogits {
		return nil, fmt.Errorf("decode: no output y (1, q, vocab)")
	}
	vocab := y[2]
	if len(y) == 4 {
		vocab *= y[3] // folded into rows (logitsShape)
	}
	s := &Session{progs: []step{{p: p, q: x[1]}}, m: m, hidden: x[2], vocab: vocab}
	s.softcap = llm.HostSoftcap(p.Meta)
	if l, ok := m.(llm.TokenInputs); ok {
		s.lookups = l.TokenInputs()
	}
	if err := s.checkLookups(p); err != nil {
		return nil, err
	}
	var err error
	if s.kv, s.t, err = caches(p); err != nil {
		return nil, err
	}
	if p.RunPart != nil {
		if _, ok := m.(llm.Paged); !ok {
			return nil, fmt.Errorf("decode: the program is split at its attention, the model cannot be")
		}
		s.chunks = map[string][]Buf{}
		s.chunk = s.chunkFor(0)
		for i := range s.kv {
			if s.kv[i].chunk > 0 {
				s.kv[i].chunk = s.chunk
			}
		}
		if err := s.progs[0].checkPaged(s); err != nil {
			return nil, err
		}
	}
	s.Reset()
	return s, nil
}

// checkPaged checks a program split at its attention and makes its step's buffers.
func (st *step) checkPaged(s *Session) error {
	p := st.p
	if p.Attn == nil || p.Attn.NewBuf == nil || p.Attn.Run == nil || p.Attn.Chunk <= 0 {
		return fmt.Errorf("decode: a program split at its attention has no attention program")
	}
	if p.Attn.Chunk < st.q {
		return fmt.Errorf("decode: chunks of %d for %d tokens per step", p.Attn.Chunk, st.q)
	}
	for i, a := range p.Smaller {
		next := p.Attn
		if i+1 < len(p.Smaller) {
			next = p.Smaller[i+1]
		}
		switch {
		case a == nil || a.Run == nil:
			return fmt.Errorf("decode: attention program %d of the shorter chunks is missing", i)
		case a.Chunk < st.q || a.Chunk >= next.Chunk || next.Chunk%a.Chunk != 0:
			return fmt.Errorf("decode: a shorter chunk of %d before %d, for %d tokens per step: each must divide the next and hold a step", a.Chunk, next.Chunk, st.q)
		case len(a.Q) != len(p.Attn.Q) || len(a.O) != len(p.Attn.O) || len(a.M) != len(p.Attn.M) || len(a.L) != len(p.Attn.L):
			return fmt.Errorf("decode: the attention program over chunks of %d does not take the queries of the one over %d", a.Chunk, p.Attn.Chunk)
		}
	}
	var heads, d int
	one := false
	for _, layer := range p.Attend {
		src := s.kvLayer(layer)
		if src != layer && !slices.Contains(p.Attend, src) || src > layer {
			return fmt.Errorf("decode: layer %d attends over the cache of layer %d, which the program does not attend for before it", layer, src)
		}
		for _, name := range []string{fmt.Sprintf("kv_k_%d_new", src), fmt.Sprintf("kv_v_%d_new", src)} {
			c := s.cache(name)
			if c == nil || c.chunk == 0 {
				return fmt.Errorf("decode: the program attends for layer %d but has no output %s", layer, name)
			}
		}
		qs, ok := p.OutShape[fmt.Sprintf("attn_q_%d", layer)]
		if !ok || len(qs) != 4 || qs[2] != st.q && (qs[2] != 1 || rows(p) != 1) {
			// 1 row: a layer after which only the last token's row is needed (llm.Options.LastRow)
			return fmt.Errorf("decode: no output attn_q_%d (1, heads, %d, head_dim)", layer, st.q)
		}
		if ts, ok := p.InShape[fmt.Sprintf("attn_ot_%d", layer)]; ok {
			if want := []int{1, qs[2], qs[1], qs[3]}; !slices.Equal(ts, want) {
				return fmt.Errorf("decode: input attn_ot_%d has shape %v, want %v", layer, ts, want)
			}
		} else if os := p.InShape[fmt.Sprintf("attn_o_%d", layer)]; !slices.Equal(os, qs) {
			return fmt.Errorf("decode: input attn_o_%d has shape %v, want %v", layer, os, qs)
		}
		heads, d = qs[1], qs[3]
		if qs[2] < st.q {
			one = true
		} else if one {
			return fmt.Errorf("decode: layer %d attends for every row after a layer of the last row alone", layer)
		}
	}
	if len(p.Attend) == 0 && !p.NoLogits {
		return fmt.Errorf("decode: a program split at its attention attends for no layer")
	}
	if len(p.Attend) > 0 && (2*heads*st.q*d != len(p.Attn.Q) || len(p.Attn.O) != len(p.Attn.Q) || 2*heads*st.q != len(p.Attn.M) || len(p.Attn.L) != len(p.Attn.M)) {
		return fmt.Errorf("decode: the attention program does not take %d heads of %d for %d tokens", heads, d, st.q)
	}
	st.merge = paged.NewMerge(heads*st.q, d)
	st.o = [3][]float32{make([]float32, heads*st.q*d), make([]float32, heads*st.q), make([]float32, heads*st.q)}
	st.res = make([]float32, heads*st.q*d)
	return st.setChunk(s.chunk)
}

// setChunk makes the step's masks for chunks of c positions.
func (st *step) setChunk(c int) error {
	for _, b := range append(append([]Buf{st.zero}, st.spare...), slices.Collect(maps.Values(st.masks))...) {
		if b != nil {
			b.Close()
		}
	}
	st.spare, st.masks, st.zero, st.chunk = nil, map[int]Buf{}, nil, c
	var err error
	st.zero, err = st.p.Attn.NewBuf(2 * st.q * c)
	return err
}

// attention is the program's attention program over chunks of c positions.
func (p Program) attention(c int) *Attention {
	for _, a := range p.Smaller {
		if a.Chunk == c {
			return a
		}
	}
	return p.Attn
}

// chunkFor is the shortest chunk of the session's attention programs that holds n positions, or
// the longest.
func (s *Session) chunkFor(n int) int {
	p := s.progs[0].p
	for _, a := range p.Smaller {
		if a.Chunk >= n {
			return a.Chunk
		}
	}
	return p.Attn.Chunk
}

// fit moves the cache the session holds into the shortest chunk of its attention programs
// (Program.Smaller) that holds the sequence's n positions, before a step or rows written up to
// n: a longer one as the sequence grows, a shorter one after it went back (Reset, Rewind,
// Truncate). The rows keep their positions; when the chunks get shorter only those before n
// move, which are all a step after it reads.
func (s *Session) fit(n int) error {
	if s.chunks == nil {
		return nil
	}
	want := s.chunkFor(n)
	if want == s.chunk {
		return nil
	}
	for i := range s.kv {
		c := &s.kv[i]
		if c.chunk == 0 {
			continue
		}
		old := s.chunks[c.out]
		end := len(old) * c.chunk // every row held, into longer chunks
		if want < c.chunk {
			end = min(end, n) // within the first chunk
		}
		var moved []Buf
		row := 2 * c.d
		for pos := 0; pos < end; {
			j, k := pos/c.chunk, pos/want
			run := min(c.chunk-pos%c.chunk, want-pos%want, end-pos)
			if j < len(old) && old[j] != nil {
				for len(moved) <= k {
					moved = append(moved, nil)
				}
				if moved[k] == nil {
					b, err := s.progs[0].p.Attn.NewBuf(2 * c.heads * want * c.d)
					if err != nil {
						return fmt.Errorf("decode: a chunk of the cache: %w", err)
					}
					moved[k] = b
				}
				src, dst := old[j].Bytes(), moved[k].Bytes()
				so, do := pos%c.chunk, pos%want
				for h := range c.heads {
					copy(dst[(h*want+do)*row:(h*want+do+run)*row], src[(h*c.chunk+so)*row:(h*c.chunk+so+run)*row])
				}
			}
			pos += run
		}
		for _, b := range old {
			if b != nil {
				b.Close()
			}
		}
		s.chunks[c.out] = moved
		c.chunk = want
	}
	return s.setChunk(want)
}

// setChunk records the chunk of the cache the session holds and makes its steps' masks for it.
func (s *Session) setChunk(c int) error {
	s.chunk = c
	for i := range s.kv {
		if s.kv[i].chunk > 0 {
			s.kv[i].chunk = c
		}
	}
	for i := range s.progs {
		if s.progs[i].p.RunPart != nil {
			if err := s.progs[i].setChunk(c); err != nil {
				return err
			}
		}
	}
	return nil
}

// kvLayer is the layer whose cache the attention of layer attends over (llm.Paged.PagedKV).
func (s *Session) kvLayer(layer int) int {
	return s.m.(llm.Paged).PagedKV(layer)
}

// cache returns the cache whose step rows are the output out, or nil.
func (s *Session) cache(out string) *kvCache {
	for i := range s.kv {
		if s.kv[i].out == out {
			return &s.kv[i]
		}
	}
	return nil
}

// caches gathers the program's KV cache inputs into caches, and gives the longest's length. For
// a program split at its attention, the step outputs kv_NAME_new without an input are caches the
// session holds (in chunks of Attn.Chunk), and the length is Context.
func caches(p Program) ([]kvCache, int, error) {
	var kv []kvCache
	t := 0
	chunks := map[string]map[int]string{}
	for name, sh := range p.InShape {
		if strings.HasPrefix(name, "kv_") {
			if len(sh) != 4 {
				return nil, 0, fmt.Errorf("decode: %s has shape %v, want (1, heads, cache, head_dim)", name, sh)
			}
			base, j := kvBase(name)
			if chunks[base] == nil {
				chunks[base] = map[int]string{}
			}
			chunks[base][j] = name
		}
		if cacheInput(name) {
			if _, ok := p.OutShape[newOutput(name)]; !ok {
				return nil, 0, fmt.Errorf("decode: input %s has no output %s", name, newOutput(name))
			}
		}
	}
	for base, parts := range chunks {
		c := kvCache{out: base + "_new"}
		names := []string{parts[-1]}
		if _, whole := parts[-1]; !whole {
			names = names[:0]
			for j := range len(parts) {
				if parts[j] == "" {
					return nil, 0, fmt.Errorf("decode: the chunks of %s are not %s_c0 to %s_c%d", base, base, base, len(parts)-1)
				}
				names = append(names, parts[j])
			}
		} else if len(parts) > 1 {
			return nil, 0, fmt.Errorf("decode: %s is both whole and in chunks", base)
		}
		for _, name := range names {
			c.parts = append(c.parts, name)
			c.lens = append(c.lens, p.InShape[name][2])
			c.total += p.InShape[name][2]
		}
		sh := p.InShape[names[0]]
		c.heads, c.d = sh[1], sh[3]
		kv = append(kv, c)
		t = max(t, c.total) // the longest cache; shorter ones are rings
	}
	if p.RunPart != nil {
		if p.Attn == nil || p.Context <= 0 {
			return nil, 0, fmt.Errorf("decode: a program split at its attention needs its attention program and its context")
		}
		t = p.Context
		for name, sh := range p.OutShape {
			base, ok := strings.CutSuffix(name, "_new")
			if !ok || !strings.HasPrefix(name, "kv_") || chunks[base] != nil {
				continue
			}
			if len(sh) != 4 {
				return nil, 0, fmt.Errorf("decode: %s has shape %v, want (1, heads, q, head_dim)", name, sh)
			}
			kv = append(kv, kvCache{out: name, total: t, long: true, chunk: p.Attn.Chunk, heads: sh[1], d: sh[3]})
		}
	}
	if t == 0 {
		return nil, 0, fmt.Errorf("decode: the program has no KV cache inputs (kv_*)")
	}
	for i := range kv {
		if kv[i].chunk == 0 && kv[i].total == t {
			kv[i].long = true
		}
	}
	slices.SortFunc(kv, func(a, b kvCache) int { return strings.Compare(a.out, b.out) })
	return kv, t, nil
}

// SetInputF16 copies b, fp16 values, into the input name of every program of the session that
// has it: an input the session neither holds (llm.CacheInput) nor writes each step (x, the token
// lookups, last and StepInputs'), which then stays as set across steps, marks and rewinds, until
// set again. Whisper's decoder takes its cross attention's keys and values of a clip this way.
func (s *Session) SetInputF16(name string, b []byte) error {
	if name == "x" || name == "last" || cacheInput(name) {
		return fmt.Errorf("decode: the session writes the input %s itself", name)
	}
	for _, lk := range s.lookups {
		if lk.Input == name {
			return fmt.Errorf("decode: the session writes the input %s itself", name)
		}
	}
	found := false
	for _, st := range s.progs {
		in, ok := st.p.In[name]
		if !ok {
			continue
		}
		if len(in) != len(b) {
			return fmt.Errorf("decode: input %s takes %d bytes, not %d", name, len(in), len(b))
		}
		copy(in, b)
		found = true
	}
	if !found {
		return fmt.Errorf("decode: no program has an input %s", name)
	}
	return nil
}

// Add adds a program of the same model that takes another number of tokens per step. Its cache
// inputs (kv_*, win_*, st_*: llm.CacheInput) must be the same memory as the first program's (chain.Share), so that
// steps of either program continue the same sequence.
func (s *Session) Add(p Program) error {
	x, ok := p.InShape["x"]
	if !ok || len(x) != 3 || x[2] != s.hidden {
		return fmt.Errorf("decode: no input x (1, q, %d)", s.hidden)
	}
	if !p.NoLogits && !logitsShape(p.OutShape["y"], rows(p), s.vocab) {
		return fmt.Errorf("decode: no output y (1, %d, %d)", rows(p), s.vocab)
	}
	if c := llm.HostSoftcap(p.Meta); !p.NoLogits && c != s.softcap {
		return fmt.Errorf("decode: the programs leave different soft caps of the logits to the host (%g and %g)", s.softcap, c)
	}
	first := s.progs[0].p
	for name, b := range first.In {
		if !cacheInput(name) {
			continue
		}
		o := p.In[name]
		if len(o) != len(b) || unsafe.SliceData(o) != unsafe.SliceData(b) {
			return fmt.Errorf("decode: the programs do not share the cache input %s", name)
		}
		if _, ok := p.OutShape[newOutput(name)]; !ok {
			return fmt.Errorf("decode: input %s has no output %s", name, newOutput(name))
		}
	}
	for _, o := range s.progs {
		if o.q == x[1] {
			return fmt.Errorf("decode: a program of %d tokens per step is there already", x[1])
		}
	}
	if err := s.checkLookups(p); err != nil {
		return err
	}
	st := step{p: p, q: x[1]}
	if (p.RunPart != nil) != (first.RunPart != nil) {
		return fmt.Errorf("decode: one program is split at its attention and the other is not")
	}
	if p.RunPart != nil {
		if p.Attn == nil || p.Attn.Chunk != first.Attn.Chunk || p.Context != first.Context || len(p.Smaller) != len(first.Smaller) {
			return fmt.Errorf("decode: the programs split at their attention hold different caches")
		}
		for i, a := range p.Smaller {
			if a == nil || a.Chunk != first.Smaller[i].Chunk {
				return fmt.Errorf("decode: the programs split at their attention attend over different chunks")
			}
		}
		for _, c := range s.kv {
			if c.chunk > 0 && !slices.Equal(p.OutShape[c.out], []int{1, c.heads, x[1], c.d}) {
				return fmt.Errorf("decode: the program has no output %s (1, %d, %d, %d)", c.out, c.heads, x[1], c.d)
			}
		}
		if err := st.checkPaged(s); err != nil {
			return err
		}
		if s.sparse != nil && p.Attn.Mass == nil {
			return fmt.Errorf("decode: the session has the two-level attention, the program of %d tokens per step no mass program", st.q)
		}
	}
	s.progs = append(s.progs, st)
	slices.SortFunc(s.progs, func(a, b step) int { return a.q - b.q })
	return nil
}

// checkLookups checks that p takes the model's lookup inputs.
func (s *Session) checkLookups(p Program) error {
	q := p.InShape["x"][1]
	for _, lk := range s.lookups {
		if sh := p.InShape[lk.Input]; !slices.Equal(sh, []int{1, q, lk.Width}) {
			return fmt.Errorf("decode: input %s has shape %v, want (1, %d, %d)", lk.Input, sh, q, lk.Width)
		}
	}
	return nil
}

func cacheInput(name string) bool { return llm.CacheInput(name) }

// Q is the most tokens one step takes.
func (s *Session) Q() int { return s.progs[len(s.progs)-1].q }

// Qs is the tokens per step of each program, fewest first.
func (s *Session) Qs() []int {
	var qs []int
	for _, o := range s.progs {
		qs = append(qs, o.q)
	}
	return qs
}

// Cache is the number of tokens the longest cache holds: the sequence length the session can
// reach. Shorter caches (a sliding layer's ring) hold the last tokens only.
func (s *Session) Cache() int { return s.t }

// Pos is the number of tokens in the sequence so far.
func (s *Session) Pos() int { return s.pos }

// Migrate moves the session to np, a program of the same model whose caches are at least as long
// (the next tier of a tiered model): the cached rows are copied over, and np becomes the session's
// only program; add programs of other step lengths again (Add) with buffers shared with np. A ring
// (a cache shorter than the longest) keeps its length and is copied as is; the other caches move
// row by row to the same positions, whether the old or the new one is in chunks.
func (s *Session) Migrate(np Program) error {
	if s.chunks == nil && np.RunPart != nil {
		return s.migratePaged(np)
	}
	if s.chunks != nil || np.RunPart != nil {
		return fmt.Errorf("decode: a program split at its attention has no tiers to move to")
	}
	x, ok := np.InShape["x"]
	if !ok || len(x) != 3 || x[2] != s.hidden {
		return fmt.Errorf("decode: no input x (1, q, %d)", s.hidden)
	}
	if !logitsShape(np.OutShape["y"], rows(np), s.vocab) {
		return fmt.Errorf("decode: no output y (1, %d, %d)", rows(np), s.vocab)
	}
	nkv, nt, err := caches(np)
	if err != nil {
		return err
	}
	if nt < s.t {
		return fmt.Errorf("decode: the new program caches %d tokens, fewer than the session's %d", nt, s.t)
	}
	if err := s.checkLookups(np); err != nil {
		return err
	}
	first := s.progs[0].p
	for name, in := range first.In {
		if llm.HeldInput(name) {
			out := np.In[name]
			if len(out) != len(in) {
				return fmt.Errorf("decode: %s has shape %v in the new program, %v in the session", name, np.InShape[name], first.InShape[name])
			}
			copy(out, in)
		}
	}
	for _, c := range s.kv {
		var nc *kvCache
		for i := range nkv {
			if nkv[i].out == c.out {
				nc = &nkv[i]
			}
		}
		if nc == nil {
			return fmt.Errorf("decode: the new program has no cache %s", strings.TrimSuffix(c.out, "_new"))
		}
		sh, nsh := first.InShape[c.parts[0]], np.InShape[nc.parts[0]]
		if sh[1] != nsh[1] || sh[3] != nsh[3] {
			return fmt.Errorf("decode: %s has shape %v in the new program, %v in the session", nc.parts[0], nsh, sh)
		}
		for _, part := range nc.parts {
			clear(np.In[part])
		}
		if c.total < s.t { // a ring keeps its length
			if nc.total != c.total {
				return fmt.Errorf("decode: the ring %s holds %d in the new program, %d in the session", c.parts[0], nc.total, c.total)
			}
			copy(np.In[nc.parts[0]], first.In[c.parts[0]])
			continue
		}
		// (1, heads, L, d) -> (1, heads, nL, d): each head's rows at the same positions, in
		// whichever chunks hold them
		heads, d := sh[1], sh[3]
		for pos := 0; pos < c.total; pos++ {
			j, slot := c.locate(pos)
			nj, nslot := nc.locate(pos)
			from, to := first.In[c.parts[j]], np.In[nc.parts[nj]]
			for h := 0; h < heads; h++ {
				o, no := (h*c.lens[j]+slot)*2*d, (h*nc.lens[nj]+nslot)*2*d
				copy(to[no:no+2*d], from[o:o+2*d])
			}
		}
	}
	s.progs, s.t, s.kv = []step{{p: np, q: x[1]}}, nt, nkv
	return nil
}

// migratePaged moves a session of a whole program, its caches inside, to np, a program of the
// same model split at its attention (the next tier: a whole program with a short cache runs a
// step in fewer calls, and the one split at its attention costs in proportion to the
// conversation past it). The rows of a cache that holds every position go into the chunks the
// session holds, or into a ring of np's at position mod its length; a ring of the same length
// is copied as is, and so are the windows and states. np becomes the session's only program.
func (s *Session) migratePaged(np Program) error {
	x, ok := np.InShape["x"]
	if !ok || len(x) != 3 || x[2] != s.hidden {
		return fmt.Errorf("decode: no input x (1, q, %d)", s.hidden)
	}
	if np.NoLogits || !logitsShape(np.OutShape["y"], rows(np), s.vocab) {
		return fmt.Errorf("decode: no output y (1, %d, %d)", rows(np), s.vocab)
	}
	if _, ok := s.m.(llm.Paged); !ok {
		return fmt.Errorf("decode: the program is split at its attention, the model cannot be")
	}
	if c := llm.HostSoftcap(np.Meta); c != s.softcap {
		return fmt.Errorf("decode: the programs leave different soft caps of the logits to the host (%g and %g)", s.softcap, c)
	}
	nkv, nt, err := caches(np)
	if err != nil {
		return err
	}
	if nt < s.t {
		return fmt.Errorf("decode: the new program caches %d tokens, fewer than the session's %d", nt, s.t)
	}
	if err := s.checkLookups(np); err != nil {
		return err
	}
	first := s.progs[0].p
	for name, in := range first.In {
		if llm.HeldInput(name) {
			out := np.In[name]
			if len(out) != len(in) {
				return fmt.Errorf("decode: %s has shape %v in the new program, %v in the session", name, np.InShape[name], first.InShape[name])
			}
			copy(out, in)
		}
	}
	old := s.kv
	pos := s.pos
	s.progs, s.kv, s.t = []step{{p: np, q: x[1]}}, nkv, nt
	s.chunks = map[string][]Buf{}
	s.chunk = s.chunkFor(pos)
	for i := range s.kv {
		if s.kv[i].chunk > 0 {
			s.kv[i].chunk = s.chunk
		}
	}
	for i := range s.kv {
		nc := &s.kv[i]
		var c *kvCache
		for j := range old {
			if old[j].out == nc.out {
				c = &old[j]
			}
		}
		if c == nil {
			return fmt.Errorf("decode: the session has no cache %s for the new program", strings.TrimSuffix(nc.out, "_new"))
		}
		if c.heads != nc.heads || c.d != nc.d {
			return fmt.Errorf("decode: %s has %d heads of %d in the new program, %d of %d in the session", nc.out, nc.heads, nc.d, c.heads, c.d)
		}
		if !c.long && (nc.chunk > 0 || nc.total != c.total) {
			return fmt.Errorf("decode: the ring %s holds %d in the session, the new program's %d", c.parts[0], c.total, nc.total)
		}
		if !c.long { // a ring of the same length
			for _, part := range nc.parts {
				clear(np.In[part])
			}
			copy(np.In[nc.parts[0]], first.In[c.parts[0]])
			continue
		}
		for _, part := range nc.parts {
			clear(np.In[part])
		}
		// the positions before the session's, each head's row where the new cache holds it
		for p := 0; p < pos; p++ {
			j, slot := c.locate(p)
			from, L := first.In[c.parts[j]], c.lens[j]
			np2 := p
			if !nc.long {
				np2 %= nc.total
			}
			nj, nslot := nc.locate(np2)
			var to []byte
			if nc.chunk > 0 {
				if to, err = s.part(nc, nj, true); err != nil {
					return err
				}
			} else {
				to = np.In[nc.parts[nj]]
			}
			nL := nc.span(nj)
			for h := 0; h < c.heads; h++ {
				o, no := (h*L+slot)*2*c.d, (h*nL+nslot)*2*c.d
				copy(to[no:no+2*c.d], from[o:o+2*c.d])
			}
		}
	}
	return s.progs[0].checkPaged(s)
}

// locate gives the chunk that holds a position of the cache and its row there.
func (c *kvCache) locate(pos int) (chunk, row int) {
	if c.chunk > 0 {
		return pos / c.chunk, pos % c.chunk
	}
	for pos >= c.lens[chunk] {
		pos -= c.lens[chunk]
		chunk++
	}
	return chunk, pos
}

// span is the positions chunk j of the cache holds.
func (c *kvCache) span(j int) int {
	if c.chunk > 0 {
		return c.chunk
	}
	return c.lens[j]
}

// part returns the bytes of chunk j of the cache, (1, heads, span, d) in fp16. A chunk the session
// holds is allocated when alloc is set and it is not there yet; otherwise it is nil.
func (s *Session) part(c *kvCache, j int, alloc bool) ([]byte, error) {
	if c.chunk == 0 {
		return s.progs[0].p.In[c.parts[j]], nil
	}
	have := s.chunks[c.out]
	if j < len(have) && have[j] != nil {
		return have[j].Bytes(), nil
	}
	if !alloc {
		return nil, nil
	}
	if j*c.chunk >= c.total {
		return nil, fmt.Errorf("decode: chunk %d is past the cache of %d", j, c.total)
	}
	for len(have) <= j {
		have = append(have, nil)
	}
	b, err := s.progs[0].p.Attn.NewBuf(2 * c.heads * c.chunk * c.d)
	if err != nil {
		return nil, fmt.Errorf("decode: a chunk of the cache: %w", err)
	}
	have[j] = b
	s.chunks[c.out] = have
	return b.Bytes(), nil
}

// Paged is the chunk length of a session split at its attention, 0 for another.
func (s *Session) Paged() int {
	if s.chunks == nil {
		return 0
	}
	return s.progs[0].p.Attn.Chunk
}

// HeldBytes is the memory of the chunks a session split at its attention holds, with the
// two-level attention's summaries and keys read exactly (they stay
// allocated through Reset, to be used again; Close frees them).
func (s *Session) HeldBytes() int64 {
	var n int64
	for _, cs := range s.chunks {
		for _, b := range cs {
			if b != nil {
				n += int64(len(b.Bytes()))
			}
		}
	}
	if s.sparse != nil {
		for _, b := range s.sparse.bufs() {
			n += int64(len(b.Bytes()))
		}
	}
	return n
}

// Close frees the chunks of the cache the session holds and the masks of its steps. The session
// cannot be used after.
func (s *Session) Close() error {
	for _, cs := range s.chunks {
		for _, b := range cs {
			if b != nil {
				b.Close()
			}
		}
	}
	s.chunks = nil
	if s.sparse != nil {
		for _, b := range s.sparse.bufs() {
			b.Close()
		}
		s.sparse = nil
	}
	for i := range s.progs {
		st := &s.progs[i]
		bufs := append(append([]Buf{st.zero}, st.spare...), slices.Collect(maps.Values(st.masks))...)
		if st.sp != nil {
			bufs = append(bufs, st.sp.bufs()...)
			st.sp = nil
		}
		for _, b := range bufs {
			if b != nil {
				b.Close()
			}
		}
		st.zero, st.spare, st.masks = nil, nil, nil
	}
	return nil
}

// Limit lowers the longest sequence of a session split at its attention to n tokens (at most
// what its programs hold): its cache grows as the sequence does, and the caller may keep it
// within the memory there is.
func (s *Session) Limit(n int) error {
	if s.chunks == nil {
		return fmt.Errorf("decode: only a program split at its attention can be limited")
	}
	if n <= 0 || n > s.progs[0].p.Context {
		return fmt.Errorf("decode: a limit of %d for a cache of %d", n, s.progs[0].p.Context)
	}
	if s.pos > n {
		return fmt.Errorf("decode: the sequence is %d tokens, over a limit of %d", s.pos, n)
	}
	s.t = n
	return nil
}

// Reset empties the cache. The chunks a session split at its attention holds stay allocated:
// their rows past the position are masked (the next step moves them into the shortest chunk
// that holds the sequence, Program.Smaller).
func (s *Session) Reset() {
	for name, b := range s.progs[0].p.In {
		if cacheInput(name) {
			clear(b)
		}
	}
	s.pos = 0
	s.sparseTouch(0)
}

// Mark is a point of a sequence that the session can go back to: the position, and a copy of
// what the steps after it overwrite in place, the rings of the sliding layers and the
// convolution windows. The longest caches are not copied: their rows past the position are
// masked, and the steps after it write only past it. A mark from Save also holds the rows of the
// longest caches before it.
type Mark struct {
	pos  int
	bufs map[string][]byte
	rows []byte // of Save: ExportRows(0, pos)
}

// Pos is the position of the mark.
func (m *Mark) Pos() int { return m.pos }

// Mark returns the current point of the sequence.
func (s *Session) Mark() *Mark {
	m := &Mark{pos: s.pos, bufs: map[string][]byte{}}
	in := s.progs[0].p.In
	for name, b := range in {
		if llm.HeldInput(name) {
			m.bufs[name] = slices.Clone(b)
		}
	}
	for _, c := range s.kv {
		if !c.long { // a ring
			m.bufs[c.parts[0]] = s.ringRows(c.parts[0], in[c.parts[0]])
		}
	}
	return m
}

// ringRows copies a ring (1, kv heads, slots, head_dim) for a mark: whole once the sequence has
// filled it, else only each head's slots before the position. Before then, slot j at or past the
// position holds no position the masks let a query see (StepInputs: the latest position before
// pos congruent to j is j - slots < 0), whatever it was left with, so a mark early in a
// conversation is a small part of a long window's rings (PLaMo 3's 21 rings of 2048 are 88 MB).
func (s *Session) ringRows(name string, b []byte) []byte {
	sh := s.progs[0].p.InShape[name]
	if len(sh) != 4 || s.pos >= sh[2] {
		return slices.Clone(b)
	}
	heads, slots, row := sh[1], sh[2], 2*sh[3]
	out := make([]byte, 0, heads*s.pos*row)
	for h := 0; h < heads; h++ {
		out = append(out, b[h*slots*row:(h*slots+s.pos)*row]...)
	}
	return out
}

// ringBytes is the size of a mark's copy of a ring at pos (ringRows).
func (s *Session) ringBytes(name string, pos int) int {
	sh := s.progs[0].p.InShape[name]
	if len(sh) != 4 {
		return -1
	}
	return sh[1] * min(pos, sh[2]) * 2 * sh[3]
}

// putRing writes a mark's copy of a ring back: whole, or each head's first slots (ringRows).
func (s *Session) putRing(name string, dst, b []byte) error {
	if len(b) == len(dst) {
		copy(dst, b)
		return nil
	}
	sh := s.progs[0].p.InShape[name]
	if len(sh) != 4 || sh[1] == 0 || len(b)%(sh[1]*2*sh[3]) != 0 || len(b)/(sh[1]*2*sh[3]) > sh[2] {
		return fmt.Errorf("decode: the mark's %s has %d bytes, the session's %d", name, len(b), len(dst))
	}
	heads, slots, row := sh[1], sh[2], 2*sh[3]
	n := len(b) / (heads * row)
	for h := 0; h < heads; h++ {
		copy(dst[h*slots*row:(h*slots+n)*row], b[h*n*row:(h+1)*n*row])
	}
	return nil
}

// Save returns the current point of the sequence with a copy of the rows before it, which stays
// good whatever the session does after: it can put aside one sequence for another and come back.
// It holds what the sequence uses, not the whole cache.
func (s *Session) Save() *Mark {
	m := s.Mark()
	m.rows, _ = s.ExportRows(0, s.pos)
	return m
}

// Rewind goes back to a mark of this session. For a mark from Mark, the rows of the longest
// caches before it must not have changed since it was taken: it is good until the session
// rewinds to an earlier point or restores a Save, and steps again. A mark from Save puts its
// rows back.
func (s *Session) Rewind(m *Mark) {
	if m.rows != nil {
		s.ImportRows(0, m.rows)
	}
	for name, b := range m.bufs {
		if err := s.putRing(name, s.progs[0].p.In[name], b); err != nil {
			panic(err) // a mark of this session: its rings are its own
		}
	}
	s.pos = m.pos
}

// Width is the number of values of a token's row that Step takes: the hidden size, and the widths
// of the model's lookup inputs (llm.TokenRow).
func (s *Session) Width() int {
	n := s.hidden
	for _, lk := range s.lookups {
		n += lk.Width
	}
	return n
}

// Step runs a program over the next tokens, given as their rows (at most Q): the embeddings (x)
// followed by the model's lookup inputs, as llm.TokenRow gathers them. It returns the logits after
// the last one, running the program with the fewest tokens per step that takes them all.
func (s *Session) Step(emb [][]float32) ([]float32, error) {
	p, q, err := s.run(emb, false)
	if err != nil {
		return nil, err
	}
	return s.keep(p, q, len(emb)), nil
}

// Speculate runs the tokens like Step, but keeps only as many of them as accept says, given the
// logits after each (row i, after token i, computed when asked): the first token and the drafts
// of speculative decoding after it, of which the model confirms a beginning. Each row is the
// logits the token would have been generated from, so a draft that the sampling picks from the
// row before it is what the model would have generated. The rest are not written to the cache;
// accept must keep at least 1. It returns the logits after the last token kept and how many.
func (s *Session) Speculate(emb [][]float32, accept func(row func(i int) []float32) int) ([]float32, int, error) {
	p, q, err := s.run(emb, true)
	if err != nil {
		return nil, 0, err
	}
	kept := accept(func(i int) []float32 { return s.row(p, i) })
	if kept < 1 || kept > len(emb) {
		return nil, 0, fmt.Errorf("decode: %d of %d tokens kept", kept, len(emb))
	}
	if kept < len(emb) && s.replaced(p) {
		// a state replaced as a whole is the one after all the step's tokens (llm.Cached):
		// bring the state before the step, which the inputs still hold, to the one after the
		// kept tokens on the host when the program gives what each row adds to it, or run the
		// kept ones again
		if ok, err := s.replay(p, kept); err != nil {
			return nil, 0, err
		} else if !ok {
			if p, q, err = s.run(emb[:kept], false); err != nil {
				return nil, 0, err
			}
		}
	}
	return s.keep(p, q, kept), kept, nil
}

// replay writes into each st_NAME_new of p the state after the first n rows of the step it
// ran, from st_NAME_rows (llm.StateReplayer), and reports whether it could: every state needs
// its rows.
func (s *Session) replay(p Program, n int) (bool, error) {
	r, ok := s.m.(llm.StateReplayer)
	if !ok {
		return false, nil
	}
	var names []string
	for name := range p.In {
		if strings.HasPrefix(name, "st_") {
			if _, ok := p.Out[name+"_rows"]; !ok {
				return false, nil
			}
			names = append(names, name)
		}
	}
	for _, name := range names {
		if err := r.ReplayState(name, p.In[name], p.Out[name+"_rows"], p.OutShape[name+"_rows"], n, p.Out[name+"_new"]); err != nil {
			return false, err
		}
	}
	return true, nil
}

// replaced reports whether p has state that a step replaces as a whole (st_, llm.Cached).
func (s *Session) replaced(p Program) bool {
	for name := range p.In {
		if strings.HasPrefix(name, "st_") {
			return true
		}
	}
	return false
}

// run writes the tokens' rows and the step's inputs into the program with the fewest tokens per
// step that takes them all, and runs it. With each, the program must compute the logits after
// each token, not after the last alone (llm.Options.LastRow).
func (s *Session) run(emb [][]float32, each bool) (p Program, q int, err error) {
	v := len(emb)
	if v == 0 || v > s.Q() {
		return p, 0, fmt.Errorf("decode: a step takes 1 to %d tokens, not %d", s.Q(), v)
	}
	if s.pos+v > s.t {
		return p, 0, fmt.Errorf("decode: the cache of %d tokens is full", s.t)
	}
	var st *step
	for i := range s.progs {
		if s.progs[i].q >= v && (!each || v == 1 || rows(s.progs[i].p) == s.progs[i].q) {
			st = &s.progs[i]
			p, q = st.p, st.q
			break
		}
	}
	if st == nil {
		return p, 0, fmt.Errorf("decode: no program computes the logits after each of %d tokens (the longer ones compute them after the last alone)", v)
	}
	x := p.In["x"]
	for _, e := range emb {
		if len(e) != s.Width() {
			return p, 0, fmt.Errorf("decode: a token's row has %d values, want %d", len(e), s.Width())
		}
	}
	// The rows past the tokens repeat the last token: they are masked out and not kept, but a
	// row of zeros would make 0/0 in the norms whose epsilon is under fp16's smallest value
	// (Gemma 4's), and the NaN would reach the tokens' rows through attention (0 * NaN).
	for i := range q {
		e := emb[min(i, v-1)]
		hwnpu.EncodeF16(x[2*i*s.hidden:2*(i+1)*s.hidden], e[:s.hidden])
		off := s.hidden
		for _, lk := range s.lookups {
			hwnpu.EncodeF16(p.In[lk.Input][2*i*lk.Width:2*(i+1)*lk.Width], e[off:off+lk.Width])
			off += lk.Width
		}
	}
	if last, ok := p.In["last"]; ok && rows(p) == 1 {
		// one-hot: the program's output layer reads the last token's row alone
		clear(last)
		hwnpu.EncodeF16(last[2*(v-1):2*v], []float32{1})
	}
	inputs := s.m.StepInputs
	if p.RunPart != nil {
		inputs = s.m.(llm.Paged).PagedStepInputs
	}
	for name, vals := range inputs(s.pos, v, q, s.t) {
		if b, ok := p.In[name]; ok {
			if 2*len(vals) != len(b) {
				return p, 0, fmt.Errorf("decode: %s takes %d values, the model gives %d", name, len(b)/2, len(vals))
			}
			hwnpu.EncodeF16(b, vals)
		}
	}
	if p.RunPart != nil {
		s.sparseTouch(s.pos)
		return p, q, s.runPaged(st, v)
	}
	if err := p.Run(); err != nil {
		return p, 0, err
	}
	return p, q, nil
}

// runPaged runs a program split at its attention, valid of its rows tokens: its parts, and
// between them the attention over the session's own cache. The step's keys and values go into the cache first, all q rows
// that fit (the padding's past the valid tokens are masked for them and written over later), so
// that the attention over the cache covers the step's tokens too; keep does not write them again.
func (s *Session) runPaged(st *step, valid int) error {
	p, q := st.p, st.q
	rows := min(q, s.t-s.pos) // rows that fit in the cache
	if err := s.fit(s.pos + rows); err != nil {
		return err
	}
	att := p.attention(s.chunk)
	used := (s.pos + rows + att.Chunk - 1) / att.Chunk
	if err := st.stepMasks(s.pos, q, used); err != nil {
		return err
	}
	oneMasks := false // the masks of the last token alone, made for the first layer that needs them
	prof := profStep(profKeyOf(p))
	var t0 time.Time
	for i := 0; ; i++ {
		if prof != nil {
			if i > 0 {
				prof.attend[i-1] += time.Since(t0)
			}
			t0 = time.Now()
		}
		if err := p.RunPart(i); err != nil {
			return err
		}
		if prof != nil {
			prof.part(i, time.Since(t0))
			t0 = time.Now()
		}
		if i == len(p.Attend) {
			if p.NoLogits {
				return s.writeUnattended(p, q, rows)
			}
			return nil
		}
		layer := p.Attend[i]
		src := s.kvLayer(layer)
		kc, vc := s.cache(fmt.Sprintf("kv_k_%d_new", src)), s.cache(fmt.Sprintf("kv_v_%d_new", src))
		if src == layer { // a layer reusing another's keys and values finds the step's written
			for _, c := range []*kvCache{kc, vc} {
				if err := s.writeRows(c, p.Out[c.out], q, s.pos, rows); err != nil {
					return err
				}
			}
		}
		if s.oneRow(p, layer) {
			if err := s.attendOne(st, layer, kc, vc, valid, &oneMasks); err != nil {
				return err
			}
			continue
		}
		copy(att.Q, p.Out[fmt.Sprintf("attn_q_%d", layer)])
		if s.sparse != nil && att == p.Attn {
			done, err := s.attendSparse(st, layer, kc, vc, s.pos, rows, min(valid, rows), used)
			if err != nil {
				return err
			}
			if done {
				putAttnOut(p, layer, st.res)
				continue
			}
		}
		st.merge.Reset()
		for j := 0; j < used; j++ {
			mask := st.zero
			if m, ok := st.masks[j]; ok {
				mask = m
			}
			if err := att.Run(s.chunks[kc.out][j], s.chunks[vc.out][j], mask); err != nil {
				return fmt.Errorf("decode: attention of layer %d over chunk %d: %w", layer, j, err)
			}
			hwnpu.DecodeF16(st.o[0], att.O)
			hwnpu.DecodeF16(st.o[1], att.M)
			hwnpu.DecodeF16(st.o[2], att.L)
			st.merge.Add(st.o[0], st.o[1], st.o[2])
		}
		st.merge.Result(st.res)
		putAttnOut(p, layer, st.res)
	}
}

// writeUnattended writes the step's rows of the caches p outputs but attends for no layer of
// (a program that only reads tokens into the caches: the last layer that writes one).
func (s *Session) writeUnattended(p Program, q, rows int) error {
	done := map[int]bool{}
	for _, layer := range p.Attend {
		done[s.kvLayer(layer)] = true
	}
	for i := range s.kv {
		c := &s.kv[i]
		if c.chunk == 0 {
			continue
		}
		var layer int
		if _, err := fmt.Sscanf(c.out, "kv_k_%d_new", &layer); err != nil {
			if _, err := fmt.Sscanf(c.out, "kv_v_%d_new", &layer); err != nil {
				return fmt.Errorf("decode: cache output %s", c.out)
			}
		}
		if done[layer] {
			continue
		}
		out, ok := p.Out[c.out]
		if !ok {
			return fmt.Errorf("decode: the program reads tokens into the caches but has no output %s", c.out)
		}
		if err := s.writeRows(c, out, q, s.pos, rows); err != nil {
			return err
		}
	}
	return nil
}

// oneRow reports whether p computes layer's query for the step's last token alone: a program that
// needs only its row of the layers after the one that last writes the cache (llm.Options.LastRow,
// Gemma 4's layers that reuse earlier keys and values).
func (s *Session) oneRow(p Program, layer int) bool {
	sh := p.OutShape[fmt.Sprintf("attn_q_%d", layer)]
	return sh[2] == 1 && p.InShape["x"][1] > 1
}

// attendOne runs the attention of layer for the step's last token alone, the one at position
// s.pos+valid-1, on the program with the fewest tokens per step (its row 0; that program's
// attention costs a fraction of the step's), and writes its row to the step's attn_o. masks says
// whether that program's masks are the token's already.
func (s *Session) attendOne(st *step, layer int, kc, vc *kvCache, valid int, masks *bool) error {
	sm := &s.progs[0]
	if sm == st || sm.p.RunPart == nil {
		return fmt.Errorf("decode: layer %d attends for the last token alone, and no program of fewer tokens per step runs it", layer)
	}
	a := sm.p.attention(s.chunk)
	pos := s.pos + valid - 1
	used := pos/a.Chunk + 1 // the chunks the token sees
	if !*masks {
		if err := sm.stepMasks(pos, sm.q, used); err != nil {
			return err
		}
		*masks = true
	}
	sh := st.p.OutShape[fmt.Sprintf("attn_q_%d", layer)]
	heads, d := sh[1], sh[3]
	q := st.p.Out[fmt.Sprintf("attn_q_%d", layer)]
	for h := range heads {
		copy(a.Q[2*h*sm.q*d:2*(h*sm.q+1)*d], q[2*h*d:2*(h+1)*d])
	}
	done := false
	if s.sparse != nil && a == sm.p.Attn {
		var err error
		if done, err = s.attendSparse(sm, layer, kc, vc, pos, 1, 1, used); err != nil {
			return err
		}
	}
	if !done {
		sm.merge.Reset()
		for j := 0; j < used; j++ {
			mask := sm.zero
			if m, ok := sm.masks[j]; ok {
				mask = m
			}
			if err := a.Run(s.chunks[kc.out][j], s.chunks[vc.out][j], mask); err != nil {
				return fmt.Errorf("decode: attention of layer %d over chunk %d: %w", layer, j, err)
			}
			hwnpu.DecodeF16(sm.o[0], a.O)
			hwnpu.DecodeF16(sm.o[1], a.M)
			hwnpu.DecodeF16(sm.o[2], a.L)
			sm.merge.Add(sm.o[0], sm.o[1], sm.o[2])
		}
		sm.merge.Result(sm.res)
	}
	res := make([]float32, heads*d)
	for h := range heads {
		copy(res[h*d:(h+1)*d], sm.res[h*sm.q*d:(h*sm.q+1)*d])
	}
	putAttnOut(st.p, layer, res)
	return nil
}

// putAttnOut writes a layer's attention output res, (1, heads, q, head_dim), to the program's
// input: attn_o_N as it is, or attn_ot_N with the tokens first, (1, q, heads, head_dim), which
// spares the part that goes on from the layer a transpose of its input (Core ML put the one that
// begins PLaMo 3's last part on the CPU).
func putAttnOut(p Program, layer int, res []float32) {
	b, ok := p.In[fmt.Sprintf("attn_ot_%d", layer)]
	if !ok {
		hwnpu.EncodeF16(p.In[fmt.Sprintf("attn_o_%d", layer)], res)
		return
	}
	sh := p.InShape[fmt.Sprintf("attn_ot_%d", layer)]
	q, heads, d := sh[1], sh[2], sh[3]
	t := make([]float32, len(res))
	for h := 0; h < heads; h++ {
		for i := 0; i < q; i++ {
			copy(t[(i*heads+h)*d:(i*heads+h+1)*d], res[(h*q+i)*d:(h*q+i+1)*d])
		}
	}
	hwnpu.EncodeF16(b, t)
}

// stepMasks makes the masks of the chunks the step's q tokens from pos fall in: query t (position
// pos+t) sees the positions up to its own. The chunks before them are wholly seen (zero).
func (st *step) stepMasks(pos, q, used int) error {
	c := st.chunk
	for j, b := range st.masks {
		st.spare = append(st.spare, b)
		delete(st.masks, j)
	}
	vals := make([]float32, q*c)
	for j := pos / c; j < used; j++ {
		var b Buf
		if n := len(st.spare); n > 0 {
			b, st.spare = st.spare[n-1], st.spare[:n-1]
		} else {
			var err error
			if b, err = st.p.Attn.NewBuf(2 * q * c); err != nil {
				return err
			}
		}
		for t := 0; t < q; t++ {
			row := vals[t*c : (t+1)*c]
			for k := range row {
				if j*c+k <= pos+t {
					row[k] = 0
				} else {
					row[k] = -1e4
				}
			}
		}
		hwnpu.EncodeF16(b.Bytes(), vals)
		st.masks[j] = b
	}
	return nil
}

// writeRows writes the first n of the q rows of a step output (1, heads, q, d) into the cache at
// positions from pos on.
func (s *Session) writeRows(c *kvCache, out []byte, q, pos, n int) error {
	for i := 0; i < n; i++ {
		j, slot := c.locate(pos + i)
		in, err := s.part(c, j, true)
		if err != nil {
			return err
		}
		L := c.span(j)
		for h := 0; h < c.heads; h++ {
			copy(in[2*(h*L+slot)*c.d:2*(h*L+slot+1)*c.d], out[2*(h*q+i)*c.d:2*(h*q+i+1)*c.d])
		}
	}
	return nil
}

// keep writes the first v tokens of the step just run into the cache (the program run was p, of
// q tokens per step), moves the sequence past them, and returns the logits after the last one.
func (s *Session) keep(p Program, q, v int) []float32 {
	for _, c := range s.kv {
		if c.chunk > 0 {
			continue // written before the attention (runPaged)
		}
		// (1, heads, L, d) <- (1, heads, q, d): the step's rows at their positions, in the
		// chunk that holds each, or in a cache shorter than the longest (a ring) at position
		// mod L
		out := p.Out[c.out]
		heads, d := c.heads, c.d
		for i := 0; i < v; i++ {
			pos := s.pos + i
			if !c.long {
				pos %= c.total
			}
			j, slot := c.locate(pos)
			in, L := p.In[c.parts[j]], c.lens[j]
			for h := 0; h < heads; h++ {
				copy(in[2*(h*L+slot)*d:2*(h*L+slot+1)*d], out[2*(h*q+i)*d:2*(h*q+i+1)*d])
			}
		}
	}
	for name, in := range p.In {
		if strings.HasPrefix(name, "win_") {
			// (1, w, d) <- rows v..v+w of (1, w+q, d): the window after v tokens
			sh := p.InShape[name]
			w, d := sh[1], sh[2]
			copy(in, p.Out[name+"_new"][2*v*d:2*(v+w)*d])
		}
		if strings.HasPrefix(name, "st_") {
			// the state after the step's valid tokens (llm.Cached): run kept them all
			copy(in, p.Out[name+"_new"])
		}
	}
	s.pos += v
	s.last, s.lastRows = nil, nil
	i := v - 1
	if rows(p) == 1 {
		i = 0 // the program computed the last token's row alone
	}
	if h, ok := p.Out["h"]; ok {
		// h has a row per token, or only the last token's (rows of the logits)
		hr := len(h) / (2 * s.hidden)
		from, last := 0, i
		if hr == q {
			from, last = 0, v-1
		} else {
			from = i
		}
		for r := from; r <= last; r++ {
			row := make([]float32, s.hidden)
			hwnpu.DecodeF16(row, h[2*r*s.hidden:2*(r+1)*s.hidden])
			s.lastRows = append(s.lastRows, row)
		}
		s.last = s.lastRows[len(s.lastRows)-1]
	}
	if p.NoLogits {
		return nil
	}
	return s.row(p, i)
}

// HiddenRows is the final hidden states h of the last step's kept tokens, one per token, for a
// program that outputs h for every row (llm qwen35's hidden-out); only the last token's for one
// that outputs it for the last row alone, and nil for one without h.
func (s *Session) HiddenRows() [][]float32 { return s.lastRows }

// Truncate drops the positions from n on, which the next steps write again. Only a session whose
// state is all KV caches that hold every position can go back without a mark: a window, a
// state replaced as a whole or a ring would not be as it was at n.
func (s *Session) Truncate(n int) error {
	if n >= s.pos {
		return nil
	}
	for _, c := range s.kv {
		if !c.long {
			return fmt.Errorf("decode: a ring cache cannot go back without a mark")
		}
	}
	for name := range s.progs[0].p.In {
		if llm.HeldInput(name) {
			return fmt.Errorf("decode: %s cannot go back without a mark", name)
		}
	}
	s.pos = max(n, 0)
	s.last, s.lastRows = nil, nil
	return nil
}

// rows is the number of rows of p's logits: 1 for a program that computes them after the token
// its input last picks (llm.Options.LastRow), otherwise one per token of a step.
func rows(p Program) int {
	q := p.InShape["x"][1]
	if sh := p.InShape["last"]; slices.Equal(sh, []int{1, 1, q}) && q > 1 {
		return 1
	}
	return q
}

// row returns the logits after token i of the step just run on p (row i of its output).
func (s *Session) row(p Program, i int) []float32 {
	if profile.on {
		defer profRows(profKeyOf(p), time.Now())
	}
	logits := make([]float32, s.vocab)
	hwnpu.DecodeF16(logits, p.Out["y"][2*i*s.vocab:2*(i+1)*s.vocab])
	if !s.raw {
		llm.Softcap(logits, s.softcap)
	}
	return logits
}

// Softcap is the soft cap c the host applies to the program's logits (llm.LogitSoftcap): the
// logits are c*tanh(y/c) of the program's y. 0 when the program applied any itself.
func (s *Session) Softcap() float64 { return s.softcap }

// SetRawLogits makes the steps return the program's logits before the host's soft cap, for a
// caller that applies it to the few it needs (the cap is monotonic). Without a cap it changes
// nothing.
func (s *Session) SetRawLogits(raw bool) { s.raw = raw }

// Hidden is the final hidden state after the last step's last token, for a program that outputs
// it as h (lfm3's hidden-out); nil otherwise.
func (s *Session) Hidden() []float32 { return s.last }

// Rows of the longest caches do not depend on the program's cache length or chunks: a position's
// rows are the same in every program of the model, which is what Migrate relies on. ExportRows
// packs them position by position, for each long cache in name order each head's d values in
// fp16, and ImportRows puts them back into a session of any program of the model whose cache
// holds the positions (a prompt kept on disk for a later run, internal/chat's store). The
// rings and the convolution windows are not rows: a Mark holds them.

// longCaches is the caches that hold every position (not the rings), by name.
func (s *Session) longCaches() []*kvCache {
	var cs []*kvCache
	for i := range s.kv {
		if s.kv[i].long {
			cs = append(cs, &s.kv[i])
		}
	}
	slices.SortFunc(cs, func(a, b *kvCache) int { return strings.Compare(a.out, b.out) })
	return cs
}

// RowBytes is the bytes of one position's rows in ExportRows.
func (s *Session) RowBytes() int {
	n := 0
	for _, c := range s.longCaches() {
		n += 2 * c.heads * c.d
	}
	return n
}

// ExportRows returns the rows of positions from (inclusive) to to (exclusive) of the longest
// caches, packed as RowBytes bytes per position.
func (s *Session) ExportRows(from, to int) ([]byte, error) {
	if from < 0 || to < from || to > s.t {
		return nil, fmt.Errorf("decode: rows %d to %d of a cache of %d", from, to, s.t)
	}
	out := make([]byte, 0, (to-from)*s.RowBytes())
	for pos := from; pos < to; pos++ {
		for _, c := range s.longCaches() {
			j, slot := c.locate(pos)
			in, _ := s.part(c, j, false)
			L := c.span(j)
			for h := 0; h < c.heads; h++ {
				if in == nil { // a chunk never written
					out = append(out, make([]byte, 2*c.d)...)
					continue
				}
				out = append(out, in[2*(h*L+slot)*c.d:2*(h*L+slot+1)*c.d]...)
			}
		}
	}
	return out, nil
}

// ImportRows writes rows from ExportRows (of a session of the same model) at the positions from
// from on. The sequence's position does not move: Rewind to a Mark at the end of the rows.
func (s *Session) ImportRows(from int, rows []byte) error {
	rb := s.RowBytes()
	if rb == 0 || len(rows)%rb != 0 {
		return fmt.Errorf("decode: %d bytes of rows, not a multiple of %d", len(rows), rb)
	}
	n := len(rows) / rb
	if from < 0 || from+n > s.t {
		return fmt.Errorf("decode: rows %d to %d of a cache of %d", from, from+n, s.t)
	}
	if err := s.fit(max(from+n, s.pos)); err != nil {
		return err
	}
	s.sparseTouch(from)
	off := 0
	for pos := from; pos < from+n; pos++ {
		for _, c := range s.longCaches() {
			j, slot := c.locate(pos)
			in, err := s.part(c, j, true)
			if err != nil {
				return err
			}
			L := c.span(j)
			for h := 0; h < c.heads; h++ {
				copy(in[2*(h*L+slot)*c.d:2*(h*L+slot+1)*c.d], rows[off:off+2*c.d])
				off += 2 * c.d
			}
		}
	}
	return nil
}

// Bufs is what the mark holds besides its position: the rings and the windows at it, by input
// name. The caller must not change them.
func (m *Mark) Bufs() map[string][]byte { return m.bufs }

// NewMark makes a mark from its position and buffers (Bufs), as read back from a store. Resume
// checks it against the session.
func NewMark(pos int, bufs map[string][]byte) *Mark {
	return &Mark{pos: pos, bufs: bufs}
}

// Resume brings an empty sequence to a point kept outside the session: rows (ExportRows) of the
// positions before the mark, and the mark, from a session of the same model. It checks that the
// mark's buffers are the session's rings and windows, of the same sizes, and that the rows are
// exactly the mark's position.
func (s *Session) Resume(rows []byte, m *Mark) error {
	if m.pos < 0 || m.pos > s.t {
		return fmt.Errorf("decode: a mark at %d, the cache holds %d", m.pos, s.t)
	}
	if rb := s.RowBytes(); len(rows) != m.pos*rb {
		return fmt.Errorf("decode: %d bytes of rows for a mark at %d, want %d", len(rows), m.pos, m.pos*rb)
	}
	in := s.progs[0].p.In
	want := map[string]bool{}
	for name := range in {
		if llm.HeldInput(name) {
			want[name] = true
		}
	}
	for _, c := range s.kv {
		if !c.long {
			want[c.parts[0]] = true
		}
	}
	for name, b := range m.bufs {
		if !want[name] {
			return fmt.Errorf("decode: the mark holds %s, which is not a ring or a window of the session", name)
		}
		if len(b) != len(in[name]) && (llm.HeldInput(name) || len(b) != s.ringBytes(name, m.pos)) {
			return fmt.Errorf("decode: the mark's %s has %d bytes, the session's %d", name, len(b), len(in[name]))
		}
		delete(want, name)
	}
	for name := range want {
		return fmt.Errorf("decode: the mark lacks %s", name)
	}
	s.Reset()
	if err := s.ImportRows(0, rows); err != nil {
		return err
	}
	s.Rewind(m)
	return nil
}

// logitsShape reports whether y holds the logits of q tokens: (1, q, vocab), or folded into rows
// (1, q, rows, vocab/rows), which lays out the same (Gemma 4's output stays folded so that the
// whole program is on the Neural Engine).
func logitsShape(y []int, q, vocab int) bool {
	switch len(y) {
	case 3:
		return y[0] == 1 && y[1] == q && y[2] == vocab
	case 4:
		return y[0] == 1 && y[1] == q && y[2]*y[3] == vocab
	}
	return false
}
