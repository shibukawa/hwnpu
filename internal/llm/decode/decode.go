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
	"slices"
	"strconv"
	"strings"
	"unsafe"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
)

// Program is the loaded program: its fp16 input and output buffers by name and their shapes.
type Program struct {
	In, Out           map[string][]byte
	InShape, OutShape map[string][]int
	// Run computes the outputs from the inputs. Before it, the inputs have been written; after
	// it, the outputs are read. Sync hooks for device buffers belong in Run.
	Run func() error
}

// Session is the state of one sequence.
type Session struct {
	progs  []step // by tokens per step, shortest first
	m      llm.Cached
	kv     []kvCache
	t      int // cache length
	hidden int
	vocab  int
	pos    int
}

// kvCache is one cache of keys or values: an input kv_NAME, or the chunks kv_NAME_cJ that hold
// its positions in order (llm.CacheChunks).
type kvCache struct {
	out   string   // the step's rows, kv_NAME_new
	parts []string // the inputs
	lens  []int
	total int
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
}

// New checks the program's inputs and outputs and starts an empty sequence.
func New(p Program, m llm.Cached) (*Session, error) {
	x, ok := p.InShape["x"]
	if !ok || len(x) != 3 {
		return nil, fmt.Errorf("decode: no input x (1, q, hidden)")
	}
	y, ok := p.OutShape["y"]
	if !ok || len(y) != 3 || y[1] != x[1] {
		return nil, fmt.Errorf("decode: no output y (1, q, vocab)")
	}
	s := &Session{progs: []step{{p, x[1]}}, m: m, hidden: x[2], vocab: y[2]}
	var err error
	if s.kv, s.t, err = caches(p); err != nil {
		return nil, err
	}
	s.Reset()
	return s, nil
}

// caches gathers the program's KV cache inputs into caches, and gives the longest's length.
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
		kv = append(kv, c)
		t = max(t, c.total) // the longest cache; shorter ones are rings
	}
	if t == 0 {
		return nil, 0, fmt.Errorf("decode: the program has no KV cache inputs (kv_*)")
	}
	return kv, t, nil
}

// Add adds a program of the same model that takes another number of tokens per step. Its cache
// inputs (kv_*, win_*) must be the same memory as the first program's (chain.Share), so that
// steps of either program continue the same sequence.
func (s *Session) Add(p Program) error {
	x, ok := p.InShape["x"]
	if !ok || len(x) != 3 || x[2] != s.hidden {
		return fmt.Errorf("decode: no input x (1, q, %d)", s.hidden)
	}
	if y, ok := p.OutShape["y"]; !ok || len(y) != 3 || y[1] != x[1] || y[2] != s.vocab {
		return fmt.Errorf("decode: no output y (1, %d, %d)", x[1], s.vocab)
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
	s.progs = append(s.progs, step{p, x[1]})
	slices.SortFunc(s.progs, func(a, b step) int { return a.q - b.q })
	return nil
}

func cacheInput(name string) bool {
	return strings.HasPrefix(name, "kv_") || strings.HasPrefix(name, "win_")
}

// Q is the most tokens one step takes.
func (s *Session) Q() int { return s.progs[len(s.progs)-1].q }

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
	x, ok := np.InShape["x"]
	if !ok || len(x) != 3 || x[2] != s.hidden {
		return fmt.Errorf("decode: no input x (1, q, %d)", s.hidden)
	}
	if y, ok := np.OutShape["y"]; !ok || len(y) != 3 || y[1] != x[1] || y[2] != s.vocab {
		return fmt.Errorf("decode: no output y (1, %d, %d)", x[1], s.vocab)
	}
	nkv, nt, err := caches(np)
	if err != nil {
		return err
	}
	if nt < s.t {
		return fmt.Errorf("decode: the new program caches %d tokens, fewer than the session's %d", nt, s.t)
	}
	first := s.progs[0].p
	for name, in := range first.In {
		if strings.HasPrefix(name, "win_") {
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
	s.progs, s.t, s.kv = []step{{np, x[1]}}, nt, nkv
	return nil
}

// locate gives the chunk that holds a position of the cache and its row there.
func (c *kvCache) locate(pos int) (chunk, row int) {
	for pos >= c.lens[chunk] {
		pos -= c.lens[chunk]
		chunk++
	}
	return chunk, pos
}

// Reset empties the cache.
func (s *Session) Reset() {
	for name, b := range s.progs[0].p.In {
		if cacheInput(name) {
			clear(b)
		}
	}
	s.pos = 0
}

// Mark is a point of a sequence that the session can go back to: the position, and a copy of
// what the steps after it overwrite in place, the rings of the sliding layers and the
// convolution windows. The longest caches are not copied: their rows past the position are
// masked, and the steps after it write only past it.
type Mark struct {
	pos  int
	bufs map[string][]byte
}

// Pos is the position of the mark.
func (m *Mark) Pos() int { return m.pos }

// Mark returns the current point of the sequence.
func (s *Session) Mark() *Mark {
	m := &Mark{pos: s.pos, bufs: map[string][]byte{}}
	in := s.progs[0].p.In
	for name, b := range in {
		if strings.HasPrefix(name, "win_") {
			m.bufs[name] = slices.Clone(b)
		}
	}
	for _, c := range s.kv {
		if c.total < s.t { // a ring
			m.bufs[c.parts[0]] = slices.Clone(in[c.parts[0]])
		}
	}
	return m
}

// Save returns the current point of the sequence with a copy of the whole cache, which stays
// good whatever the session does after: it can put aside one sequence for another and come back.
func (s *Session) Save() *Mark {
	m := &Mark{pos: s.pos, bufs: map[string][]byte{}}
	for name, b := range s.progs[0].p.In {
		if cacheInput(name) {
			m.bufs[name] = slices.Clone(b)
		}
	}
	return m
}

// Rewind goes back to a mark of this session. For a mark from Mark, the rows of the longest
// caches before it must not have changed since it was taken: it is good until the session
// rewinds to an earlier point or restores a Save, and steps again.
func (s *Session) Rewind(m *Mark) {
	for name, b := range m.bufs {
		copy(s.progs[0].p.In[name], b)
	}
	s.pos = m.pos
}

// Step runs a program over the next tokens, given as their embeddings (at most Q of hidden
// values each), and returns the logits after the last one. It runs the program with the fewest
// tokens per step that takes them all.
func (s *Session) Step(emb [][]float32) ([]float32, error) {
	v := len(emb)
	if v == 0 || v > s.Q() {
		return nil, fmt.Errorf("decode: a step takes 1 to %d tokens, not %d", s.Q(), v)
	}
	if s.pos+v > s.t {
		return nil, fmt.Errorf("decode: the cache of %d tokens is full", s.t)
	}
	var p Program
	var q int
	for _, o := range s.progs {
		if o.q >= v {
			p, q = o.p, o.q
			break
		}
	}
	x := p.In["x"]
	clear(x)
	for i, e := range emb {
		if len(e) != s.hidden {
			return nil, fmt.Errorf("decode: embedding of %d values, want %d", len(e), s.hidden)
		}
		hwnpu.EncodeF16(x[2*i*s.hidden:2*(i+1)*s.hidden], e)
	}
	for name, vals := range s.m.StepInputs(s.pos, v, q, s.t) {
		if b, ok := p.In[name]; ok {
			if 2*len(vals) != len(b) {
				return nil, fmt.Errorf("decode: %s takes %d values, the model gives %d", name, len(b)/2, len(vals))
			}
			hwnpu.EncodeF16(b, vals)
		}
	}
	if err := p.Run(); err != nil {
		return nil, err
	}
	for _, c := range s.kv {
		// (1, heads, L, d) <- (1, heads, q, d): the step's rows at their positions, in the
		// chunk that holds each, or in a cache shorter than the longest (a ring) at position
		// mod L
		out := p.Out[c.out]
		sh := p.InShape[c.parts[0]]
		heads, d := sh[1], sh[3]
		for i := 0; i < v; i++ {
			pos := s.pos + i
			if c.total < s.t {
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
			// (1, w, d) <- rows v..v+w of (1, w+q, d)
			sh := p.InShape[name]
			w, d := sh[1], sh[2]
			copy(in, p.Out[name+"_new"][2*v*d:2*(v+w)*d])
		}
	}
	s.pos += v
	logits := make([]float32, s.vocab)
	y := p.Out["y"]
	hwnpu.DecodeF16(logits, y[2*(v-1)*s.vocab:2*v*s.vocab])
	return logits, nil
}
