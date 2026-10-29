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
	t      int // cache length
	hidden int
	vocab  int
	pos    int
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
	for name, sh := range p.InShape {
		if strings.HasPrefix(name, "kv_") {
			if len(sh) != 4 {
				return nil, fmt.Errorf("decode: %s has shape %v, want (1, heads, cache, head_dim)", name, sh)
			}
			s.t = max(s.t, sh[2]) // the longest cache; shorter ones are rings
		}
		if strings.HasPrefix(name, "kv_") || strings.HasPrefix(name, "win_") {
			if _, ok := p.OutShape[name+"_new"]; !ok {
				return nil, fmt.Errorf("decode: input %s has no output %s_new", name, name)
			}
		}
	}
	if s.t == 0 {
		return nil, fmt.Errorf("decode: the program has no KV cache inputs (kv_*)")
	}
	s.Reset()
	return s, nil
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
		if _, ok := p.OutShape[name+"_new"]; !ok {
			return fmt.Errorf("decode: input %s has no output %s_new", name, name)
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
// (a cache shorter than the longest) keeps its length and is copied as is.
func (s *Session) Migrate(np Program) error {
	x, ok := np.InShape["x"]
	if !ok || len(x) != 3 || x[2] != s.hidden {
		return fmt.Errorf("decode: no input x (1, q, %d)", s.hidden)
	}
	if y, ok := np.OutShape["y"]; !ok || len(y) != 3 || y[1] != x[1] || y[2] != s.vocab {
		return fmt.Errorf("decode: no output y (1, %d, %d)", x[1], s.vocab)
	}
	nt := 0
	for name, sh := range np.InShape {
		if strings.HasPrefix(name, "kv_") {
			nt = max(nt, sh[2])
		}
		if cacheInput(name) {
			if _, ok := np.OutShape[name+"_new"]; !ok {
				return fmt.Errorf("decode: input %s has no output %s_new", name, name)
			}
		}
	}
	if nt < s.t {
		return fmt.Errorf("decode: the new program caches %d tokens, fewer than the session's %d", nt, s.t)
	}
	first := s.progs[0].p
	for name, in := range first.In {
		if !cacheInput(name) {
			continue
		}
		out, ok := np.In[name]
		if !ok {
			return fmt.Errorf("decode: the new program has no input %s", name)
		}
		sh, nsh := first.InShape[name], np.InShape[name]
		switch {
		case len(sh) == 4 && len(nsh) == 4 && nsh[2] > sh[2] && sh[1] == nsh[1] && sh[3] == nsh[3]:
			// (1, heads, L, d) -> (1, heads, nL, d): each head's rows at the same positions
			heads, L, nL, d := sh[1], sh[2], nsh[2], sh[3]
			clear(out)
			for h := 0; h < heads; h++ {
				copy(out[2*h*nL*d:], in[2*h*L*d:2*(h+1)*L*d])
			}
		case len(in) == len(out):
			copy(out, in)
		default:
			return fmt.Errorf("decode: %s has shape %v in the new program, %v in the session", name, nsh, sh)
		}
	}
	s.progs, s.t = []step{{np, x[1]}}, nt
	return nil
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
	for name, b := range s.progs[0].p.In {
		if strings.HasPrefix(name, "win_") || strings.HasPrefix(name, "kv_") && s.progs[0].p.InShape[name][2] < s.t {
			m.bufs[name] = slices.Clone(b)
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
	for name, in := range p.In {
		sh := p.InShape[name]
		out := p.Out[name+"_new"]
		switch {
		case strings.HasPrefix(name, "kv_"):
			// (1, heads, L, d) <- (1, heads, q, d): the step's rows at their positions, or
			// in a cache shorter than the longest (a ring) at position mod L
			heads, L, d := sh[1], sh[2], sh[3]
			for h := 0; h < heads; h++ {
				for i := 0; i < v; i++ {
					slot := s.pos + i
					if L < s.t {
						slot %= L
					}
					copy(in[2*(h*L+slot)*d:2*(h*L+slot+1)*d], out[2*(h*q+i)*d:2*(h*q+i+1)*d])
				}
			}
		case strings.HasPrefix(name, "win_"):
			// (1, w, d) <- rows v..v+w of (1, w+q, d)
			w, d := sh[1], sh[2]
			copy(in, out[2*v*d:2*(v+w)*d])
		}
	}
	s.pos += v
	logits := make([]float32, s.vocab)
	y := p.Out["y"]
	hwnpu.DecodeF16(logits, y[2*(v-1)*s.vocab:2*v*s.vocab])
	return logits, nil
}
