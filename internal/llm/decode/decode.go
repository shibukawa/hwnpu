// Package decode runs a causal model built with a KV cache (llm.Cached, Options.Cache) one step
// at a time. The cache lives in the program's own input buffers: after each step the new keys,
// values and convolution windows the program outputs are written into the inputs they belong to,
// so the next step reads them where they are. The buffers are plain fp16 bytes, so the same code
// drives Core ML buffers (internal/llm/chain) and the CPU interpreter in tests.
package decode

import (
	"fmt"
	"strings"

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
	p      Program
	m      llm.Cached
	q, t   int // tokens per step, cache length
	hidden int
	vocab  int
	pos    int
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
	s := &Session{p: p, m: m, q: x[1], hidden: x[2], vocab: y[2]}
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

// Q is the number of tokens one step takes.
func (s *Session) Q() int { return s.q }

// Cache is the number of tokens the longest cache holds: the sequence length the session can
// reach. Shorter caches (a sliding layer's ring) hold the last tokens only.
func (s *Session) Cache() int { return s.t }

// Pos is the number of tokens in the sequence so far.
func (s *Session) Pos() int { return s.pos }

// Migrate moves the session to np, a program of the same model whose caches are at least as long
// (the next tier of a tiered model): the cached rows are copied over and later steps run np. A
// ring (a cache shorter than the longest) keeps its length and is copied as is.
func (s *Session) Migrate(np Program) error {
	x, ok := np.InShape["x"]
	if !ok || len(x) != 3 || x[1] != s.q || x[2] != s.hidden {
		return fmt.Errorf("decode: the new program takes x %v, the session %v", x, []int{1, s.q, s.hidden})
	}
	nt := 0
	for name, sh := range np.InShape {
		if strings.HasPrefix(name, "kv_") {
			nt = max(nt, sh[2])
		}
	}
	if nt < s.t {
		return fmt.Errorf("decode: the new program caches %d tokens, fewer than the session's %d", nt, s.t)
	}
	for name, in := range s.p.In {
		if !strings.HasPrefix(name, "kv_") && !strings.HasPrefix(name, "win_") {
			continue
		}
		out, ok := np.In[name]
		if !ok {
			return fmt.Errorf("decode: the new program has no input %s", name)
		}
		sh, nsh := s.p.InShape[name], np.InShape[name]
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
	s.p, s.t = np, nt
	return nil
}

// Reset empties the cache.
func (s *Session) Reset() {
	for name, b := range s.p.In {
		if strings.HasPrefix(name, "kv_") || strings.HasPrefix(name, "win_") {
			clear(b)
		}
	}
	s.pos = 0
}

// Step runs the program over the next tokens, given as their embeddings (at most Q of hidden
// values each), and returns the logits after the last one.
func (s *Session) Step(emb [][]float32) ([]float32, error) {
	v := len(emb)
	if v == 0 || v > s.q {
		return nil, fmt.Errorf("decode: a step takes 1 to %d tokens, not %d", s.q, v)
	}
	if s.pos+v > s.t {
		return nil, fmt.Errorf("decode: the cache of %d tokens is full", s.t)
	}
	x := s.p.In["x"]
	clear(x)
	for i, e := range emb {
		if len(e) != s.hidden {
			return nil, fmt.Errorf("decode: embedding of %d values, want %d", len(e), s.hidden)
		}
		hwnpu.EncodeF16(x[2*i*s.hidden:2*(i+1)*s.hidden], e)
	}
	for name, vals := range s.m.StepInputs(s.pos, v, s.q, s.t) {
		if b, ok := s.p.In[name]; ok {
			if 2*len(vals) != len(b) {
				return nil, fmt.Errorf("decode: %s takes %d values, the model gives %d", name, len(b)/2, len(vals))
			}
			hwnpu.EncodeF16(b, vals)
		}
	}
	if err := s.p.Run(); err != nil {
		return nil, err
	}
	for name, in := range s.p.In {
		sh := s.p.InShape[name]
		out := s.p.Out[name+"_new"]
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
					copy(in[2*(h*L+slot)*d:2*(h*L+slot+1)*d], out[2*(h*s.q+i)*d:2*(h*s.q+i+1)*d])
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
	y := s.p.Out["y"]
	hwnpu.DecodeF16(logits, y[2*(v-1)*s.vocab:2*v*s.vocab])
	return logits, nil
}
