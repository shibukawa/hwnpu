package decode

import (
	"fmt"
	"strings"
)

// StepChunks runs the chunks of a prompt (each up to Q tokens, every one but the last Q) like
// Step one after another, and returns the logits after the last. A program split at its
// attention whose parts can hand their stages over (Program.Carry) runs them a stage at a time:
// stage i for every chunk, then stage i+1. A mixture of experts run on the host (gpt-oss's) then
// runs a layer's experts for chunk after chunk, which keeps that layer's weights in memory
// between them, where chunk by chunk every layer's go through it once a chunk (design.md 16.275).
// It takes the cache's keys and values a stage writes before the chunks after it read them, and
// the windows of the layers that attend inside the parts (gpt-oss's sliding ones) after each
// stage's chunk.
func (s *Session) StepChunks(chunks [][][]float32) ([]float32, error) {
	st, ok := s.layerMajor(chunks)
	if !ok {
		var logits []float32
		for _, c := range chunks {
			var err error
			if logits, err = s.Step(c); err != nil {
				return nil, err
			}
		}
		return logits, nil
	}
	p, q := st.p, st.q
	pos0 := s.pos
	total := 0
	for _, c := range chunks {
		total += len(c)
	}
	if pos0+total > s.t {
		return nil, fmt.Errorf("decode: the cache of %d tokens is full", s.t)
	}
	n := len(chunks)
	lastPos := pos0 + total - len(chunks[n-1])
	if err := s.fit(lastPos + min(q, s.t-lastPos)); err != nil {
		return nil, err
	}
	s.sparseTouch(pos0)
	att := p.attention(s.chunk)
	carry := make([][]byte, n)
	for i := 0; ; i++ {
		pos := pos0
		for j, emb := range chunks {
			s.pos = pos
			if err := s.prepare(st, emb); err != nil {
				return nil, err
			}
			if i > 0 {
				p.Restore(i-1, carry[j])
			}
			rows := min(q, s.t-s.pos)
			used := (s.pos + rows + att.Chunk - 1) / att.Chunk
			if err := st.stepMasks(s.pos, q, used); err != nil {
				return nil, err
			}
			if err := p.RunPart(i); err != nil {
				return nil, err
			}
			if j < n-1 || i < len(p.Attend) {
				// the windows this stage wrote, for the chunk after it (the last chunk's at
				// the end, by keep)
				s.keepWindows(p, q, len(emb), i)
			}
			if i < len(p.Attend) {
				oneMasks := false
				if err := s.attendAfter(st, i, len(emb), rows, used, &oneMasks); err != nil {
					return nil, err
				}
				carry[j] = p.Carry(i)
			}
			pos += len(emb)
		}
		if i == len(p.Attend) {
			break
		}
	}
	s.pos = lastPos
	return s.keep(p, q, len(chunks[n-1])), nil
}

// layerMajor returns the step that runs every chunk when StepChunks can run them a stage at a
// time: more than one chunk, all on one program split at its attention that hands its stages
// over, with logits, no state replaced as a whole or window input (whose rows a stage would
// need from the chunk before it in the same stage). The two-level attention keeps a state for each
// layer that moves with the positions, which a stage at a time still takes in order. The layers that attend for the last token alone (oneRow) do for each
// chunk's last token: only the last chunk's is kept.
func (s *Session) layerMajor(chunks [][][]float32) (*step, bool) {
	if len(chunks) < 2 {
		return nil, false
	}
	var st *step
	for _, c := range chunks {
		v := len(c)
		var pick *step
		for i := range s.progs {
			if s.progs[i].q >= v {
				pick = &s.progs[i]
				break
			}
		}
		if pick == nil || (st != nil && pick != st) {
			return nil, false
		}
		st = pick
	}
	p := st.p
	if p.RunPart == nil || p.Carry == nil || p.Restore == nil || p.NoLogits {
		return nil, false
	}
	for name := range p.In {
		if strings.HasPrefix(name, "st_") || strings.HasPrefix(name, "win_") {
			return nil, false
		}
	}
	return st, true
}

// keepWindows writes the step's rows (v of q, at s.pos) of the caches the parts attend over
// themselves (the windows: kvCache.chunk 0) that stage i computes, as keep does for all of them.
func (s *Session) keepWindows(p Program, q, v, stage int) {
	for _, c := range s.kv {
		if c.chunk > 0 {
			continue
		}
		if at, ok := p.OutStage[c.out]; !ok || at != stage {
			continue
		}
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
}
