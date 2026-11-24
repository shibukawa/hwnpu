package decode

import (
	"fmt"
	"slices"
	"time"

	"github.com/shibukawa/bonsaiai/internal/llm"
)

// Marked is a point of the sequence that Steps marked: after one of its steps, with the logits
// after it (nil from a program that gives none).
type Marked struct {
	Mark   *Mark
	Logits []float32
}

// waveBytes is the most a wave holds of what its steps hand from one stage to the next (the
// hidden state and the attention's output of each: 2.6 MB a step of 128 tokens of Qwen3.6-35B).
const waveBytes = 64 << 20

// Steps runs several steps one after another, each step's tokens as Step takes them, and returns
// the logits after the last. It marks the sequence after the steps whose indices marked lists,
// in order. check, when not nil, is asked between the calls of the programs whether to go on;
// after its error, as after any other, the caches are half written.
//
// The steps of a program split at its attention that tells what its stages hand on
// (Program.Carry) run a stage at a time, a wave: every step through the parts before the first
// attention layer, then every step through the next stage, and so on. The values are the ones of
// running the steps in turn (each layer still sees the steps in order; only a cache that moved to
// longer chunks sooner rounds differently). What it changes is which weights are in use together:
// a stage's alone for the whole wave. A model whose mixture of experts layers run on the host
// from a mapped file larger than memory (llm.HostRouted: a step of 128 tokens chooses nearly
// every expert of a layer) then reads a stage's experts once for the wave rather than every
// layer's for every step. A program with RunMany orders the programs inside a stage itself: each
// for every step before the next, so that a layer's experts run on the rows of all the steps.
func (s *Session) Steps(steps [][][]float32, marked []int, check func() error) ([]float32, []Marked, error) {
	var logits []float32
	var marks []Marked
	var hidden [][]float32
	for i := 0; i < len(steps); {
		if check != nil {
			if err := check(); err != nil {
				return nil, nil, err
			}
		}
		st, n := s.wave(steps[i:])
		if n < 2 {
			var err error
			if logits, err = s.Step(steps[i]); err != nil {
				return nil, nil, err
			}
			hidden = append(hidden, s.lastRows...)
			if slices.Contains(marked, i) {
				marks = append(marks, Marked{s.Mark(), slices.Clone(logits)})
			}
			i++
			continue
		}
		var at []int
		for _, m := range marked {
			if m >= i && m < i+n {
				at = append(at, m-i)
			}
		}
		lg, ms, err := s.runWave(st, steps[i:i+n], at, check)
		if err != nil {
			return nil, nil, err
		}
		logits, marks, hidden = lg, append(marks, ms...), append(hidden, s.lastRows...)
		i += n
	}
	if hidden != nil {
		s.lastRows, s.last = hidden, hidden[len(hidden)-1]
	}
	return logits, marks, nil
}

// Waves reports whether the session's program of the most tokens per step can run several steps
// a stage at a time (Steps).
func (s *Session) Waves() bool {
	st := &s.progs[len(s.progs)-1]
	return st.waves(s)
}

// waves reports whether st's program can run waves: it is split at its attention, tells what its
// stages hand on, and the session holds every cache itself (one in the program's own inputs is
// written at a step's end).
func (st *step) waves(s *Session) bool {
	if st.p.RunPart == nil || st.p.Carry == nil || st.p.StageIn == nil || len(st.p.Attend) == 0 {
		return false
	}
	for _, c := range s.kv {
		if c.chunk == 0 {
			return false
		}
	}
	return true
}

// wave is the program the first of steps runs on and how many of them, from the first, run as
// one wave: 1 when the program cannot (it is not split at its attention, does not tell what its
// stages hand on, or keeps a cache in its own inputs, which a step writes at its end), else those
// that run on the same program and fit the cache, as many as waveBytes lets it hold.
func (s *Session) wave(steps [][][]float32) (*step, int) {
	st := s.pick(len(steps[0]), false)
	if st == nil || !st.waves(s) {
		return st, 1
	}
	if st.carry == 0 {
		for i := range st.p.Attend {
			n := 0
			for _, b := range st.p.Carry(i) {
				n += len(b)
			}
			st.carry = max(st.carry, n, 1)
		}
	}
	most := max(waveBytes/st.carry, 2)
	n, pos := 0, s.pos
	for n < len(steps) && n < most {
		v := len(steps[n])
		if v == 0 || pos+v > s.t || s.pick(v, false) != st {
			break
		}
		pos += v
		n++
	}
	return st, n
}

// runWave runs the steps on st's program a stage at a time (Steps), and marks the sequence after
// those at lists.
func (s *Session) runWave(st *step, steps [][][]float32, at []int, check func() error) ([]float32, []Marked, error) {
	p, q := st.p, st.q
	n := len(steps)
	start := make([]int, n+1) // each step's position
	start[0] = s.pos
	for c, e := range steps {
		start[c+1] = start[c] + len(e)
	}
	// on any return the sequence is where the wave began or, done, where it ends
	pos := s.pos
	defer func() { s.pos = pos }()
	if err := s.fit(start[n-1] + min(q, s.t-start[n-1])); err != nil {
		return nil, nil, err
	}
	s.sparseTouch(s.pos)
	att := p.attention(s.chunk)
	marks := make([]Marked, len(at))
	for k := range marks {
		marks[k].Mark = &Mark{pos: start[at[k]+1], bufs: map[string][]byte{}}
	}
	held := make([][][]byte, n) // by step, what the stage before handed on
	var prof *profSteps
	for range steps { // the profile's times are by step
		prof = profStep(profKeyOf(p))
	}
	var logits []float32
	s.last, s.lastRows = nil, nil
	for i := 0; i <= len(p.Attend); i++ {
		var carry [][]byte
		if i > 0 {
			carry = p.Carry(i - 1)
		}
		var next [][]byte
		if i < len(p.Attend) {
			next = p.Carry(i)
		}
		var t0 time.Time
		// the step's inputs of part i, before each of its programs runs for step c
		step := func(c int) error {
			if check != nil {
				if err := check(); err != nil {
					return err
				}
			}
			s.pos = start[c]
			if err := s.writeInputs(st, steps[c]); err != nil {
				return err
			}
			for j, b := range carry {
				copy(b, held[c][j])
			}
			t0 = time.Now()
			return nil
		}
		// after one of its programs ran for step c: the states it took are kept, and after the
		// last the attention, or the step's logits
		ran := func(c int, names []string, last bool) error {
			if prof != nil {
				prof.part(i, time.Since(t0))
				t0 = time.Now()
			}
			v := len(steps[c])
			for _, name := range names {
				keepState(p, name, v)
			}
			for k, m := range at {
				if m != c {
					continue
				}
				for _, name := range names {
					if llm.HeldInput(name) {
						marks[k].Mark.bufs[name] = slices.Clone(p.In[name])
					}
				}
			}
			if !last {
				return nil
			}
			rows := min(q, s.t-s.pos)
			if i == len(p.Attend) {
				if p.NoLogits {
					if err := s.writeUnattended(p, q, rows); err != nil {
						return err
					}
				}
				r := s.keepHidden(p, q, v)
				if p.NoLogits {
					return nil
				}
				marked := slices.Contains(at, c)
				if c == n-1 || marked {
					logits = s.row(p, r)
				}
				if marked {
					marks[slices.Index(at, c)].Logits = slices.Clone(logits)
				}
				return nil
			}
			used := (s.pos + rows + att.Chunk - 1) / att.Chunk
			if err := st.stepMasks(s.pos, q, used); err != nil {
				return err
			}
			oneMasks := false
			if err := s.attend(st, att, i, rows, v, used, &oneMasks); err != nil {
				return err
			}
			old := held[c][:cap(held[c])] // their memory again, where it is as large
			held[c] = held[c][:0]
			for j, b := range next {
				var d []byte
				if j < len(old) {
					d = old[j][:0]
				}
				held[c] = append(held[c], append(d, b...))
			}
			if prof != nil {
				prof.attend[i] += time.Since(t0)
			}
			return nil
		}
		if p.RunMany != nil {
			begin := time.Now()
			var parts, attended time.Duration
			if prof != nil {
				prof.part(i, 0)
				parts, attended = prof.parts[i], prof.attend[i]
			}
			if err := p.RunMany(i, n, step, ran); err != nil {
				return nil, nil, err
			}
			if prof != nil {
				// what ran between the programs' runs is the part's too (the experts of all
				// the steps together)
				prof.parts[i] = parts + time.Since(begin) - (prof.attend[i] - attended)
			}
			continue
		}
		names := p.StageIn(i)
		for c := range steps {
			if err := step(c); err != nil {
				return nil, nil, err
			}
			if err := p.RunPart(i); err != nil {
				return nil, nil, err
			}
			if err := ran(c, names, true); err != nil {
				return nil, nil, err
			}
		}
	}
	for _, m := range marks {
		for name := range p.In {
			if llm.HeldInput(name) && m.Mark.bufs[name] == nil {
				return nil, nil, fmt.Errorf("decode: no stage of the program takes the state %s", name)
			}
		}
	}
	pos = start[n]
	if p.NoLogits {
		logits = nil
	}
	return logits, marks, nil
}
