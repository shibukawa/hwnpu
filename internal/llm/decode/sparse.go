package decode

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm/paged"
)

// Sparse is the two-level attention of a session split at its attention (SetSparse): past a
// length, each call of a layer reads the cache in two ways instead of whole. The cache before the
// window is cut into buckets of Bucket positions, each summarized by its keys' mean and its
// values' mean. The buckets that weigh most for the call's queries are read exactly, every other
// bucket as its summary, one key whose score has log Bucket added. The first Sinks positions and
// the last Window before the call are always read exactly, and so are the call's own.
//
// A bucket's weight is the softmax of each query row over the summaries, summed over the rows
// and the query heads of its kv head (internal/llm/paged's mass program), so that each kv head
// reads its own buckets and all the queries of a call share them. Checked on the CPU with
// LFM3-1.7B (docs/design.md 16.61): at 128K, with 16384 keys read exactly (32768 in the
// prompt) and 64 tokens per call, the next-token loss rose by 0.007 nats and pass phrases at
// 10, 50 and 90% of the context were still found; keys picked one by one (top-k), even the
// best 4096, lost them.
type Sparse struct {
	// Bucket is the positions a summary stands for.
	Bucket int
	// Sinks and Window are the first and the last positions always read exactly.
	Sinks, Window int
	// Exact is the chunks of keys a call of the shortest program (one that generates) reads
	// exactly: the buckets picked, the window and the call's own. Prefill is the same for the
	// longer programs (those that read a prompt), 0 for Exact.
	Exact, Prefill int
	// Refresh is the calls of a layer between two choices of the buckets, 1 for every call.
	Refresh int
}

// DefaultSparse is the two-level attention as checked: buckets of 8, 4 sinks and a window of
// 512, two chunks of 8192 read exactly when generating (the 16384 keys of the check) and four
// when reading a prompt (32768, where the errors would pile up), the buckets chosen anew on
// every call.
func DefaultSparse() Sparse {
	return Sparse{Bucket: 8, Sinks: 4, Window: 512, Exact: 2, Prefill: 4, Refresh: 1}
}

// Mass is the program without weights that weighs the summaries of one chunk of buckets for the
// call's queries (internal/llm/paged's mass program), for the tokens per step of the Program it
// belongs to.
type Mass struct {
	// Q and R are its inputs q, as the attention program's, and r (1, heads, tokens, 1); W is
	// its output w (1, kv_heads, 1, chunk), fp16.
	Q, R, W []byte
	// Run weighs the summaries' keys k (1, kv_heads, chunk, head_dim) with the mask (1, 1,
	// tokens, chunk), writing W.
	Run func(k, mask Buf) error
}

// sparse is the state of the two-level attention of a session.
type sparse struct {
	cfg    Sparse
	layers map[int]*sparseLayer
	calls  int // calls of the attention and mass programs, for the tests and the logs
}

// sparseLayer is a layer's summaries and the buckets it reads exactly.
type sparseLayer struct {
	n      int   // buckets summarized
	kb, vb []Buf // the summaries, (1, kv_heads, chunk, head_dim) per chunk of buckets
	pick   int   // buckets read exactly per kv head (the sinks' included); 0 for none chosen
	sel    [][]int32
	gk, gv []Buf // the keys and values read exactly: sel's buckets by slot, then the tail
	sk, sv Buf   // the summaries of sel's buckets past the sinks', which the merge takes out
	since  int   // calls since sel was chosen
	at     int   // the buckets there were when sel was chosen: fewer now (the sequence went back) choose again
	free   []int32
	in     []int8 // by bucket, while choosing: 1 chosen and not placed yet
}

// sparseStep is what a step's program needs besides: its masks and merges.
type sparseStep struct {
	smerge  *paged.Merge
	smasks  []rangeMask // by chunk of summaries
	gmasks  []Buf       // by chunk read exactly, for the tail
	selm    rangeMask
	norm    []float32
	r       []float32
	w       []float32
	mass    [][]float32 // by kv head, by bucket
	vals    []float32   // a mask's values
	gmaskAt []int       // by chunk read exactly: the first slot of the tail its mask was written for
	o       [3][]float32
}

// rangeMask is a mask that is log Bucket in the same span of a chunk for every query row, and
// -1e4 elsewhere, with the span it was written for.
type rangeMask struct {
	b      Buf
	lo, hi int
}

// SetSparse turns the two-level attention on for a session split at its attention whose
// programs have a mass program (Attention.Mass); a Bucket of 0 turns it off.
func (s *Session) SetSparse(sp Sparse) error {
	if sp.Bucket == 0 {
		s.sparse = nil
		return nil
	}
	if s.chunks == nil {
		return fmt.Errorf("decode: the two-level attention is for a session split at its attention")
	}
	c := s.progs[0].p.Attn.Chunk
	switch {
	case sp.Bucket < 1 || c%sp.Bucket != 0:
		return fmt.Errorf("decode: buckets of %d do not divide chunks of %d", sp.Bucket, c)
	case sp.Sinks < 0 || sp.Window < 0 || sp.Refresh < 1 || sp.Exact < 1 || sp.Prefill < 0:
		return fmt.Errorf("decode: two-level attention %+v", sp)
	}
	for _, st := range s.progs {
		if st.p.Attn.Mass == nil {
			return fmt.Errorf("decode: the program of %d tokens per step has no mass program", st.q)
		}
		if _, err := s.layout(sp, st.q, st.q == s.progs[0].q); err != nil {
			return err
		}
	}
	s.sparse = &sparse{cfg: sp, layers: map[int]*sparseLayer{}}
	return nil
}

// Sparse is the session's two-level attention, a Bucket of 0 when it is off.
func (s *Session) Sparse() Sparse {
	if s.sparse == nil {
		return Sparse{}
	}
	return s.sparse.cfg
}

// AttentionCalls is the calls of the attention and mass programs so far of the two-level
// attention (the whole cache's calls are not counted).
func (s *Session) AttentionCalls() int {
	if s.sparse == nil {
		return 0
	}
	return s.sparse.calls
}

// layout is the buckets a call of q tokens reads exactly per kv head, the sinks' included, so
// that they, the window and the call's own fill the chunks read exactly.
func (s *Session) layout(sp Sparse, q int, shortest bool) (int, error) {
	c := s.progs[0].p.Attn.Chunk
	n := sp.Exact
	if !shortest && sp.Prefill > 0 {
		n = sp.Prefill
	}
	pick := (n*c - (sp.Window + sp.Bucket + q)) / sp.Bucket
	if pick <= s.firstBuckets(sp) {
		return 0, fmt.Errorf("decode: %d chunks of %d read exactly hold no bucket past the sinks, the window of %d and %d tokens per call", n, c, sp.Window, q)
	}
	if pick-s.firstBuckets(sp) > c {
		// their summaries, which the merge takes out, are one chunk
		return 0, fmt.Errorf("decode: %d chunks of %d read exactly are over %d buckets", n, c, c)
	}
	return pick, nil
}

func (s *Session) firstBuckets(sp Sparse) int { return (sp.Sinks + sp.Bucket - 1) / sp.Bucket }

// sparseTouch forgets what the two-level attention holds of the positions from pos on: the cache
// is written there.
func (s *Session) sparseTouch(pos int) {
	if s.sparse == nil {
		return
	}
	S := s.sparse.cfg.Bucket
	for _, l := range s.sparse.layers {
		if pos < l.n*S {
			l.n = pos / S
			l.pick = 0 // its buckets may be past n: choose again
		}
	}
}

// attendSparse runs the two-level attention of a layer into st.res, the step's rows of the layer
// already in the cache, the layer's q in the attention program's input: its rows at the positions
// from pos on, rows of them in the cache. The buckets are chosen for the valid rows (the call's
// tokens, not its padding). It reports false, doing nothing, when reading the whole cache costs
// no more calls.
func (s *Session) attendSparse(st *step, layer int, kc, vc *kvCache, pos, rows, valid, used int) (bool, error) {
	sp := s.sparse
	cfg := sp.cfg
	att := st.p.Attn
	C, S, q := att.Chunk, cfg.Bucket, st.q
	nB := max(0, (pos-cfg.Window)/S)
	first := s.firstBuckets(cfg)
	pick, err := s.layout(cfg, q, st.q == s.progs[0].q)
	if err != nil {
		return false, err
	}
	if nB <= pick {
		return false, nil
	}
	nsc := (nB + C - 1) / C
	exact := (pick*S + pos + rows - nB*S + C - 1) / C
	if 2*nsc+1+exact >= used {
		return false, nil
	}
	l := sp.layers[layer]
	if l == nil {
		l = &sparseLayer{}
		sp.layers[layer] = l
	}
	if st.sp == nil {
		st.sp = s.newSparseStep(st)
	}
	ss := st.sp
	for len(ss.smasks) < nsc {
		ss.smasks = append(ss.smasks, rangeMask{})
	}
	if err := s.summarize(l, kc, vc, nB); err != nil {
		return false, err
	}
	run := func(k, v Buf, mask Buf) error {
		sp.calls++
		if err := att.Run(k, v, mask); err != nil {
			return err
		}
		hwnpu.DecodeF16(ss.o[0], att.O)
		hwnpu.DecodeF16(ss.o[1], att.M)
		hwnpu.DecodeF16(ss.o[2], att.L)
		return nil
	}
	// every summary but the sinks', which are read exactly
	ss.smerge.Reset()
	for j := range nsc {
		m, err := s.rangeMask(&ss.smasks[j], q, max(first-j*C, 0), min(nB-j*C, C))
		if err != nil {
			return false, err
		}
		if err := run(l.kb[j], l.vb[j], m); err != nil {
			return false, fmt.Errorf("decode: summaries of layer %d, chunk %d: %w", layer, j, err)
		}
		ss.smerge.Add(ss.o[0], ss.o[1], ss.o[2])
	}
	if l.pick != pick || l.since >= cfg.Refresh || nB < l.at {
		if err := s.choose(st, l, layer, kc, vc, nB, pick, valid); err != nil {
			return false, err
		}
	}
	l.since++
	// the tail: the window and the call's own, the same positions for every kv head
	tail := pos + rows - nB*S
	for _, c := range []struct {
		c *kvCache
		g []Buf
	}{{kc, l.gk}, {vc, l.gv}} {
		for i := range tail {
			pos := nB*S + i
			j, slot := c.c.locate(pos)
			src, err := s.part(c.c, j, false)
			if err != nil || src == nil {
				return false, fmt.Errorf("decode: position %d of layer %d is not in the cache", pos, layer)
			}
			gs := pick*S + i
			dst := c.g[gs/C].Bytes()
			row := gs % C
			for h := range c.c.heads {
				copy(dst[2*(h*C+row)*c.c.d:2*(h*C+row+1)*c.c.d], src[2*(h*C+slot)*c.c.d:2*(h*C+slot+1)*c.c.d])
			}
		}
	}
	st.merge.Reset()
	for g := range exact {
		mask := st.zero
		if (g+1)*C > pick*S {
			if mask, err = s.tailMask(ss, g, C, q, pick*S, nB*S, tail, pos); err != nil {
				return false, err
			}
		}
		if err := run(l.gk[g], l.gv[g], mask); err != nil {
			return false, fmt.Errorf("decode: exact keys of layer %d, chunk %d: %w", layer, g, err)
		}
		st.merge.Add(ss.o[0], ss.o[1], ss.o[2])
	}
	if pick > first {
		m, err := s.rangeMask(&ss.selm, q, 0, pick-first)
		if err != nil {
			return false, err
		}
		if err := run(l.sk, l.sv, m); err != nil {
			return false, fmt.Errorf("decode: summaries read exactly of layer %d: %w", layer, err)
		}
		ss.smerge.Sub(ss.o[0], ss.o[1], ss.o[2])
	}
	st.merge.AddMerge(ss.smerge)
	st.merge.Result(st.res)
	return true, nil
}

func (s *Session) newSparseStep(st *step) *sparseStep {
	att := st.p.Attn
	rows := len(att.M) / 2
	d := len(att.O) / len(att.M)
	kvh := len(att.Mass.W) / 2 / att.Chunk
	ss := &sparseStep{
		smerge: paged.NewMerge(rows, d),
		norm:   make([]float32, rows),
		r:      make([]float32, rows),
		w:      make([]float32, kvh*att.Chunk),
		mass:   make([][]float32, kvh),
		o:      [3][]float32{make([]float32, rows*d), make([]float32, rows), make([]float32, rows)},
	}
	return ss
}

// summarize makes the summaries of the buckets up to n.
func (s *Session) summarize(l *sparseLayer, kc, vc *kvCache, n int) error {
	if n <= l.n {
		return nil
	}
	C, S := s.progs[0].p.Attn.Chunk, s.sparse.cfg.Bucket
	for len(l.kb) < (n+C-1)/C {
		for _, x := range []*[]Buf{&l.kb, &l.vb} {
			b, err := s.progs[0].p.Attn.NewBuf(2 * kc.heads * C * kc.d)
			if err != nil {
				return fmt.Errorf("decode: a chunk of summaries: %w", err)
			}
			*x = append(*x, b)
		}
	}
	d := kc.d
	vals := make([]float32, S*d)
	mean := make([]float32, d)
	for b := l.n; b < n; b++ {
		j, slot := kc.locate(b * S)
		for _, c := range []struct {
			c   *kvCache
			out []Buf
		}{{kc, l.kb}, {vc, l.vb}} {
			src, err := s.part(c.c, j, false)
			if err != nil || src == nil {
				return fmt.Errorf("decode: bucket %d is not in the cache", b)
			}
			L := c.c.span(j)
			dst := c.out[b/C].Bytes()
			for h := range c.c.heads {
				hwnpu.DecodeF16(vals, src[2*(h*L+slot)*d:2*(h*L+slot+S)*d])
				clear(mean)
				for r := range S {
					for i, x := range vals[r*d : (r+1)*d] {
						mean[i] += x
					}
				}
				for i := range mean {
					mean[i] /= float32(S)
				}
				hwnpu.EncodeF16(dst[2*(h*C+b%C)*d:2*(h*C+b%C+1)*d], mean)
			}
		}
	}
	l.n = n
	return nil
}

// choose weighs the summaries of buckets first..n for the call's queries (the attention over
// them just run, its normalizers in st.sp.smerge) and reads the heaviest pick-first of each kv
// head exactly, after the sinks' buckets: their rows go into the chunks read exactly, and their
// summaries into sk and sv.
func (s *Session) choose(st *step, l *sparseLayer, layer int, kc, vc *kvCache, n, pick, valid int) error {
	att, ss := st.p.Attn, st.sp
	mass := att.Mass
	C, S, q := att.Chunk, s.sparse.cfg.Bucket, st.q
	first := s.firstBuckets(s.sparse.cfg)
	G, d := kc.heads, kc.d
	ss.smerge.Norm(ss.norm)
	heads := len(ss.norm) / q
	for h := range heads {
		for t := range q {
			i := h*q + t
			ss.r[i] = -1e4
			if t < valid && !math.IsInf(float64(ss.norm[i]), -1) {
				ss.r[i] = -ss.norm[i]
			}
		}
	}
	hwnpu.EncodeF16(mass.R, ss.r)
	copy(mass.Q, att.Q)
	for g := range G {
		if cap(ss.mass[g]) < n {
			ss.mass[g] = make([]float32, n, 2*n)
		}
		ss.mass[g] = ss.mass[g][:n]
	}
	for j := range (n + C - 1) / C {
		m, err := s.rangeMask(&ss.smasks[j], q, max(first-j*C, 0), min(n-j*C, C))
		if err != nil {
			return err
		}
		s.sparse.calls++
		if err := mass.Run(l.kb[j], m); err != nil {
			return fmt.Errorf("decode: weights of the summaries of layer %d, chunk %d: %w", layer, j, err)
		}
		hwnpu.DecodeF16(ss.w, mass.W)
		for g := range G {
			copy(ss.mass[g][j*C:min(n, (j+1)*C)], ss.w[g*C:g*C+min(n-j*C, C)])
		}
	}
	rebuild := l.pick != pick
	if rebuild {
		l.sel = make([][]int32, G)
		for g := range G {
			l.sel[g] = make([]int32, pick)
			for i := range l.sel[g] {
				l.sel[g][i] = -1
			}
		}
		l.pick = pick
		exact := (pick*S + s.sparse.cfg.Window + S + q + C - 1) / C
		for len(l.gk) < exact {
			for _, x := range []*[]Buf{&l.gk, &l.gv} {
				b, err := att.NewBuf(2 * G * C * d)
				if err != nil {
					return fmt.Errorf("decode: a chunk of keys read exactly: %w", err)
				}
				*x = append(*x, b)
			}
		}
		if l.sk == nil {
			var err error
			if l.sk, err = att.NewBuf(2 * G * C * d); err != nil {
				return err
			}
			if l.sv, err = att.NewBuf(2 * G * C * d); err != nil {
				return err
			}
		}
	}
	if len(l.in) < n {
		l.in = make([]int8, n)
	}
	in := l.in[:n]
	for g := range G {
		top := topK(ss.mass[g][first:], pick-first)
		clear(in)
		for b := range first {
			in[b] = 1
		}
		for _, b := range top {
			in[int(b)+first] = 1
		}
		// keep the buckets chosen again where they are, put the new ones in the slots freed
		sel := l.sel[g]
		l.free = l.free[:0]
		for i, b := range sel {
			if b >= 0 && int(b) < n && in[b] == 1 {
				in[b] = 0
				continue
			}
			l.free = append(l.free, int32(i))
		}
		f := 0
		for b := range n {
			if in[b] != 1 {
				continue
			}
			i := l.free[f]
			f++
			sel[i] = int32(b)
			for _, c := range []struct {
				c   *kvCache
				out []Buf
			}{{kc, l.gk}, {vc, l.gv}} {
				j, slot := c.c.locate(b * S)
				src, err := s.part(c.c, j, false)
				if err != nil || src == nil {
					return fmt.Errorf("decode: bucket %d of layer %d is not in the cache", b, layer)
				}
				L := c.c.span(j)
				dst := c.out[int(i)*S/C].Bytes()
				row := int(i) * S % C
				copy(dst[2*(g*C+row)*d:2*(g*C+row+S)*d], src[2*(g*L+slot)*d:2*(g*L+slot+S)*d])
			}
		}
		// the summaries of the buckets read exactly past the sinks', in any order
		k := 0
		for _, b := range sel {
			if int(b) < first {
				continue
			}
			for _, c := range []struct {
				from []Buf
				to   Buf
			}{{l.kb, l.sk}, {l.vb, l.sv}} {
				src := c.from[int(b)/C].Bytes()
				copy(c.to.Bytes()[2*(g*C+k)*d:2*(g*C+k+1)*d], src[2*(g*C+int(b)%C)*d:2*(g*C+int(b)%C+1)*d])
			}
			k++
		}
	}
	l.since, l.at = 0, n
	return nil
}

// rangeMask returns the mask m, allocated or rewritten when its span is not lo..hi: log Bucket
// there for every one of the q rows, -1e4 elsewhere.
func (s *Session) rangeMask(m *rangeMask, q, lo, hi int) (Buf, error) {
	C := s.progs[0].p.Attn.Chunk
	if m.b != nil && m.lo == lo && m.hi == hi {
		return m.b, nil
	}
	// only the columns whose value changes are written (the span grows by a bucket at a time)
	a, b := min(lo, m.lo), max(hi, m.hi)
	if m.b == nil {
		var err error
		if m.b, err = s.progs[0].p.Attn.NewBuf(2 * q * C); err != nil {
			return nil, err
		}
		a, b = 0, C
	}
	a, b = max(a, 0), min(b, C)
	logS := float32(math.Log(float64(s.sparse.cfg.Bucket)))
	row := make([]float32, max(b-a, 0))
	for i := range row {
		row[i] = -1e4
		if a+i >= lo && a+i < hi {
			row[i] = logS
		}
	}
	enc := make([]byte, 2*len(row))
	hwnpu.EncodeF16(enc, row)
	for t := range q {
		copy(m.b.Bytes()[2*(t*C+a):2*(t*C+b)], enc)
	}
	m.lo, m.hi = lo, hi
	return m.b, nil
}

// tailMask writes the mask of chunk g read exactly: its slots before tail0 (the buckets picked)
// seen by every row, then the tail, the positions from pos0 on, seen by the rows at or after
// them (row t at position pos+t), and nothing after.
func (s *Session) tailMask(ss *sparseStep, g, C, q, tail0, pos0, tail, pos int) (Buf, error) {
	for len(ss.gmasks) <= g {
		ss.gmasks = append(ss.gmasks, nil)
		ss.gmaskAt = append(ss.gmaskAt, -1)
	}
	if ss.gmasks[g] == nil {
		b, err := s.progs[0].p.Attn.NewBuf(2 * q * C)
		if err != nil {
			return nil, err
		}
		ss.gmasks[g] = b
	}
	// the whole chunk the first time for these buckets, then only the columns the tail can reach
	// (the window, a bucket and the call's own)
	a, b := 0, C
	if ss.gmaskAt[g] == tail0 {
		a = max(tail0-g*C, 0)
		b = min(tail0+s.sparse.cfg.Window+s.sparse.cfg.Bucket+q-g*C, C)
	}
	ss.gmaskAt[g] = tail0
	if b <= a {
		return ss.gmasks[g], nil
	}
	if len(ss.vals) < b-a {
		ss.vals = make([]float32, C)
	}
	vals := ss.vals[:b-a]
	for t := range q {
		for x := a; x < b; x++ {
			gs := g*C + x
			v := float32(-1e4)
			switch {
			case gs < tail0:
				v = 0
			case gs < tail0+tail && pos0+gs-tail0 <= pos+t:
				v = 0
			}
			vals[x-a] = v
		}
		hwnpu.EncodeF16(ss.gmasks[g].Bytes()[2*(t*C+a):2*(t*C+b)], vals)
	}
	return ss.gmasks[g], nil
}

// topK returns the indices of the k largest values (all of them when there are no more), in no
// order, by selection rather than a sort.
func topK(v []float32, k int) []int32 {
	if k >= len(v) {
		out := make([]int32, len(v))
		for i := range out {
			out[i] = int32(i)
		}
		return out
	}
	if k <= 0 {
		return nil
	}
	tmp := append([]float32(nil), v...)
	for i, x := range tmp {
		if x != x { // NaN: weighs nothing
			tmp[i], v[i] = -1, -1
		}
	}
	thr := kth(tmp, len(tmp)-k) // the k-th largest
	out := make([]int32, 0, k)
	for i, x := range v {
		if x > thr {
			out = append(out, int32(i))
		}
	}
	for i, x := range v {
		if len(out) == k {
			break
		}
		if x == thr {
			out = append(out, int32(i))
		}
	}
	return out
}

// kth returns the value that would be at index i of v sorted ascending, reordering v.
func kth(v []float32, i int) float32 {
	lo, hi := 0, len(v)-1
	for lo < hi {
		p := v[(lo+hi)/2]
		a, b := lo, hi
		for a <= b {
			for v[a] < p {
				a++
			}
			for v[b] > p {
				b--
			}
			if a <= b {
				v[a], v[b] = v[b], v[a]
				a++
				b--
			}
		}
		switch {
		case i <= b:
			hi = b
		case i >= a:
			lo = a
		default:
			return v[i]
		}
	}
	return v[i]
}

// sparseBufs is the buffers the two-level attention holds.
func (sp *sparse) bufs() []Buf {
	var out []Buf
	for _, l := range sp.layers {
		out = append(out, l.kb...)
		out = append(out, l.vb...)
		out = append(out, l.gk...)
		out = append(out, l.gv...)
		if l.sk != nil {
			out = append(out, l.sk, l.sv)
		}
	}
	return out
}

func (ss *sparseStep) bufs() []Buf {
	var out []Buf
	for _, m := range ss.smasks {
		if m.b != nil {
			out = append(out, m.b)
		}
	}
	for _, b := range ss.gmasks {
		if b != nil {
			out = append(out, b)
		}
	}
	if ss.selm.b != nil {
		out = append(out, ss.selm.b)
	}
	return out
}
