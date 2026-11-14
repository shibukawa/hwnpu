package decode

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
)

// The fake model of these tests has one layer whose attention the host runs: its queries, keys
// and values at each position come from tables, the position read from x, and the attention's
// output is kept for the test to compare with a two-level attention computed directly.

const (
	fH, fG, fD = 4, 2, 8 // heads, kv heads, head size
	fChunk     = 64
)

type fakeModel struct{ llm.Paged }

func (fakeModel) StepInputs(pos, valid, q, cache int) map[string][]float32      { return nil }
func (fakeModel) PagedStepInputs(pos, valid, q, cache int) map[string][]float32 { return nil }
func (fakeModel) PagedKV(layer int) int                                         { return layer }

type cpuBuf []byte

func (b cpuBuf) Bytes() []byte { return b }
func (b cpuBuf) Close() error  { return nil }

// tables are the fake layer's queries, keys and values by position.
type tables struct {
	q, k, v [][]float32 // by position: (heads or kv heads) × fD
}

func newTables(n int, seed uint64) *tables {
	rng := rand.New(rand.NewPCG(seed, 1))
	t := &tables{}
	for range n {
		row := func(heads int, scale float64) []float32 {
			x := make([]float32, heads*fD)
			for i := range x {
				x[i] = float32(scale * rng.NormFloat64())
			}
			return x
		}
		t.q = append(t.q, row(fH, 2))
		t.k = append(t.k, row(fG, 1))
		t.v = append(t.v, row(fG, 1))
	}
	return t
}

// fakeProgram is a program of q tokens per step over a context of n, its attention's output after
// each step in *out (heads × q × fD).
func fakeProgram(q, n int, tab **tables, out *[]float32, mass bool) Program {
	p := Program{
		In:      map[string][]byte{"x": make([]byte, 2*q), "attn_o_1": make([]byte, 2*fH*q*fD)},
		InShape: map[string][]int{"x": {1, q, 1}, "attn_o_1": {1, fH, q, fD}},
		Out: map[string][]byte{"y": make([]byte, 2*q), "attn_q_1": make([]byte, 2*fH*q*fD),
			"kv_k_1_new": make([]byte, 2*fG*q*fD), "kv_v_1_new": make([]byte, 2*fG*q*fD)},
		OutShape: map[string][]int{"y": {1, q, 1}, "attn_q_1": {1, fH, q, fD},
			"kv_k_1_new": {1, fG, q, fD}, "kv_v_1_new": {1, fG, q, fD}},
		Attend: []int{1}, Context: n,
	}
	x := make([]float32, q)
	p.RunPart = func(i int) error {
		if i == 1 {
			*out = make([]float32, fH*q*fD)
			hwnpu.DecodeF16(*out, p.In["attn_o_1"])
			return nil
		}
		hwnpu.DecodeF16(x, p.In["x"])
		put := func(name string, heads int, rows [][]float32) {
			vals := make([]float32, heads*q*fD)
			for t := range q {
				r := rows[int(x[t])]
				for h := range heads {
					copy(vals[(h*q+t)*fD:(h*q+t+1)*fD], r[h*fD:(h+1)*fD])
				}
			}
			hwnpu.EncodeF16(p.Out[name], vals)
		}
		put("attn_q_1", fH, (*tab).q)
		put("kv_k_1_new", fG, (*tab).k)
		put("kv_v_1_new", fG, (*tab).v)
		return nil
	}
	p.Attn = fakeAttention(q, mass)
	return p
}

func dec(b []byte) []float32 {
	x := make([]float32, len(b)/2)
	hwnpu.DecodeF16(x, b)
	return x
}

// fakeAttention is the attention program (and the mass program) in Go, its outputs in fp16.
func fakeAttention(q int, mass bool) *Attention { return fakeAttentionOf(q, fChunk, mass) }

// fakeAttentionOf is fakeAttention over chunks of chunk positions (with a mass program: fChunk).
func fakeAttentionOf(q, chunk int, mass bool) *Attention {
	a := &Attention{Chunk: chunk, Q: make([]byte, 2*fH*q*fD), O: make([]byte, 2*fH*q*fD), M: make([]byte, 2*fH*q), L: make([]byte, 2*fH*q)}
	a.NewBuf = func(n int) (Buf, error) { return cpuBuf(make([]byte, n)), nil }
	scale := 1 / math.Sqrt(fD)
	scores := func(qs, ks, mask []float32, h, t int) []float64 {
		g := h / (fH / fG)
		sc := make([]float64, chunk)
		for c := range chunk {
			var dot float64
			for i := range fD {
				dot += float64(qs[(h*q+t)*fD+i]) * float64(ks[(g*chunk+c)*fD+i])
			}
			sc[c] = dot*scale + float64(mask[t*chunk+c])
		}
		return sc
	}
	a.Run = func(k, v, mask Buf) error {
		qs, ks, vs, ms := dec(a.Q), dec(k.Bytes()), dec(v.Bytes()), dec(mask.Bytes())
		o, m, l := make([]float32, fH*q*fD), make([]float32, fH*q), make([]float32, fH*q)
		for h := range fH {
			g := h / (fH / fG)
			for t := range q {
				sc := scores(qs, ks, ms, h, t)
				top := slices.Max(sc)
				var sum float64
				acc := make([]float64, fD)
				for c, s := range sc {
					e := math.Exp(s - top)
					sum += e
					for i := range fD {
						acc[i] += e * float64(vs[(g*chunk+c)*fD+i])
					}
				}
				for i := range fD {
					o[(h*q+t)*fD+i] = float32(acc[i] / sum)
				}
				m[h*q+t], l[h*q+t] = float32(top), float32(sum)
			}
		}
		hwnpu.EncodeF16(a.O, o)
		hwnpu.EncodeF16(a.M, m)
		hwnpu.EncodeF16(a.L, l)
		return nil
	}
	if mass {
		ms := &Mass{Q: make([]byte, 2*fH*q*fD), R: make([]byte, 2*fH*q), W: make([]byte, 2*fG*fChunk)}
		ms.Run = func(k, mask Buf) error {
			qs, ks, mk, r := dec(ms.Q), dec(k.Bytes()), dec(mask.Bytes()), dec(ms.R)
			w := make([]float32, fG*fChunk)
			for h := range fH {
				g := h / (fH / fG)
				for t := range q {
					for c, s := range scores(qs, ks, mk, h, t) {
						w[g*fChunk+c] += float32(math.Exp(s + float64(r[h*q+t])))
					}
				}
			}
			hwnpu.EncodeF16(ms.W, w)
			return nil
		}
		a.Mass = ms
	}
	return a
}

// twoLevel is the two-level attention computed directly, as the CPU check of LFM3 did it, for
// the rows of a call of q tokens from pos, valid of them: heads × q × fD. The buckets read
// exactly past the sinks' are chosen[g] for kv head g, or when chosen is nil the heaviest. It
// also returns the buckets' weights and the heaviest buckets by kv head.
func twoLevel(tab *tables, sp Sparse, pick, pos, valid, q int, chosen [][]int) ([]float32, [][]float64, [][]int) {
	S := sp.Bucket
	nB := max(0, (pos-sp.Window)/S)
	first := (sp.Sinks + S - 1) / S
	scale := 1 / math.Sqrt(fD)
	dot := func(a, b []float32) float64 {
		var s float64
		for i := range a {
			s += float64(a[i]) * float64(b[i])
		}
		return s * scale
	}
	r16 := func(x []float32) []float32 { // through fp16, as the cache holds them
		b := make([]byte, 2*len(x))
		hwnpu.EncodeF16(b, x)
		return dec(b)
	}
	mean := func(rows [][]float32, g, b int) []float32 {
		m := make([]float32, fD)
		for p := b * S; p < (b+1)*S; p++ {
			x := r16(rows[p][g*fD : (g+1)*fD])
			for i := range m {
				m[i] += x[i] / float32(S)
			}
		}
		return m
	}
	out := make([]float32, fH*q*fD)
	logS := math.Log(float64(S))
	weights, heaviest := make([][]float64, fG), make([][]int, fG)
	for g := range fG {
		kb, vb := make([][]float32, nB), make([][]float32, nB)
		for b := range nB {
			kb[b], vb[b] = mean(tab.k, g, b), mean(tab.v, g, b)
		}
		// the weight of each bucket past the sinks': each row's softmax over them, summed
		w := make([]float64, nB)
		for h := g * fH / fG; h < (g+1)*fH/fG; h++ {
			for t := range valid {
				qr := r16(tab.q[pos+t][h*fD : (h+1)*fD])
				sc := make([]float64, nB)
				top := math.Inf(-1)
				for b := first; b < nB; b++ {
					sc[b] = dot(qr, kb[b]) + logS
					top = max(top, sc[b])
				}
				var sum float64
				for b := first; b < nB; b++ {
					sum += math.Exp(sc[b] - top)
				}
				for b := first; b < nB; b++ {
					w[b] += math.Exp(sc[b]-top) / sum
				}
			}
		}
		order := make([]int, 0, nB)
		for b := first; b < nB; b++ {
			order = append(order, b)
		}
		slices.SortStableFunc(order, func(a, b int) int {
			switch {
			case w[a] > w[b]:
				return -1
			case w[a] < w[b]:
				return 1
			}
			return 0
		})
		weights[g], heaviest[g] = w, order[:pick-first]
		use := order[:pick-first]
		if chosen != nil {
			use = chosen[g]
		}
		exact := map[int]bool{}
		for b := range first {
			exact[b] = true
		}
		for _, b := range use {
			exact[b] = true
		}
		for h := g * fH / fG; h < (g+1)*fH/fG; h++ {
			for t := range valid {
				qr := r16(tab.q[pos+t][h*fD : (h+1)*fD])
				var keys, vals [][]float32
				var bias []float64
				for b := range nB {
					if exact[b] {
						for p := b * S; p < (b+1)*S; p++ {
							keys, vals, bias = append(keys, r16(tab.k[p][g*fD:(g+1)*fD])), append(vals, r16(tab.v[p][g*fD:(g+1)*fD])), append(bias, 0)
						}
					} else if b >= first {
						keys, vals, bias = append(keys, kb[b]), append(vals, vb[b]), append(bias, logS)
					}
				}
				for p := nB * S; p <= pos+t; p++ {
					keys, vals, bias = append(keys, r16(tab.k[p][g*fD:(g+1)*fD])), append(vals, r16(tab.v[p][g*fD:(g+1)*fD])), append(bias, 0)
				}
				sc := make([]float64, len(keys))
				for i := range keys {
					sc[i] = dot(qr, keys[i]) + bias[i]
				}
				top := slices.Max(sc)
				var sum float64
				acc := make([]float64, fD)
				for i, s := range sc {
					e := math.Exp(s - top)
					sum += e
					for j := range fD {
						acc[j] += e * float64(vals[i][j])
					}
				}
				for j := range fD {
					out[(h*q+t)*fD+j] = float32(acc[j] / sum)
				}
			}
		}
	}
	return out, weights, heaviest
}

func testSparse() Sparse { return Sparse{Bucket: 4, Sinks: 2, Window: 16, Exact: 2, Refresh: 1} }

func relErr(got, want []float32, q, valid int) float64 {
	var num, den float64
	for h := range fH {
		for t := range valid {
			for i := range fD {
				j := (h*q+t)*fD + i
				e := float64(got[j] - want[j])
				num += e * e
				den += float64(want[j]) * float64(want[j])
			}
		}
	}
	return math.Sqrt(num / den)
}

// exactAttention is the attention over the whole cache, computed directly.
func exactAttention(tab *tables, pos, valid, q int) []float32 {
	sp := Sparse{Bucket: 1, Window: pos + q}
	out, _, _ := twoLevel(tab, sp, 0, pos, valid, q, nil)
	return out
}

// chosen is the buckets the session reads exactly past the sinks', by kv head.
func chosen(s *Session, layer int) [][]int {
	l := s.sparse.layers[layer]
	first := s.firstBuckets(s.sparse.cfg)
	out := make([][]int, len(l.sel))
	for g, sel := range l.sel {
		for _, b := range sel {
			if int(b) >= first {
				out[g] = append(out[g], int(b))
			}
		}
	}
	return out
}

// steps runs the tokens from pos to end, up to q at a time, through the session, checking each call
// past start: a call of the two-level attention against it computed directly with the buckets
// the session chose, and those buckets against the heaviest (fp16's rounding of the weights may
// swap two of nearly the same weight); a call over the whole cache against the exact attention.
// It returns the calls of the two-level attention checked.
func steps(t *testing.T, s *Session, tab *tables, out *[]float32, pos, end, q, start int, pick int) int {
	t.Helper()
	worst, short := 0.0, 0.0
	sparse := 0
	for i := 0; pos < end; i++ {
		v := min(q, end-pos, 1+i%(q+2)) // calls of fewer tokens than the program takes, padded
		emb := make([][]float32, v)
		for i := range emb {
			emb[i] = []float32{float32(pos + i)}
		}
		calls := s.AttentionCalls()
		if _, err := s.Step(emb); err != nil {
			t.Fatal(err)
		}
		// the program run: the one of fewest tokens per call that takes them
		q := q
		for _, st := range s.progs {
			if st.q >= v {
				q = st.q
				break
			}
		}
		pick, _ := s.layout(s.sparse.cfg, q, q == s.progs[0].q)
		if pos >= start {
			if s.AttentionCalls() == calls {
				worst = max(worst, relErr(*out, exactAttention(tab, pos, v, q), q, v))
			} else {
				sel := chosen(s, 1)
				want, w, top := twoLevel(tab, s.sparse.cfg, pick, pos, v, q, sel)
				worst = max(worst, relErr(*out, want, q, v))
				for g := range sel {
					var got, best float64
					for _, b := range sel[g] {
						got += w[g][b]
					}
					for _, b := range top[g] {
						best += w[g][b]
					}
					short = max(short, 1-got/best)
				}
				sparse++
			}
		}
		pos += v
	}
	if worst > 1e-2 || math.IsNaN(worst) {
		t.Errorf("the attention differs from the direct one by up to %g (relative)", worst)
	}
	if short > 1e-2 {
		t.Errorf("the buckets chosen weigh up to %.3g less than the heaviest", short)
	}
	return sparse
}

// The session's two-level attention is the direct one: past the length where it costs fewer
// calls than the whole cache, over a sequence long enough for several chunks of summaries, on
// every call of one token and of several, the call's padding rows aside.
func TestSparseMatchesDirect(t *testing.T) {
	for _, q := range []int{1, 4} {
		const n = 1200
		tab := newTables(n, 1)
		var out []float32
		s, err := New(fakeProgram(q, n, &tab, &out, true), fakeModel{})
		if err != nil {
			t.Fatal(err)
		}
		if err := s.SetSparse(testSparse()); err != nil {
			t.Fatal(err)
		}
		pick, _ := s.layout(testSparse(), q, true)
		if steps(t, s, tab, &out, 0, n, q, 0, pick) == 0 {
			t.Fatal("the two-level attention never ran")
		}
	}
}

// Below the length where it pays, the session reads the whole cache: the attention is exact.
func TestSparseShortIsExact(t *testing.T) {
	const n, q = 200, 4
	tab := newTables(n, 2)
	var out []float32
	s, err := New(fakeProgram(q, n, &tab, &out, true), fakeModel{})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetSparse(testSparse()); err != nil {
		t.Fatal(err)
	}
	for pos := 0; pos < n; pos += q {
		emb := [][]float32{{float32(pos)}, {float32(pos + 1)}, {float32(pos + 2)}, {float32(pos + 3)}}
		if _, err := s.Step(emb); err != nil {
			t.Fatal(err)
		}
	}
	if s.AttentionCalls() != 0 {
		t.Fatalf("%d calls of the two-level attention over %d tokens", s.AttentionCalls(), n)
	}
}

// Choosing the buckets every few calls only (Refresh) keeps the choice in between: the buckets
// change on every fourth call of the two-level attention at most, and each call is the direct
// attention with the buckets the session holds.
func TestSparseRefresh(t *testing.T) {
	const n, q = 1200, 4
	tab := newTables(n, 3)
	var out []float32
	s, err := New(fakeProgram(q, n, &tab, &out, true), fakeModel{})
	if err != nil {
		t.Fatal(err)
	}
	sp := testSparse()
	sp.Refresh = 4
	if err := s.SetSparse(sp); err != nil {
		t.Fatal(err)
	}
	pick, _ := s.layout(sp, q, true)
	var last [][]int
	calls, changes, worst := 0, 0, 0.0
	for pos := 0; pos < n; pos += q {
		before := s.AttentionCalls()
		emb := [][]float32{{float32(pos)}, {float32(pos + 1)}, {float32(pos + 2)}, {float32(pos + 3)}}
		if _, err := s.Step(emb); err != nil {
			t.Fatal(err)
		}
		if s.AttentionCalls() == before {
			continue
		}
		sel := chosen(s, 1)
		for g := range sel {
			sel[g] = slices.Clone(sel[g])
			slices.Sort(sel[g])
		}
		if last != nil && !slices.EqualFunc(sel, last, slices.Equal) {
			changes++
			if calls%sp.Refresh != 0 {
				t.Fatalf("call %d of the two-level attention chose anew", calls)
			}
		}
		last = sel
		want, _, _ := twoLevel(tab, sp, pick, pos, q, q, chosen(s, 1))
		worst = max(worst, relErr(out, want, q, q))
		calls++
	}
	if calls < 2*sp.Refresh || changes == 0 {
		t.Fatalf("%d calls of the two-level attention, %d choices changed", calls, changes)
	}
	if worst > 1e-2 {
		t.Errorf("the attention differs from the direct one by up to %g", worst)
	}
}

// Going back and writing other tokens over the cache (Rewind, then other rows) drops the
// summaries of the positions written over: the attention is that of the new sequence. So does a
// Reset, and rows put back with ImportRows.
func TestSparseRewindAndImport(t *testing.T) {
	const n, q = 1200, 4
	a, b := newTables(n, 4), newTables(n, 5)
	tab := a
	var out []float32
	s, err := New(fakeProgram(q, n, &tab, &out, true), fakeModel{})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetSparse(testSparse()); err != nil {
		t.Fatal(err)
	}
	pick, _ := s.layout(testSparse(), q, true)
	if steps(t, s, a, &out, 0, 600, q, 0, pick) == 0 {
		t.Fatal("the two-level attention never ran")
	}
	at600 := s.Mark()
	steps(t, s, a, &out, 600, 800, q, 600, pick)
	mark := s.Save()
	// back to 600 and b's tokens from there
	s.Rewind(at600)
	mixed := &tables{q: append(slices.Clone(a.q[:600]), b.q[600:]...), k: append(slices.Clone(a.k[:600]), b.k[600:]...), v: append(slices.Clone(a.v[:600]), b.v[600:]...)}
	tab = mixed
	if steps(t, s, mixed, &out, 600, 1000, q, 600, pick) == 0 {
		t.Fatal("the two-level attention never ran after the rewind")
	}
	// all of b from a Reset
	s.Reset()
	tab = b
	steps(t, s, b, &out, 0, 900, q, 0, pick)
	// a's 800 rows put back (Save, Rewind), then a's sequence on
	s.Rewind(mark)
	tab = a
	if steps(t, s, a, &out, 800, 1100, q, 800, pick) == 0 {
		t.Fatal("the two-level attention never ran after the rows were put back")
	}
}

// Going back a little (less than the window) and on again reads the buckets chosen before the
// rewind no longer, those of them now in the window included: they would count twice.
func TestSparseShortRewind(t *testing.T) {
	const n, q = 1200, 1
	tab := newTables(n, 7)
	var out []float32
	s, err := New(fakeProgram(q, n, &tab, &out, true), fakeModel{})
	if err != nil {
		t.Fatal(err)
	}
	sp := testSparse()
	if err := s.SetSparse(sp); err != nil {
		t.Fatal(err)
	}
	pick, _ := s.layout(sp, q, true)
	if steps(t, s, tab, &out, 0, 1000, q, 0, pick) == 0 {
		t.Fatal("the two-level attention never ran")
	}
	// from now on no choice but the forced ones. Going back past the buckets summarized writes
	// over them, which drops them and chooses anew (sparseTouch); going back less, to within
	// the window of the last choice, leaves the buckets the choice took from there behind the
	// new window's start but still chosen, unless the session chooses again
	s.sparse.cfg.Refresh = 1000
	sp.Refresh = 1000
	l := s.sparse.layers[1]
	back := l.n*sp.Bucket + 2
	if (back-sp.Window)/sp.Bucket >= l.at {
		t.Fatalf("going back to %d leaves the %d buckets of the last choice", back, l.at)
	}
	s.Rewind(NewMark(back, nil))
	worst, checked := 0.0, 0
	for pos := back; pos < back+40; pos++ {
		calls := s.AttentionCalls()
		if _, err := s.Step([][]float32{{float32(pos)}}); err != nil {
			t.Fatal(err)
		}
		if s.AttentionCalls() == calls {
			continue // the whole cache read: the choice is not used
		}
		checked++
		want, _, _ := twoLevel(tab, sp, pick, pos, 1, q, chosen(s, 1))
		worst = max(worst, relErr(out, want, q, 1))
		nB := (pos - sp.Window) / sp.Bucket
		if l.at > nB {
			t.Fatalf("at %d the choice is of %d buckets, there are %d", pos, l.at, nB)
		}
		for _, sel := range l.sel {
			for _, b := range sel {
				if int(b) >= nB {
					t.Fatalf("at %d bucket %d is read exactly and in the window", pos, b)
				}
			}
		}
	}
	if checked == 0 {
		t.Fatal("no call of the two-level attention after the rewind")
	}
	if worst > 1e-2 {
		t.Errorf("after the rewind the attention differs from the direct one by up to %g", worst)
	}
}

// The longer program (reading a prompt) has its own exact chunks (Prefill) and the shorter its
// own; switching between them chooses anew.
func TestSparsePrefillThenDecode(t *testing.T) {
	const n = 1400
	tab := newTables(n, 6)
	var out []float32
	long := fakeProgram(8, n, &tab, &out, true)
	short := fakeProgram(1, n, &tab, &out, true)
	short.Attn.NewBuf = long.Attn.NewBuf
	s, err := New(long, fakeModel{})
	if err != nil {
		t.Fatal(err)
	}
	// the programs share no cache inputs: the cache is the session's own
	if err := s.Add(short); err != nil {
		t.Fatal(err)
	}
	sp := testSparse()
	sp.Prefill = 3
	if err := s.SetSparse(sp); err != nil {
		t.Fatal(err)
	}
	pl, _ := s.layout(sp, 8, false)
	ps, _ := s.layout(sp, 1, true)
	if pl == ps {
		t.Fatalf("the prefill reads %d buckets exactly, as the decode", pl)
	}
	if steps(t, s, tab, &out, 0, 1200, 8, 0, pl) == 0 {
		t.Fatal("the two-level attention never ran in the prefill")
	}
	if steps(t, s, tab, &out, 1200, 1300, 1, 1200, ps) == 0 {
		t.Fatal("the two-level attention never ran in the decode")
	}
}

func TestTopK(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 9))
	for _, n := range []int{1, 5, 100, 1000} {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.IntN(50)) // ties
		}
		for _, k := range []int{0, 1, n / 2, n, n + 1} {
			got := topK(slices.Clone(v), k)
			if len(got) != min(k, n) {
				t.Fatalf("n %d k %d: %d indices", n, k, len(got))
			}
			sorted := slices.Clone(v)
			slices.Sort(sorted)
			slices.Reverse(sorted)
			var sum, want float64
			for _, i := range got {
				sum += float64(v[i])
			}
			for _, x := range sorted[:min(k, n)] {
				want += float64(x)
			}
			if sum != want {
				t.Fatalf("n %d k %d: the indices sum to %g, the k largest to %g", n, k, sum, want)
			}
		}
	}
}
