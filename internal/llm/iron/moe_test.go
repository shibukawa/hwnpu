package iron

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"testing"
)

func TestTopK(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	for trial := 0; trial < 200; trial++ {
		vals := make([]float32, 256)
		for i := range vals {
			vals[i] = float32(r.Intn(40)) // ties
		}
		order := make([]int, len(vals))
		for i := range order {
			order[i] = i
		}
		sort.SliceStable(order, func(a, b int) bool { return vals[order[a]] > vals[order[b]] })
		best := make([]int, 8)
		topK(best, vals)
		for k := range best {
			if best[k] != order[k] {
				t.Fatalf("trial %d: top %v, a stable sort's %v", trial, best, order[:8])
			}
		}
	}
}

// TestSigmoidRouter checks LFM2-MoE's routing against its definition: the top of sigmoid + bias
// (a bias can choose an expert with a smaller score), the weights the chosen's scores over their
// sum times the scale, in bf16.
func TestSigmoidRouter(t *testing.T) {
	const E, L, top = 32, 64, 4
	r := rand.New(rand.NewSource(2))
	h := &HostStep{Experts: E, Top: top, Bias: make([]float32, E), Scale: 1.5}
	for i := range h.Bias {
		h.Bias[i] = float32(r.NormFloat64()) * 0.2
	}
	rt := newSigmoidRouter(h)
	const rows = 50
	logits := make([]byte, 2*rows*L)
	for i := 0; i < rows*L; i++ {
		putBF16(logits, i, float32(r.NormFloat64())*2)
	}
	biased := 0
	for row := 0; row < rows; row++ {
		s, pick := make([]float64, E), make([]float32, E)
		for i := range s {
			s[i] = 1 / (1 + math.Exp(-float64(bf16At(logits, row*L+i))))
			pick[i] = float32(s[i]) + h.Bias[i]
		}
		order := make([]int, E)
		for i := range order {
			order[i] = i
		}
		sort.SliceStable(order, func(a, b int) bool { return pick[order[a]] > pick[order[b]] })
		byScore := make([]int, E)
		copy(byScore, order)
		sort.SliceStable(byScore, func(a, b int) bool { return s[byScore[a]] > s[byScore[b]] })
		var sum float64
		for _, e := range order[:top] {
			sum += s[e]
		}
		rt.route(logits, row, L)
		var wsum float64
		for k, e := range rt.best {
			if e != order[k] {
				t.Fatalf("row %d: experts %v, by score + bias %v", row, rt.best, order[:top])
			}
			if e != byScore[k] {
				biased++
			}
			want := s[e] / sum * 1.5
			if d := math.Abs(float64(rt.w[k]) - want); d > want/128 {
				t.Fatalf("row %d: expert %d's weight %g, want %g", row, e, rt.w[k], want)
			}
			if rt.w[k] != roundBF16(rt.w[k]) {
				t.Fatalf("row %d: weight %g is no bf16", row, rt.w[k])
			}
			wsum += float64(rt.w[k])
		}
		if math.Abs(wsum-1.5) > 0.02 {
			t.Fatalf("row %d: the weights sum to %g", row, wsum)
		}
	}
	if biased == 0 {
		t.Fatal("the bias changed no choice: the test does not cover it")
	}
	// no scale in the step: 1
	h.Scale = 0
	rt = newSigmoidRouter(h)
	rt.route(logits, 0, L)
	var wsum float32
	for _, w := range rt.w {
		wsum += w
	}
	if math.Abs(float64(wsum)-1) > 0.02 {
		t.Fatalf("without a scale the weights sum to %g", wsum)
	}
}

// TestGatherSlots checks the slots of a chunk's rows by expert: every choice in one slot of its
// expert, at the row among the slots' rows that at names, no slot over its rows, an expert's
// rows in the chunk's order.
func TestGatherSlots(t *testing.T) {
	const E, top, R = 32, 4, 16
	r := rand.New(rand.NewSource(3))
	for _, n := range []int{1, 5, 97, 300} {
		ids := make([]int32, n*top)
		for tok := 0; tok < n; tok++ {
			// a few experts take most rows, so that some need several slots; a row's are distinct
			p, q := r.Perm(8), r.Perm(E-8)
			for k := 0; k < top; k++ {
				ids[tok*top+k] = int32(p[k])
				if r.Intn(4) == 0 {
					ids[tok*top+k] = int32(8 + q[k])
				}
			}
		}
		slots, at := gatherSlots(ids, E, top, R)
		rows := 0
		for i, s := range slots {
			if len(s.rows) == 0 || len(s.rows) > R || s.first != rows {
				t.Fatalf("n %d: slot %d of %d rows from %d, after %d rows", n, i, len(s.rows), s.first, rows)
			}
			if i > 0 && slots[i-1].expert == s.expert && len(slots[i-1].rows) != R {
				t.Fatalf("n %d: expert %d has a further slot after one of %d rows", n, s.expert, len(slots[i-1].rows))
			}
			if i > 0 && slots[i-1].expert > s.expert {
				t.Fatalf("n %d: the slots are not in order of the experts", n)
			}
			for j := 1; j < len(s.rows); j++ {
				if s.rows[j] <= s.rows[j-1] {
					t.Fatalf("n %d: slot %d's rows %v are not in order", n, i, s.rows)
				}
			}
			rows += len(s.rows)
		}
		if rows != n*top {
			t.Fatalf("n %d: %d rows in the slots, %d choices", n, rows, n*top)
		}
		seen := make([]bool, rows)
		for c, a := range at {
			if seen[a] {
				t.Fatalf("n %d: two choices at row %d", n, a)
			}
			seen[a] = true
			var s gatherSlot
			for _, x := range slots {
				if int(a) >= x.first && int(a) < x.first+len(x.rows) {
					s = x
				}
			}
			if int32(s.expert) != ids[c] || s.rows[int(a)-s.first] != int32(c/top) {
				t.Fatalf("n %d: choice %d (row %d, expert %d) at %d, in expert %d's slot", n, c, c/top, ids[c], a, s.expert)
			}
		}
	}
}

// TestSoftmaxRouter checks softmaxRouter (and routeRows) bit for bit against the loop
// moeHostRoute and moeHostRouteMX4 ran before they shared it: the top of the first E logits,
// ties to the lower index, weights exp(l - l_top) over their sum in float64, rounded to bf16.
func TestSoftmaxRouter(t *testing.T) {
	const E, L, top, n = 256, 320, 8, 64
	r := rand.New(rand.NewSource(4))
	logits := make([]byte, 2*n*L)
	for i := 0; i < n*L; i++ {
		putBF16(logits, i, float32(r.Intn(64)-32)/8) // ties
	}
	sr := newSoftmaxRouter(E, top)
	var shared []float32
	ids, w := routeRows(sr, logits, n, L, func(t int) { shared = append(shared, sr.vals[E]) })
	order := make([]int, top)
	vals := make([]float32, L)
	for row := 0; row < n; row++ {
		for i := range vals {
			vals[i] = bf16At(logits, row*L+i)
		}
		topK(order, vals[:E])
		var sum float64
		e := make([]float64, top)
		for k := 0; k < top; k++ {
			e[k] = math.Exp(float64(vals[order[k]] - vals[order[0]]))
			sum += e[k]
		}
		for k := 0; k < top; k++ {
			if ids[row*top+k] != int32(order[k]) || math.Float32bits(w[row*top+k]) != math.Float32bits(roundBF16(float32(e[k]/sum))) {
				t.Fatalf("row %d rank %d: expert %d weight %g, the loop's %d %g", row, k, ids[row*top+k], w[row*top+k], order[k], roundBF16(float32(e[k]/sum)))
			}
		}
		if shared[row] != vals[E] {
			t.Fatalf("row %d: vals[E] %g, want %g", row, shared[row], vals[E])
		}
	}
}

// TestCombineRows checks combineRows bit for bit against the loops moe_combine (with a shared
// expert's row and gate) and moe_gather_combine (without) ran before they shared it.
func TestCombineRows(t *testing.T) {
	const n, K, top = 37, 128, 4
	r := rand.New(rand.NewSource(5))
	bf := func(m int) []byte {
		b := make([]byte, 2*m)
		for i := 0; i < m; i++ {
			putBF16(b, i, float32(r.NormFloat64()))
		}
		return b
	}
	resid, sh, ys := bf(n*K), bf(n*K), bf(n*top*K)
	gate, w := make([]float32, n), make([]float32, n*top)
	for i := range gate {
		gate[i] = roundBF16(r.Float32())
	}
	for i := range w {
		w[i] = roundBF16(r.Float32())
	}
	at := r.Perm(n * top)
	row := func(c int) []byte { return ys[2*at[c]*K : 2*(at[c]+1)*K] }
	for _, withShared := range []bool{true, false} {
		got := make([]byte, 2*n*K)
		if withShared {
			combineRows(got, resid, sh, gate, w, n, K, top, row)
		} else {
			combineRows(got, resid, nil, nil, w, n, K, top, row)
		}
		want := make([]byte, 2*n*K)
		acc := make([]float32, K)
		for tok := 0; tok < n; tok++ {
			for i := range acc {
				acc[i] = bf16At(resid[2*tok*K:], i)
				if withShared {
					acc[i] = bf16At(resid[2*tok*K:], i) + gate[tok]*bf16At(sh[2*tok*K:], i)
				}
			}
			for k := 0; k < top; k++ {
				c := tok*top + k
				for i := range acc {
					acc[i] += w[c] * bf16At(row(c), i)
				}
			}
			for i, x := range acc {
				putBF16(want[2*tok*K:], i, x)
			}
		}
		if string(got) != string(want) {
			t.Fatalf("shared %v: combineRows differs from the loop", withShared)
		}
	}
}

// TestByExpert checks byExpert against the loop moe_route's passes grouped their choices by:
// each expert's choices (row × top + rank) of a run of rows, in the rows' order.
func TestByExpert(t *testing.T) {
	const E, top, n, size = 64, 8, 50, 256
	r := rand.New(rand.NewSource(6))
	ids := make([]int32, n*top)
	for i := range ids {
		ids[i] = int32(r.Intn(E))
	}
	lo, hi := 7, 31 // a pass's rows
	got := byExpert(ids[lo*top:hi*top], size, lo*top)
	want := make([][]int, size)
	for tok := lo; tok < hi; tok++ {
		for k := 0; k < top; k++ {
			e := ids[tok*top+k]
			want[e] = append(want[e], tok*top+k)
		}
	}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("byExpert %v, the loop's %v", got, want)
	}
}
