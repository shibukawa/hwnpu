package iron

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

// testPool is a pool of P places for E experts in L layers, the places holding experts 0..P-1.
func testPool(L, E, P int) *expertPool {
	p := &expertPool{L: L, E: E, P: P, places: make([]poolPlaces, L), core: make([]bool, L*E),
		score: make([]float64, L*E), last: make([]int64, L*E), init: make([]float64, L*E), filled: true}
	for l := range L {
		pl := poolPlaces{at: make([]int, P), of: make([]int, E)}
		for e := range E {
			pl.of[e] = -1
		}
		for i := range P {
			pl.at[i], pl.of[i] = i, i
		}
		p.places[l] = pl
	}
	return p
}

// TestPoolPasses: every choice of a chunk runs in a pass where its expert is in the place its
// slot reads, the places' copies come before, and a pass's slots and rows are within bounds.
func TestPoolPasses(t *testing.T) {
	const E, P, top, cap, kt, in = 32, 16, 4, 56, 320, 2880
	S := P + 8
	names := make([]string, S)
	for i := P; i < S; i++ {
		names[i] = fmt.Sprintf("slot%d", i)
	}
	rng := rand.New(rand.NewPCG(1, 2))
	for trial, n := range []int{1, 7, 64, 256, 256} {
		m := &Model{pool: testPool(1, E, P)}
		if trial == 4 { // a skewed chunk: some experts past cap rows
			m.pool.core[3] = true
		}
		h := &HostStep{Layer: 0, Experts: E, Top: top, Cap: cap, Tokens: 256, InFeatures: in, KT: kt, Slots: S,
			SlotValues: names, Places: P, Writes: []Region{{Bytes: 4 * 64}, {Bytes: 2 * S * in * cap}}}
		ids := make([]int32, n*top)
		for t := 0; t < n; t++ {
			seen := map[int32]bool{}
			for k := 0; k < top; k++ {
				var e int32
				for {
					if trial == 4 && k < 2 {
						e = int32(rng.IntN(3))
					} else {
						e = int32(rng.IntN(E))
					}
					if !seen[e] {
						break
					}
				}
				seen[e] = true
				ids[t*top+k] = e
			}
		}
		rt := &moeRoute{pass: make([]int32, n*top), row: make([]int32, n*top)}
		before := append([]int(nil), m.pool.places[0].at...)
		if err := m.poolPasses(h, rt, ids, n, 3072); err != nil {
			t.Fatal(err)
		}
		// replay the passes on the places
		at := before
		for q, ps := range rt.passes {
			for _, ld := range ps.loads {
				at[ld[1]] = ld[0]
			}
			slotExpert := make([]int, S)
			for i := range P {
				slotExpert[i] = at[i]
			}
			for i := P; i < S; i++ {
				slotExpert[i] = at[ps.values[names[i]]]
			}
			rows := make([]int, S)
			for c := range ids {
				if int(rt.pass[c]) != q {
					continue
				}
				sl, r := int(rt.row[c])/cap, int(rt.row[c])%cap
				if slotExpert[sl] != int(ids[c]) {
					t.Fatalf("n %d pass %d: choice %d of expert %d in slot %d of expert %d", n, q, c, ids[c], sl, slotExpert[sl])
				}
				if r >= ps.slotRows[sl] {
					t.Fatalf("n %d pass %d: row %d of slot %d past its %d", n, q, r, sl, ps.slotRows[sl])
				}
				rows[sl]++
			}
			for sl := range S {
				if rows[sl] != ps.slotRows[sl] {
					t.Fatalf("n %d pass %d: slot %d holds %d rows, counts %d", n, q, sl, rows[sl], ps.slotRows[sl])
				}
			}
		}
		for i, e := range at {
			if m.pool.places[0].at[i] != e || m.pool.places[0].of[e] != i {
				t.Fatalf("n %d: the pool's place %d is not the passes' %d", n, i, e)
			}
		}
		t.Logf("%d tokens: %d passes", n, len(rt.passes))
	}
}
