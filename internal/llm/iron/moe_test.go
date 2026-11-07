package iron

import (
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
