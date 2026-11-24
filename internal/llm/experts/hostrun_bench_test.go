package experts

import (
	"math/rand/v2"
	"os"
	"slices"
	"testing"
	"time"
)

// syntheticHost is a Host of one layer of gpt-oss-20b's shape with random weights in the heap
// (425 MB: a row's 4 experts, 53 MB, are out of the caches).
func syntheticHost(tb testing.TB) (*Host, *rand.Rand) {
	sh := Shape{Hidden: 2880, Inter: 2880, Experts: 32, TopK: 4}
	E, H, I := sh.Experts, sh.Hidden, sh.Inter
	r := rand.New(rand.NewPCG(11, 12))
	fill := func(n int, scales bool) []byte {
		b := make([]byte, n)
		for i := 0; i+8 <= n; i += 8 {
			v := r.Uint64()
			for j := range 8 {
				b[i+j] = byte(v >> (8 * j))
				if scales {
					b[i+j] = 118 + b[i+j]%8
				}
			}
		}
		return b
	}
	hl := &hostLayer{gw: fill(E*2*I*H/2, false), ge: fill(E*2*I*H/32, true), dw: fill(E*H*I/2, false), de: fill(E*H*I/32, true),
		gb: make([]float32, E*2*I), db: make([]float32, E*H)}
	return &Host{shape: sh, act: Act{Kind: ActGptOss, Alpha: 1.702, Limit: 7}, layers: map[int]*hostLayer{0: hl}}, r
}

// BenchmarkHostRun is a generated token's experts of a layer through Host.Run (one row, 4
// experts): internal/mxfp4's BenchmarkGenLayer is the two matrix products alone.
func BenchmarkHostRun(b *testing.B) {
	h, r := syntheticHost(b)
	x := [][]float32{make([]float32, 2880)}
	for j := range x[0] {
		x[0][j] = float32(r.NormFloat64())
	}
	out := [][]float32{make([]float32, 2880)}
	w := [][]float32{{0.4, 0.3, 0.2, 0.1}}
	b.SetBytes(4 * h.expertBytes())
	b.ReportAllocs()
	for b.Loop() {
		idx := [][]int{r.Perm(32)[:4]}
		if err := h.Run(0, x, idx, w, out); err != nil {
			b.Fatal(err)
		}
	}
}

// TestHostRunGap times Host.Run on synthetic weights with a pause between the layers
// (BONSAIAI_TEST_GPTOSS_GAP, as the Neural Engine's part between them: the workers sleep): the median
// and the tail.
func TestHostRunGap(t *testing.T) {
	gap, _ := time.ParseDuration(os.Getenv("BONSAIAI_TEST_GPTOSS_GAP"))
	if gap <= 0 {
		t.Skip("BONSAIAI_TEST_GPTOSS_GAP: the pause between layers")
	}
	h, r := syntheticHost(t)
	x := [][]float32{make([]float32, 2880)}
	for j := range x[0] {
		x[0][j] = float32(r.NormFloat64())
	}
	out := [][]float32{make([]float32, 2880)}
	w := [][]float32{{0.4, 0.3, 0.2, 0.1}}
	for pass := range 3 {
		var ds []time.Duration
		var sum time.Duration
		for range 1500 {
			idx := [][]int{r.Perm(32)[:4]}
			for t0 := time.Now(); time.Since(t0) < gap; {
				time.Sleep(gap / 4)
			}
			start := time.Now()
			if err := h.Run(0, x, idx, w, out); err != nil {
				t.Fatal(err)
			}
			d := time.Since(start)
			ds = append(ds, d)
			sum += d
		}
		slices.Sort(ds)
		t.Logf("pass %d, gap %v: mean %v, median %v, p90 %v, p99 %v", pass, gap, (sum / time.Duration(len(ds))).Round(time.Microsecond),
			ds[len(ds)/2].Round(time.Microsecond), ds[len(ds)*9/10].Round(time.Microsecond), ds[len(ds)*99/100].Round(time.Microsecond))
	}
}
