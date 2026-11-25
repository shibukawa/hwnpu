package experts

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"
	"time"
)

// TestHostTokenTime times a generated token's experts (one row, 4 of 32 a layer, every layer) on
// gpt-oss-20b's checkpoint (BONSAIAI_TEST_GPTOSS_CKPT_DIR), the experts picked at random.
func TestHostTokenTime(t *testing.T) {
	h, r := openGptOss(t)
	defer h.Close()
	x := [][]float32{make([]float32, 2880)}
	for j := range x[0] {
		x[0][j] = float32(r.NormFloat64())
	}
	out := [][]float32{make([]float32, 2880)}
	w := [][]float32{{0.25, 0.25, 0.25, 0.25}}
	// BONSAIAI_TEST_GPTOSS_GAP: a pause between layers (the Neural Engine's part of a step), not timed
	var gap time.Duration
	if v := os.Getenv("BONSAIAI_TEST_GPTOSS_GAP"); v != "" {
		gap, _ = time.ParseDuration(v)
	}
	for pass := range 3 {
		var el time.Duration
		const tokens = 20
		for range tokens {
			for l := range 24 {
				idx := [][]int{r.Perm(32)[:4]}
				if gap > 0 {
					for t0 := time.Now(); time.Since(t0) < gap; {
						time.Sleep(gap / 4)
					}
				}
				start := time.Now()
				if err := h.Run(l, x, idx, w, out); err != nil {
					t.Fatal(err)
				}
				el += time.Since(start)
			}
		}
		per := el / tokens
		t.Logf("pass %d: %v a token, %v a layer (gap %v)", pass, per, per/24, gap)
	}
}

// TestHostRowsTime times a prompt chunk's experts (BONSAIAI_TEST_GPTOSS_ROWS rows, 128 by default, 4 of
// 32 each, every layer), after every page is touched.
func TestHostRowsTime(t *testing.T) {
	h, r := openGptOss(t)
	defer h.Close()
	h.warm.Wait()
	n := 128
	if v := os.Getenv("BONSAIAI_TEST_GPTOSS_ROWS"); v != "" {
		fmt.Sscan(v, &n)
	}
	x, out, idx, w := make([][]float32, n), make([][]float32, n), make([][]int, n), make([][]float32, n)
	for i := range x {
		x[i], out[i] = make([]float32, 2880), make([]float32, 2880)
		for j := range x[i] {
			x[i][j] = float32(r.NormFloat64())
		}
		w[i] = []float32{0.25, 0.25, 0.25, 0.25}
	}
	layers := 24
	if v := os.Getenv("BONSAIAI_TEST_GPTOSS_LAYERS"); v != "" {
		fmt.Sscan(v, &layers)
	}
	for pass := range 3 {
		start := time.Now()
		for l := range layers {
			for i := range idx {
				idx[i] = r.Perm(32)[:4]
			}
			if err := h.Run(l, x, idx, w, out); err != nil {
				t.Fatal(err)
			}
		}
		el := time.Since(start)
		t.Logf("pass %d: %d rows of %d layers %v, %v a layer, %v a layer row", pass, n, layers, el, el/time.Duration(layers), el/time.Duration(layers*n))
	}
}

func openGptOss(t *testing.T) (*Host, *rand.Rand) {
	dir := os.Getenv("BONSAIAI_TEST_GPTOSS_CKPT_DIR")
	if dir == "" {
		t.Skip("BONSAIAI_TEST_GPTOSS_CKPT_DIR: gpt-oss-20b's checkpoint")
	}
	sh := Shape{Hidden: 2880, Inter: 2880, Experts: 32, TopK: 4}
	rec := &HostRecord{Checkpoint: filepath.Join(dir, "model.safetensors.index.json"), Act: Act{Kind: ActGptOss, Alpha: 1.702, Limit: 7}, Layers: map[int]MXFP4{}}
	for l := range 24 {
		p := fmt.Sprintf("model.layers.%d.mlp.experts.", l)
		rec.Layers[l] = MXFP4{GateUpBlocks: p + "gate_up_proj_blocks", GateUpScales: p + "gate_up_proj_scales", GateUpBias: p + "gate_up_proj_bias", DownBlocks: p + "down_proj_blocks", DownScales: p + "down_proj_scales", DownBias: p + "down_proj_bias"}
	}
	h, err := OpenHost(rec, sh)
	if err != nil {
		t.Fatal(err)
	}
	return h, rand.New(rand.NewPCG(5, 6))
}

// TestHostTokenTimeQwen36 times a generated token's experts (one row, 8 of 256 a layer, 40
// layers) on Qwen3.6-35B-A3B's checkpoint with MXFP4 experts (BONSAIAI_TEST_QWEN36_CKPT_DIR, bonsaiai-dev
// mxfp4): the same experts every token (in memory after the first: what the kernels and the
// host's steps around them take), then experts picked at random (read from the file as chosen,
// on a machine whose memory does not hold them all). BONSAIAI_TEST_QWEN36_GAP pauses between the layers,
// as the Neural Engine's part of a step does: asleep, or with BONSAIAI_TEST_QWEN36_SPIN the caller's core
// kept awake.
func TestHostTokenTimeQwen36(t *testing.T) {
	dir := os.Getenv("BONSAIAI_TEST_QWEN36_CKPT_DIR")
	if dir == "" {
		t.Skip("BONSAIAI_TEST_QWEN36_CKPT_DIR: Qwen3.6-35B-A3B's checkpoint with MXFP4 experts")
	}
	const L, E, K, H = 40, 256, 8, 2048
	sh := Shape{Hidden: H, Inter: 512, Experts: E, TopK: K}
	rec := &HostRecord{Checkpoint: filepath.Join(dir, "model.safetensors.index.json"), Act: Act{Kind: ActSiLU}, Layers: map[int]MXFP4{}}
	for l := range L {
		p := fmt.Sprintf("model.language_model.layers.%d.", l)
		rec.Layers[l] = MXFP4{GateUpBlocks: p + "mlp.experts.gate_up_proj_blocks", GateUpScales: p + "mlp.experts.gate_up_proj_scales",
			DownBlocks: p + "mlp.experts.down_proj_blocks", DownScales: p + "mlp.experts.down_proj_scales",
			Router: p + "mlp.gate.weight", Norm: p + "post_attention_layernorm.weight", NormOffset: 1}
	}
	h, err := OpenHost(rec, sh)
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	var ms runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&ms)
	t.Logf("the host's own memory (Go's heap in use): %.1f MB", float64(ms.HeapInuse)/1e6)
	r := rand.New(rand.NewPCG(5, 6))
	x := [][]float32{make([]float32, H)}
	for j := range x[0] {
		x[0][j] = float32(r.NormFloat64())
	}
	out := [][]float32{make([]float32, H)}
	w := [][]float32{{0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125}}
	var gap time.Duration
	if v := os.Getenv("BONSAIAI_TEST_QWEN36_GAP"); v != "" {
		gap, _ = time.ParseDuration(v)
	}
	fixed := make([][]int, L)
	for l := range fixed {
		fixed[l] = r.Perm(E)[:K]
	}
	for _, c := range []struct {
		name   string
		tokens int
		pick   func(l int) []int
	}{{"the same experts", 20, func(l int) []int { return fixed[l] }}, {"the same experts", 20, func(l int) []int { return fixed[l] }},
		{"experts at random", 10, func(int) []int { return r.Perm(E)[:K] }}} {
		var el time.Duration
		for range c.tokens {
			for l := range L {
				idx := [][]int{c.pick(l)}
				for t0 := time.Now(); gap > 0 && time.Since(t0) < gap; {
					if os.Getenv("BONSAIAI_TEST_QWEN36_SPIN") != "" {
						continue // the caller's core kept awake
					}
					time.Sleep(gap / 4)
				}
				start := time.Now()
				if err := h.Run(l, x, idx, w, out); err != nil {
					t.Fatal(err)
				}
				el += time.Since(start)
			}
		}
		per := el / time.Duration(c.tokens)
		t.Logf("%s: %v a token, %v a layer (gap %v)", c.name, per, per/L, gap)
	}
}

// qwen36Host opens Qwen3.6-35B-A3B's checkpoint with MXFP4 experts (BONSAIAI_TEST_QWEN36_CKPT_DIR).
func qwen36Host(t *testing.T) (*Host, Shape) {
	dir := os.Getenv("BONSAIAI_TEST_QWEN36_CKPT_DIR")
	if dir == "" {
		t.Skip("BONSAIAI_TEST_QWEN36_CKPT_DIR: Qwen3.6-35B-A3B's checkpoint with MXFP4 experts")
	}
	const L, E, K, H = 40, 256, 8, 2048
	sh := Shape{Hidden: H, Inter: 512, Experts: E, TopK: K}
	rec := &HostRecord{Checkpoint: filepath.Join(dir, "model.safetensors.index.json"), Act: Act{Kind: ActSiLU}, Layers: map[int]MXFP4{}}
	for l := range L {
		p := fmt.Sprintf("model.language_model.layers.%d.", l)
		rec.Layers[l] = MXFP4{GateUpBlocks: p + "mlp.experts.gate_up_proj_blocks", GateUpScales: p + "mlp.experts.gate_up_proj_scales",
			DownBlocks: p + "mlp.experts.down_proj_blocks", DownScales: p + "mlp.experts.down_proj_scales",
			Router: p + "mlp.gate.weight", Norm: p + "post_attention_layernorm.weight", NormOffset: 1}
	}
	h, err := OpenHost(rec, sh)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(h.Close)
	return h, sh
}

// TestHostRowsTimeQwen36 times a layer's experts on the rows of a prompt (8 of 256 experts a
// row), by how many rows one call takes: the rows of a step of 128 tokens, and those of several
// steps together. The experts are picked evenly (the fewest rows an expert) or as tokens pick
// them (BONSAIAI_TEST_QWEN36_SKEW=1: a few experts take most rows).
func TestHostRowsTimeQwen36(t *testing.T) {
	h, sh := qwen36Host(t)
	E, K, H := sh.Experts, sh.TopK, sh.Hidden
	r := rand.New(rand.NewPCG(5, 6))
	skew := os.Getenv("BONSAIAI_TEST_QWEN36_SKEW") != ""
	pick := func() []int {
		if !skew {
			return r.Perm(E)[:K]
		}
		var idx []int
		for len(idx) < K {
			e := int(float64(E) * math.Pow(r.Float64(), 3))
			if !slices.Contains(idx, e) {
				idx = append(idx, e)
			}
		}
		return idx
	}
	const layer = 7
	for _, n := range []int{128, 128, 256, 512, 1024, 2048} {
		x, out, idx, w := make([][]float32, n), make([][]float32, n), make([][]int, n), make([][]float32, n)
		used := map[int]bool{}
		for i := range x {
			x[i], out[i] = make([]float32, H), make([]float32, H)
			for j := range x[i] {
				x[i][j] = float32(r.NormFloat64())
			}
			idx[i], w[i] = pick(), make([]float32, K)
			for k := range w[i] {
				w[i][k] = 1 / float32(K)
				used[idx[i][k]] = true
			}
		}
		best := time.Duration(math.MaxInt64)
		for range 3 {
			start := time.Now()
			if err := h.Run(layer, x, idx, w, out); err != nil {
				t.Fatal(err)
			}
			best = min(best, time.Since(start))
		}
		macs := float64(n) * float64(K) * float64(3*H*sh.Inter)
		t.Logf("%4d rows (%d experts, %.1f rows each): %v, %.3f ms a row, %.0f GMAC/s", n, len(used), float64(n*K)/float64(len(used)), best, best.Seconds()*1e3/float64(n), macs/best.Seconds()/1e9)
	}
}
