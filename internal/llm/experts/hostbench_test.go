package experts

import (
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestHostTokenTime times a generated token's experts (one row, 4 of 32 a layer, every layer) on
// gpt-oss-20b's checkpoint (HWNPU_GPTOSS_CKPT), the experts picked at random.
func TestHostTokenTime(t *testing.T) {
	h, r := openGptOss(t)
	defer h.Close()
	x := [][]float32{make([]float32, 2880)}
	for j := range x[0] {
		x[0][j] = float32(r.NormFloat64())
	}
	out := [][]float32{make([]float32, 2880)}
	w := [][]float32{{0.25, 0.25, 0.25, 0.25}}
	// HWNPU_GPTOSS_GAP: a pause between layers (the Neural Engine's part of a step), not timed
	var gap time.Duration
	if v := os.Getenv("HWNPU_GPTOSS_GAP"); v != "" {
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

// TestHostRowsTime times a prompt chunk's experts (HWNPU_GPTOSS_ROWS rows, 128 by default, 4 of
// 32 each, every layer), after every page is touched.
func TestHostRowsTime(t *testing.T) {
	h, r := openGptOss(t)
	defer h.Close()
	h.warm.Wait()
	n := 128
	if v := os.Getenv("HWNPU_GPTOSS_ROWS"); v != "" {
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
	if v := os.Getenv("HWNPU_GPTOSS_LAYERS"); v != "" {
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
	dir := os.Getenv("HWNPU_GPTOSS_CKPT")
	if dir == "" {
		t.Skip("HWNPU_GPTOSS_CKPT: gpt-oss-20b's checkpoint")
	}
	sh := Shape{Hidden: 2880, Inter: 2880, Experts: 32, TopK: 4}
	rec := &HostRecord{Checkpoint: filepath.Join(dir, "model.safetensors.index.json"), Act: Act{Kind: ActGptOss, Alpha: 1.702, Limit: 7}, Layers: map[int]MXFP4{}}
	for l := range 24 {
		p := fmt.Sprintf("model.layers.%d.mlp.experts.", l)
		rec.Layers[l] = MXFP4{p + "gate_up_proj_blocks", p + "gate_up_proj_scales", p + "gate_up_proj_bias", p + "down_proj_blocks", p + "down_proj_scales", p + "down_proj_bias"}
	}
	h, err := OpenHost(rec, sh)
	if err != nil {
		t.Fatal(err)
	}
	return h, rand.New(rand.NewPCG(5, 6))
}
