//go:build darwin

package experts

import (
	"context"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"syscall"
	"testing"

	"github.com/shibukawa/hwnpu"
)

func lockNPUForTest(t *testing.T) {
	t.Helper()
	home, err := os.UserHomeDir()
	if err != nil {
		return
	}
	lockPath := filepath.Join(home, "Library", "Caches", "hwnpu", "neural-engine.lock")
	lf, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return
	}
	_ = syscall.Flock(int(lf.Fd()), syscall.LOCK_EX)
	t.Cleanup(func() {
		_ = syscall.Flock(int(lf.Fd()), syscall.LOCK_UN)
		lf.Close()
	})
}

func TestHostPrefillNPU(t *testing.T) {
	lockNPUForTest(t)
	ctx := context.Background()
	cacheDir := t.TempDir()
	t.Setenv("HWNPU_CACHE", cacheDir)
	dev, err := hwnpu.Open(ctx, hwnpu.WithBackend("coreai"), hwnpu.WithCacheDir(cacheDir))
	if err != nil {
		t.Skipf("Core AI unavailable: %v", err)
	}
	defer dev.Close()
	gr, err := dev.Graph()
	if err != nil {
		t.Fatal(err)
	}

	sh := Shape{Hidden: 512, Inter: 256, Experts: 16, TopK: 4}
	for _, tc := range []struct {
		name string
		act  Act
		bias bool
	}{
		{"silu", Act{Kind: ActSiLU}, false},
		{"gptoss_bias", Act{Kind: ActGptOss, Alpha: 1.702, Limit: 7}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rec := writeHost(t, sh, 1, tc.act, tc.bias)

			hCPU, err := OpenHost(rec, sh)
			if err != nil {
				t.Fatal(err)
			}
			defer hCPU.Close()
			hCPU.warm.Wait()

			hNPU, err := OpenHost(rec, sh)
			if err != nil {
				t.Fatal(err)
			}
			defer hNPU.Close()
			hNPU.warm.Wait()
			hNPU.EnablePrefillNPU(gr, "")
			if !hNPU.HasPrefillNPU() {
				t.Fatal("expected HasPrefillNPU to be true")
			}

			// 320 rows with skewed expert selection so hot experts (>=24 rows) run on ANE
			// and cold experts (<24 rows) run concurrently on CPU.
			const nrows = 320
			r := rand.New(rand.NewPCG(17, 19))
			rows := make([][]float32, nrows)
			idx := make([][]int, nrows)
			w := make([][]float32, nrows)
			outCPU := make([][]float32, nrows)
			outNPU := make([][]float32, nrows)
			for i := range nrows {
				rows[i] = make([]float32, sh.Hidden)
				outCPU[i] = make([]float32, sh.Hidden)
				outNPU[i] = make([]float32, sh.Hidden)
				for j := range sh.Hidden {
					rows[i][j] = float32(r.NormFloat64()) * 0.5
				}
				var chosen []int
				for len(chosen) < sh.TopK {
					e := int(float64(sh.Experts) * math.Pow(r.Float64(), 2.2))
					if !slices.Contains(chosen, e) {
						chosen = append(chosen, e)
					}
				}
				idx[i] = chosen
				w[i] = []float32{0.45, 0.25, 0.20, 0.10}
			}

			if err := hCPU.Run(0, rows, idx, w, outCPU); err != nil {
				t.Fatal(err)
			}
			if err := hNPU.Run(0, rows, idx, w, outNPU); err != nil {
				t.Fatal(err)
			}
			if hNPU.npu.g == nil {
				t.Skip("Core AI placed synthetic 512x256 prefill graph off NPU")
			}
			if hNPU.npu.slots[0] == nil || hNPU.npu.slots[1] == nil {
				t.Fatal("expected prefill slots to be allocated during multi-row run")
			}

			var dot, na, nb float64
			for i := range nrows {
				for j := range sh.Hidden {
					dot += float64(outCPU[i][j]) * float64(outNPU[i][j])
					na += float64(outCPU[i][j]) * float64(outCPU[i][j])
					nb += float64(outNPU[i][j]) * float64(outNPU[i][j])
				}
			}
			cos := dot / math.Sqrt(na*nb)
			if cos < 0.9998 {
				t.Fatalf("prefill NPU cosine = %.6f, want >= 0.9998", cos)
			}

			// Verify 1-row decode releases prefill slots immediately and matches CPU bit-for-bit.
			decCPU := [][]float32{make([]float32, sh.Hidden)}
			decNPU := [][]float32{make([]float32, sh.Hidden)}
			if err := hCPU.Run(0, rows[:1], idx[:1], w[:1], decCPU); err != nil {
				t.Fatal(err)
			}
			if err := hNPU.Run(0, rows[:1], idx[:1], w[:1], decNPU); err != nil {
				t.Fatal(err)
			}
			if hNPU.npu.slots[0] != nil || hNPU.npu.slots[1] != nil {
				t.Fatal("expected prefill slots to be released on 1-row decode")
			}
			for j := range sh.Hidden {
				if decCPU[0][j] != decNPU[0][j] {
					t.Fatalf("1-row decode mismatch at %d: %g vs %g", j, decCPU[0][j], decNPU[0][j])
				}
			}
		})
	}
}

func TestHostMinRatio(t *testing.T) {
	sh := Shape{Hidden: 128, Inter: 64, Experts: 8, TopK: 4}
	rec := writeHost(t, sh, 1, Act{Kind: ActSiLU}, false)

	t.Setenv("HWNPU_DEV", "moe.min-ratio=0.15")
	h, err := OpenHost(rec, sh)
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()

	x := [][]float32{make([]float32, sh.Hidden)}
	for j := range sh.Hidden {
		x[0][j] = float32(j%7-3) * 0.2
	}
	out := [][]float32{make([]float32, sh.Hidden)}
	// Expert weights: 0.70, 0.20, 0.06 (< 0.15*0.70=0.105), 0.04 -> keeps first 2 renormalized to 0.7778, 0.2222.
	if err := h.Run(0, x, [][]int{{1, 3, 5, 7}}, [][]float32{{0.70, 0.20, 0.06, 0.04}}, out); err != nil {
		t.Fatal(err)
	}

	t.Setenv("HWNPU_DEV", "")
	hRef, err := OpenHost(rec, sh)
	if err != nil {
		t.Fatal(err)
	}
	defer hRef.Close()
	want := [][]float32{make([]float32, sh.Hidden)}
	if err := hRef.Run(0, x, [][]int{{1, 3}}, [][]float32{{0.70 / 0.90, 0.20 / 0.90}}, want); err != nil {
		t.Fatal(err)
	}
	for j := range sh.Hidden {
		if math.Abs(float64(out[0][j]-want[0][j])) > 1e-5 {
			t.Fatalf("idx %d: got %g, want %g", j, out[0][j], want[0][j])
		}
	}
}
