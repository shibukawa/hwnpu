package chat

import (
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/safetensors"
	"github.com/shibukawa/hwnpu/internal/vecops"
)

func TestDraftScanSynthetic(t *testing.T) {
	const V, D = 8192, 128
	rng := rand.New(rand.NewPCG(1, 2))
	emb := make([]float32, V*D)
	// Realistic clustered embeddings + strong shared bias (anisotropy).
	bias := make([]float32, D)
	for j := range D {
		bias[j] = float32(rng.NormFloat64()) * 0.5
	}
	const nCenters = 64
	centers := make([]float32, nCenters*D)
	for i := range centers {
		centers[i] = float32(rng.NormFloat64()) * 0.25
	}
	for r := range V {
		c := r % nCenters
		for j := range D {
			emb[r*D+j] = bias[j] + centers[c*D+j] + float32(rng.NormFloat64())*0.05
		}
	}
	ds, err := BuildDraftScan(V, D, func(r int, dst []float32) error {
		copy(dst, emb[r*D:(r+1)*D])
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	maxCluster := 0
	for c := range ds.C {
		n := int(ds.Offsets[c+1] - ds.Offsets[c])
		if n > maxCluster {
			maxCluster = n
		}
	}
	t.Logf("V=%d D=%d -> C=%d clusters, maxCluster=%d", V, D, ds.C, maxCluster)

	hits := 0
	const Q = 32
	for q := range Q {
		target := rng.IntN(V)
		h := make([]float32, D)
		for j := range D {
			h[j] = emb[target*D+j]*2.0 + float32(rng.NormFloat64())*0.02
		}
		// Exact argmax
		exactID, exactV := int32(0), float32(-1e30)
		for r := range V {
			var s float32
			for j := range D {
				s += h[j] * emb[r*D+j]
			}
			if s > exactV {
				exactV, exactID = s, int32(r)
			}
		}
		got := ds.Scan(h)
		if got == exactID {
			hits++
		}
		_ = q
	}
	t.Logf("synthetic top-1 recall: %d/%d", hits, Q)
	if hits < Q*3/4 {
		t.Errorf("top-1 recall %d/%d too low", hits, Q)
	}

	// Verify disk cache Save + LoadDraftScan round-trip produces identical Scan results
	// and rejects stale srcSize/srcMtime.
	cachePath := filepath.Join(t.TempDir(), draftScanFile)
	if err := ds.Save(cachePath, 12345, 67890); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := LoadDraftScan(cachePath, V, D, 12345, 67890)
	if err != nil {
		t.Fatalf("LoadDraftScan: %v", err)
	}
	if _, err := LoadDraftScan(cachePath, V, D, 99999, 67890); err == nil {
		t.Errorf("expected stale srcSize error")
	}
	q := make([]float32, D)
	copy(q, emb[17*D:18*D])
	if a, b := ds.Scan(q), loaded.Scan(q); a != b {
		t.Errorf("loaded DraftScan gave %d, want %d", b, a)
	}
}

// TestDraftScanRealQwen35 benchmarks DraftScan on real Qwen3.5-0.8B 248,320 x 1024 embeddings
// when present in the local hwnpu cache.
func TestDraftScanRealQwen35(t *testing.T) {
	cacheDir, err := os.UserCacheDir()
	if err != nil {
		t.Skip(err)
	}
	path := filepath.Join(cacheDir, "hwnpu", "models", "Qwen3.5-0.8B", "tables.safetensors")
	sf, err := safetensors.Open(path)
	if err != nil {
		t.Skip("no local Qwen3.5-0.8B tables.safetensors:", err)
	}
	defer sf.Close()
	const name = "model.language_model.embed_tokens.weight"
	tt, ok := sf.Lookup(name)
	if !ok {
		t.Skip("tensor missing")
	}
	V, D := tt.Shape[0], tt.Shape[1]

	tBuild := time.Now()
	ds, err := BuildDraftScan(V, D, func(r int, dst []float32) error {
		v, err := sf.Float32Range(name, r*D, (r+1)*D)
		if err != nil {
			return err
		}
		copy(dst, v)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	buildDur := time.Since(tBuild)
	maxCluster := 0
	for c := range ds.C {
		n := int(ds.Offsets[c+1] - ds.Offsets[c])
		if n > maxCluster {
			maxCluster = n
		}
	}
	t.Logf("Built real Qwen3.5-0.8B DraftScan (V=%d, D=%d, C=%d, maxCluster=%d) in %v", V, D, ds.C, maxCluster, buildDur)
	cachePath := filepath.Join(t.TempDir(), draftScanFile)
	tSave := time.Now()
	if err := ds.Save(cachePath, 1, 2); err != nil {
		t.Fatalf("Save: %v", err)
	}
	saveDur := time.Since(tSave)
	tLoad := time.Now()
	if _, err := LoadDraftScan(cachePath, V, D, 1, 2); err != nil {
		t.Fatalf("LoadDraftScan: %v", err)
	}
	t.Logf("Disk cache Save=%v, Load=%v", saveDur, time.Since(tLoad))

	rng := rand.New(rand.NewPCG(42, 7))
	const Q = 32
	queries := make([][]float32, Q)
	exact := make([]int32, Q)
	row := make([]float32, D)
	for q := range Q {
		h := make([]float32, D)
		tok0 := rng.IntN(V)
		if q%2 == 1 {
			tok0 = 100000 + rng.IntN(V-100000) // half high-ID Japanese/rare tokens
		}
		row, _ = sf.Float32Range(name, tok0*D, (tok0+1)*D)
		for j := range D {
			h[j] = row[j]*2.5 + float32(rng.NormFloat64())*0.03
		}
		for k := range 3 {
			tok := rng.IntN(V)
			row, _ = sf.Float32Range(name, tok*D, (tok+1)*D)
			w := float32(0.5) / float32(k+1)
			for j := range D {
				h[j] += row[j] * w
			}
		}
		queries[q] = h
		// Full exact int8 scan via ScanTop(h, ds.C)
		exact[q] = ds.ScanTop(h, ds.C)
	}

	// Benchmark full scan vs DraftScan B=16, 32, 64
	tFull := time.Now()
	for _, h := range queries {
		_ = ds.ScanTop(h, ds.C)
	}
	fullPerQ := time.Since(tFull) / Q
	t.Logf("Full-vocab NEON SDOT scan (248K rows): %v / query", fullPerQ)

	for _, B := range []int{32, 64, 96, 128} {
		hits := 0
		t0 := time.Now()
		for q, h := range queries {
			if ds.ScanTop(h, B) == exact[q] {
				hits++
			}
		}
		dt := time.Since(t0) / Q
		t.Logf("DraftScan B=%3d: %v / query (%.1fx vs full), top1 recall = %d/%d (%.1f%%)",
			B, dt, float64(fullPerQ)/float64(dt), hits, Q, 100*float64(hits)/Q)
		if B == 96 && hits < Q*3/4 {
			t.Errorf("B=96 recall %d/%d too low", hits, Q)
		}
	}
	_ = vecops.DotI8
}
