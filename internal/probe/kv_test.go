package probe

import (
	"fmt"
	"math"
	"math/rand/v2"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/milcpu"
)

// The cache in chunks attends as the plain cache does: the same cache bytes and the same visible
// positions give the same hidden state, through the softmax over the chunks and the token.
func TestKVChunkMatchesHost(t *testing.T) {
	defer func(c int) { KVChunk = c }(KVChunk)
	KVChunk = 64
	const layers, cache, visible = 2, 256, 150 // chunks 3 and 4 partly and wholly masked
	run := func(mode string, in func(name string, shape []int) []float32) []float32 {
		dir := filepath.Join(t.TempDir(), mode+".mlpackage")
		if err := WriteKVPackage(dir, mode, layers, cache, 7); err != nil {
			t.Fatal(err)
		}
		p, f, err := milcpu.OpenPackage(dir)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		ins := map[string]milcpu.Tensor{}
		for name, sh := range p.Inputs() {
			ins[name] = milcpu.Tensor{Shape: sh, Data: in(name, sh)}
		}
		res, err := p.Run(ins, milcpu.Options{})
		if err != nil {
			t.Fatal(err)
		}
		return res["y"].Data
	}
	values := map[string][]float32{}
	rng := rand.New(rand.NewPCG(1, 2))
	same := func(name string, n int) []float32 { // the same random values for both modes
		if v, ok := values[name]; ok {
			return v
		}
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		values[name] = v
		return v
	}
	size := func(sh []int) int {
		n := 1
		for _, d := range sh {
			n *= d
		}
		return n
	}
	mask := func(n int, own bool) []float32 {
		m := make([]float32, n)
		for i := range m {
			if i >= visible && !(own && i == n-1) {
				m[i] = -1e4
			}
		}
		return m
	}
	host := run("host", func(name string, sh []int) []float32 {
		if name == "mask" {
			return mask(cache+1, true)
		}
		return same(name, size(sh))
	})
	chunk := run("host-chunk", func(name string, sh []int) []float32 {
		if name == "x" {
			return same(name, size(sh))
		}
		// mask_J and kv_K_L_J: chunk J of the plain mask and cache
		var j int
		if _, err := fmt.Sscanf(name, "mask_%d", &j); err == nil {
			return mask(cache, false)[j*64 : (j+1)*64]
		}
		var kv string
		var layer int
		if _, err := fmt.Sscanf(strings.ReplaceAll(name, "_", " "), "kv %s %d %d", &kv, &layer, &j); err != nil {
			t.Fatalf("input %s", name)
		}
		full := same(fmt.Sprintf("kv_%s_%d", kv, layer), 4*cache*HeadDim)
		out := make([]float32, 0, size(sh))
		for h := 0; h < 4; h++ {
			out = append(out, full[(h*cache+j*64)*HeadDim:(h*cache+(j+1)*64)*HeadDim]...)
		}
		return out
	})
	var num, den float64
	for i := range host {
		d := float64(chunk[i] - host[i])
		num += d * d
		den += float64(host[i]) * float64(host[i])
	}
	if e := math.Sqrt(num / den); e > 1e-4 || math.IsNaN(e) {
		t.Errorf("relative error %g between the cache in chunks and the plain cache", e)
	} else {
		t.Logf("relative error %.2g", e)
	}
}
