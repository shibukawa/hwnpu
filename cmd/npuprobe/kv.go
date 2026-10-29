package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu/internal/probe"
)

// kvResult is one KV cache probe model run on the NPU: the cache in host buffers bound as inputs
// and outputs ("host"), or in Core ML states ("state").
type kvResult struct {
	Mode       string `json:"mode"`
	Cache      int    `json:"cache"`       // tokens the cache holds
	CacheBytes int64  `json:"cache_bytes"` // keys and values of every layer
	result
}

// kv measures the KV cache probe model of each mode and cache length.
func (p *prober) kvs(rep *report, layers int, caches []int, modes []string, save func()) {
	fmt.Printf("kv, %d layers (lut4) with attention over the cache:\n", layers)
	for _, c := range caches {
		for _, m := range modes {
			if p.ctx.Err() != nil {
				return
			}
			var k kvResult
			if err := p.child(childSpec{Kind: "kv", Layers: layers, N: c, Layout: m}, &k); err != nil {
				k = kvResult{Mode: m, Cache: c, CacheBytes: probe.KVCacheBytes(m, layers, c), result: p.fail(childSpec{Format: "lut4", Layers: layers}, err.Error())}
			}
			fmt.Printf("  %-5s cache %5d (%6.1f MiB): ", k.Mode, k.Cache, float64(k.CacheBytes)/(1<<20))
			logResult(k.result)
			rep.KV = append(rep.KV, k)
			save()
		}
	}
}

// kv is the child side: build, load on the NPU and time one model.
func (p *prober) kv(mode string, layers, cache int) kvResult {
	k := kvResult{Mode: mode, Cache: cache, CacheBytes: probe.KVCacheBytes(mode, layers, cache)}
	r := &k.result
	path := filepath.Join(p.dir, fmt.Sprintf("probe_kv_%s_%d_%d.mlpackage", mode, layers, cache))
	if err := probe.WriteKVPackage(path, mode, layers, cache, splitSeed); err != nil {
		r.Error = "build: " + err.Error()
		return k
	}
	defer os.RemoveAll(path)
	l := p.loadPart("lut4", probe.Whole(layers), "npu", splitSeed, nil, &sharedPart{path: path, fileBytes: probe.DirSize(path)}, r)
	if l == nil {
		return k
	}
	defer l.close()
	r.OnNPU = r.Error == "" && r.Verified && r.NPUOps > 0 && r.OtherOps == 0
	if err := p.time(l, r, false); err != nil {
		r.Error = "run: " + err.Error()
	}
	return k
}
