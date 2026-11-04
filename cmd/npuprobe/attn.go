package main

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/probe"
)

// attnResult is one chunk attention probe model (probe.WriteChunkAttnPackage): its placement and
// time per call with one chunk bound, and passes over caches of several used lengths.
type attnResult struct {
	Form   string     `json:"form"`   // grouped or tile
	Chunk  int        `json:"chunk"`  // tokens of a chunk
	Passes []attnPass `json:"passes"` // result.Tokens is the query rows per call
	result
}

// attnPass is the attention of one step over a cache of Used tokens: for each of Layers layers,
// the program called once per chunk holding the used tokens, each chunk its own buffers, and the
// outputs merged on the host (probe.ChunkMerge).
type attnPass struct {
	Used      int     `json:"used"`
	Layers    int     `json:"layers"`
	Chunks    int     `json:"chunks"`   // per layer
	KVBytes   int64   `json:"kv_bytes"` // the buffers of every layer's chunks
	MsPerPass float64 `json:"ms_per_pass,omitempty"`
	MsMerge   float64 `json:"ms_merge,omitempty"` // of a pass, reading the outputs and merging them
	CPUUse    float64 `json:"cpu_use,omitempty"`
	MaxRSS    int64   `json:"max_rss,omitempty"` // of the measuring process, after this pass
	Error     string  `json:"error,omitempty"`
}

// msPerCall is the time of one call within a pass, merging included.
func (a attnPass) msPerCall() float64 {
	if a.MsPerPass == 0 || a.Chunks == 0 {
		return 0
	}
	return a.MsPerPass / float64(a.Layers*a.Chunks)
}

// attns measures the chunk attention program of each form, query rows and chunk length.
func (p *prober) attns(rep *report, layers int, chunks, tokens []int, forms []string, used []int, save func()) {
	fmt.Printf("attn, attention without weights over one chunk, %d layers per pass:\n", layers)
	for _, f := range forms {
		for _, q := range tokens {
			for _, c := range chunks {
				if p.ctx.Err() != nil {
					return
				}
				var a attnResult
				if err := p.child(childSpec{Kind: "attn", Layers: layers, Tokens: q, N: c, Layout: f, Used: used}, &a); err != nil {
					a = attnResult{Form: f, Chunk: c, result: result{Format: "none", Tokens: q, Units: "npu", Error: err.Error()}}
				}
				fmt.Printf("  %-7s Q=%-3d chunk %5d: ", a.Form, a.Tokens, a.Chunk)
				logResult(a.result)
				for _, s := range a.Passes {
					fmt.Printf("      used %6d: %4d chunks/layer, %7.2f ms/pass (%.3f ms/call, merge %.2f ms), KV %.0f MiB, peak RSS %.2f GiB%s\n",
						s.Used, s.Chunks, s.MsPerPass, s.msPerCall(), s.MsMerge, float64(s.KVBytes)/(1<<20), probe.GiB(s.MaxRSS), errSuffix(s.Error))
				}
				rep.Attn = append(rep.Attn, a)
				save()
			}
		}
	}
}

func errSuffix(e string) string {
	if e == "" {
		return ""
	}
	return "  error: " + short(e, 160)
}

// attn is the child side: build, load on the NPU, time one call, then the passes.
func (p *prober) attn(form string, layers, tokens, chunk int, used []int) attnResult {
	a := attnResult{Form: form, Chunk: chunk}
	r := &a.result
	path := filepath.Join(p.dir, fmt.Sprintf("probe_attn_%s_q%d_c%d.mlpackage", form, tokens, chunk))
	if err := probe.WriteChunkAttnPackage(path, form, tokens, chunk); err != nil {
		r.Error = "build: " + err.Error()
		return a
	}
	defer os.RemoveAll(path)
	l := p.loadPart("none", probe.Whole(0), "npu", splitSeed, nil, &sharedPart{path: path, fileBytes: probe.DirSize(path)}, r)
	r.Tokens = tokens
	if l == nil {
		return a
	}
	defer l.close()
	r.OnNPU = r.Error == "" && r.Verified && r.NPUOps > 0 && r.OtherOps == 0
	// Core ML puts a program with too little work on the CPU (Q=1 over 1024 keys, M3, macOS
	// 27.0.1). The passes run wherever it is placed: attention over a chunk is small enough for
	// the CPU, and its time there is the alternative to weigh against the Neural Engine's.
	if err := p.time(l, r, false); err != nil {
		r.Error = "run: " + err.Error()
		return a
	}
	a.Passes = p.attnPasses(l, layers, tokens, chunk, used)
	return a
}

// attnPasses times a step's attention over caches of the used lengths, ascending, the chunk
// buffers allocated as the cache grows and kept for the longer ones.
func (p *prober) attnPasses(l *loaded, layers, tokens, chunk int, used []int) []attnPass {
	arg := map[string]hwnpu.ArgInfo{}
	bound := map[string]*hwnpu.Buffer{}
	for i, a := range l.g.Args() {
		arg[a.Name] = a
		bound[a.Name] = l.bufs[i]
	}
	template := bound["k"].Bytes() // filled by loadPart; every chunk gets a copy
	type kv struct{ k, v *hwnpu.Buffer }
	cache := make([][]kv, layers)
	defer func() {
		for _, c := range cache {
			for _, b := range c {
				b.k.Close()
				b.v.Close()
			}
		}
	}()
	full := bound["mask"] // every chunk but a partly used last one: all zero
	clear(full.Bytes())
	full.SyncToDevice(0, full.Size())
	tail, err := p.dev.AllocBuffer(arg["mask"].Bytes)
	if err != nil {
		return []attnPass{{Error: "buffer: " + err.Error()}}
	}
	defer tail.Close()
	merge := probe.NewChunkMerge(probe.QHeads*tokens, probe.HeadDim)
	o := make([]float32, probe.QHeads*tokens*probe.HeadDim)
	m, s := make([]float32, probe.QHeads*tokens), make([]float32, probe.QHeads*tokens)
	out := make([]float32, len(o))

	var passes []attnPass
	for _, u := range used {
		if p.ctx.Err() != nil {
			break
		}
		n := (u + chunk - 1) / chunk
		pass := attnPass{Used: u, Layers: layers, Chunks: n, KVBytes: int64(layers*n) * probe.AttnChunkBytes(chunk)}
		for i := range cache {
			for len(cache[i]) < n && pass.Error == "" {
				var c kv
				if c.k, err = p.dev.AllocBuffer(len(template)); err == nil {
					if c.v, err = p.dev.AllocBuffer(len(template)); err != nil {
						c.k.Close()
					}
				}
				if err != nil {
					pass.Error = "buffer: " + err.Error()
					break
				}
				copy(c.k.Bytes(), template)
				copy(c.v.Bytes(), template)
				c.k.SyncToDevice(0, len(template))
				c.v.SyncToDevice(0, len(template))
				cache[i] = append(cache[i], c)
			}
		}
		if pass.Error != "" {
			passes = append(passes, pass)
			break
		}
		lastMask := full
		if rest := u % chunk; rest != 0 {
			vals := make([]float32, tokens*chunk)
			for row := 0; row < tokens; row++ {
				for c := rest; c < chunk; c++ {
					vals[row*chunk+c] = -1e4
				}
			}
			hwnpu.EncodeF16(tail.Bytes(), vals)
			tail.SyncToDevice(0, tail.Size())
			lastMask = tail
		}
		var merging time.Duration
		bind := func(name string, b *hwnpu.Buffer) error { return l.req.Bind(arg[name].Index, b) }
		run := func() error {
			for i := 0; i < layers; i++ {
				merge.Reset()
				for j, c := range cache[i][:n] {
					mask := full
					if j == n-1 {
						mask = lastMask
					}
					if err := bind("k", c.k); err != nil {
						return err
					}
					if err := bind("v", c.v); err != nil {
						return err
					}
					if err := bind("mask", mask); err != nil {
						return err
					}
					if err := l.req.Run(p.ctx); err != nil {
						return err
					}
					t := time.Now()
					for _, x := range []struct {
						name string
						dst  []float32
					}{{"o", o}, {"m", m}, {"l", s}} {
						b := bound[x.name]
						b.SyncFromDevice(0, b.Size())
						hwnpu.DecodeF16(x.dst, b.Bytes()[:arg[x.name].Bytes])
					}
					merge.Add(o, m, s)
					merging += time.Since(t)
				}
				merge.Result(out)
			}
			return nil
		}
		var r result
		calls0 := 0
		err := p.measure(func() error { calls0++; return run() }, &r, false)
		if err != nil {
			pass.Error = "run: " + err.Error()
		} else {
			pass.MsPerPass, pass.CPUUse = r.MsPerCall, r.CPUUse
			pass.MsMerge = float64(merging.Microseconds()) / 1000 / float64(calls0)
		}
		pass.MaxRSS = maxRSS()
		passes = append(passes, pass)
		if err != nil {
			break
		}
	}
	return passes
}
