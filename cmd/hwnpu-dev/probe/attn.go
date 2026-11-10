package probe

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
	a, err := p.newAttnRunner(l, layers, tokens, chunk)
	if err != nil {
		return []attnPass{{Error: err.Error()}}
	}
	defer a.close()
	var passes []attnPass
	for _, u := range used {
		if p.ctx.Err() != nil {
			break
		}
		pass := attnPass{Used: u, Layers: layers, Chunks: a.chunks(u), KVBytes: int64(layers*a.chunks(u)) * probe.AttnChunkBytes(chunk)}
		if err := a.use(u); err != nil {
			pass.Error = err.Error()
			passes = append(passes, pass)
			break
		}
		var r result
		a.merging, a.steps = 0, 0
		err := p.measure(a.step, &r, false)
		if err != nil {
			pass.Error = "run: " + err.Error()
		} else {
			pass.MsPerPass, pass.CPUUse = r.MsPerCall, r.CPUUse
			pass.MsMerge = float64(a.merging.Microseconds()) / 1000 / float64(a.steps)
		}
		pass.MaxRSS = maxRSS()
		passes = append(passes, pass)
		if err != nil {
			break
		}
	}
	return passes
}

// attnRunner calls a loaded chunk attention program over the chunks of a cache of several
// layers, each chunk in buffers of its own, and merges the outputs on the host.
type attnRunner struct {
	p              *prober
	l              *loaded
	layers, tokens int
	chunk, n       int // chunk length, and the chunks in use
	arg            map[string]hwnpu.ArgInfo
	bound          map[string]*hwnpu.Buffer
	cache          [][]chunkKV // per layer
	full, tail     *hwnpu.Buffer
	last           *hwnpu.Buffer // the mask of the last chunk in use
	merge          *probe.ChunkMerge
	o, m, s, out   []float32
	merging        time.Duration // spent reading the outputs and merging, since reset
	steps          int           // steps run, since reset
}

type chunkKV struct{ k, v *hwnpu.Buffer }

func (p *prober) newAttnRunner(l *loaded, layers, tokens, chunk int) (*attnRunner, error) {
	a := &attnRunner{p: p, l: l, layers: layers, tokens: tokens, chunk: chunk,
		arg: map[string]hwnpu.ArgInfo{}, bound: map[string]*hwnpu.Buffer{}, cache: make([][]chunkKV, layers)}
	for i, x := range l.g.Args() {
		a.arg[x.Name] = x
		a.bound[x.Name] = l.bufs[i]
	}
	a.full = a.bound["mask"] // every chunk but a partly used last one: all zero
	clear(a.full.Bytes())
	a.full.SyncToDevice(0, a.full.Size())
	var err error
	if a.tail, err = p.dev.AllocBuffer(a.arg["mask"].Bytes); err != nil {
		return nil, fmt.Errorf("buffer: %w", err)
	}
	rows := probe.QHeads * tokens
	a.merge = probe.NewChunkMerge(rows, probe.HeadDim)
	a.o, a.out = make([]float32, rows*probe.HeadDim), make([]float32, rows*probe.HeadDim)
	a.m, a.s = make([]float32, rows), make([]float32, rows)
	return a, nil
}

func (a *attnRunner) close() {
	for _, c := range a.cache {
		for _, b := range c {
			b.k.Close()
			b.v.Close()
		}
	}
	a.tail.Close()
}

func (a *attnRunner) chunks(used int) int { return (used + a.chunk - 1) / a.chunk }

// use sets the cache to used tokens: it allocates the chunks every layer is missing, filled with
// the keys loadPart wrote, and masks the unused end of the last.
func (a *attnRunner) use(used int) error {
	a.n = a.chunks(used)
	template := a.bound["k"].Bytes()
	for i := range a.cache {
		for len(a.cache[i]) < a.n {
			var c chunkKV
			var err error
			if c.k, err = a.p.dev.AllocBuffer(len(template)); err == nil {
				if c.v, err = a.p.dev.AllocBuffer(len(template)); err != nil {
					c.k.Close()
				}
			}
			if err != nil {
				return fmt.Errorf("buffer: %w", err)
			}
			copy(c.k.Bytes(), template)
			copy(c.v.Bytes(), template)
			c.k.SyncToDevice(0, len(template))
			c.v.SyncToDevice(0, len(template))
			a.cache[i] = append(a.cache[i], c)
		}
	}
	a.last = a.full
	if rest := used % a.chunk; rest != 0 {
		vals := make([]float32, a.tokens*a.chunk)
		for row := 0; row < a.tokens; row++ {
			for c := rest; c < a.chunk; c++ {
				vals[row*a.chunk+c] = -1e4
			}
		}
		hwnpu.EncodeF16(a.tail.Bytes(), vals)
		a.tail.SyncToDevice(0, a.tail.Size())
		a.last = a.tail
	}
	return nil
}

// layer runs layer i's attention: the program once per chunk in use, the outputs merged.
func (a *attnRunner) layer(i int) error {
	bind := func(name string, b *hwnpu.Buffer) error { return a.l.req.Bind(a.arg[name].Index, b) }
	a.merge.Reset()
	for j, c := range a.cache[i][:a.n] {
		mask := a.full
		if j == a.n-1 {
			mask = a.last
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
		if err := a.l.req.Run(a.p.ctx); err != nil {
			return err
		}
		t := time.Now()
		for _, x := range []struct {
			name string
			dst  []float32
		}{{"o", a.o}, {"m", a.m}, {"l", a.s}} {
			b := a.bound[x.name]
			b.SyncFromDevice(0, b.Size())
			hwnpu.DecodeF16(x.dst, b.Bytes()[:a.arg[x.name].Bytes])
		}
		a.merge.Add(a.o, a.m, a.s)
		a.merging += time.Since(t)
	}
	a.merge.Result(a.out)
	return nil
}

// step runs every layer's attention once.
func (a *attnRunner) step() error {
	for i := 0; i < a.layers; i++ {
		if err := a.layer(i); err != nil {
			return err
		}
	}
	a.steps++
	return nil
}
