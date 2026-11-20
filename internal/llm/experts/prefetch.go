package experts

import (
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/shibukawa/hwnpu/internal/vecops"
)

// prefetcher guesses the next layer's experts while a row's experts run (one row: a generated
// token) and brings the pages of the guessed ones that are not wired into memory in the
// background, during the Neural Engine's part between the layers (Fate, arXiv 2502.12224; the
// Mixtral offloading of arXiv 2312.17238). The guess is the next layer's router on this layer's
// input: moe_x_l is rmsnorm(h)·w_l, so rmsnorm(h)·w_{l+1} = moe_x_l / w_l · w_{l+1}, and the
// router of l+1 runs on that (the residual before layer l+1's attention). The outputs do not
// change: the guess only reads pages.
type prefetcher struct {
	h      *Host
	n      int // the guessed experts brought in (the top n of the guess)
	keep   int // the guess's length, for the debug counts: twice n, at least 8
	router map[int]*router
	mu     sync.Mutex
	guess  map[int][]int // by layer: the guessed experts, best first (keep)
	queue  chan [2]int   // the experts to bring in: layer, expert
	rows   chan prefRow  // the rows to guess the next layer's experts from

	// for HWNPU_MOE_DEBUG: how many of the experts run the guess had in its top n and its
	// whole, the runs of experts not wired, and those of them brought in before
	topN, topAll, runs, misses, covered atomic.Int64
	wg                                  sync.WaitGroup
	quit                                chan struct{}
}

// prefRow is a row of layer l (moe_x_l), from which layer l+1's experts are guessed.
type prefRow struct {
	l int
	x []float32
}

// router is a layer's router (experts, hidden) with its bias (experts, nil for none) and the ffn
// norm's weight (hidden). The weight is float32 (w), or the checkpoint's bf16 mapped (raw).
type router struct {
	w, b, norm []float32
	raw        []byte
}

// logits writes the router's logits of y, with row a scratch of hidden values.
func (r *router) logits(dst, y, row []float32) {
	H := len(y)
	for e := range dst {
		w := row
		if r.raw != nil {
			vecops.BF16ToF32(row, r.raw[2*e*H:2*(e+1)*H])
		} else {
			w = r.w[e*H : (e+1)*H]
		}
		dst[e] = dot(w, y)
		if r.b != nil {
			dst[e] += r.b[e]
		}
	}
}

// hostRouters reads the routers and the ffn norms of rec's layers: the record's names (MXFP4's
// Router and Norm), or gpt-oss's beside the experts' tensors; nil when they are not there.
func hostRouters(rec *HostRecord, sh Shape, floats func(string, int) ([]float32, error), bf16 func(string, int) []byte) map[int]*router {
	out := map[int]*router{}
	for l, n := range rec.Layers {
		if n.Router == "" {
			p, ok := strings.CutSuffix(n.GateUpBlocks, "mlp.experts.gate_up_proj_blocks")
			if !ok || rec.Act.Kind != ActGptOss {
				return nil
			}
			n.Router, n.RouterBias, n.Norm = p+"mlp.router.weight", p+"mlp.router.bias", p+"post_attention_layernorm.weight"
		}
		r := &router{}
		var err error
		if r.raw = bf16(n.Router, sh.Experts*sh.Hidden); r.raw == nil {
			r.w, err = floats(n.Router, sh.Experts*sh.Hidden)
		}
		if err == nil {
			if r.b, err = floats(n.RouterBias, sh.Experts); err == nil {
				r.norm, err = floats(n.Norm, sh.Hidden)
			}
		}
		if err == nil && n.NormOffset != 0 {
			r.norm = append([]float32(nil), r.norm...) // a copy: floats may give the mapped file's
			for i := range r.norm {
				r.norm[i] += float32(n.NormOffset)
			}
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "experts: no guess of the next layer's experts: %v\n", err)
			return nil
		}
		out[l] = r
	}
	return out
}

// prefetchN is how many guessed experts are brought in: HWNPU_MOE_DEBUG's prefetch=N (0: none),
// else as many as a token runs (topK), at least 4: gpt-oss-20b's 4 of 32 (the guess's top 4 hold
// 86 % of the experts run), Qwen3.6-35B-A3B's 8 of 256 (its top 4 hold 44–47 %, its top 8
// 72–80 %).
func prefetchN(topK int) int {
	if v, ok := DebugOpt("prefetch"); ok {
		n, err := strconv.Atoi(v)
		if err != nil || n < 0 {
			return 0
		}
		return min(n, 32)
	}
	return max(topK, 4)
}

// prefetchReaders are the goroutines that bring pages in: the reads of several experts go on at
// once (an SSD reads several ranges faster than one after the other).
const prefetchReaders = 4

func newPrefetcher(h *Host, routers map[int]*router, n int) *prefetcher {
	p := &prefetcher{h: h, n: n, keep: max(2*n, 8), router: routers, guess: map[int][]int{}, queue: make(chan [2]int, 256), rows: make(chan prefRow, 4), quit: make(chan struct{})}
	for range prefetchReaders {
		p.wg.Add(1)
		go p.loop()
	}
	p.wg.Add(1)
	go p.guesser()
	return p
}

// loop brings the experts queued into memory: each is read whole from the file where it is not
// in memory (Host.readIn), then every page is touched (a page in the page cache still costs the
// process a fault when first touched).
func (p *prefetcher) loop() {
	defer p.wg.Done()
	page := os.Getpagesize()
	var sink byte
	for {
		select {
		case <-p.quit:
			touched.Add(uint32(sink))
			return
		case q := <-p.queue:
			p.h.readIn(q[0], q[1])
			for _, b := range p.h.spans(q[0], q[1]) {
				for j := 0; j < len(b); j += page {
					sink += b[j]
				}
			}
		}
	}
}

// ran is told the experts idx a row of layer l runs, with the row x (moe_x_l): it scores the
// last guess for l and has l+1 guessed (guesser), off the run's own time.
func (p *prefetcher) ran(l int, x []float32, idx []int) {
	E := p.h.shape.Experts
	p.mu.Lock()
	g := p.guess[l]
	delete(p.guess, l)
	p.mu.Unlock()
	for _, e := range idx {
		if g != nil {
			p.runs.Add(1)
			if i := slices.Index(g, e); i >= 0 {
				if i < p.n {
					p.topN.Add(1)
				}
				p.topAll.Add(1)
			}
		}
		if pl := p.h.place; pl != nil && e >= 0 && e < E && !pl.on[l*E+e].Load() {
			p.misses.Add(1)
			if g != nil && slices.Index(g, e) >= 0 && slices.Index(g, e) < p.n {
				p.covered.Add(1)
			}
		}
	}
	if p.router[l] == nil || p.router[l+1] == nil {
		return
	}
	select {
	case p.rows <- prefRow{l, append([]float32(nil), x...)}:
	default: // behind: no guess for this layer
	}
}

// guesser guesses the next layer's experts from the rows ran gives it: a router of Qwen3.6's
// (256 x 2048) takes about as long as the row's experts themselves, so it runs beside them.
func (p *prefetcher) guesser() {
	defer p.wg.Done()
	for {
		select {
		case <-p.quit:
			return
		case r := <-p.rows:
			p.guessNext(r.l, r.x)
		}
	}
}

func (p *prefetcher) guessNext(l int, x []float32) {
	E := p.h.shape.Experts
	cur, next := p.router[l], p.router[l+1]
	H := p.h.shape.Hidden
	y := make([]float32, H)
	for i, v := range x {
		if w := cur.norm[i]; w > 1e-6 || w < -1e-6 {
			y[i] = v / w * next.norm[i]
		}
	}
	logits := make([]float32, E)
	next.logits(logits, y, make([]float32, H))
	order := make([]int, E)
	for i := range order {
		order[i] = i
	}
	slices.SortFunc(order, func(a, b int) int {
		switch {
		case logits[a] > logits[b]:
			return -1
		case logits[a] < logits[b]:
			return 1
		}
		return 0
	})
	order = order[:min(p.keep, E)]
	p.mu.Lock()
	p.guess[l+1] = order
	p.mu.Unlock()
	pl := p.h.place
	for _, e := range order[:min(p.n, len(order))] {
		if pl != nil && pl.on[(l+1)*E+e].Load() {
			continue // wired: in memory already
		}
		select {
		case p.queue <- [2]int{l + 1, e}:
		default: // behind: drop the guess rather than wait
		}
	}
}

// dot is the dot product of a and b (as long), in four sums.
func dot(a, b []float32) float32 {
	var s0, s1, s2, s3 float32
	n := len(a) &^ 3
	b = b[:len(a)]
	for i := 0; i < n; i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for i := n; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return s0 + s1 + s2 + s3
}

func (p *prefetcher) stop() {
	close(p.quit)
	p.wg.Wait()
	if DebugTime() && p.runs.Load() > 0 {
		r := float64(p.runs.Load())
		line := fmt.Sprintf("experts: the next layer's guess had %.1f%% of the experts run in its top %d, %.1f%% in its top %d (%d runs)",
			100*float64(p.topN.Load())/r, p.n, 100*float64(p.topAll.Load())/r, p.keep, p.runs.Load())
		if m := p.misses.Load(); m > 0 {
			line += fmt.Sprintf("; of %d runs of experts not wired, %.1f%% brought in before (top %d)", m, 100*float64(p.covered.Load())/float64(m), p.n)
		}
		fmt.Fprintln(os.Stderr, line)
	}
}
