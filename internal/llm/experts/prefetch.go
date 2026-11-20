package experts

import (
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
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
	router map[int]*router
	mu     sync.Mutex
	guess  map[int][]int // by layer: the guessed experts, best first (8)
	queue  chan []byte
	rows   chan guessRow         // the rows to guess the next layer from, off the experts' path
	ranN   map[int]*atomic.Int64 // by layer: the rows run (ran)

	// for HWNPU_MOE_DEBUG: how many of the experts run the guess had in its top 4 and top 8,
	// the runs of experts not wired, and those of them brought in before
	top4, top8, runs, misses, covered, late atomic.Int64
	wg                                      sync.WaitGroup
	quit                                    chan struct{}
}

// guessRow is a row x (moe_x_l) of layer l, copied for the guess.
type guessRow struct {
	l    int
	x    []float32
	seen int64 // layer l+1's rows run when x was sent: a guess made later than l+1's row is dropped
}

type router struct {
	w, b, norm []float32 // (experts, hidden), (experts), (hidden): the router and the ffn norm's weight
}

// gptOssRouters reads the routers and the ffn norms of rec's layers (gpt-oss's names, beside
// the experts' tensors); nil when they are not there.
func gptOssRouters(rec *HostRecord, sh Shape, floats func(string, int) ([]float32, error)) map[int]*router {
	out := map[int]*router{}
	for l, n := range rec.Layers {
		p, ok := strings.CutSuffix(n.GateUpBlocks, "mlp.experts.gate_up_proj_blocks")
		if !ok {
			return nil
		}
		r := &router{}
		var err error
		if r.w, err = floats(p+"mlp.router.weight", sh.Experts*sh.Hidden); err == nil {
			if r.b, err = floats(p+"mlp.router.bias", sh.Experts); err == nil {
				r.norm, err = floats(p+"post_attention_layernorm.weight", sh.Hidden)
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
// else 4.
func prefetchN() int {
	if v, ok := DebugOpt("prefetch"); ok {
		n, err := strconv.Atoi(v)
		if err != nil || n < 0 {
			return 0
		}
		return min(n, 8)
	}
	return 4
}

func newPrefetcher(h *Host, routers map[int]*router, n int) *prefetcher {
	p := &prefetcher{h: h, n: n, router: routers, guess: map[int][]int{}, queue: make(chan []byte, 256), rows: make(chan guessRow, 4), ranN: map[int]*atomic.Int64{}, quit: make(chan struct{})}
	for l := range routers {
		p.ranN[l] = new(atomic.Int64)
	}
	p.wg.Add(2)
	go p.loop()
	go p.guessLoop()
	return p
}

// loop brings the pieces queued into memory, on readers goroutines: a piece out of memory is
// read from its file (pread on several at once: 2.7 GB/s on an M2's SSD, where the kernel's
// read-ahead of madvise(WILLNEED) gives 1.8 and faults a page at a time 0.6), then every page is
// touched (a page in the page cache still costs the process a fault when first touched).
func (p *prefetcher) loop() {
	defer p.wg.Done()
	var wg sync.WaitGroup
	for range readers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			page := os.Getpagesize()
			buf := make([]byte, piece)
			var sink byte
			for {
				select {
				case <-p.quit:
					touched.Add(uint32(sink))
					return
				case b := <-p.queue:
					if !incore(b) {
						if f, off, ok := p.h.fileAt(b); ok {
							f.ReadAt(buf[:len(b)], off)
						} else {
							willneed(b)
						}
					}
					for j := 0; j < len(b); j += page {
						sink += b[j]
					}
				}
			}
		}()
	}
	wg.Wait()
}

// readers and piece are the goroutines that bring guessed experts in, and the bytes each reads at
// a time (an expert of gpt-oss-20b is 13.3 MB).
const (
	readers = 4
	piece   = 1 << 20
)

// ran is told the experts idx a row of layer l runs, with the row x (moe_x_l): it scores the
// last guess for l and sends x for l+1's guess (guessLoop).
func (p *prefetcher) ran(l int, x []float32, idx []int) {
	E := p.h.shape.Experts
	if c := p.ranN[l]; c != nil {
		c.Add(1)
	}
	p.mu.Lock()
	g := p.guess[l]
	delete(p.guess, l)
	p.mu.Unlock()
	for _, e := range idx {
		if g != nil {
			p.runs.Add(1)
			if i := slices.Index(g, e); i >= 0 {
				if i < 4 {
					p.top4.Add(1)
				}
				p.top8.Add(1)
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
	case p.rows <- guessRow{l, slices.Clone(x), p.ranN[l+1].Load()}:
	default: // behind: no guess for l+1 rather than wait
	}
}

// guessLoop guesses the next layers of the rows sent (ran) and queues the pages of the guessed
// experts: a layer's guess (2880 × 32 multiplies, about 0.13 ms on an M2) runs beside the experts
// rather than before them.
func (p *prefetcher) guessLoop() {
	defer p.wg.Done()
	for {
		select {
		case <-p.quit:
			return
		case r := <-p.rows:
			p.guessNext(r)
		}
	}
}

// guessNext guesses layer l+1's experts from row x of layer l and queues their pages.
func (p *prefetcher) guessNext(r guessRow) {
	l, x := r.l, r.x
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
	for e := range E {
		s := next.b[e]
		row := next.w[e*H : (e+1)*H]
		for i, v := range y {
			s += row[i] * v
		}
		logits[e] = s
	}
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
	order = order[:min(8, E)]
	p.mu.Lock()
	late := p.ranN[l+1].Load() != r.seen
	if !late {
		p.guess[l+1] = order
	}
	p.mu.Unlock()
	if late {
		p.late.Add(1)
		return
	}
	pl := p.h.place
	for _, e := range order[:min(p.n, len(order))] {
		if pl != nil && pl.on[(l+1)*E+e].Load() {
			continue // wired: in memory already
		}
		for _, b := range p.h.spans(l+1, e) {
			for lo := 0; lo < len(b); lo += piece {
				select {
				case p.queue <- b[lo:min(lo+piece, len(b))]:
				default: // behind: drop the guess rather than wait
				}
			}
		}
	}
}

// guessed is whether e was among the experts of layer l brought in (the top n of l's guess).
func (p *prefetcher) guessed(l, e int) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	i := slices.Index(p.guess[l], e)
	return i >= 0 && i < p.n
}

func (p *prefetcher) stop() {
	close(p.quit)
	p.wg.Wait()
	if DebugTime() && p.runs.Load() > 0 {
		r := float64(p.runs.Load())
		line := fmt.Sprintf("experts: the next layer's guess had %.1f%% of the experts run in its top 4, %.1f%% in its top 8 (%d runs)",
			100*float64(p.top4.Load())/r, 100*float64(p.top8.Load())/r, p.runs.Load())
		if n := p.late.Load(); n > 0 {
			line += fmt.Sprintf("; %d guesses too late", n)
		}
		if m := p.misses.Load(); m > 0 {
			line += fmt.Sprintf("; of %d runs of experts not wired, %.1f%% brought in before (top %d)", m, 100*float64(p.covered.Load())/float64(m), p.n)
		}
		fmt.Fprintln(os.Stderr, line)
	}
}
