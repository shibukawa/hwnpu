package mxfp4

import (
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
)

// Job is one matrix's products with vectors: Out[k][r] is row r's dot product with V[k], the
// rows' blocks W (rows, blocks, 16) and their scales E (rows, blocks). A row is read once for all
// the vectors (the rows of a prompt that chose the same expert).
type Job struct {
	Out  [][]float32
	W, E []byte
	V    []*Vec
}

// Rows is the number of the matrix's rows.
func (j Job) Rows() int { return len(j.Out[0]) }

// chunk is the rows a core takes at a time: the cores of an M3 are not equally fast (4 of
// performance, 4 of efficiency), so they take the next chunk as they finish one.
const chunk = 32

// padFrom is how many vectors past a multiple of 8 take a group of SMMLA padded with zeros
// (HWNPU_MXFP4=pad=N).
var padFrom = 6

// Run runs the jobs over the CPU's cores.
func Run(jobs ...Job) {
	type span struct{ job, lo int }
	var spans []span
	for j, jb := range jobs {
		for lo := 0; lo < jb.Rows(); lo += chunk {
			spans = append(spans, span{j, lo})
		}
	}
	// with SMMLA, the vectors in groups of 8 as 4 pairs (mm28Kernel): 2 rows at a time. The
	// vectors past the last 8 make a group of their own padded with zeros when there are padFrom
	// of them or more (their products go nowhere): a group costs about what 8 vectors do, and the
	// kernels of 4 vectors and of one decode a row again for each call.
	pairs := make([][]pairVec, len(jobs))
	if haveI8MM {
		for j, jb := range jobs {
			if nb := len(jb.E) / max(jb.Rows(), 1); nb == 0 || nb%2 != 0 {
				continue
			}
			n := len(jb.V) / 8 * 8
			if len(jb.V)-n >= padFrom {
				n += 8
			}
			if n == 0 {
				continue
			}
			var zero *Vec
			vec := func(k int) *Vec {
				if k < len(jb.V) {
					return jb.V[k]
				}
				if zero == nil {
					zero = NewVec(len(jb.V[0].Q))
				}
				return zero
			}
			for k := 0; k < n; k += 2 {
				pairs[j] = append(pairs[j], interleave(vec(k), vec(k+1)))
			}
		}
	}
	var next atomic.Int64
	work := func() {
		var sixteen [16]float32
		for {
			i := int(next.Add(1)) - 1
			if i >= len(spans) {
				return
			}
			jb := jobs[spans[i].job]
			rows := jb.Rows()
			nb := len(jb.E) / rows
			ps := pairs[spans[i].job]
			hi := min(spans[i].lo+chunk, rows)
			for r := spans[i].lo; r < hi; r++ {
				w, e := jb.W[r*nb*16:(r+1)*nb*16], jb.E[r*nb:(r+1)*nb]
				k := 0
				if len(ps) > 0 && r+1 < hi {
					w2, e2 := jb.W[r*nb*16:(r+2)*nb*16], jb.E[r*nb:(r+2)*nb]
					for g := 0; g < len(ps); g += 4 {
						mm28Kernel(w2, e2, (*[4]pairVec)(ps[g:g+4]), &sixteen)
						for p := range 4 {
							if a := 2 * (g + p); a < len(jb.Out) {
								jb.Out[a][r], jb.Out[a][r+1] = sixteen[4*p], sixteen[4*p+2]
							}
							if b := 2*(g+p) + 1; b < len(jb.Out) {
								jb.Out[b][r], jb.Out[b][r+1] = sixteen[4*p+1], sixteen[4*p+3]
							}
						}
					}
					k = min(2*len(ps), len(jb.V))
					w1, e1 := w2[nb*16:], e2[nb:]
					var four [4]float32
					for ; k+4 <= len(jb.V); k += 4 {
						v4 := (*[4]*Vec)(jb.V[k : k+4])
						dot4Kernel(w, e, v4, &four)
						for i, f := range four {
							jb.Out[k+i][r] = f
						}
						dot4Kernel(w1, e1, v4, &four)
						for i, f := range four {
							jb.Out[k+i][r+1] = f
						}
					}
					for ; k < len(jb.V); k++ {
						jb.Out[k][r] = Dot(w, e, jb.V[k])
						jb.Out[k][r+1] = Dot(w1, e1, jb.V[k])
					}
					r++
					continue
				}
				var four [4]float32
				for ; k+4 <= len(jb.V) && nb > 0; k += 4 {
					if !dot4Kernel(w, e, (*[4]*Vec)(jb.V[k:k+4]), &four) {
						break
					}
					for i, f := range four {
						jb.Out[k+i][r] = f
					}
				}
				for ; k < len(jb.V); k++ {
					jb.Out[k][r] = Dot(w, e, jb.V[k])
				}
			}
		}
	}
	workers := min(runtime.GOMAXPROCS(0), len(spans))
	if workers > 1 && pool.TryLock() {
		pool.run(work, workers-1)
		pool.Unlock()
		return
	}
	var wg sync.WaitGroup
	for range workers - 1 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			work()
		}()
	}
	work()
	wg.Wait()
}

// Keeping the workers awake over a layer's three Runs alone (polling in Go between them, 2026-11-20)
// was slower too: Qwen3.6-35B-A3B's experts of a layer 0.55 → 1.0 ms back to back, the polling
// goroutines in the way of the scheduler. One Run a layer instead (experts.Host.runRow).
//
// Workers sleep between Runs. Polling instead (tried 2026-11-19, M3) keeps them on fast cores:
// gpt-oss-20b's experts of a layer take 0.93 ms back to back, 1.77 ms with 1 ms between and 1.08
// with every worker polling for 5 ms. In hwnpu-chat the Neural Engine's parts run between them,
// and polling workers slowed those parts as much as they saved (2 of 8: the experts 1.62 to 1.27
// ms a layer, 15.8 tokens/s either way; 4: 12.0 tokens/s), so it was taken out
// (docs/design/performance.md).

func init() {
	if v, ok := Opt("pad"); ok {
		fmt.Sscan(v, &padFrom)
	}
}

// Opt reads an item of HWNPU_MXFP4, a list of NAME=VALUE separated by commas, for measuring:
// i8mm=0 runs without SMMLA (the SDOT kernels), pad=N the vectors past a multiple of 8 that take
// a padded group of SMMLA (6), bench=LAYERS the layers TestTokenTime times.
func Opt(name string) (string, bool) {
	for _, it := range strings.Split(os.Getenv("HWNPU_MXFP4"), ",") {
		if k, v, _ := strings.Cut(strings.TrimSpace(it), "="); k == name {
			return v, true
		}
	}
	return "", false
}

// workers are goroutines kept for Run, started on first use.
type workers struct {
	sync.Mutex // held by a Run that uses them; another at the same time starts its own
	started    int
	cur        atomic.Pointer[poolRun] // the Run they are woken for
	mu         sync.Mutex
	cond       *sync.Cond
	sleeping   int
}

// poolRun is one Run's work for the workers.
type poolRun struct {
	work   func()
	want   atomic.Int64 // the workers it takes yet
	active atomic.Int64 // the workers at it
	closed atomic.Bool  // its work is given out: a worker that wakes now has none
}

var pool workers

// run runs work on the caller and on up to n workers, and returns when all of it is done: work
// returns to a worker when nothing of it is left to take. The caller does not wait for the workers
// to wake: one that wakes after the caller's work returned finds the Run closed. (It waited for
// every one to wake and report, 2026-11-20: after a pause the last of 7 took as long as the work.)
func (p *workers) run(work func(), n int) {
	if p.cond == nil {
		p.cond = sync.NewCond(&p.mu)
	}
	for p.started < n {
		go p.loop()
		p.started++
	}
	r := &poolRun{work: work}
	r.want.Store(int64(n))
	p.cur.Store(r)
	p.mu.Lock()
	if p.sleeping > 0 {
		p.cond.Broadcast()
	}
	p.mu.Unlock()
	work()
	// (a worker counts itself at the Run before it looks whether it is closed, so one of the
	// two sees the other)
	r.closed.Store(true)
	for r.active.Load() != 0 {
		runtime.Gosched()
	}
}

func (p *workers) loop() {
	var last *poolRun
	for {
		r := p.cur.Load()
		if r == last {
			p.mu.Lock()
			for r = p.cur.Load(); r == last; r = p.cur.Load() {
				p.sleeping++
				p.cond.Wait()
				p.sleeping--
			}
			p.mu.Unlock()
		}
		last = r
		// take a place in this Run, or sit it out (a Run of fewer workers than started)
		if r.want.Add(-1) < 0 {
			continue
		}
		r.active.Add(1)
		if !r.closed.Load() {
			r.work()
		}
		r.active.Add(-1)
	}
}

// Parallel runs f(0) to f(n-1) over the CPU's cores (Run's workers), each i once.
func Parallel(n int, f func(i int)) {
	var next atomic.Int64
	work := func() {
		for {
			i := int(next.Add(1)) - 1
			if i >= n {
				return
			}
			f(i)
		}
	}
	workers := min(runtime.GOMAXPROCS(0), n)
	if workers > 1 && pool.TryLock() {
		pool.run(work, workers-1)
		pool.Unlock()
		return
	}
	work()
}

// MatVec is Run of one job.
func MatVec(out []float32, w, e []byte, v *Vec) {
	Run(Job{Out: [][]float32{out}, W: w, E: e, V: []*Vec{v}})
}
