package mxfp4

import (
	"fmt"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
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

// Run runs the jobs over the CPU's cores.
func Run(jobs ...Job) {
	type span struct{ job, lo int }
	var spans []span
	for j, jb := range jobs {
		for lo := 0; lo < jb.Rows(); lo += chunk {
			spans = append(spans, span{j, lo})
		}
	}
	// with SMMLA, the vectors in groups of 8 as 4 pairs (mm28Kernel): 2 rows at a time
	pairs := make([][]pairVec, len(jobs))
	if haveI8MM {
		for j, jb := range jobs {
			if nb := len(jb.E) / max(jb.Rows(), 1); nb == 0 || nb%2 != 0 {
				continue
			}
			for k := 0; k+8 <= len(jb.V); k += 2 {
				pairs[j] = append(pairs[j], interleave(jb.V[k], jb.V[k+1]))
			}
			pairs[j] = pairs[j][:len(pairs[j])/4*4]
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
							a, b := 2*(g+p), 2*(g+p)+1
							jb.Out[a][r], jb.Out[b][r] = sixteen[4*p], sixteen[4*p+1]
							jb.Out[a][r+1], jb.Out[b][r+1] = sixteen[4*p+2], sixteen[4*p+3]
						}
					}
					k = 2 * len(ps)
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

// Spin is how long the pool's workers that poll (Spinners) keep polling for the next Run before
// they sleep (HWNPU_MXFP4_SPIN, a duration). Workers that sleep between Runs wake slowly onto cores
// the system has slowed: on an M3, gpt-oss-20b's experts of a layer take 0.93 ms back to back,
// 1.77 ms with 1 ms between and 2.91 ms with 3 ms, and 1.08 and 1.14 ms with every worker polling
// for 5 ms. In hwnpu-chat the Neural Engine's parts run between them, and polling workers slow
// those parts as much as they save (2 of 8: the experts 1.62 to 1.27 ms a layer, 15.8 tokens/s
// either way; 4: 12.0 tokens/s), so none polls by default.
var Spin = 5 * time.Millisecond

// Spinners is how many of the workers poll (HWNPU_MXFP4_SPINNERS); the others sleep at once.
var Spinners = 0

func init() {
	if v := os.Getenv("HWNPU_MXFP4_SPIN"); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			Spin = d
		}
	}
	if v := os.Getenv("HWNPU_MXFP4_SPINNERS"); v != "" {
		fmt.Sscan(v, &Spinners)
	}
}

// workers are goroutines kept for Run, started on first use.
type workers struct {
	sync.Mutex // held by a Run that uses them; another at the same time starts its own
	started    int
	gen        atomic.Uint64 // bumped for each Run
	task       atomic.Pointer[func()]
	want       atomic.Int64 // the workers the current Run takes
	done       sync.WaitGroup
	mu         sync.Mutex
	cond       *sync.Cond
	sleeping   int
}

var pool workers

func (p *workers) run(work func(), n int) {
	if p.cond == nil {
		p.cond = sync.NewCond(&p.mu)
	}
	for p.started < n {
		go p.loop(p.started < Spinners)
		p.started++
	}
	p.task.Store(&work)
	p.want.Store(int64(n))
	p.done.Add(n)
	p.gen.Add(1)
	p.mu.Lock()
	if p.sleeping > 0 {
		p.cond.Broadcast()
	}
	p.mu.Unlock()
	work()
	p.done.Wait()
}

func (p *workers) loop(spin bool) {
	var last uint64
	for {
		g := p.gen.Load()
		if g == last && spin {
			for t := time.Now(); g == last && time.Since(t) < Spin; g = p.gen.Load() {
			}
		}
		if g == last {
			p.mu.Lock()
			for g = p.gen.Load(); g == last; g = p.gen.Load() {
				p.sleeping++
				p.cond.Wait()
				p.sleeping--
			}
			p.mu.Unlock()
		}
		last = g
		// take a place in this Run, or sit it out (a Run of fewer workers than started)
		if p.want.Add(-1) < 0 {
			continue
		}
		(*p.task.Load())()
		p.done.Done()
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
