package experts

import (
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

// reader brings experts' pages into memory from the checkpoint's files, on readers goroutines.
// A piece out of memory is read from its file (pread on several at once: 2.7 GB/s on an M2's
// SSD, where the kernel's read-ahead of madvise(WILLNEED) gives 1.8 and faults a page at a time,
// what running an expert out of memory does, 0.6), then every page is touched (a page in the page
// cache still costs the process a fault when first touched).
//
// Two kinds of work: the prefetcher's guesses (queue, dropped when the readers are behind) and
// the experts a run is about to use that are out of memory (demand, which the run waits for and
// reads with them).
type reader struct {
	h      *Host
	queue  chan []byte
	urgent chan *demand
	quit   chan struct{}
	wg     sync.WaitGroup
	bufs   sync.Pool

	// for HWNPU_DEV's moe.time: the runs that waited for a demand, its pieces read and the wait
	demands, read atomic.Int64
	waited        atomic.Int64
	coldAt        atomic.Int64 // when a run last found experts out of memory (UnixNano)
}

// shortFor is how long after a run found experts out of memory the memory counts as short.
const shortFor = 10 * time.Second

// short is whether a run found experts out of memory lately: the page cache does not hold the
// experts not wired, and what is read for a guess pushes out pages in use.
func (r *reader) short() bool {
	return time.Duration(time.Now().UnixNano()-r.coldAt.Load()) < shortFor
}

// readers and piece are the goroutines that bring experts in, and the bytes each reads at a
// time (an expert of gpt-oss-20b is 13.3 MB).
const (
	readers = 4
	piece   = 1 << 20
)

// demand is the pieces a run waits for, taken one by one by the readers and the run itself.
type demand struct {
	pieces [][]byte
	next   atomic.Int64
	wg     sync.WaitGroup
}

func newReader(h *Host) *reader {
	r := &reader{h: h, queue: make(chan []byte, 256), urgent: make(chan *demand, readers), quit: make(chan struct{})}
	r.bufs.New = func() any { b := make([]byte, piece); return &b }
	for range readers {
		r.wg.Add(1)
		go func() {
			defer r.wg.Done()
			buf := make([]byte, piece)
			for {
				// a run's demand before the guesses
				select {
				case d := <-r.urgent:
					r.work(d, buf)
					continue
				default:
				}
				select {
				case <-r.quit:
					return
				case d := <-r.urgent:
					r.work(d, buf)
				case b := <-r.queue:
					r.bring(b, buf)
				}
			}
		}()
	}
	return r
}

// bring has b (a piece) in memory and mapped; it reports whether it was read from the file.
func (r *reader) bring(b, buf []byte) (read bool) {
	if !incore(b) {
		if f, off, ok := r.h.fileAt(b); ok {
			f.ReadAt(buf[:len(b)], off)
		} else {
			willneed(b)
		}
		read = true
	}
	page := os.Getpagesize()
	var sink byte
	for j := 0; j < len(b); j += page {
		sink += b[j]
	}
	touched.Add(uint32(sink))
	return read
}

func (r *reader) work(d *demand, buf []byte) {
	for {
		i := int(d.next.Add(1)) - 1
		if i >= len(d.pieces) {
			return
		}
		if r.bring(d.pieces[i], buf) {
			r.read.Add(1)
		}
		d.wg.Done()
	}
}

// pieces cuts spans into pieces.
func pieces(dst [][]byte, spans [][]byte) [][]byte {
	for _, b := range spans {
		for lo := 0; lo < len(b); lo += piece {
			dst = append(dst, b[lo:min(lo+piece, len(b))])
		}
	}
	return dst
}

// guess queues the pages of expert e of layer l for the readers, as far as they keep up.
func (r *reader) guess(l, e int) {
	for _, b := range pieces(nil, r.h.spans(l, e)) {
		select {
		case r.queue <- b:
		default: // behind: drop the guess rather than wait
		}
	}
}

// need has the experts idx (each row's) of layer l in memory before they run: those not wired
// with pages out of memory (cold) are read now by the readers and the caller together. Running
// them from the mapping instead faults their pages in one at a time: on an M2 an expert out of
// memory then costs a row's layer 15–17 ms, and 5 read like this.
func (r *reader) need(l int, idx [][]int) {
	E := r.h.shape.Experts
	var seen [64]bool
	var ps [][]byte
	for _, row := range idx {
		for _, e := range row {
			if e < 0 || e >= E || e < len(seen) && seen[e] {
				continue
			}
			if e < len(seen) {
				seen[e] = true
			}
			if r.h.cold(l, e) {
				ps = pieces(ps, r.h.spans(l, e))
			}
		}
	}
	if len(ps) == 0 {
		return
	}
	start, before := time.Now(), r.read.Load()
	d := &demand{pieces: ps}
	d.wg.Add(len(ps))
	for range min(readers, len(ps)-1) {
		select {
		case r.urgent <- d:
		default:
		}
	}
	buf := r.bufs.Get().(*[]byte)
	r.work(d, *buf)
	r.bufs.Put(buf)
	d.wg.Wait()
	if r.read.Load() > before {
		r.coldAt.Store(time.Now().UnixNano())
	}
	r.demands.Add(1)
	r.waited.Add(int64(time.Since(start)))
}

// cold is whether expert e of layer l is not wired and has pages out of memory, by 8 pages spread
// over its spans (asking for every page takes 0.2 ms an expert, these 4 µs): the system drops
// an expert's pages together, having last used them together.
func (h *Host) cold(l, e int) bool {
	if pl := h.place; pl != nil && pl.on[l*h.shape.Experts+e].Load() {
		return false
	}
	page := os.Getpagesize()
	for _, b := range h.spans(l, e) {
		n := len(b) / page
		k := max(min(n, len(b)*8/int(h.expertBytes())+1), 1) // the pages asked of this span: by its share of 8
		for i := range k {
			off := n * i / k * page
			if !incore(b[off:min(off+page, len(b))]) {
				return true
			}
		}
	}
	return false
}

func (r *reader) stop() {
	close(r.quit)
	r.wg.Wait()
	if n := r.demands.Load(); n > 0 && DebugTime() {
		fmt.Fprintf(os.Stderr, "experts: %d runs waited for experts out of memory, %.2f ms each (%.1f GB read)\n", n,
			float64(r.waited.Load())/1e6/float64(n), float64(r.read.Load())*piece/1e9)
	}
}
