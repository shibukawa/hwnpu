//go:build darwin

package experts

import (
	"os"
	"sync"
	"testing"
	"time"
	"unsafe"

	"golang.org/x/sys/unix"
)

// resident is the share of b's pages in memory.
func resident(b []byte) float64 {
	page := os.Getpagesize()
	vec := make([]byte, (len(b)+page-1)/page)
	if _, _, e := unix.Syscall(unix.SYS_MINCORE, uintptr(unsafe.Pointer(&b[0])), uintptr(len(b)), uintptr(unsafe.Pointer(&vec[0]))); e != 0 {
		return -1
	}
	n := 0
	for _, v := range vec {
		if v&1 != 0 {
			n++
		}
	}
	return float64(n) / float64(len(vec))
}

// TestColdRead times reading experts that are out of memory (HWNPU_GPTOSS_CKPT) by ways the
// prefetcher could: touching the pages on one goroutine, madvise(WILLNEED) first, and touching
// on several goroutines at once.
func TestColdRead(t *testing.T) {
	h, _ := openGptOss(t)
	defer h.Close()
	type cand struct{ l, e int }
	var cold []cand
	for l := range 24 {
		for e := range 32 {
			r := 0.0
			sp := h.spans(l, e)
			for _, b := range sp {
				r += resident(b)
			}
			if r/float64(len(sp)) < 0.05 {
				cold = append(cold, cand{l, e})
			}
		}
	}
	t.Logf("%d of 768 experts less than 5%% resident", len(cold))
	page := os.Getpagesize()
	touch := func(b []byte) (s byte) {
		for j := 0; j < len(b); j += page {
			s += b[j]
		}
		return
	}
	var sink byte
	ways := []struct {
		name string
		run  func(sp [][]byte)
	}{
		{"touch 1", func(sp [][]byte) {
			for _, b := range sp {
				sink += touch(b)
			}
		}},
		{"willneed + touch 1", func(sp [][]byte) {
			for _, b := range sp {
				unix.Madvise(b, unix.MADV_WILLNEED)
			}
			for _, b := range sp {
				sink += touch(b)
			}
		}},
		{"willneed + touch 4", func(sp [][]byte) {
			for _, b := range sp {
				unix.Madvise(b, unix.MADV_WILLNEED)
			}
			par(sp, 4, page, touch)
		}},
		{"willneed 1MB pieces + touch 1", func(sp [][]byte) {
			for _, b := range sp {
				for lo := 0; lo < len(b); lo += 1 << 20 {
					unix.Madvise(b[lo:min(lo+1<<20, len(b))], unix.MADV_WILLNEED)
				}
			}
			for _, b := range sp {
				sink += touch(b)
			}
		}},
		{"willneed, wait mincore", func(sp [][]byte) {
			for _, b := range sp {
				unix.Madvise(b, unix.MADV_WILLNEED)
			}
			for _, b := range sp {
				for resident(b) < 0.999 {
					time.Sleep(50 * time.Microsecond)
				}
			}
		}},
		{"pread 1MB, 4 threads", func(sp [][]byte) { h.pread(sp, 4, 1<<20) }},
		{"pread 1MB, 8 threads", func(sp [][]byte) { h.pread(sp, 8, 1<<20) }},
		{"pread 4MB, 4 threads", func(sp [][]byte) { h.pread(sp, 4, 4<<20) }},
		{"pread 256KB, 16 threads", func(sp [][]byte) { h.pread(sp, 16, 256<<10) }},
		{"willneed + touch 1 (again)", func(sp [][]byte) {
			for _, b := range sp {
				unix.Madvise(b, unix.MADV_WILLNEED)
			}
			for _, b := range sp {
				sink += touch(b)
			}
		}},
	}
	const each = 6 // experts a way
	for i, w := range ways {
		if len(cold) < (i+1)*each {
			t.Skip("not enough cold experts")
		}
		var bytes int
		start := time.Now()
		for _, c := range cold[i*each : (i+1)*each] {
			sp := h.spans(c.l, c.e)
			for _, b := range sp {
				bytes += len(b)
			}
			w.run(sp)
		}
		el := time.Since(start)
		t.Logf("%-20s %d experts %.0f MB in %v: %.2f GB/s, %.2f ms an expert", w.name, each, float64(bytes)/1e6, el, float64(bytes)/el.Seconds()/1e9, float64(el.Microseconds())/1e3/each)
	}
	_ = unsafe.Pointer(&sink)
}

// par touches the spans' pages in pieces of 1 MB over n goroutines.
func par(sp [][]byte, n, page int, touch func([]byte) byte) {
	ch := make(chan []byte, 1024)
	for _, b := range sp {
		for lo := 0; lo < len(b); lo += 1 << 20 {
			ch <- b[lo:min(lo+1<<20, len(b))]
		}
	}
	close(ch)
	var wg sync.WaitGroup
	for range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for b := range ch {
				touch(b)
			}
		}()
	}
	wg.Wait()
}

// pread reads the spans from their files in pieces over n goroutines (the page cache keeps them
// for the mapping).
func (h *Host) pread(sp [][]byte, n, piece int) {
	type job struct {
		f   int
		off int64
		n   int
	}
	ch := make(chan job, 4096)
	for _, b := range sp {
		for i, f := range h.files {
			off, ok := f.MappedAt(b)
			if !ok {
				continue
			}
			for lo := 0; lo < len(b); lo += piece {
				ch <- job{i, off + int64(lo), min(piece, len(b)-lo)}
			}
		}
	}
	close(ch)
	var wg sync.WaitGroup
	for range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			buf := make([]byte, piece)
			for j := range ch {
				h.files[j.f].ReadAt(buf[:j.n], j.off)
			}
		}()
	}
	wg.Wait()
}
