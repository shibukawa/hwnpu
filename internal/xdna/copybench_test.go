package xdna

import (
	"os"
	"sync"
	"syscall"
	"testing"
	"time"
)

// TestCopyExpertCost: an expert's bytes from a mapped file (in the page cache) into another
// buffer, then flushed, as a miss of the experts' slots would (BONSAIAI_TEST_COPYBENCH_FILE=FILE).
func TestCopyExpertCost(t *testing.T) {
	path := os.Getenv("BONSAIAI_TEST_COPYBENCH_FILE")
	if path == "" {
		t.Skip("BONSAIAI_TEST_COPYBENCH_FILE=FILE")
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	const n = 14 << 20
	src, err := syscall.Mmap(int(f.Fd()), 0, 64*n, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		t.Fatal(err)
	}
	dst := make([]byte, n)
	for i := range dst {
		dst[i] = 1
	}
	for _, th := range []int{1, 2, 4, 8} {
		for k := 0; k < 3; k++ {
			s := src[(k+1)*n : (k+2)*n]
			t0 := time.Now()
			var wg sync.WaitGroup
			for j := 0; j < th; j++ {
				wg.Add(1)
				go func(j int) {
					defer wg.Done()
					lo, hi := j*n/th, (j+1)*n/th
					copy(dst[lo:hi], s[lo:hi])
					FlushCPU(dst[lo:hi])
				}(j)
			}
			wg.Wait()
			t.Logf("threads %d: copy+flush %.2f ms", th, time.Since(t0).Seconds()*1000)
		}
	}
	t0 := time.Now()
	FlushCPU(dst)
	t.Logf("flush alone (not dirty): %.2f ms", time.Since(t0).Seconds()*1000)
}
