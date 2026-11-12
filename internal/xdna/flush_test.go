//go:build amd64

// The test flushes with CLFLUSH (flush_amd64.go); elsewhere FlushCPU does nothing.

package xdna

import "testing"

func TestFlushCPU(t *testing.T) {
	b := make([]byte, 3*cacheLine+5)
	for i := range b {
		b[i] = byte(i)
	}
	for _, r := range [][2]int{{0, len(b)}, {1, 2}, {63, 65}, {10, 10}} {
		FlushCPU(b[r[0]:r[1]])
	}
	for i := range b {
		if b[i] != byte(i) {
			t.Fatal("flushing changed the data")
		}
	}
}
