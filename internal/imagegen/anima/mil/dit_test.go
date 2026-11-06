package mil

import "testing"

// TestDitQChunk keeps a chunk's self-attention scores within 64 MiB: 512 at 1024 x 1024 (4096
// patches, exactly 64 MiB), halved past it.
func TestDitQChunk(t *testing.T) {
	for _, c := range []struct{ w, h, in, want int }{
		{1024, 1024, 512, 512},
		{1024, 1024, 1024, 512},
		{1280, 1280, 512, 256},
		{512, 512, 512, 512},
		{512, 512, 0, 1024}, // all 1024 queries: 32 MiB
		{2048, 2048, 512, 128},
	} {
		if got := DitQChunk(c.w, c.h, c.in); got != c.want {
			t.Errorf("%dx%d, %d: %d, want %d", c.w, c.h, c.in, got, c.want)
		}
	}
}
