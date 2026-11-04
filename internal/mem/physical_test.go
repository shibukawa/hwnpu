package mem

import (
	"runtime"
	"testing"
)

// The machine has at least 1 GiB and less than 64 TiB, where the system tells.
func TestPhysical(t *testing.T) {
	n := Physical()
	if runtime.GOOS != "darwin" && runtime.GOOS != "linux" {
		t.Skip("not known here")
	}
	if n < 1<<30 || n > 64<<40 {
		t.Errorf("physical memory %d bytes", n)
	}
	t.Logf("%.1f GiB", float64(n)/(1<<30))
}
