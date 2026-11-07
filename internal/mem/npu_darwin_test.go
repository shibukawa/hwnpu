package mem

import "testing"

// The Neural Engine's driver site is found on a Mac that has one, and reading it twice does not
// leak or fail.
func TestNPUWired(t *testing.T) {
	for range 3 {
		n, ok := NPUWired()
		if !ok {
			t.Skip("no Neural Engine driver site on this Mac")
		}
		if n < 0 || n > 64<<30 {
			t.Fatalf("Neural Engine wired memory %d bytes", n)
		}
		t.Logf("%.1f MiB", float64(n)/(1<<20))
	}
}
