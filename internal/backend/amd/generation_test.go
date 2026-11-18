package amd

import "testing"

// TestXDNAOf: the NPU ids models.yaml's npu.amd generations name.
func TestXDNAOf(t *testing.T) {
	for dev, want := range map[string]string{"0x1502": "xdna1", "0x17F0": "xdna2", "0x9999": ""} {
		if got := xdnaOf(Node{Device: dev}); got != want {
			t.Errorf("%s: %q, want %q", dev, got, want)
		}
	}
}
