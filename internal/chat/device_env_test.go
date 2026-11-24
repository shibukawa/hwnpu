package chat

import (
	"testing"
)

// BONSAIAI_BACKEND naming Intel's programs or Core ML makes the machine Intel's or Apple's for the
// commands, so that BONSAIAI_BACKEND=openvino runs their Intel path on a machine without that NPU.
func TestHereVendorFollowsBackendEnv(t *testing.T) {
	for env, want := range map[string]string{"openvino": "intel", "levelzero": "intel", "intel": "intel", "coreml": "apple"} {
		t.Setenv("BONSAIAI_BACKEND", env)
		if got := hereVendor(); got != want {
			t.Errorf("BONSAIAI_BACKEND=%s: %q, want %q", env, got, want)
		}
	}
}
