package main

import (
	"runtime"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
)

// TestAudioDevice checks that on a Mac whatever HWNPU_BACKEND names of Apple's Neural Engine
// (nothing, coreml, coreai) opens Core AI's device, and that Intel's names pass.
func TestAudioDevice(t *testing.T) {
	mac := runtime.GOOS == "darwin"
	apple := map[bool]string{true: llm.BackendCoreAI, false: ""}[mac]
	for _, c := range []struct{ env, want string }{
		{"", apple},
		{"coreai", "coreai"},
		{"coreml", map[bool]string{true: "coreai", false: "coreml"}[mac]},
		{"intel", "intel"},
		{"openvino", "openvino"},
	} {
		t.Setenv(llm.BackendEnv, c.env)
		if got := audioDevice(); got != c.want {
			t.Errorf("HWNPU_BACKEND %q: %q, want %q", c.env, got, c.want)
		}
	}
}
