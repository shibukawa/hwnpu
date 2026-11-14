package main

import (
	"runtime"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
)

// TestChatDevice checks that on a Mac whatever names Apple's Neural Engine (nothing, coreml,
// coreai, from --backend or HWNPU_BACKEND) is Core AI's device, and that Intel's names pass.
func TestChatDevice(t *testing.T) {
	apple := ""
	if runtime.GOOS == "darwin" {
		apple = llm.BackendCoreAI
	}
	for _, c := range []struct{ flag, env, want string }{
		{"", "", apple},
		{"coreai", "", "coreai"},
		{"", "coreml", map[bool]string{true: "coreai", false: "coreml"}[apple != ""]},
		{"intel", "coreml", "intel"},
		{"openvino", "", "openvino"},
	} {
		t.Setenv(llm.BackendEnv, c.env)
		if got := chatDevice(c.flag); got != c.want {
			t.Errorf("--backend %q, HWNPU_BACKEND %q: %q, want %q", c.flag, c.env, got, c.want)
		}
	}
}
