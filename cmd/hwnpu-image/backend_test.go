package main

import (
	"runtime"
	"slices"
	"testing"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
)

// TestImageDevice checks that on a Mac whatever names Apple's Neural Engine (nothing, coreml,
// coreai, from --backend or HWNPU_BACKEND) opens Core AI's device, and that Intel's names pass.
func TestImageDevice(t *testing.T) {
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
		if got := imageDevice(c.flag); got != c.want {
			t.Errorf("--backend %q, HWNPU_BACKEND %q: %q, want %q", c.flag, c.env, got, c.want)
		}
	}
	if got := exportDevice(llm.BackendCoreML, ""); got != llm.BackendCoreAI {
		t.Errorf("a Core ML export opens %q, want coreai", got)
	}
}

// TestAMDNames: the names of an export for AMD's NPU, as tools/amd writes them and the sources
// hold them (anima-turbo-1024.amd-npu2, chroma1-flash-512.amd-npu2).
func TestAMDNames(t *testing.T) {
	m := chat.Model{Name: "Chroma1-Flash"}
	for size, want := range map[int][]string{
		0:    {"chroma1-flash.amd-npu2", "chroma1-flash-1024.amd-npu2"},
		1024: {"chroma1-flash.amd-npu2", "chroma1-flash-1024.amd-npu2"},
		512:  {"chroma1-flash-512.amd-npu2"},
	} {
		if got := amdNames(m, size); !slices.Equal(got, want) {
			t.Errorf("%d: %v, want %v", size, got, want)
		}
	}
}
