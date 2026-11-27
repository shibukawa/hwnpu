package imagecmd

import (
	"slices"
	"testing"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagewire"
	"github.com/shibukawa/hwnpu/internal/llm"
)

// TestAMDNames: the names of an export for AMD's NPU, as tools/amd writes them and the sources
// hold them (anima-turbo-1024.amd-npu2, anima-turbo-512.amd-npu2).
func TestAMDNames(t *testing.T) {
	m := chat.Model{Name: "Anima-Turbo"}
	for size, want := range map[int][]string{
		0:    {"anima-turbo.amd-npu2", "anima-turbo-1024.amd-npu2"},
		1024: {"anima-turbo.amd-npu2", "anima-turbo-1024.amd-npu2"},
		512:  {"anima-turbo-512.amd-npu2"},
	} {
		if got := chat.AMDPictureNames(m, size); !slices.Equal(got, want) {
			t.Errorf("%d: %v, want %v", size, got, want)
		}
	}
}

// TestVendor: imagewire.Vendor reads a backend's name as internal/chat's HereVendor reads
// HWNPU_BACKEND.
func TestVendor(t *testing.T) {
	for _, b := range []string{"", "coreml", "coreai", "intel", "levelzero", "openvino", "amd", "cpu"} {
		want := ""
		switch p := llm.ProgramBackend(b); {
		case b == llm.BackendIntel || p == llm.BackendIntel:
			want = "intel"
		case p == llm.BackendCoreML:
			want = "apple"
		}
		if got := imagewire.Vendor(b); got != want {
			t.Errorf("%q: %q, want %q", b, got, want)
		}
	}
}

// TestConvertPaths: a conversion takes absolute paths alone (hwnpud's working directory is not
// the client's).
func TestConvertPaths(t *testing.T) {
	if err := Convert(imagewire.Convert{Dir: "Anima", Out: "x.coreai", Size: "1:1"}, nil); err == nil {
		t.Error("relative paths: no error")
	}
	if err := Convert(imagewire.Convert{Dir: "/a", Out: "/b", Size: "5:7"}, nil); err == nil {
		t.Error("--size 5:7: no error")
	}
}
