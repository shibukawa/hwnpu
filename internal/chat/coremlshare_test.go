package chat

import (
	"testing"

	"github.com/shibukawa/hwnpu/internal/cache"
)

// The models the definitions put on the CPU somewhere (device: cpu) have their Core ML packages
// versioned as their Core AI exports; the others have no such target (nothing shared under a
// version for them).
func TestCoreMLTarget(t *testing.T) {
	for name, want := range map[string]bool{"YuNet": true, "RTMDet-tiny": true, "RTMO-s": true, "swift-f0": true, "basic-pitch": true, "EfficientSAM-Ti": true, "PP-OCRv6-small": true, "LFM3-1.7B": false, "whisper-large-v3-turbo": false, "supertonic-3": false} {
		m, ok := Lookup(name)
		if !ok {
			t.Fatalf("%s: not in the definitions", name)
		}
		v, has := m.Targets[cache.CoreMLTarget]
		if has != want || has && v != m.Targets[cache.CoreAITarget] {
			t.Errorf("%s: coreml version %d (%v), coreai %d; want a coreml target %v", name, v, has, m.Targets[cache.CoreAITarget], want)
		}
	}
	if !isCoreML("/x/yunet-r1-fp16-320x256.mlpackage") || isCoreML("/x/yunet-r1-fp16-320x256.coreai") {
		t.Error("isCoreML")
	}
}
