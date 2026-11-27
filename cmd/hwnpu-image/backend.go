package main

import (
	"github.com/shibukawa/hwnpu/internal/mem"
)

// backendFlag is --backend of the commands that run converted programs (ocr, detect, pose,
// face-detect, depth, segment, label, embed). On a Mac hwnpu-image runs its programs on Core AI
// alone (macOS 27; internal/imagedev.Device in hwnpud's engine), converted for it (.aimodel).
type backendFlag struct {
	Backend string `enum:",coreai,intel,levelzero,openvino" default:"" help:"The device backend for the converted programs: coreai (Apple's Neural Engine through Core AI, macOS 27: a Mac's only one) or intel (Intel's NPU); default: HWNPU_BACKEND, else the NPU found."`
}

// npuHeld logs, when called after the pictures, how much more memory the Neural Engine holds
// than before (mem.NPUWired, the whole machine's: the programs' weights and buffers hwnpud
// loaded, outside any process's footprint), as internal/imagedev.NPUHeld does.
func npuHeld(logf func(string, ...any)) func() {
	before, ok := mem.NPUWired()
	return func() {
		if after, ok2 := mem.NPUWired(); ok && ok2 && after > before {
			logf("the Neural Engine holds %.0f MB more", float64(after-before)/1e6)
		}
	}
}
