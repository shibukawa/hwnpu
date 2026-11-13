package main

// backendFlag is --backend of the commands that run converted programs (ocr, detect, pose,
// face-detect, depth, segment): coreai runs the same Core ML packages on Core AI (macOS 27,
// internal/backend/coreai), lowered to Core AI programs by internal/coreai/ailower.
type backendFlag struct {
	Backend string `enum:",coreml,coreai,intel,levelzero,openvino" default:"" help:"The device backend for the converted programs: coreml (Apple's Neural Engine), coreai (the same programs on Core AI, macOS 27) or intel (Intel's NPU); default: HWNPU_BACKEND, else the NPU found."`
}
