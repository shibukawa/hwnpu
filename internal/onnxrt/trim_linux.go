package onnxrt

import (
	"sync"

	"github.com/shibukawa/hwnpu/internal/dl"
)

var (
	libcOnce   sync.Once
	mallocTrim func(pad uintptr) int32
	mallopt    func(param, value int32) int32
)

// glibc's mallopt parameter M_MMAP_THRESHOLD.
const mMmapThreshold = -3

func libc() {
	libcOnce.Do(func() {
		lib, err := dl.Open("libc.so.6")
		if err != nil {
			return
		}
		if lib.Bind(&mallocTrim, "malloc_trim") != nil {
			mallocTrim = nil
		}
		if lib.Bind(&mallopt, "mallopt") != nil {
			mallopt = nil
		}
	})
}

// trim gives the C heap's free memory back to the system (glibc's malloc_trim): what ONNX
// Runtime took while it made a session (the model's parsed copy, its graph's rewrites) is freed
// but kept by malloc otherwise, 40 to 110 MB of RTMO's and RTMDet's. Without glibc, nothing.
func trim() {
	libc()
	if mallocTrim != nil {
		mallocTrim(0)
	}
}

// mallocTune fixes glibc's mmap threshold at 4 MB, once ONNX Runtime is loaded: without its
// arena a session's tensors are malloc's, and the threshold glibc moves up (to 32 MB) as they
// are freed puts the next ones on the heap, which keeps them. The largest RSS of OCR over nine
// pictures 760 → 370 MB as fast as with the arena (at 1 MB 360 MB, 5 to 25 % slower: the
// mmaps of the small tensors), of basic-pitch over two minutes 340 → 205 MB (at 16 MB 265).
// Only C's allocations are malloc's (Go has its own).
func mallocTune() {
	libc()
	if mallopt != nil {
		mallopt(mMmapThreshold, 4<<20)
	}
}
