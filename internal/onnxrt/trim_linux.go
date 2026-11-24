package onnxrt

import (
	"sync"

	"github.com/shibukawa/hwnpu/internal/dl"
)

var (
	trimOnce   sync.Once
	mallocTrim func(pad uintptr) int32
)

// trim gives the C heap's free memory back to the system (glibc's malloc_trim): what ONNX
// Runtime took while it made a session (the model's parsed copy, its graph's rewrites) is freed
// but kept by malloc otherwise, 40 to 110 MB of RTMO's and RTMDet's. Without glibc, nothing.
func trim() {
	trimOnce.Do(func() {
		if lib, err := dl.Open("libc.so.6"); err == nil {
			if lib.Bind(&mallocTrim, "malloc_trim") != nil {
				mallocTrim = nil
			}
		}
	})
	if mallocTrim != nil {
		mallocTrim(0)
	}
}
