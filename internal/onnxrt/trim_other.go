//go:build !linux

package onnxrt

// trim is glibc's malloc_trim on Linux: nothing here.
func trim() {}
