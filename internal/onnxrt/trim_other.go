//go:build !linux

package onnxrt

// trim and mallocTune are glibc's malloc_trim and mallopt on Linux: nothing here.
func trim() {}

func mallocTune() {}
