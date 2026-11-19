//go:build !arm64

package mxfp4

const haveKernel = false

func dotKernel(w, e []byte, v *Vec) float32 { return dotGo(w, e, v) }

func dot4Kernel(w, e []byte, v *[4]*Vec, out *[4]float32) bool { return false }
