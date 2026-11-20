//go:build !arm64

package mxfp4

const haveKernel = false

func dotKernel(w, e []byte, v *Vec) float32 { return dotGo(w, e, v) }

func dot4Kernel(w, e []byte, v *[4]*Vec, out *[4]float32) bool { return false }

const haveI8MM = false

func mm28Kernel(w []byte, e []byte, p *[4]pairVec, out *[16]float32) {}
func rowsKernel(out []float32, w, e []byte, v *Vec) bool             { return false }
