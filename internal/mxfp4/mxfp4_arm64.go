package mxfp4

const haveKernel = true

//go:noescape
func dotNEON(w, e *byte, q *int8, s *float32, nb int, lut *int8) float32

func dotKernel(w, e []byte, v *Vec) float32 {
	return dotNEON(&w[0], &e[0], &v.Q[0], &v.S[0], len(e), &doubled[0]) / 2
}

//go:noescape
func dot4NEON(w, e *byte, q *[4]*int8, s *[4]*float32, nb int, lut *int8, out *[4]float32)

// dot4Kernel is dotKernel of one row with 4 vectors, the row decoded once: false when the row's
// blocks are odd (the kernel takes them in pairs).
func dot4Kernel(w, e []byte, v *[4]*Vec, out *[4]float32) bool {
	if len(e)%2 != 0 {
		return false
	}
	q := [4]*int8{&v[0].Q[0], &v[1].Q[0], &v[2].Q[0], &v[3].Q[0]}
	s := [4]*float32{&v[0].S[0], &v[1].S[0], &v[2].S[0], &v[3].S[0]}
	dot4NEON(&w[0], &e[0], &q, &s, len(e), &doubled[0], out)
	for i := range out {
		out[i] /= 2
	}
	return true
}
