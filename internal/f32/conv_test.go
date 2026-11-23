package f32

import (
	"math/rand/v2"
	"testing"
)

// The vector loops against the plain ones, at lengths around the vectors' and from offsets that
// are not aligned.
func TestDepthwise3(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, m := range []int{1, 7, 8, 9, 16, 20, 40, 78, 80, 81} {
		for _, relu := range []bool{false, true} {
			var w [9]float32
			for i := range w {
				w[i] = rng.Float32()*2 - 1
			}
			rows := make([][]float32, 3)
			for k := range rows {
				rows[k] = make([]float32, m+3)
				for i := range rows[k] {
					rows[k][i] = rng.Float32()*4 - 2
				}
				rows[k] = rows[k][1:] // unaligned
			}
			got := make([]float32, m)
			Depthwise3(got, rows[0], rows[1], rows[2], &w, 0.25, relu)
			for i := range m {
				var want float32 = 0.25
				for k := range 3 {
					for j := range 3 {
						want += w[3*k+j] * rows[k][i+j]
					}
				}
				if relu {
					want = max(want, 0)
				}
				if d := got[i] - want; d > 1e-5 || d < -1e-5 {
					t.Fatalf("m %d relu %v [%d]: %v, want %v", m, relu, i, got[i], want)
				}
			}
		}
	}
}

func TestMaxPool2AndReLU(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	for _, m := range []int{1, 5, 8, 10, 16, 20, 31, 32, 40, 80, 100} {
		a, b := make([]float32, 2*m+1), make([]float32, 2*m+1)
		for i := range a {
			a[i], b[i] = rng.Float32()*2-1, rng.Float32()*2-1
		}
		a, b = a[1:], b[1:]
		got := make([]float32, m)
		MaxPool2(got, a, b)
		for i := range m {
			if want := max(a[2*i], a[2*i+1], b[2*i], b[2*i+1]); got[i] != want {
				t.Fatalf("max pool m %d [%d]: %v, want %v", m, i, got[i], want)
			}
		}
		r := make([]float32, 2*m)
		ReLU(r, a)
		for i := range r {
			if want := max(a[i], 0); r[i] != want {
				t.Fatalf("relu m %d [%d]: %v, want %v", 2*m, i, r[i], want)
			}
		}
	}
}

func TestWeightedRows(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	for _, k := range []int{1, 2, 3, 9, 27} {
		for _, m := range []int{1, 15, 16, 17, 40, 160} {
			for _, relu := range []bool{false, true} {
				rows, w := make([]*float32, k), make([]float32, k)
				src := make([][]float32, k)
				for j := range k {
					src[j] = make([]float32, m+1)
					for i := range src[j] {
						src[j][i] = rng.Float32()*2 - 1
					}
					rows[j], w[j] = &src[j][1], rng.Float32()*2-1
				}
				got := make([]float32, m)
				WeightedRows(got, rows, w, -0.1, relu)
				for i := range m {
					var want float32 = -0.1
					for j := range k {
						want += w[j] * src[j][1+i]
					}
					if relu {
						want = max(want, 0)
					}
					if d := got[i] - want; d > 1e-5 || d < -1e-5 {
						t.Fatalf("k %d m %d relu %v [%d]: %v, want %v", k, m, relu, i, got[i], want)
					}
				}
			}
		}
	}
}

func TestWeightedRows4(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 10))
	for _, k := range []int{1, 3, 27, 64} {
		for _, m := range []int{1, 15, 16, 33, 128, 160} {
			for _, relu := range []bool{false, true} {
				rows, w := make([]*float32, k), make([]float32, 4*k)
				src := make([][]float32, k)
				for j := range k {
					src[j] = make([]float32, m+1)
					for i := range src[j] {
						src[j][i] = rng.Float32()*2 - 1
					}
					rows[j] = &src[j][1]
				}
				for i := range w {
					w[i] = rng.Float32()*2 - 1
				}
				bias := [4]float32{0.1, -0.2, 0.3, -0.4}
				outs := make([][]float32, 4)
				var dst [4]*float32
				for o := range 4 {
					outs[o] = make([]float32, m)
					dst[o] = &outs[o][0]
				}
				WeightedRows4(&dst, m, rows, w, &bias, relu)
				for o := range 4 {
					for i := range m {
						want := bias[o]
						for j := range k {
							want += w[4*j+o] * src[j][1+i]
						}
						if relu {
							want = max(want, 0)
						}
						if d := outs[o][i] - want; d > 1e-4 || d < -1e-4 {
							t.Fatalf("k %d m %d relu %v out %d [%d]: %v, want %v", k, m, relu, o, i, outs[o][i], want)
						}
					}
				}
			}
		}
	}
}

func TestDeinterleave2(t *testing.T) {
	for _, m := range []int{1, 7, 8, 9, 16, 161} {
		src := make([]float32, 2*m+1)
		for i := range src {
			src[i] = float32(i)
		}
		e, o := make([]float32, m), make([]float32, m)
		Deinterleave2(e, o, src[1:])
		for i := range m {
			if e[i] != src[1+2*i] || o[i] != src[2+2*i] {
				t.Fatalf("m %d [%d]: %v %v", m, i, e[i], o[i])
			}
		}
	}
}
