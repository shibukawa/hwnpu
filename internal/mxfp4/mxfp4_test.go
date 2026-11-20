package mxfp4

import (
	"math"
	"math/rand/v2"
	"testing"
)

func randomRow(r *rand.Rand, nb int) ([]byte, []byte) {
	w, e := make([]byte, 16*nb), make([]byte, nb)
	for i := range w {
		w[i] = byte(r.IntN(256))
	}
	for i := range e {
		e[i] = byte(118 + r.IntN(8))
	}
	return w, e
}

// Dot (the kernel on arm64) is the dequantized row's dot product with the quantized vector.
func TestDot(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, nb := range []int{1, 2, 3, 90} {
		w, e := randomRow(r, nb)
		x := make([]float32, nb*Block)
		for i := range x {
			x[i] = float32(r.NormFloat64())
		}
		v := NewVec(len(x))
		v.Quantize(x)
		row := make([]float32, len(x))
		Dequantize(row, w, e)
		var want, exact, nr, nx float64
		for b := range nb {
			for j := range Block / 2 {
				want += float64(row[b*Block+2*j]) * float64(v.Q[b*Block+j]) * float64(v.S[b])
				want += float64(row[b*Block+2*j+1]) * float64(v.Q[b*Block+16+j]) * float64(v.S[b])
			}
		}
		for i := range x {
			exact += float64(row[i]) * float64(x[i])
			nr += float64(row[i]) * float64(row[i])
			nx += float64(x[i]) * float64(x[i])
		}
		for name, got := range map[string]float32{"Dot": Dot(w, e, v), "dotGo": dotGo(w, e, v)} {
			if math.Abs(float64(got)-want) > 1e-4*math.Max(1, math.Abs(want)) {
				t.Errorf("%d blocks: %s %g, want %g", nb, name, got, want)
			}
		}
		// int8 by blocks: an error of a few thousandths of the norms' product
		if math.Abs(want-exact) > 0.01*math.Sqrt(nr*nx) {
			t.Errorf("%d blocks: quantized %g, exact %g", nb, want, exact)
		}
	}
}

func BenchmarkDot(b *testing.B) {
	r := rand.New(rand.NewPCG(1, 2))
	const rows, nb = 512, 90
	w, e := randomRow(r, rows*nb)
	v := NewVec(nb * Block)
	x := make([]float32, nb*Block)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	v.Quantize(x)
	b.SetBytes(int64(rows * nb * 17))
	for b.Loop() {
		for i := range rows {
			Dot(w[i*nb*16:(i+1)*nb*16], e[i*nb:(i+1)*nb], v)
		}
	}
}

// Run with 4 or more vectors (the kernel of a row decoded once for 4) gives Dot's values.
func TestRunVectors(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	for _, rows := range []int{70, 71} {
		const nb = 90
		w, e := randomRow(r, rows*nb)
		for _, n := range []int{1, 4, 5, 8, 9, 12, 16, 19} {
			vs := make([]*Vec, n)
			out := make([][]float32, n)
			for k := range vs {
				x := make([]float32, nb*Block)
				for i := range x {
					x[i] = float32(r.NormFloat64())
				}
				vs[k] = NewVec(len(x))
				vs[k].Quantize(x)
				out[k] = make([]float32, rows)
			}
			Run(Job{Out: out, W: w, E: e, V: vs})
			for k, v := range vs {
				for row := range rows {
					want := Dot(w[row*nb*16:(row+1)*nb*16], e[row*nb:(row+1)*nb], v)
					if got := out[k][row]; math.Abs(float64(got-want)) > 1e-4*math.Max(1, math.Abs(float64(want))) {
						t.Fatalf("%d rows, %d vectors: vector %d row %d: %g, want %g", rows, n, k, row, got, want)
					}
				}
			}
		}
	}
}

// BenchmarkRunVectors is Run of an expert's gate_up (5760 rows of 2880) with 16 vectors, the
// rows of a prompt's chunk that chose it (HWNPU_MXFP4=i8mm=0 for the kernel without SMMLA).
func BenchmarkRunVectors(b *testing.B) {
	r := rand.New(rand.NewPCG(5, 6))
	const rows, nb, n = 5760, 90, 16
	w, e := randomRow(r, rows*nb)
	vs := make([]*Vec, n)
	out := make([][]float32, n)
	for k := range vs {
		x := make([]float32, nb*Block)
		for i := range x {
			x[i] = float32(r.NormFloat64())
		}
		vs[k] = NewVec(len(x))
		vs[k].Quantize(x)
		out[k] = make([]float32, rows)
	}
	b.SetBytes(int64(rows * nb * 17))
	for b.Loop() {
		Run(Job{Out: out, W: w, E: e, V: vs})
	}
}
