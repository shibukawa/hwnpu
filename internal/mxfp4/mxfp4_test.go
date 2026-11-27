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
// rows of a prompt's chunk that chose it.
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

// Encode rounds each value to the nearest of its block's scale, and picks the scale of the
// smallest error among the rule's and its neighbours.
func TestEncode(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	const nb = 64
	row := make([]float32, nb*Block)
	for i := range row {
		row[i] = float32(r.NormFloat64() * 0.02)
	}
	row[5] = 0
	for i := Block; i < 2*Block; i++ {
		row[i] = 0 // a block of zeros
	}
	w, e := make([]byte, 16*nb), make([]byte, nb)
	Encode(w, e, row)
	got := make([]float32, len(row))
	Dequantize(got, w, e)
	var se, sr float64
	for b := range nb {
		blk, dq := row[b*Block:(b+1)*Block], got[b*Block:(b+1)*Block]
		s := float64(scale(e[b]))
		for j, v := range blk {
			d := float64(v - dq[j])
			se, sr = se+d*d, sr+float64(v)*float64(v)
			// nearest at its scale: within half a step (the steps are at most 2 scales apart)
			if v != 0 && math.Abs(d) > s+1e-12 && math.Abs(float64(v))/s < 6 {
				t.Fatalf("block %d value %d: %g encoded as %g at scale %g", b, j, v, dq[j], s)
			}
		}
	}
	if got[5] != 0 || got[Block] != 0 {
		t.Errorf("zeros encoded as %g, %g", got[5], got[Block])
	}
	// e2m1 of normal values: about a tenth of the norm
	if rel := math.Sqrt(se / sr); rel > 0.12 {
		t.Errorf("relative error %.3f", rel)
	} else {
		t.Logf("relative error %.4f", rel)
	}
	for _, tc := range []struct {
		x    float64
		want byte
	}{{0.25, 0}, {0.26, 1}, {0.75, 2}, {1.25, 2}, {1.75, 4}, {2.5, 4}, {3.5, 6}, {5, 6}, {5.1, 7}, {100, 7}, {-0.6, 9}, {-0.1, 0}} {
		if got := nearest(tc.x); got != tc.want {
			t.Errorf("nearest(%g) = %d, want %d", tc.x, got, tc.want)
		}
	}
	// Reference check across subnormals, normal floats, huge values, ties, and negative zero.
	nearestRef := func(x float64) byte {
		var sign byte
		if x < 0 {
			sign, x = 8, -x
		}
		c := byte(7)
		for k := range byte(7) {
			mid := float64(Values[k]+Values[k+1]) / 2
			if x < mid || x == mid && k%2 == 0 {
				c = k
				break
			}
			if x == mid {
				c = k + 1
				break
			}
		}
		if c == 0 {
			return 0
		}
		return sign | c
	}
	edge := make([]float32, 6*Block)
	for i := range Block {
		edge[i] = math.Float32frombits(uint32(i + 1)) // subnormals
		edge[Block+i] = float32(r.NormFloat64()) * 1e-35
		edge[2*Block+i] = float32(r.NormFloat64()) * 1e20
		edge[3*Block+i] = Values[i%len(Values)]
		edge[4*Block+i] = float32(r.NormFloat64() * 0.1)
	}
	edge[5*Block] = float32(math.Copysign(0, -1))
	ew, ee := make([]byte, 16*6), make([]byte, 6)
	Encode(ew, ee, edge)
	for b := range 6 {
		blk := edge[b*Block : (b+1)*Block]
		var m float32
		for _, v := range blk {
			m = max(m, float32(math.Abs(float64(v))))
		}
		best, bestErr := byte(0), math.Inf(1)
		if m > 0 {
			_, exp := math.Frexp(float64(m))
			rule := exp - 3
			for _, x := range []int{rule - 1, rule, rule + 1} {
				eb := x + 127
				if eb < 1 || eb > 254 {
					continue
				}
				s := float64(scale(byte(eb)))
				var se float64
				for _, v := range blk {
					d := float64(v) - float64(Values[nearestRef(float64(v)/s)])*s
					se += d * d
				}
				if se < bestErr {
					best, bestErr = byte(eb), se
				}
			}
		}
		if ee[b] != best {
			t.Fatalf("block %d: scale %d, want %d", b, ee[b], best)
		}
		if best == 0 {
			continue
		}
		s := float64(scale(best))
		for j := range Block / 2 {
			want := nearestRef(float64(blk[2*j])/s) | nearestRef(float64(blk[2*j+1])/s)<<4
			if ew[16*b+j] != want {
				t.Fatalf("block %d byte %d: got 0x%02x, want 0x%02x", b, j, ew[16*b+j], want)
			}
		}
	}
}

func BenchmarkEncode(b *testing.B) {
	r := rand.New(rand.NewPCG(9, 10))
	const nb = 1024
	row := make([]float32, nb*Block)
	for i := range row {
		row[i] = float32(r.NormFloat64() * 0.02)
	}
	w, e := make([]byte, 16*nb), make([]byte, nb)
	b.SetBytes(int64(len(row) * 4))
	for b.Loop() {
		Encode(w, e, row)
	}
}

// Rows gives Dot's value for every row, bit for bit: with blocks in fours, with two left over
// (gpt-oss's 90) and with an odd number of them (no kernel of rows).
func TestRowsMatchesDot(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	for _, c := range []struct{ rows, cols int }{{37, 2048}, {64, 512}, {5, 2880}, {9, 64}, {3, 96}, {1, 32}} {
		nb := c.cols / Block
		w, e := make([]byte, c.rows*nb*16), make([]byte, c.rows*nb)
		for i := range w {
			w[i] = byte(r.Uint32())
		}
		for i := range e {
			e[i] = byte(110 + r.IntN(20))
		}
		x := make([]float32, c.cols)
		for i := range x {
			x[i] = float32(r.NormFloat64()) * float32(math.Exp(r.NormFloat64()))
		}
		v := NewVec(c.cols)
		v.Quantize(x)
		out := make([]float32, c.rows)
		Rows(out, w, e, v)
		for i := range out {
			if want := Dot(w[i*nb*16:(i+1)*nb*16], e[i*nb:(i+1)*nb], v); out[i] != want || want == 0 {
				t.Fatalf("%d rows of %d: row %d is %g, Dot gives %g", c.rows, c.cols, i, out[i], want)
			}
		}
	}
}
