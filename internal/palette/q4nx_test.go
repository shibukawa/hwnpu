package palette

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"testing"
)

func relRMS(a, b []float32) float64 {
	var d, n float64
	for i := range a {
		d += float64(a[i]-b[i]) * float64(a[i]-b[i])
		n += float64(b[i]) * float64(b[i])
	}
	return math.Sqrt(d / n)
}

func TestQ4NX(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	rows, cols := 64, 512
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(rng.NormFloat64()) * 0.02
	}
	w[5] = 0.3 // an outlier: clipping should help its group
	plain := QuantizeQ4NX(w, rows, cols, false)
	searched := QuantizeQ4NX(w, rows, cols, true)
	ep, es := relRMS(plain.Dequantize(), w), relRMS(searched.Dequantize(), w)
	// 16 levels over a group's range: about 10% for Gaussian weights
	if ep > 0.15 || es > ep {
		t.Errorf("rel RMS error: min/max %.4f, searched %.4f", ep, es)
	}
	// every weight within half a step of its value (min/max: the range covers the group)
	d := plain.Dequantize()
	for i := range w {
		g := (i/cols)*(cols/Q4NXGroup) + (i%cols)/Q4NXGroup
		if s := bf16val(plain.Scales[g]); math.Abs(float64(d[i]-w[i])) > float64(s)*0.51+1e-3*math.Abs(float64(w[i])) {
			t.Fatalf("weight %d: %g for %g, scale %g", i, d[i], w[i], s)
		}
	}
}

func TestQ4NXConstantGroup(t *testing.T) {
	w := make([]float32, 32)
	for i := range w {
		w[i] = -0.5
	}
	q := QuantizeQ4NX(w, 1, 32, true)
	for i, v := range q.Dequantize() {
		if v != -0.5 {
			t.Fatalf("element %d: %g", i, v)
		}
	}
	if bf16val(q.Scales[0]) == 0 {
		t.Error("zero scale: min / scale is not finite")
	}
}

// Blocks lays the matrix out as the kernel reads it; reading it back by that layout gives the
// same codes, scales and minima.
func TestQ4NXBlocks(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	rows, cols := 64, 512
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	q := QuantizeQ4NX(w, rows, cols, false)
	b := q.Blocks()
	if len(b) != rows/32*cols/256*Q4NXBlockBytes {
		t.Fatalf("%d bytes", len(b))
	}
	ng := cols / Q4NXGroup
	for rb := 0; rb < rows/32; rb++ {
		for cb := 0; cb < cols/256; cb++ {
			blk := b[(rb*(cols/256)+cb)*Q4NXBlockBytes:]
			for g := 0; g < 8; g++ {
				for r := 0; r < 32; r++ {
					gi := (rb*32+r)*ng + cb*8 + g
					if s := binary.LittleEndian.Uint16(blk[2*(g*32+r):]); s != q.Scales[gi] {
						t.Fatalf("block %d,%d scale g%d r%d", rb, cb, g, r)
					}
					if m := binary.LittleEndian.Uint16(blk[512+2*(g*32+r):]); m != q.Mins[gi] {
						t.Fatalf("block %d,%d min g%d r%d", rb, cb, g, r)
					}
				}
			}
			codes := blk[1024:Q4NXBlockBytes]
			for strip := 0; strip < 2; strip++ {
				for c := 0; c < 256; c++ {
					for r := 0; r < 16; r++ {
						n := (strip*256+c)*16 + r
						got := int(codes[n/2]>>(4*(n%2))) & 0xf
						if want := q.Code((rb*32+strip*16+r)*cols + cb*256 + c); got != want {
							t.Fatalf("block %d,%d code row %d col %d: %d, want %d", rb, cb, strip*16+r, c, got, want)
						}
					}
				}
			}
		}
	}
}

func TestBF16(t *testing.T) {
	for _, c := range []struct {
		f    float32
		want uint16
	}{{1, 0x3f80}, {-2, 0xc000}, {1 + 1.0/256, 0x3f80}, {1 + 3.0/256, 0x3f82}, {float32(math.Inf(1)), 0x7f80}} {
		if got := bf16(c.f); got != c.want {
			t.Errorf("bf16(%g) = %#x, want %#x", c.f, got, c.want)
		}
	}
}
