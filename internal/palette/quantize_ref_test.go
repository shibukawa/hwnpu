package palette

import (
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/shibukawa/hwnpu/internal/vecops"
)

// quantizeRef is QuantizeBits written plainly, the way it was before its blocks were read in
// place and its samples sorted by radix: every block copied out, the sample sorted by
// sortValues, and the indices packed bit by bit.
func quantizeRef(w []float32, rows, cols, group, colBlock, bits, iters int, mode Mode) LUT4 {
	if colBlock <= 0 || colBlock > cols {
		colBlock = cols
	}
	k := 1 << bits
	nr, nc := rows/group, cols/colBlock
	out := LUT4{Rows: rows, Cols: cols, Group: group, ColBlock: colBlock, Bits: bits, LUT: make([]uint16, nr*nc*k)}
	idx := make([]uint8, rows*cols)
	for g := 0; g < nr*nc; g++ {
		rb, cb := g/nc, g%nc
		var vals []float32
		for r := rb * group; r < (rb+1)*group; r++ {
			vals = append(vals, w[r*cols+cb*colBlock:r*cols+(cb+1)*colBlock]...)
		}
		c := make([]float32, k)
		if mode != KMeans {
			lo, hi := float32(math.MaxFloat32), float32(-math.MaxFloat32)
			for _, v := range vals {
				lo, hi = min(lo, v), max(hi, v)
			}
			for i := range c {
				c[i] = lo + (hi-lo)*float32(i)/float32(k-1)
			}
			if mode != Uniform && hi != lo {
				lloyd(c, vals, iters, &scratch{})
			}
		} else {
			var sample []float32
			if len(vals) > 1<<16 {
				step := len(vals) / (1 << 16)
				for i := 0; i < len(vals); i += step {
					sample = append(sample, vals[i])
				}
			} else {
				sample = append([]float32(nil), vals...)
			}
			sortValues(sample)
			if sample[0] == sample[len(sample)-1] {
				for i := range c {
					c[i] = sample[0]
				}
			} else {
				for i := range c {
					c[i] = sample[(2*i+1)*len(sample)/(2*k)]
				}
				lloyd(c, vals, iters, &scratch{})
			}
		}
		for i := range c {
			out.LUT[g*k+i] = f16(c[i])
			c[i] = f16val(out.LUT[g*k+i])
		}
		ids := make([]uint8, len(vals))
		vecops.NewThresholds(midpoints(make([]float32, k-1), c)).Assign(ids, vals)
		for r := range group {
			copy(idx[(rb*group+r)*cols+cb*colBlock:][:colBlock], ids[r*colBlock:(r+1)*colBlock])
		}
	}
	out.Indices = make([]byte, (len(idx)*bits+7)/8)
	for i, q := range idx {
		for b := range bits {
			if q>>b&1 != 0 {
				out.Indices[(i*bits+b)/8] |= 1 << ((i*bits + b) % 8)
			}
		}
	}
	return out
}

func TestQuantizeBitsReference(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	negZero := float32(math.Copysign(0, -1))
	shapes := []struct{ rows, cols, group, colBlock int }{
		{64, 96, 16, 0},    // whole rows, read in place
		{12, 13, 3, 0},     // whole rows of 39 values: not whole bytes, copied
		{32, 64, 8, 16},    // blocks of columns, copied
		{128, 1100, 64, 0}, // 70400 values: the k-means sample is every other one
		{16, 520, 16, 0},   // a sample of 8320 for the radix sort
		{8, 40, 8, 0},      // a sample under 256: sortValues
	}
	for _, sh := range shapes {
		for _, zeros := range []string{"none", "positive", "both"} {
			w := make([]float32, sh.rows*sh.cols)
			for i := range w {
				v := float32(rng.NormFloat64() * 0.02)
				if rng.Intn(40) == 0 {
					switch zeros {
					case "positive":
						v = 0
					case "both":
						v = 0
						if rng.Intn(2) == 0 {
							v = negZero
						}
					}
				}
				w[i] = v
			}
			for _, bits := range []int{4, 6, 8} {
				for _, mode := range []Mode{KMeans, Uniform, KMeansMinMax} {
					got := QuantizeBits(w, sh.rows, sh.cols, sh.group, sh.colBlock, bits, 4, mode)
					want := quantizeRef(w, sh.rows, sh.cols, sh.group, sh.colBlock, bits, 4, mode)
					if !slices.Equal(got.LUT, want.LUT) || !slices.Equal(got.Indices, want.Indices) {
						t.Fatalf("%+v zeros %s bits %d mode %d: differs from the reference", sh, zeros, bits, mode)
					}
				}
			}
		}
	}
}

func TestRadixSort(t *testing.T) {
	rng := rand.New(rand.NewSource(12))
	for _, n := range []int{1, 2, 300, 5000} {
		for trial := range 5 {
			x := make([]float32, n)
			for i := range x {
				switch rng.Intn(10) {
				case 0:
					x[i] = math.Float32frombits(rng.Uint32() & 0x807fffff) // subnormals, -0
				case 1:
					x[i] = float32(math.Inf(1 - 2*rng.Intn(2)))
				default:
					x[i] = float32(rng.NormFloat64() * math.Pow(10, float64(trial)))
				}
				if x[i] == 0 {
					x[i] = 1 // no zeros: radixSort orders -0 before +0
				}
			}
			want := slices.Clone(x)
			slices.Sort(want)
			radixSort(x, make([]float32, n))
			for i := range x {
				if math.Float32bits(x[i]) != math.Float32bits(want[i]) {
					t.Fatalf("n %d trial %d: %v at %d, want %v", n, trial, x[i], i, want[i])
				}
			}
		}
	}
}
