package vecops

import (
	"math"
	"math/rand"
	"slices"
	"testing"
)

// withGo runs f with the kernels turned off.
func withGo(f func()) {
	saved := haveKernels
	haveKernels = false
	defer func() { haveKernels = saved }()
	f()
}

// weights draws n values like a weight matrix's: mostly small, a few large, some exact zeros of
// both signs, rounded to BF16 like a checkpoint's.
func weights(rng *rand.Rand, n int) []float32 {
	w := make([]float32, n)
	for i := range w {
		v := float32(rng.NormFloat64() * 0.02)
		switch rng.Intn(50) {
		case 0:
			v *= 30
		case 1:
			v = 0
		case 2:
			v = float32(math.Copysign(0, -1))
		}
		w[i] = math.Float32frombits(math.Float32bits(v) &^ 0xffff)
	}
	return w
}

func sameBits(a, b float32) bool { return math.Float32bits(a) == math.Float32bits(b) }

// sizes covers the kernels' blocks and the Go tails after them.
var sizes = []int{0, 1, 7, 31, 32, 33, 63, 64, 95, 96, 100, 1000, 4096 + 17}

func TestAssign(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, levels := range []int{1, 2, 3, 4, 5, 6, 7, 8} {
		for trial := range 20 {
			k := 1 << levels
			vals := weights(rng, 4096+37)
			// centers: sorted, with duplicates and, sometimes, infinities
			c := make([]float32, k)
			for i := range c {
				c[i] = vals[rng.Intn(len(vals))]
			}
			if trial%5 == 1 {
				c[0] = float32(math.Inf(-1))
			}
			if trial%5 == 2 {
				c[k-1] = float32(math.Inf(1))
			}
			if trial%5 == 3 {
				for i := range c {
					c[i] = c[0] // all equal
				}
			}
			slices.Sort(c)
			mids := make([]float32, k-1)
			for i := range mids {
				mids[i] = (c[i] + c[i+1]) / 2
			}
			// values exactly on the thresholds, and NaNs and infinities
			for i := 0; i < len(vals); i += 7 {
				vals[i] = mids[rng.Intn(len(mids))]
			}
			vals[3] = float32(math.NaN())
			vals[40] = float32(math.Inf(1))
			vals[41] = float32(math.Inf(-1))
			th := NewThresholds(mids)
			for _, n := range sizes {
				got := make([]uint8, n)
				want := make([]uint8, n)
				th.Assign(got, vals[:n])
				assignGo(mids, want, vals[:n])
				for i := range n {
					if got[i] != want[i] {
						t.Fatalf("levels %d trial %d n %d: value %d (%g) got %d want %d", levels, trial, n, i, vals[i], got[i], want[i])
					}
				}
			}
		}
	}
}

// The old search over the centers, for the record: Assign over the midpoints must match it.
func nearest(c []float32, v float32) int {
	lo, hi := 0, len(c)-1
	for lo < hi {
		mid := (lo + hi) / 2
		if v > (c[mid]+c[mid+1])/2 {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return lo
}

func TestAssignMatchesNearest(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	vals := weights(rng, 10000)
	for _, k := range []int{16, 64, 256} {
		c := make([]float32, k)
		for i := range c {
			c[i] = vals[rng.Intn(len(vals))]
		}
		slices.Sort(c)
		mids := make([]float32, k-1)
		for i := range mids {
			mids[i] = (c[i] + c[i+1]) / 2
		}
		got := make([]uint8, len(vals))
		NewThresholds(mids).Assign(got, vals)
		for i, v := range vals {
			if int(got[i]) != nearest(c, v) {
				t.Fatalf("k %d: %g got %d want %d", k, v, got[i], nearest(c, v))
			}
		}
	}
}

func TestMinMaxAbsMax(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for trial := range 50 {
		vals := weights(rng, 5000)
		switch trial % 5 {
		case 1:
			vals[rng.Intn(len(vals))] = float32(math.NaN())
		case 2:
			vals[rng.Intn(len(vals))] = float32(math.Inf(1))
		case 3:
			vals[rng.Intn(len(vals))] = float32(math.Inf(-1))
		}
		for _, n := range sizes {
			lo, hi, ok := MinMax(vals[:n])
			var glo, ghi float32
			var gok bool
			withGo(func() { glo, ghi, gok = MinMax(vals[:n]) })
			if ok != gok || ok && (lo != glo || hi != ghi) {
				t.Fatalf("trial %d n %d: MinMax got %g %g %v want %g %g %v", trial, n, lo, hi, ok, glo, ghi, gok)
			}
			top, ok := AbsMax(vals[:n])
			var gtop float32
			withGo(func() { gtop, gok = AbsMax(vals[:n]) })
			if ok != gok || ok && !sameBits(top, gtop) {
				t.Fatalf("trial %d n %d: AbsMax got %g %v want %g %v", trial, n, top, ok, gtop, gok)
			}
		}
	}
}

var clips = []float32{1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5}

func TestClipErrors(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	for _, bits := range []int{4, 8} {
		qmax := float32(int(1)<<(bits-1) - 1)
		qmin := -qmax - 1
		for trial := range 30 {
			vals := weights(rng, 5000)
			if trial%6 == 5 {
				clear(vals) // scales of 0
			}
			top, _ := AbsMax(vals)
			ns := len(clips)
			if trial%3 == 1 {
				ns = 1 + rng.Intn(MaxScales)
			}
			scales := make([]float32, ns)
			for i := range scales {
				scales[i] = top * clips[i%len(clips)] / qmax
				if trial%4 == 2 && i == 3 {
					scales[i] = 0
				}
				if trial%7 == 3 && i == 4 {
					scales[i] = float32(math.Inf(1)) // an fp16 scale that overflowed
				}
			}
			for _, n := range sizes {
				got := make([]float32, ns)
				want := make([]float32, ns)
				ClipErrors(got, vals[:n], scales, qmin, qmax)
				withGo(func() { ClipErrors(want, vals[:n], scales, qmin, qmax) })
				for i := range got {
					if !sameBits(got[i], want[i]) && !(got[i] != got[i] && want[i] != want[i]) {
						t.Fatalf("bits %d trial %d n %d scale %d: got %g want %g", bits, trial, n, i, got[i], want[i])
					}
				}
			}
		}
	}
}

func TestQuantizeSym(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for _, bits := range []int{4, 8} {
		qmax := float32(int(1)<<(bits-1) - 1)
		qmin := -qmax - 1
		for trial := range 30 {
			vals := weights(rng, 5000)
			top, _ := AbsMax(vals)
			s := top * clips[trial%len(clips)] / qmax
			if trial == 7 {
				s = 0
			}
			// values that land exactly halfway between two integers
			for i := 0; i < len(vals); i += 5 {
				vals[i] = (float32(rng.Intn(int(2*qmax))) - qmax + 0.5) * s
			}
			for _, n := range sizes {
				got := make([]int8, n)
				want := make([]int8, n)
				QuantizeSym(got, vals[:n], s, qmin, qmax)
				withGo(func() { QuantizeSym(want, vals[:n], s, qmin, qmax) })
				if !slices.Equal(got, want) {
					for i := range got {
						if got[i] != want[i] {
							t.Fatalf("bits %d trial %d n %d: value %d (%g / %g) got %d want %d", bits, trial, n, i, vals[i], s, got[i], want[i])
						}
					}
				}
			}
		}
	}
}

func TestBF16ToF32(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	src := make([]byte, 2*5000)
	rng.Read(src)
	for _, n := range sizes {
		got := make([]float32, n)
		BF16ToF32(got, src)
		for i, v := range got {
			want := uint32(src[2*i]) | uint32(src[2*i+1])<<8
			if math.Float32bits(v) != want<<16 {
				t.Fatalf("n %d: value %d got %08x want %08x", n, i, math.Float32bits(v), want<<16)
			}
		}
	}
}

func benchVals(n int) []float32 { return weights(rand.New(rand.NewSource(7)), n) }

func BenchmarkAssign(b *testing.B) {
	vals := benchVals(1 << 16)
	for _, levels := range []int{4, 6, 8} {
		k := 1 << levels
		c := make([]float32, k)
		lo, hi, _ := MinMax(vals)
		for i := range c {
			c[i] = lo + (hi-lo)*float32(i)/float32(k-1)
		}
		mids := make([]float32, k-1)
		for i := range mids {
			mids[i] = (c[i] + c[i+1]) / 2
		}
		th := NewThresholds(mids)
		dst := make([]uint8, len(vals))
		for _, kern := range []bool{false, true} {
			name := map[bool]string{false: "go", true: "simd"}[kern]
			b.Run(name+"/lut"+string(rune('0'+levels)), func(b *testing.B) {
				saved := haveKernels
				haveKernels = haveKernels && kern
				defer func() { haveKernels = saved }()
				b.SetBytes(int64(4 * len(vals)))
				for b.Loop() {
					th.Assign(dst, vals)
				}
			})
		}
	}
}

func BenchmarkClipErrors(b *testing.B) {
	vals := benchVals(4096)
	top, _ := AbsMax(vals)
	scales := make([]float32, len(clips))
	for i := range scales {
		scales[i] = top * clips[i] / 7
	}
	errs := make([]float32, len(scales))
	for _, kern := range []bool{false, true} {
		b.Run(map[bool]string{false: "go", true: "simd"}[kern], func(b *testing.B) {
			saved := haveKernels
			haveKernels = haveKernels && kern
			defer func() { haveKernels = saved }()
			b.SetBytes(int64(4 * len(vals)))
			for b.Loop() {
				ClipErrors(errs, vals, scales, -8, 7)
			}
		})
	}
}

func BenchmarkQuantizeSym(b *testing.B) {
	vals := benchVals(4096)
	top, _ := AbsMax(vals)
	dst := make([]int8, len(vals))
	for _, kern := range []bool{false, true} {
		b.Run(map[bool]string{false: "go", true: "simd"}[kern], func(b *testing.B) {
			saved := haveKernels
			haveKernels = haveKernels && kern
			defer func() { haveKernels = saved }()
			b.SetBytes(int64(4 * len(vals)))
			for b.Loop() {
				QuantizeSym(dst, vals, top/7, -8, 7)
			}
		})
	}
}

func BenchmarkBF16(b *testing.B) {
	src := make([]byte, 2<<16)
	dst := make([]float32, 1<<16)
	for _, kern := range []bool{false, true} {
		b.Run(map[bool]string{false: "go", true: "simd"}[kern], func(b *testing.B) {
			saved := haveKernels
			haveKernels = haveKernels && kern
			defer func() { haveKernels = saved }()
			b.SetBytes(int64(len(src)))
			for b.Loop() {
				BF16ToF32(dst, src)
			}
		})
	}
}
