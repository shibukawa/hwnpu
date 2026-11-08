package vecops

import (
	"math"
	"math/rand"
	"slices"
	"testing"
)

// withGo runs f with the kernels turned off.
func withGo(f func()) {
	saved, savedF16 := haveKernels, haveF16
	haveKernels, haveF16 = false, false
	defer func() { haveKernels, haveF16 = saved, savedF16 }()
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

// packRef is the plain bit by bit packing Pack must match.
func packRef(idx []uint8, bits int) []byte {
	out := make([]byte, PackedLen(len(idx), bits))
	for i, q := range idx {
		for b := range bits {
			if q>>b&1 != 0 {
				bit := i*bits + b
				out[bit/8] |= 1 << (bit % 8)
			}
		}
	}
	return out
}

func TestPack(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for bits := 1; bits <= 8; bits++ {
		for _, n := range append(sizes, 127, 128, 129, 191, 192, 4096+63) {
			idx := make([]uint8, n)
			for i := range idx {
				idx[i] = uint8(rng.Intn(256)) // the bits above the width must be ignored
			}
			want := packRef(idx, bits)
			for _, kern := range []bool{false, true} {
				got := make([]byte, len(want)+8)
				for i := range got {
					got[i] = 0xa5 // overwritten, and nothing written past the end
				}
				run := func() { Pack(got, idx, bits) }
				if kern {
					run()
				} else {
					withGo(run)
				}
				if !slices.Equal(got[:len(want)], want) {
					t.Fatalf("bits %d n %d kernels %v: got %x want %x", bits, n, kern, got[:len(want)], want)
				}
				for _, b := range got[len(want):] {
					if b != 0xa5 {
						t.Fatalf("bits %d n %d kernels %v: wrote past the end", bits, n, kern)
					}
				}
			}
		}
	}
}

func BenchmarkPack(b *testing.B) {
	idx := make([]uint8, 1<<16)
	for i := range idx {
		idx[i] = uint8(i * 37)
	}
	dst := make([]byte, len(idx))
	for _, bits := range []int{4, 6} {
		for _, kern := range []bool{false, true} {
			b.Run(map[bool]string{false: "go", true: "simd"}[kern]+"/"+string(rune('0'+bits)), func(b *testing.B) {
				saved := haveKernels
				haveKernels = haveKernels && kern
				defer func() { haveKernels = saved }()
				b.SetBytes(int64(len(idx)))
				for b.Loop() {
					Pack(dst, idx, bits)
				}
			})
		}
	}
}

// halfCases are float32 values around the edges of half precision: subnormals and the
// ties of their rounding, the largest finite half and the overflow past it, and infinities.
func halfCases() []float32 {
	var c []float32
	for _, b := range []uint32{
		0, 0x80000000, 1, 0x00800000, // float32 zeros and subnormals
		0x33000000, 0x33000001, 0x337fffff, 0x33800000, 0x33c00000, 0x33a00000, // 2^-25 and around the smallest half
		0x387fc000, 0x387fe000, 0x38800000, // the largest subnormal half and the tie above it
		0x477fe000, 0x477fefff, 0x477ff000, 0x47800000, // 65504, below and at the tie to infinity
		0x7f7fffff, 0x7f800000, 0xff800000, // the largest float32, infinities
		0x3f801000, 0x3f803000, 0x3f800fff, 0x3f801001, // ties to even and around them
	} {
		c = append(c, math.Float32frombits(b), math.Float32frombits(b|0x80000000))
	}
	return c
}

func TestF16(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for _, n := range append(sizes, 16, 17, 4096+15) {
		for trial := range 4 {
			src := make([]float32, n)
			for i := range src {
				switch rng.Intn(4) {
				case 0:
					src[i] = math.Float32frombits(rng.Uint32()) // any bits, NaNs among them
				case 1:
					c := halfCases()
					src[i] = c[rng.Intn(len(c))]
				default:
					src[i] = float32(rng.NormFloat64() * math.Pow(2, float64(rng.Intn(40)-25)))
				}
				if trial < 2 && src[i] != src[i] {
					src[i] = 1 // the first trials without NaNs, so the kernels run
				}
			}
			want := make([]byte, 2*n)
			for i, v := range src {
				h := F16Bits(v)
				want[2*i], want[2*i+1] = byte(h), byte(h>>8)
			}
			got := make([]byte, 2*n)
			F32ToF16(got, src)
			if !slices.Equal(got, want) {
				for i := range n {
					if got[2*i] != want[2*i] || got[2*i+1] != want[2*i+1] {
						t.Fatalf("n %d: %08x to half got %02x%02x want %02x%02x", n, math.Float32bits(src[i]), got[2*i+1], got[2*i], want[2*i+1], want[2*i])
					}
				}
			}
			// every half, NaNs and signaling NaNs among them in the later trials
			halves := make([]byte, 2*n)
			for i := range n {
				h := uint16(rng.Intn(1 << 16))
				if trial < 2 && h&0x7c00 == 0x7c00 && h&0x3ff != 0 {
					h &^= 0x0400
				}
				halves[2*i], halves[2*i+1] = byte(h), byte(h>>8)
			}
			back := make([]float32, n)
			F16ToF32(back, halves)
			for i, v := range back {
				w := F16Value(uint16(halves[2*i]) | uint16(halves[2*i+1])<<8)
				if math.Float32bits(v) != math.Float32bits(w) {
					t.Fatalf("n %d: half %02x%02x got %08x want %08x", n, halves[2*i+1], halves[2*i], math.Float32bits(v), math.Float32bits(w))
				}
			}
		}
	}
	// all 65536 halves, without NaNs, round trip through the kernels
	all := make([]byte, 0, 1<<17)
	for h := range 1 << 16 {
		if h&0x7c00 == 0x7c00 && h&0x3ff != 0 {
			continue
		}
		all = append(all, byte(h), byte(h>>8))
	}
	all = all[:len(all)&^63]
	f := make([]float32, len(all)/2)
	F16ToF32(f, all)
	again := make([]byte, len(all))
	F32ToF16(again, f)
	if !slices.Equal(again, all) {
		t.Fatal("halves do not round trip")
	}
}

func BenchmarkF16(b *testing.B) {
	src := benchVals(1 << 16)
	half := make([]byte, 2*len(src))
	for _, kern := range []bool{false, true} {
		name := map[bool]string{false: "go", true: "simd"}[kern]
		set := func(b *testing.B) func() {
			saved := haveF16
			haveF16 = haveF16 && kern
			return func() { haveF16 = saved }
		}
		b.Run(name+"/encode", func(b *testing.B) {
			defer set(b)()
			b.SetBytes(int64(4 * len(src)))
			for b.Loop() {
				F32ToF16(half, src)
			}
		})
		b.Run(name+"/decode", func(b *testing.B) {
			defer set(b)()
			b.SetBytes(int64(4 * len(src)))
			for b.Loop() {
				F16ToF32(src, half)
			}
		})
	}
}
