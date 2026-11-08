// Package palette compresses weights to 4 bit lookup tables, the format the Apple Neural Engine
// reads at full speed: every group of rows shares a table of 16 fp16 values and each weight
// becomes a 4 bit index into it (constexpr_lut_to_dense, per_grouped_channel in coremltools).
// Tables of 64 and 256 entries (6 and 8 bit indices) are the same with more levels.
package palette

import (
	"math"
	"runtime"
	"slices"
	"sort"
	"sync"
	"sync/atomic"

	"github.com/shibukawa/hwnpu/internal/vecops"
)

// LUT4 is a palettized matrix: Indices holds one Bits wide entry per element in row major
// order, packed low bits first (two per byte with the low nibble first for 4 bits); LUT holds
// 2^Bits fp16 values per block of Group rows by ColBlock columns, blocks in row major order.
type LUT4 struct {
	Rows, Cols int
	Group      int // rows per table
	ColBlock   int // columns per table (Cols for whole rows)
	Bits       int // 4, 6 or 8
	Indices    []byte
	LUT        []uint16
}

// Mode selects how a table is fitted.
type Mode int

const (
	// KMeans minimizes the squared error: Lloyd's algorithm from quantile centers. Lowest weight
	// error, but it spends few levels on the large weights, which the network depends on most.
	KMeans Mode = iota
	// Uniform spaces the 16 levels evenly between the block's minimum and maximum, like a linear
	// 4 bit quantizer: large weights keep their size.
	Uniform
	// KMeansMinMax runs Lloyd's algorithm from the Uniform levels; centers in the empty tails
	// stay put, so it keeps the outliers while refining the bulk.
	KMeansMinMax
)

// Quantize palettizes a row major rows x cols matrix with one 16 entry table per block of group
// rows by colBlock columns (0 means whole rows). iters is the number of Lloyd refinement passes
// for the k-means modes; 8 is plenty for a 16 entry table.
func Quantize(w []float32, rows, cols, group, colBlock, iters int, mode Mode) LUT4 {
	return QuantizeBits(w, rows, cols, group, colBlock, 4, iters, mode)
}

// QuantizeBits is Quantize with tables of 2^bits entries (bits 4, 6 or 8).
func QuantizeBits(w []float32, rows, cols, group, colBlock, bits, iters int, mode Mode) LUT4 {
	if colBlock <= 0 || colBlock > cols {
		colBlock = cols
	}
	if rows%group != 0 || cols%colBlock != 0 {
		panic("palette: shape not a multiple of the block")
	}
	if bits != 4 && bits != 6 && bits != 8 {
		panic("palette: tables of 4, 6 or 8 bits only")
	}
	k := 1 << bits
	nr, nc := rows/group, cols/colBlock
	out := LUT4{Rows: rows, Cols: cols, Group: group, ColBlock: colBlock, Bits: bits, LUT: make([]uint16, nr*nc*k)}
	out.Indices = make([]byte, vecops.PackedLen(rows*cols, bits))
	// A block of whole rows is a run of w, and its indices a run of whole bytes of the output
	// when its length is a multiple of 8: it is read in place and packed there. Other blocks are
	// copied out, and their indices packed at the end.
	inPlace := colBlock == cols && group*cols%8 == 0
	var idx []uint8
	if !inPlace {
		idx = make([]uint8, rows*cols)
	}
	var next atomic.Int64
	var wg sync.WaitGroup
	for range min(runtime.GOMAXPROCS(0), nr*nc) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var sc scratch
			c, mids := make([]float32, k), make([]float32, k-1)
			var th vecops.Thresholds
			for {
				g := int(next.Add(1) - 1)
				if g >= nr*nc {
					return
				}
				rb, cb := g/nc, g%nc
				var vals []float32
				if inPlace {
					vals = w[rb*group*cols : (rb+1)*group*cols]
				} else {
					vals = grow(&sc.vals, group*colBlock)
					for r := range group {
						copy(vals[r*colBlock:], w[(rb*group+r)*cols+cb*colBlock:][:colBlock])
					}
				}
				fit(c, vals, iters, mode, &sc)
				for i := range c {
					out.LUT[g*k+i] = f16(c[i])
				}
				// assign with the rounded table so the indices match what the device will use
				for i := range c {
					c[i] = f16val(out.LUT[g*k+i])
				}
				ids := grow(&sc.ids, len(vals))
				if inPlace && bits == 8 {
					ids = out.Indices[len(vals)*g:][:len(vals)] // one byte per index: no packing
				}
				th.Reset(midpoints(mids, c))
				th.Assign(ids, vals)
				if inPlace {
					if bits != 8 {
						vecops.Pack(out.Indices[len(vals)*g*bits/8:], ids, bits)
					}
					continue
				}
				for r := range group {
					copy(idx[(rb*group+r)*cols+cb*colBlock:][:colBlock], ids[r*colBlock:(r+1)*colBlock])
				}
			}
		}()
	}
	wg.Wait()
	if !inPlace {
		packParallel(out.Indices, idx, bits)
	}
	return out
}

// packParallel is vecops.Pack over the processors, in runs of whole bytes.
func packParallel(dst []byte, idx []uint8, bits int) {
	const chunk = 1 << 20 // entries, a multiple of 8
	if len(idx) <= chunk {
		vecops.Pack(dst, idx, bits)
		return
	}
	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.GOMAXPROCS(0))
	for i := 0; i < len(idx); i += chunk {
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			vecops.Pack(dst[i/8*bits:], idx[i:min(i+chunk, len(idx))], bits)
		}()
	}
	wg.Wait()
}

// pack packs indices of the given width, low bits first.
func pack(idx []uint8, bits int) []byte {
	out := make([]byte, vecops.PackedLen(len(idx), bits))
	packParallel(out, idx, bits)
	return out
}

// scratch holds a worker's buffers, reused from block to block: the values of a block that is
// not a run of w, its indices, and fit's and lloyd's own.
type scratch struct {
	vals, sample, tmp []float32
	ids, lloydIDs     []uint8
}

// grow returns (*buf)[:n], growing *buf when it is shorter.
func grow[T any](buf *[]T, n int) []T {
	if cap(*buf) < n {
		*buf = make([]T, n)
	}
	return (*buf)[:n]
}

// index reads entry i of indices packed by pack.
func (l LUT4) index(i int) int {
	bits := l.Bits
	if bits == 0 {
		bits = 4
	}
	bit := i * bits
	v := uint16(l.Indices[bit/8])
	if bit/8+1 < len(l.Indices) {
		v |= uint16(l.Indices[bit/8+1]) << 8
	}
	return int(v>>(bit%8)) & (1<<bits - 1)
}

// fit runs k-means with k centers over vals. The centers start at the quantiles of the values,
// so that every center begins with the same share of them: weights are heavy tailed, and a start
// spread evenly between the extremes leaves most centers in the empty tails, where Lloyd's
// algorithm never moves them. Centers are kept sorted so that assignment is a binary search over
// the midpoints.
func fit(c, vals []float32, iters int, mode Mode, sc *scratch) {
	k := len(c)
	if mode != KMeans {
		lo, hi, ok := vecops.MinMax(vals)
		if !ok { // NaNs: the order of min and max decides
			lo, hi = float32(math.MaxFloat32), float32(-math.MaxFloat32)
			for _, v := range vals {
				lo = min(lo, v)
				hi = max(hi, v)
			}
		}
		for i := range c {
			c[i] = lo + (hi-lo)*float32(i)/float32(k-1)
		}
		if mode == Uniform || hi == lo {
			return
		}
		lloyd(c, vals, iters, sc)
		return
	}
	var sample []float32
	if len(vals) > 1<<16 {
		step := len(vals) / (1 << 16)
		sample = grow(&sc.sample, (len(vals)+step-1)/step)
		for i := range sample {
			sample[i] = vals[i*step]
		}
	} else {
		sample = grow(&sc.sample, len(vals))
		copy(sample, vals)
	}
	sortSample(sample, sc)
	if sample[0] == sample[len(sample)-1] {
		for i := range c {
			c[i] = sample[0]
		}
		return
	}
	for i := range c {
		c[i] = sample[(2*i+1)*len(sample)/(2*k)]
	}
	lloyd(c, vals, iters, sc)
}

// lloyd refines sorted centers c (at most 256) over vals in place.
func lloyd(c []float32, vals []float32, iters int, sc *scratch) {
	var sum [256]float64
	var cnt [256]int
	ids := grow(&sc.lloydIDs, len(vals))
	mids := make([]float32, len(c)-1)
	var th vecops.Thresholds
	for it := 0; it < iters; it++ {
		clear(sum[:])
		clear(cnt[:])
		th.Reset(midpoints(mids, c))
		th.Assign(ids, vals)
		for i, v := range vals {
			k := ids[i]
			sum[k] += float64(v)
			cnt[k]++
		}
		for k := range c {
			if cnt[k] > 0 {
				c[k] = float32(sum[k] / float64(cnt[k]))
			}
		}
		sortValues(c)
	}
}

// sortValues sorts x ascending. slices.Sort runs the same pdqsort as sort.Slice with <, so it
// leaves even zeros of either sign in the same order, only without sort.Slice's indirect calls;
// NaNs, which it orders differently, take sort.Slice.
func sortValues(x []float32) {
	if slices.ContainsFunc(x, func(v float32) bool { return v != v }) {
		sort.Slice(x, func(i, j int) bool { return x[i] < x[j] })
		return
	}
	slices.Sort(x)
}

// sortSample is sortValues for the k-means sample, by radix sort when that gives the same
// order: values that compare equal are then the same bits, except the zeros of either sign,
// whose order only pdqsort itself decides, and NaNs. With both zeros, or a NaN, sortValues
// sorts.
func sortSample(x []float32, sc *scratch) {
	var pos, neg bool
	for _, v := range x {
		if v != v {
			sortValues(x)
			return
		}
		if v == 0 {
			if math.Signbit(float64(v)) {
				neg = true
			} else {
				pos = true
			}
		}
	}
	if pos && neg || len(x) < 256 {
		sortValues(x)
		return
	}
	radixSort(x, grow(&sc.tmp, len(x)))
}

// radixSort sorts x, without NaNs, ascending by the order of its bits made monotone (negative
// values with all bits flipped, others with the sign set): three passes of 11 bits, between x
// and tmp, skipping a pass whose digit all values share.
func radixSort(x, tmp []float32) {
	key := func(v float32) uint32 {
		b := math.Float32bits(v)
		return b ^ (uint32(int32(b)>>31) | 0x80000000)
	}
	var counts [3][2048]int
	for _, v := range x {
		k := key(v)
		counts[0][k&2047]++
		counts[1][k>>11&2047]++
		counts[2][k>>22]++
	}
	src, dst := x, tmp
	for p, shift := range [3]uint{0, 11, 22} {
		cnt := &counts[p]
		if cnt[key(src[0])>>shift&2047] == len(src) {
			continue
		}
		sum := 0
		for i, n := range cnt {
			cnt[i] = sum
			sum += n
		}
		for _, v := range src {
			d := key(v) >> shift & 2047
			dst[cnt[d]] = v
			cnt[d]++
		}
		src, dst = dst, src
	}
	if &src[0] != &x[0] {
		copy(x, src)
	}
}

// midpoints writes to mids the midpoints between neighbouring centers c, the thresholds of the
// nearest center for sorted c: a value goes to the number of midpoints below it.
func midpoints(mids, c []float32) []float32 {
	for i := range mids {
		mids[i] = (c[i] + c[i+1]) / 2
	}
	return mids
}

// Dequantize expands the matrix back to float32, for checking.
func (l LUT4) Dequantize() []float32 {
	out := make([]float32, l.Rows*l.Cols)
	nc := l.Cols / l.ColBlock
	bits := l.Bits
	if bits == 0 {
		bits = 4
	}
	for i := range out {
		r, c := i/l.Cols, i%l.Cols
		out[i] = f16val(l.LUT[(r/l.Group*nc+c/l.ColBlock)<<bits+l.index(i)])
	}
	return out
}

func f16(f float32) uint16 {
	x := math.Float32bits(f)
	sign := uint16(x>>16) & 0x8000
	exp := int((x>>23)&0xff) - 127 + 15
	mant := x & 0x7fffff
	switch {
	case (x>>23)&0xff == 0xff:
		if mant != 0 {
			return sign | 0x7e00
		}
		return sign | 0x7c00
	case exp >= 0x1f:
		return sign | 0x7c00
	case exp <= 0:
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint(14 - exp)
		h := uint16(mant >> shift)
		rem := mant & (1<<shift - 1)
		half := uint32(1) << (shift - 1)
		if rem > half || (rem == half && h&1 == 1) {
			h++
		}
		return sign | h
	}
	h := sign | uint16(exp)<<10 | uint16(mant>>13)
	rem := mant & 0x1fff
	if rem > 0x1000 || (rem == 0x1000 && h&1 == 1) {
		h++
	}
	return h
}

func f16val(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		e := uint32(127 - 15 + 1)
		for mant&0x400 == 0 {
			mant <<= 1
			e--
		}
		mant &= 0x3ff
		return math.Float32frombits(sign | e<<23 | mant<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | mant<<13)
	}
	return math.Float32frombits(sign | (exp+127-15)<<23 | mant<<13)
}
