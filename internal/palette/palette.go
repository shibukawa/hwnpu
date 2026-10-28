// Package palette compresses weights to 4 bit lookup tables, the format the Apple Neural Engine
// reads at full speed: every group of rows shares a table of 16 fp16 values and each weight
// becomes a 4 bit index into it (constexpr_lut_to_dense, per_grouped_channel in coremltools).
package palette

import (
	"math"
	"runtime"
	"sort"
	"sync"
)

// LUT4 is a palettized matrix: Indices holds one 4 bit entry per element in row major order, two
// per byte with the low nibble first; LUT holds 16 fp16 values per block of Group rows by
// ColBlock columns, blocks in row major order.
type LUT4 struct {
	Rows, Cols int
	Group      int // rows per table
	ColBlock   int // columns per table (Cols for whole rows)
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
	if colBlock <= 0 || colBlock > cols {
		colBlock = cols
	}
	if rows%group != 0 || cols%colBlock != 0 {
		panic("palette: shape not a multiple of the block")
	}
	nr, nc := rows/group, cols/colBlock
	out := LUT4{Rows: rows, Cols: cols, Group: group, ColBlock: colBlock, Indices: make([]byte, (rows*cols+1)/2), LUT: make([]uint16, nr*nc*16)}
	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.GOMAXPROCS(0))
	for g := 0; g < nr*nc; g++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(g int) {
			defer wg.Done()
			defer func() { <-sem }()
			rb, cb := g/nc, g%nc
			vals := make([]float32, 0, group*colBlock)
			for r := rb * group; r < (rb+1)*group; r++ {
				vals = append(vals, w[r*cols+cb*colBlock:r*cols+(cb+1)*colBlock]...)
			}
			c := fit(vals, iters, mode)
			for i := range c {
				out.LUT[g*16+i] = f16(c[i])
			}
			// assign with the rounded table so the indices match what the device will use
			for i := range c {
				c[i] = f16val(out.LUT[g*16+i])
			}
			i := 0
			for r := rb * group; r < (rb+1)*group; r++ {
				for j := r*cols + cb*colBlock; j < r*cols+(cb+1)*colBlock; j++ {
					k := nearest(c, vals[i])
					i++
					// blocks of one row block never share a byte with another goroutine's: cols is
					// even, so a row's indices start on a byte boundary
					out.Indices[j/2] |= byte(k) << (4 * uint(j%2))
				}
			}
		}(g)
	}
	wg.Wait()
	return out
}

// fit runs k-means with 16 centers over vals. The centers start at the quantiles of the values,
// so that every center begins with the same share of them: weights are heavy tailed, and a start
// spread evenly between the extremes leaves most centers in the empty tails, where Lloyd's
// algorithm never moves them. Centers are kept sorted so that assignment is a binary search over
// the midpoints.
func fit(vals []float32, iters int, mode Mode) [16]float32 {
	var c [16]float32
	if mode != KMeans {
		lo, hi := float32(math.MaxFloat32), float32(-math.MaxFloat32)
		for _, v := range vals {
			lo = min(lo, v)
			hi = max(hi, v)
		}
		for i := range c {
			c[i] = lo + (hi-lo)*float32(i)/15
		}
		if mode == Uniform || hi == lo {
			return c
		}
		return lloyd(c, vals, iters)
	}
	sample := vals
	if len(sample) > 1<<16 {
		step := len(vals) / (1 << 16)
		sample = make([]float32, 0, 1<<16)
		for i := 0; i < len(vals); i += step {
			sample = append(sample, vals[i])
		}
	} else {
		sample = append([]float32(nil), vals...)
	}
	sort.Slice(sample, func(i, j int) bool { return sample[i] < sample[j] })
	if sample[0] == sample[len(sample)-1] {
		for i := range c {
			c[i] = sample[0]
		}
		return c
	}
	for i := range c {
		c[i] = sample[(2*i+1)*len(sample)/32]
	}
	return lloyd(c, vals, iters)
}

// lloyd refines sorted centers c over vals.
func lloyd(c [16]float32, vals []float32, iters int) [16]float32 {
	var sum [16]float64
	var cnt [16]int
	for it := 0; it < iters; it++ {
		sum = [16]float64{}
		cnt = [16]int{}
		for _, v := range vals {
			k := nearest(c, v)
			sum[k] += float64(v)
			cnt[k]++
		}
		for k := range c {
			if cnt[k] > 0 {
				c[k] = float32(sum[k] / float64(cnt[k]))
			}
		}
		sort.Slice(c[:], func(i, j int) bool { return c[i] < c[j] })
	}
	return c
}

// nearest returns the index of the closest center; c must be sorted.
func nearest(c [16]float32, v float32) int {
	lo, hi := 0, 15
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

// Dequantize expands the matrix back to float32, for checking.
func (l LUT4) Dequantize() []float32 {
	out := make([]float32, l.Rows*l.Cols)
	nc := l.Cols / l.ColBlock
	for i := range out {
		q := l.Indices[i/2] >> (4 * uint(i%2)) & 15
		r, c := i/l.Cols, i%l.Cols
		out[i] = f16val(l.LUT[(r/l.Group*nc+c/l.ColBlock)*16+int(q)])
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
