package palette

import (
	"runtime"
	"sort"
	"sync"
	"sync/atomic"

	"github.com/shibukawa/hwnpu/internal/vecops"
)

// QuantizeWeighted is QuantizeBits with each table fitted to its block weighted by colWeight, a
// weight per column (the mean square of the layer's input at that in-feature, E[x^2]): the error
// of a weight then counts as much as it moves the layer's output, so the columns the inputs are
// large at get the levels. Weighted Lloyd's algorithm from the weighted quantiles, iters passes.
func QuantizeWeighted(w []float32, rows, cols, group, colBlock, bits, iters int, colWeight []float32) LUT4 {
	return quantizeWeighted(w, rows, cols, group, colBlock, bits, iters, colWeight, false)
}

// QuantizeWeightedMinMax is QuantizeWeighted with each table's weighted Lloyd started from the
// uniform levels between the block's minimum and maximum (as KMeansMinMax) in place of the
// weighted quantiles: a quantile start puts the top level at the block's 99.8 % point and clips
// the few large weights, which FLUX.3 klein's DiT does not survive (its picture breaks; started
// from the extremes, its lut8 matches per-row int8: docs/design.md, FLUX.3 klein).
func QuantizeWeightedMinMax(w []float32, rows, cols, group, colBlock, bits, iters int, colWeight []float32) LUT4 {
	return quantizeWeighted(w, rows, cols, group, colBlock, bits, iters, colWeight, true)
}

func quantizeWeighted(w []float32, rows, cols, group, colBlock, bits, iters int, colWeight []float32, minmax bool) LUT4 {
	if colBlock <= 0 || colBlock > cols {
		colBlock = cols
	}
	if rows%group != 0 || cols%colBlock != 0 {
		panic("palette: shape not a multiple of the block")
	}
	if bits != 4 && bits != 6 && bits != 8 {
		panic("palette: tables of 4, 6 or 8 bits only")
	}
	if len(colWeight) != cols {
		panic("palette: a weight per column")
	}
	k := 1 << bits
	nr, nc := rows/group, cols/colBlock
	out := LUT4{Rows: rows, Cols: cols, Group: group, ColBlock: colBlock, Bits: bits, LUT: make([]uint16, nr*nc*k)}
	idx := make([]uint8, rows*cols)
	var next atomic.Int64
	var wg sync.WaitGroup
	for range min(runtime.GOMAXPROCS(0), nr*nc) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			vals := make([]float32, group*colBlock)
			wts := make([]float32, group*colBlock)
			ids := make([]uint8, group*colBlock)
			c, mids := make([]float32, k), make([]float32, k-1)
			var th vecops.Thresholds
			for {
				g := int(next.Add(1) - 1)
				if g >= nr*nc {
					return
				}
				rb, cb := g/nc, g%nc
				for r := range group {
					copy(vals[r*colBlock:], w[(rb*group+r)*cols+cb*colBlock:][:colBlock])
					copy(wts[r*colBlock:], colWeight[cb*colBlock:][:colBlock])
				}
				if minmax {
					fitWeightedMinMax(c, vals, wts, iters, ids)
				} else {
					fitWeighted(c, vals, wts, iters, ids)
				}
				for i := range c {
					out.LUT[g*k+i] = f16(c[i])
					c[i] = f16val(out.LUT[g*k+i])
				}
				th.Reset(midpoints(mids, c))
				th.Assign(ids, vals)
				for r := range group {
					copy(idx[(rb*group+r)*cols+cb*colBlock:][:colBlock], ids[r*colBlock:(r+1)*colBlock])
				}
			}
		}()
	}
	wg.Wait()
	out.Indices = make([]byte, vecops.PackedLen(rows*cols, bits))
	packParallel(out.Indices, idx, bits)
	return out
}

// fitWeighted fits sorted centers c to vals weighted by wts: started at the weighted quantiles
// (each center the same share of the weight), then iters passes of weighted Lloyd. ids is scratch
// of len(vals).
func fitWeighted(c, vals, wts []float32, iters int, ids []uint8) {
	k := len(c)
	order := make([]int, len(vals))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return vals[order[a]] < vals[order[b]] })
	var total float64
	for _, x := range wts {
		total += float64(x)
	}
	if total <= 0 || vals[order[0]] == vals[order[len(order)-1]] {
		for i := range c {
			c[i] = vals[order[len(order)/2]]
		}
		return
	}
	var acc float64
	next := 0
	for _, i := range order {
		acc += float64(wts[i])
		for next < k && acc >= total*(float64(2*next+1)/float64(2*k)) {
			c[next] = vals[i]
			next++
		}
	}
	for ; next < k; next++ {
		c[next] = vals[order[len(order)-1]]
	}
	weightedLloyd(c, vals, wts, iters, ids)
}

// fitWeightedMinMax fits sorted centers c to vals weighted by wts: started at the levels evenly
// spaced between the values' minimum and maximum, then iters passes of weighted Lloyd. ids is
// scratch of len(vals).
func fitWeightedMinMax(c, vals, wts []float32, iters int, ids []uint8) {
	lo, hi := vals[0], vals[0]
	for _, v := range vals {
		lo, hi = min(lo, v), max(hi, v)
	}
	k := len(c)
	for i := range c {
		c[i] = lo + (hi-lo)*float32(i)/float32(k-1)
	}
	if hi == lo {
		return
	}
	weightedLloyd(c, vals, wts, iters, ids)
}

// weightedLloyd refines sorted centers c over vals weighted by wts, iters passes (a center no value
// is nearest to stays).
func weightedLloyd(c, vals, wts []float32, iters int, ids []uint8) {
	k := len(c)
	mids := make([]float32, k-1)
	var th vecops.Thresholds
	var sum, sw [256]float64
	for range iters {
		clear(sum[:])
		clear(sw[:])
		th.Reset(midpoints(mids, c))
		th.Assign(ids, vals)
		for i, v := range vals {
			sum[ids[i]] += float64(wts[i]) * float64(v)
			sw[ids[i]] += float64(wts[i])
		}
		for j := range c {
			if sw[j] > 0 {
				c[j] = float32(sum[j] / sw[j])
			}
		}
		sortValues(c)
	}
}

// Sub is the block of rows [r0, r1) and columns [c0, c1) of l as a matrix of its own, with the
// same values: its tables are those of the blocks it covers and its indices the block's. The rows
// are whole groups; the columns whole column blocks, or anything with tables of whole rows (each
// row group's table then serves the columns kept).
func (l LUT4) Sub(r0, r1, c0, c1 int) LUT4 {
	if r0%l.Group != 0 || r1%l.Group != 0 || r0 >= r1 || r1 > l.Rows || c0 >= c1 || c1 > l.Cols {
		panic("palette: Sub rows not whole groups")
	}
	whole := l.ColBlock == l.Cols
	if !whole && (c0%l.ColBlock != 0 || c1%l.ColBlock != 0) {
		panic("palette: Sub columns not whole column blocks")
	}
	bits := l.Bits
	if bits == 0 {
		bits = 4
	}
	k := 1 << bits
	rows, cols := r1-r0, c1-c0
	out := LUT4{Rows: rows, Cols: cols, Group: l.Group, ColBlock: l.ColBlock, Bits: bits}
	nc := l.Cols / l.ColBlock
	cb0, cb1 := c0/l.ColBlock, (c1+l.ColBlock-1)/l.ColBlock
	if whole {
		out.ColBlock, cb0, cb1 = cols, 0, 1
	}
	for rb := r0 / l.Group; rb < r1/l.Group; rb++ {
		for cb := cb0; cb < cb1; cb++ {
			out.LUT = append(out.LUT, l.LUT[(rb*nc+cb)*k:(rb*nc+cb+1)*k]...)
		}
	}
	idx := make([]uint8, rows*cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			idx[r*cols+c] = uint8(l.index((r0+r)*l.Cols + c0 + c))
		}
	}
	out.Indices = make([]byte, vecops.PackedLen(len(idx), bits))
	packParallel(out.Indices, idx, bits)
	return out
}
