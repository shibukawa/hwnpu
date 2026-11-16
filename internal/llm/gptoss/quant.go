package gptoss

import (
	"math"
	"slices"
	"sort"

	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// The experts' MXFP4 values are few in a group of 16 rows (e2m1's 15 values times the two or
// three powers of two of the row blocks: 24 to 40 in gpt-oss-20b), which the Neural Engine's
// tables of 16 rows cannot hold in 16 entries exactly. The k-means of internal/palette leaves
// 16 to 34 % of their magnitude off (uniform tables 28 to 98 %); the best 16 entries for each
// group, 1-D k-means solved exactly over its distinct values, leave 6.5 to 11.6 %: a layer's
// change from 29-58 % off down to 10-14 % against transformers' (layers 0 to 2). A power of two
// per row taken out first (multiplied back on the output) takes the gates' to 4 to 8 %, but the
// layers' outputs no closer (11 to 12.5 %), so it is an option (expert-tables exact-rows), not
// the default (docs/design.md 16.229). An exact table needs 6 bits a weight: 14 GB for
// gpt-oss-20b's experts.

// maxDistinct is the most distinct values of a group exactTables fits exactly; a matrix with a
// group of more (a dense checkpoint's bf16 experts) takes palette's tables instead.
const maxDistinct = 1024

// exactTables fits a table of 2^bits entries to each block of w (rows, cols) of group rows and
// colBlock columns (0: whole rows), the best in the squared error over the block's distinct
// values (optimalTable), and returns the indices packed low bits first and the tables in fp16,
// blocks in row major order, as mlpackage.Builder.LUT takes them; ok is false when a block has
// more than maxDistinct values.
func exactTables(w []float32, rows, cols, group, colBlock, bits int) (indices []byte, lut []uint16, ok bool) {
	if colBlock == 0 {
		colBlock = cols
	}
	k := 1 << bits
	idx := make([]uint8, len(w))
	for r0 := 0; r0 < rows; r0 += group {
		for c0 := 0; c0 < cols; c0 += colBlock {
			var vals []float32
			for r := r0; r < r0+group; r++ {
				vals = append(vals, w[r*cols+c0:r*cols+c0+colBlock]...)
			}
			centers, fits := optimalTable(vals, k)
			if !fits {
				return nil, nil, false
			}
			h := make([]uint16, k)
			c := make([]float64, k)
			for i, x := range centers {
				h[i] = mlpackage.Float16Bits(float32(x))
				c[i] = float64(mlpackage.Float16Value(h[i]))
			}
			lut = append(lut, h...)
			for r := r0; r < r0+group; r++ {
				for j := c0; j < c0+colBlock; j++ {
					idx[r*cols+j] = nearest(c, float64(w[r*cols+j]))
				}
			}
		}
	}
	return pack(idx, bits), lut, true
}

// nearest is the index of the center closest to x; centers are sorted.
func nearest(c []float64, x float64) uint8 {
	i := sort.SearchFloat64s(c, x)
	switch {
	case i == 0:
		return 0
	case i == len(c):
		return uint8(len(c) - 1)
	case x-c[i-1] <= c[i]-x:
		return uint8(i - 1)
	}
	return uint8(i)
}

// pack packs indices of bits bits (4 or 8) low first, as the tables' indices are stored.
func pack(idx []uint8, bits int) []byte {
	if bits == 8 {
		return slices.Clone(idx)
	}
	out := make([]byte, (len(idx)+1)/2)
	for i, v := range idx {
		out[i/2] |= v << (4 * (i % 2))
	}
	return out
}

// optimalTable is the best table of k entries for vals in the squared error: 1-D k-means solved
// exactly by a dynamic program over the sorted distinct values (an optimal cluster is a run of
// them). The entries come sorted; with fewer distinct values than k, the rest repeat the last.
// ok is false past maxDistinct values.
func optimalTable(vals []float32, k int) (centers []float64, ok bool) {
	count := map[float32]float64{}
	for _, v := range vals {
		count[v]++
		if len(count) > maxDistinct {
			return nil, false
		}
	}
	xs := make([]float64, 0, len(count))
	for v := range count {
		xs = append(xs, float64(v))
	}
	sort.Float64s(xs)
	n := len(xs)
	if n <= k {
		centers = slices.Clone(xs)
		for len(centers) < k {
			centers = append(centers, xs[n-1])
		}
		return centers, true
	}
	pc, ps, pq := make([]float64, n+1), make([]float64, n+1), make([]float64, n+1)
	for i, x := range xs {
		c := count[float32(x)]
		pc[i+1], ps[i+1], pq[i+1] = pc[i]+c, ps[i]+c*x, pq[i]+c*x*x
	}
	cost := func(i, j int) float64 { // values i..j-1 as one cluster
		c, s := pc[j]-pc[i], ps[j]-ps[i]
		return pq[j] - pq[i] - s*s/c
	}
	dp, cut := make([][]float64, k+1), make([][]int, k+1)
	for c := range dp {
		dp[c], cut[c] = make([]float64, n+1), make([]int, n+1)
		for j := range dp[c] {
			dp[c][j] = math.Inf(1)
		}
	}
	dp[0][0] = 0
	for c := 1; c <= k; c++ {
		for j := c; j <= n-(k-c); j++ {
			for i := c - 1; i < j; i++ {
				if v := dp[c-1][i] + cost(i, j); v < dp[c][j] {
					dp[c][j], cut[c][j] = v, i
				}
			}
		}
	}
	centers = make([]float64, k)
	for c, j := k, n; c >= 1; c-- {
		i := cut[c][j]
		centers[c-1] = (ps[j] - ps[i]) / (pc[j] - pc[i])
		j = i
	}
	return centers, true
}

// rowScales is a power of two for each row of w (rows, cols): the mean over its blocks of 32
// columns (MXFP4's) of log2 of their largest magnitude, rounded; 1 for a row of zeros. Divided
// out of the row before its table is fitted and multiplied back on the output, it takes the rows'
// own scale out of the 16 that share a table.
func rowScales(w []float32, rows, cols int) []float32 {
	out := make([]float32, rows)
	for r := 0; r < rows; r++ {
		var s, n float64
		for c0 := 0; c0+32 <= cols; c0 += 32 {
			var mx float64
			for _, x := range w[r*cols+c0 : r*cols+c0+32] {
				mx = math.Max(mx, math.Abs(float64(x)))
			}
			if mx > 0 {
				s += math.Log2(mx)
				n++
			}
		}
		out[r] = 1
		if n > 0 {
			out[r] = float32(math.Exp2(math.Round(s / n)))
		}
	}
	return out
}
