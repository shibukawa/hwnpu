package palette

import (
	"math"
	"runtime"
	"sync"
	"unsafe"

	"github.com/shibukawa/hwnpu/internal/vecops"
)

// Linear is a symmetrically quantized matrix: each weight is a Bits wide two's complement
// integer q and stands for q times the scale of its block, one block per row and ColBlock
// columns. This is the compressed weight layout of Intel's NPU (NNCF's symmetric int4 and int8:
// a Convert to fp16 and a Multiply by the scales before the MatMul), and of Core ML's
// constexpr_blockwise_shift_scale.
type Linear struct {
	Rows, Cols int
	ColBlock   int // columns per scale (Cols for one scale per row)
	Bits       int // 4 or 8
	// Data holds the integers in row major order: one byte each for 8 bits, two per byte with
	// the low nibble first for 4.
	Data []byte
	// Scales holds one fp16 scale per block, blocks in row major order.
	Scales []uint16
}

// clips are the fractions of a block's largest magnitude tried as the clipping point with
// search: clipping the few largest weights lets the bulk use more of the levels (including
// heavy-tailed outlier blocks down to 0.25).
var clips = []float32{1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25}

// QuantizeLinear quantizes a row major rows x cols matrix to bits (4 or 8) with one scale per
// colBlock columns of a row (0 means whole rows). The scale maps the block's largest magnitude
// to the largest level (7 or 127); with search it is instead the clipping point, from clips,
// that gives the smallest squared error.
func QuantizeLinear(w []float32, rows, cols, colBlock, bits int, search bool) Linear {
	if colBlock <= 0 || colBlock > cols {
		colBlock = cols
	}
	if cols%colBlock != 0 {
		panic("palette: columns not a multiple of the block")
	}
	if bits != 4 && bits != 8 {
		panic("palette: linear quantization of 4 or 8 bits only")
	}
	nc := cols / colBlock
	l := Linear{Rows: rows, Cols: cols, ColBlock: colBlock, Bits: bits, Scales: make([]uint16, rows*nc)}
	if bits == 8 {
		l.Data = make([]byte, rows*cols)
	} else {
		l.Data = make([]byte, (rows*cols+1)/2)
	}
	// Symmetric clipping range [-7, 7] / [-127, 127] (avoiding asymmetric -8 / -128 DC bias
	// and signed overflow on hardware INT4/INT8 MACs).
	qmax := float32(int(1)<<(bits-1) - 1)
	qmin := -qmax
	// rows are independent; 4 bit rows share no byte only for even cols, so hand out rows in
	// pairs
	work := make(chan int, rows)
	for r := 0; r < rows; r += 2 {
		work <- r
	}
	close(work)
	var wg sync.WaitGroup
	for range runtime.GOMAXPROCS(0) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			q := make([]int8, colBlock)
			scales := make([]float32, len(clips))
			errs := make([]float32, len(clips))
			for r0 := range work {
				for r := r0; r < min(r0+2, rows); r++ {
					for b := 0; b < nc; b++ {
						blk := w[r*cols+b*colBlock : r*cols+(b+1)*colBlock]
						top, ok := vecops.AbsMax(blk)
						if !ok {
							l.Scales[r*nc+b] = quantizeNaN(q, blk, qmin, qmax, search)
						} else {
							best, bestErr := f16(top/qmax), float32(math.Inf(1))
							if search {
								for i, c := range clips {
									scales[i] = f16val(f16(top * c / qmax))
								}
								vecops.ClipErrors(errs, blk, scales, qmin, qmax)
								for i, c := range clips {
									if errs[i] < bestErr {
										best, bestErr = f16(top*c/qmax), errs[i]
									}
								}
							}
							l.Scales[r*nc+b] = best
							vecops.QuantizeSym(q, blk, f16val(best), qmin, qmax)
						}
						base := r*cols + b*colBlock
						if bits == 8 {
							for j, v := range q {
								l.Data[base+j] = byte(v)
							}
						} else if base%2 == 0 && colBlock%2 == 0 {
							// whole bytes of their own: the low nibbles of q, packed
							vecops.Pack(l.Data[base/2:], unsafe.Slice((*uint8)(unsafe.Pointer(&q[0])), len(q)), 4)
						} else {
							for j, v := range q {
								i := base + j
								l.Data[i/2] |= byte(v&0xf) << (4 * (i % 2))
							}
						}
					}
				}
			}
		}()
	}
	wg.Wait()
	return l
}

// MinVal is an asymmetrically block-quantized matrix (MINVAL): each weight is an unsigned
// Bits-wide integer q in [0, 2^Bits-1] standing for Min + q * Scale, giving all 16 (4-bit) or
// 256 (8-bit) levels across each block's [min, max] range without wasting levels on empty signs.
type MinVal struct {
	Rows, Cols int
	ColBlock   int
	Bits       int // 4 or 8
	Data       []byte
	Scales     []uint16
	Mins       []uint16
}

// QuantizeMinVal quantizes a row-major rows x cols matrix to unsigned bits (4 or 8) with
// per-block fp16 minimum and scale: v ≈ Min + q * Scale.
func QuantizeMinVal(w []float32, rows, cols, colBlock, bits int) MinVal {
	if colBlock <= 0 || colBlock > cols {
		colBlock = cols
	}
	if cols%colBlock != 0 {
		panic("palette: columns not a multiple of the block")
	}
	if bits != 4 && bits != 8 {
		panic("palette: minval quantization of 4 or 8 bits only")
	}
	nc := cols / colBlock
	m := MinVal{
		Rows: rows, Cols: cols, ColBlock: colBlock, Bits: bits,
		Scales: make([]uint16, rows*nc),
		Mins:   make([]uint16, rows*nc),
	}
	if bits == 8 {
		m.Data = make([]byte, rows*cols)
	} else {
		m.Data = make([]byte, (rows*cols+1)/2)
	}
	qmax := float32(int(1)<<bits - 1)
	work := make(chan int, rows)
	for r := 0; r < rows; r += 2 {
		work <- r
	}
	close(work)
	var wg sync.WaitGroup
	for range runtime.GOMAXPROCS(0) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			q := make([]uint8, colBlock)
			for r0 := range work {
				for r := r0; r < min(r0+2, rows); r++ {
					for b := 0; b < nc; b++ {
						blk := w[r*cols+b*colBlock : r*cols+(b+1)*colBlock]
						lo, hi, ok := vecops.MinMax(blk)
						if !ok {
							lo, hi = 0, 0
						}
						minH := f16(lo)
						minV := f16val(minH)
						span := hi - minV
						if span < 0 {
							span = 0
						}
						scH := f16(span / qmax)
						scV := f16val(scH)
						m.Mins[r*nc+b] = minH
						m.Scales[r*nc+b] = scH
						if scV == 0 {
							clear(q)
						} else {
							inv := 1.0 / scV
							maxI := int(qmax)
							for j, v := range blk {
								qi := int(math.Round(float64((v - minV) * inv)))
								if qi < 0 {
									qi = 0
								} else if qi > maxI {
									qi = maxI
								}
								q[j] = uint8(qi)
							}
						}
						base := r*cols + b*colBlock
						if bits == 8 {
							copy(m.Data[base:], q)
						} else if base%2 == 0 && colBlock%2 == 0 {
							vecops.Pack(m.Data[base/2:], q, 4)
						} else {
							for j, v := range q {
								i := base + j
								m.Data[i/2] |= (v & 0xf) << (4 * (i % 2))
							}
						}
					}
				}
			}
		}()
	}
	wg.Wait()
	return m
}

// Dequantize unpacks MinVal back into float32 values matching hardware MINVAL dequantization.
func (m MinVal) Dequantize() []float32 {
	out := make([]float32, m.Rows*m.Cols)
	nc := m.Cols / m.ColBlock
	for r := range m.Rows {
		for b := range nc {
			minV := f16val(m.Mins[r*nc+b])
			scV := f16val(m.Scales[r*nc+b])
			base := r*m.Cols + b*m.ColBlock
			for j := range m.ColBlock {
				i := base + j
				var q uint8
				if m.Bits == 8 {
					q = m.Data[i]
				} else {
					q = (m.Data[i/2] >> (4 * (i % 2))) & 0xf
				}
				out[i] = minV + float32(q)*scV
			}
		}
	}
	return out
}

// quantizeNaN is QuantizeLinear's block with NaNs, in Go: NaNs take part in the maximum and
// errors as they come (the vecops kernels do not promise that), and the scale is returned.
func quantizeNaN(dst []int8, blk []float32, qmin, qmax float32, search bool) uint16 {
	quant := func(v, s float32) int {
		if s == 0 {
			return 0
		}
		q := float32(math.RoundToEven(float64(v / s)))
		return int(min(max(q, qmin), qmax))
	}
	var top float32
	for _, v := range blk {
		top = max(top, float32(math.Abs(float64(v))))
	}
	best, bestErr := f16(top/qmax), float32(math.Inf(1))
	if search {
		for _, c := range clips {
			s := f16val(f16(top * c / qmax))
			var e float32
			for _, v := range blk {
				d := v - float32(float32(quant(v, s))*s)
				e = float32(e + float32(d*d))
			}
			if e < bestErr {
				best, bestErr = f16(top*c/qmax), e
			}
		}
	}
	s := f16val(best)
	for j, v := range blk {
		dst[j] = int8(quant(v, s))
	}
	return best
}

// Dequantize returns the matrix the quantized values stand for.
func (l Linear) Dequantize() []float32 {
	out := make([]float32, l.Rows*l.Cols)
	nc := l.Cols / l.ColBlock
	for i := range out {
		var q int
		if l.Bits == 8 {
			q = int(int8(l.Data[i]))
		} else {
			q = int(l.Data[i/2]>>(4*(i%2))) & 0xf
			if q >= 8 {
				q -= 16
			}
		}
		r, c := i/l.Cols, i%l.Cols
		out[i] = float32(q) * f16val(l.Scales[r*nc+c/l.ColBlock])
	}
	return out
}
