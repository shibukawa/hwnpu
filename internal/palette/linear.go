package palette

import (
	"math"
	"runtime"
	"sync"
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
// search: clipping the few largest weights lets the bulk use more of the levels.
var clips = []float32{1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5}

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
	qmax := float32(int(1)<<(bits-1) - 1)
	qmin := -qmax - 1
	quant := func(v, s float32) int {
		if s == 0 {
			return 0
		}
		q := float32(math.RoundToEven(float64(v / s)))
		return int(min(max(q, qmin), qmax))
	}
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
			for r0 := range work {
				for r := r0; r < min(r0+2, rows); r++ {
					for b := 0; b < nc; b++ {
						blk := w[r*cols+b*colBlock : r*cols+(b+1)*colBlock]
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
									d := v - float32(quant(v, s))*s
									e += d * d
								}
								if e < bestErr {
									best, bestErr = f16(top*c/qmax), e
								}
							}
						}
						l.Scales[r*nc+b] = best
						s := f16val(best)
						for j, v := range blk {
							i := r*cols + b*colBlock + j
							q := quant(v, s)
							if bits == 8 {
								l.Data[i] = byte(int8(q))
							} else {
								l.Data[i/2] |= byte(q&0xf) << (4 * (i % 2))
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
