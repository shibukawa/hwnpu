package palette

import (
	"encoding/binary"
	"math"
	"runtime"
	"sync"
)

// Q4NXGroup is the number of columns that share a scale and a minimum in q4nx.
const Q4NXGroup = 32

// Q4NX is an asymmetrically quantized matrix in the q4nx format of FastFlowLM's AMD NPU kernels
// (mlir-aie aie_kernels/flm_gemma4/q4nx.h): each weight is a 4 bit unsigned code q and stands
// for scale * q + min, with one bf16 scale and one bf16 minimum per row and Q4NXGroup columns.
type Q4NX struct {
	Rows, Cols int
	// Codes holds the codes in row major order, two per byte with the low nibble first.
	Codes []byte
	// Scales and Mins hold one bf16 value per group, groups in row major order (row, col/32).
	Scales, Mins []uint16
}

// QuantizeQ4NX quantizes a row major rows x cols matrix (cols a multiple of Q4NXGroup) to q4nx.
// A group's range is its smallest to its largest weight; with search it is instead that range
// narrowed about its middle by a factor from clips, whichever gives the smallest squared error.
func QuantizeQ4NX(w []float32, rows, cols int, search bool) Q4NX {
	if cols%Q4NXGroup != 0 {
		panic("palette: q4nx needs columns in groups of 32")
	}
	ng := cols / Q4NXGroup
	q := Q4NX{Rows: rows, Cols: cols, Codes: make([]byte, (rows*cols+1)/2), Scales: make([]uint16, rows*ng), Mins: make([]uint16, rows*ng)}
	code := func(v, s, m float32) int {
		if s == 0 {
			return 0
		}
		c := math.RoundToEven(float64((v - m) / s))
		return int(min(max(c, 0), 15))
	}
	// fit returns the bf16 scale and minimum for the range lo..hi of blk.
	fit := func(lo, hi float32) (uint16, uint16) {
		if hi == lo {
			// one value: scale * 0 + min, with a nonzero scale so that min / scale is finite
			return bf16(max(float32(math.Abs(float64(lo))), 1e-30)), bf16(lo)
		}
		return bf16((hi - lo) / 15), bf16(lo)
	}
	work := make(chan int, rows)
	for r := 0; r < rows; r++ {
		work <- r
	}
	close(work)
	var wg sync.WaitGroup
	for range runtime.GOMAXPROCS(0) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for r := range work {
				for g := 0; g < ng; g++ {
					blk := w[r*cols+g*Q4NXGroup : r*cols+(g+1)*Q4NXGroup]
					lo, hi := blk[0], blk[0]
					for _, v := range blk {
						lo, hi = min(lo, v), max(hi, v)
					}
					bs, bm := fit(lo, hi)
					if search && hi > lo {
						bestErr := float32(math.Inf(1))
						mid, half := (lo+hi)/2, (hi-lo)/2
						for _, c := range clips {
							cs, cm := fit(mid-half*c, mid+half*c)
							s, m := bf16val(cs), bf16val(cm)
							var e float32
							for _, v := range blk {
								d := v - (s*float32(code(v, s, m)) + m)
								e += d * d
							}
							if e < bestErr {
								bs, bm, bestErr = cs, cm, e
							}
						}
					}
					q.Scales[r*ng+g], q.Mins[r*ng+g] = bs, bm
					s, m := bf16val(bs), bf16val(bm)
					for j, v := range blk {
						i := r*cols + g*Q4NXGroup + j
						q.Codes[i/2] |= byte(code(v, s, m)) << (4 * (i % 2))
					}
				}
			}
		}()
	}
	wg.Wait()
	return q
}

// Code returns the code of element i (row major).
func (q Q4NX) Code(i int) int { return int(q.Codes[i/2]>>(4*(i%2))) & 0xf }

// Dequantize returns the matrix the codes stand for: scale * q + min.
func (q Q4NX) Dequantize() []float32 {
	out := make([]float32, q.Rows*q.Cols)
	ng := q.Cols / Q4NXGroup
	for i := range out {
		g := (i/q.Cols)*ng + (i%q.Cols)/Q4NXGroup
		out[i] = bf16val(q.Scales[g])*float32(q.Code(i)) + bf16val(q.Mins[g])
	}
	return out
}

// Block sizes of the AMD NPU kernel's q4nx blocks (q4nx_block<32, 256>).
const (
	Q4NXBlockRows  = 32
	Q4NXBlockCols  = 256
	Q4NXBlockBytes = Q4NXBlockRows*Q4NXBlockCols/2 + 4*Q4NXBlockRows*Q4NXBlockCols/Q4NXGroup
)

// Blocks returns the matrix as the kernel's 32 x 256 blocks, row blocks outer and column blocks
// inner. A block holds the scales, then the minima (bf16, little endian, indexed [column group,
// row]), then the codes indexed [row / 16, column, row % 16], two per byte with the low nibble
// first. Rows must be a multiple of 32 and columns of 256.
func (q Q4NX) Blocks() []byte {
	if q.Rows%Q4NXBlockRows != 0 || q.Cols%Q4NXBlockCols != 0 {
		panic("palette: q4nx blocks need rows in 32s and columns in 256s")
	}
	ng := q.Cols / Q4NXGroup
	const gpb = Q4NXBlockCols / Q4NXGroup // groups per block
	out := make([]byte, 0, q.Rows/Q4NXBlockRows*q.Cols/Q4NXBlockCols*Q4NXBlockBytes)
	for r0 := 0; r0 < q.Rows; r0 += Q4NXBlockRows {
		for c0 := 0; c0 < q.Cols; c0 += Q4NXBlockCols {
			for _, tab := range [][]uint16{q.Scales, q.Mins} {
				for g := 0; g < gpb; g++ {
					for r := 0; r < Q4NXBlockRows; r++ {
						out = binary.LittleEndian.AppendUint16(out, tab[(r0+r)*ng+c0/Q4NXGroup+g])
					}
				}
			}
			var codes [Q4NXBlockRows * Q4NXBlockCols]byte
			n := 0
			for strip := 0; strip < Q4NXBlockRows/16; strip++ {
				for c := 0; c < Q4NXBlockCols; c++ {
					for r := 0; r < 16; r++ {
						codes[n] = byte(q.Code((r0+strip*16+r)*q.Cols + c0 + c))
						n++
					}
				}
			}
			for i := 0; i < len(codes); i += 2 {
				out = append(out, codes[i]|codes[i+1]<<4)
			}
		}
	}
	return out
}

// bf16 rounds f to the nearest bfloat16, ties to even.
func bf16(f float32) uint16 {
	b := math.Float32bits(f)
	if b&0x7f800000 == 0x7f800000 { // inf, nan
		return uint16(b >> 16)
	}
	b += 0x7fff + (b>>16)&1
	return uint16(b >> 16)
}

func bf16val(h uint16) float32 { return math.Float32frombits(uint32(h) << 16) }

// BF16 and BF16Value are bf16 and bf16val for other packages (the CPU model of the kernel).
func BF16(f float32) uint16      { return bf16(f) }
func BF16Value(h uint16) float32 { return bf16val(h) }

// ShiftScale returns the fp16 scales and offsets of the matrix as (code - offset) * scale
// (Core ML's constexpr_blockwise_shift_scale): offset is -min / scale. The fp16 offset is within
// 2^-11 of its value, so a weight moves by less than 2^-11 * 15 steps of its group.
func (q Q4NX) ShiftScale() (scales, offsets []uint16) {
	scales, offsets = make([]uint16, len(q.Scales)), make([]uint16, len(q.Scales))
	for i, s := range q.Scales {
		sv, mv := bf16val(s), bf16val(q.Mins[i])
		scales[i] = f16(sv)
		if sv != 0 {
			offsets[i] = f16(-mv / sv)
		}
	}
	return scales, offsets
}
