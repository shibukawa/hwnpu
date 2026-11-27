package chat

import (
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/shibukawa/hwnpu"
)

// Hadamard-rotated Lloyd-Max low-bit KV compression (modeled after openset-llm's kvq.h).
// Each vector of length D (power of two, e.g. 64 or 128) is Walsh-Hadamard rotated so its
// coordinates are nearly Gaussian with a single scale, then quantized to b bits (5 or 6) using
// Lloyd-Max optimal levels for a unit Gaussian (rounded to int8) and a least-squares f32 gain.
//
// Layout per block of D values: f32 gain (4 bytes) + ceil(D*b/8) packed little-endian b-bit codes.
// For D=64:  f16/bf16 is 128 B; kv6 is  52 B (40.6%), kv5 is  44 B (34.4%).
// For D=128: f16/bf16 is 256 B; kv6 is 100 B (39.1%), kv5 is  84 B (32.8%).

var (
	kvqLv4 = []int8{6, 18, 30, 43, 58, 74, 95, 126}
	kvqLv5 = []int8{3, 8, 13, 18, 23, 28, 34, 40, 46, 53, 60, 68, 77, 88, 102, 124}
	kvqLv6 = []int8{
		1, 3, 6, 8, 10, 12, 15, 17, 19, 22, 24, 27, 29, 32, 34, 37,
		40, 43, 46, 49, 52, 55, 59, 63, 67, 72, 77, 83, 89, 97, 108, 125,
	}
	kvqUnit = [7]float32{0, 0, 0, 0, 46.0, 37.81, 33.68}
)

func kvqLevels(bits int) []int8 {
	switch bits {
	case 4:
		return kvqLv4
	case 5:
		return kvqLv5
	case 6:
		return kvqLv6
	default:
		return nil
	}
}

func kvqBlockBytes(dim, bits int) int {
	return 4 + (dim*bits+7)/8
}

func kvqDTypeOK(dtype string) bool {
	return dtype == "f16" || dtype == "bf16"
}

func decodeKVQElems(dst []float32, src []byte, dtype string) {
	if dtype == "bf16" {
		for i := range dst {
			u := uint32(binary.LittleEndian.Uint16(src[2*i:])) << 16
			dst[i] = math.Float32frombits(u)
		}
		return
	}
	hwnpu.DecodeF16(dst, src)
}

func encodeKVQElems(dst []byte, src []float32, dtype string) {
	if dtype == "bf16" {
		for i, v := range src {
			u := math.Float32bits(v)
			if u&0x7fffffff > 0x7f800000 { // NaN
				binary.LittleEndian.PutUint16(dst[2*i:], uint16(u>>16)|0x0040)
				continue
			}
			// Round to nearest even.
			u += 0x7fff + ((u >> 16) & 1)
			binary.LittleEndian.PutUint16(dst[2*i:], uint16(u>>16))
		}
		return
	}
	hwnpu.EncodeF16(dst, src)
}

// fwhtInPlace applies the normalized Walsh-Hadamard transform in place to x (len must be a power of 2).
// Since H is symmetric orthogonal, applying fwhtInPlace twice recovers x.
func fwhtInPlace(x []float32) {
	n := len(x)
	for h := 1; h < n; h <<= 1 {
		for i := 0; i < n; i += h << 1 {
			for j := i; j < i+h; j++ {
				u, v := x[j], x[j+h]
				x[j] = u + v
				x[j+h] = u - v
			}
		}
	}
	scale := float32(1 / math.Sqrt(float64(n)))
	for i := range x {
		x[i] *= scale
	}
}

// kvqCompress compresses fp16 or bf16 KV bytes into Hadamard-rotated b-bit Lloyd-Max blocks of dim values.
func kvqCompress(src []byte, dtype string, dim, bits int) ([]byte, error) {
	if !kvqDTypeOK(dtype) {
		return nil, fmt.Errorf("kvq: unsupported dtype %q", dtype)
	}
	lv := kvqLevels(bits)
	if lv == nil {
		return nil, fmt.Errorf("kvq: unsupported bits %d", bits)
	}
	if dim <= 0 || dim&(dim-1) != 0 {
		return nil, fmt.Errorf("kvq: dim %d must be a power of 2", dim)
	}
	nvals := len(src) / 2
	if len(src)%2 != 0 || nvals%dim != 0 {
		return nil, fmt.Errorf("kvq: %d bytes not a multiple of dim %d %s", len(src), dim, dtype)
	}
	nblk := nvals / dim
	bs := kvqBlockBytes(dim, bits)
	out := make([]byte, nblk*bs)

	nl := 1 << (bits - 1)
	var thr [32]float32
	for t := 0; t+1 < nl; t++ {
		thr[t] = 0.5 * float32(int(lv[t])+int(lv[t+1]))
	}
	unit := kvqUnit[bits]

	runRange := func(loBlk, hiBlk int) {
		buf := make([]float32, dim)
		idx := make([]int, dim)
		for b := loBlk; b < hiBlk; b++ {
			decodeKVQElems(buf, src[b*dim*2:(b+1)*dim*2], dtype)
			fwhtInPlace(buf)
			var ss float32
			for _, v := range buf {
				ss += v * v
			}
			rms := float32(math.Sqrt(float64(ss) / float64(dim)))
			var is float32
			if rms > 0 {
				is = unit / rms
			}
			var xc, cc float32
			for i, v := range buf {
				av := v
				if av < 0 {
					av = -av
				}
				a := av * is
				lo, hi := 0, nl-1
				for lo < hi {
					mid := (lo + hi) >> 1
					if a > thr[mid] {
						lo = mid + 1
					} else {
						hi = mid
					}
				}
				idx[i] = lo
				l := float32(lv[lo])
				xc += av * l
				cc += l * l
			}
			var d float32
			if cc > 0 {
				d = xc / cc
			}
			dst := out[b*bs : (b+1)*bs]
			binary.LittleEndian.PutUint32(dst[:4], math.Float32bits(d))
			qs := dst[4:]
			for i, v := range buf {
				code := uint32(idx[i])
				if v < 0 {
					code |= uint32(nl)
				}
				bit := i * bits
				by, sh := bit>>3, bit&7
				val := code << sh
				qs[by] |= byte(val)
				if val>>8 != 0 {
					qs[by+1] |= byte(val >> 8)
				}
			}
		}
	}

	workers := runtime.GOMAXPROCS(0)
	if workers <= 1 || nblk < 256 {
		runRange(0, nblk)
		return out, nil
	}
	if workers > nblk/64 {
		workers = max(1, nblk/64)
	}
	var wg sync.WaitGroup
	for w := range workers {
		lo := w * nblk / workers
		hi := (w + 1) * nblk / workers
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			runRange(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
	return out, nil
}

// kvqDecompress restores fp16 or bf16 KV bytes from Hadamard-rotated b-bit Lloyd-Max blocks.
func kvqDecompress(src []byte, dtype string, dim, bits int) ([]byte, error) {
	if !kvqDTypeOK(dtype) {
		return nil, fmt.Errorf("kvq: unsupported dtype %q", dtype)
	}
	lv := kvqLevels(bits)
	if lv == nil {
		return nil, fmt.Errorf("kvq: unsupported bits %d", bits)
	}
	if dim <= 0 || dim&(dim-1) != 0 {
		return nil, fmt.Errorf("kvq: dim %d must be a power of 2", dim)
	}
	bs := kvqBlockBytes(dim, bits)
	if len(src)%bs != 0 {
		return nil, fmt.Errorf("kvq: %d bytes not a multiple of block size %d", len(src), bs)
	}
	nblk := len(src) / bs
	out := make([]byte, nblk*dim*2)
	nl := 1 << (bits - 1)
	mask := uint32((1 << bits) - 1)
	var lut [64]float32
	for c := 0; c < nl; c++ {
		lut[c] = float32(lv[c])
		lut[c+nl] = -float32(lv[c])
	}
	dataBytes := bs - 4

	runRange := func(loBlk, hiBlk int) {
		buf := make([]float32, dim)
		for b := loBlk; b < hiBlk; b++ {
			blk := src[b*bs : (b+1)*bs]
			d := math.Float32frombits(binary.LittleEndian.Uint32(blk[:4]))
			qs := blk[4:]
			for i := range dim {
				bit := i * bits
				by, sh := bit>>3, bit&7
				w := uint32(qs[by])
				if by+1 < dataBytes {
					w |= uint32(qs[by+1]) << 8
				}
				c := (w >> sh) & mask
				buf[i] = d * lut[c]
			}
			fwhtInPlace(buf)
			encodeKVQElems(out[b*dim*2:(b+1)*dim*2], buf, dtype)
		}
	}

	workers := runtime.GOMAXPROCS(0)
	if workers <= 1 || nblk < 256 {
		runRange(0, nblk)
		return out, nil
	}
	if workers > nblk/64 {
		workers = max(1, nblk/64)
	}
	var wg sync.WaitGroup
	for w := range workers {
		lo := w * nblk / workers
		hi := (w + 1) * nblk / workers
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			runRange(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
	return out, nil
}
