// Package tensorops implements element-type-agnostic multi-dimensional array layout operations
// shared by CPU tensor evaluators (internal/onnx and internal/milcpu).
package tensorops

// Numel returns the product of shape's dimensions (1 for a scalar empty shape).
func Numel(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

// Strides returns row-major strides for shape.
func Strides(shape []int) []int {
	st := make([]int, len(shape))
	s := 1
	for i := len(shape) - 1; i >= 0; i-- {
		st[i] = s
		s *= shape[i]
	}
	return st
}

// Permute copies elements from src into dst (len == Numel(shape)), starting at offset start in src
// and advancing by srcStrides[d] along output axis d.
func Permute[T any](dst, src []T, start int, shape, srcStrides []int) {
	idx := make([]int, len(shape))
	off := start
	for i := range dst {
		dst[i] = src[off]
		for d := len(shape) - 1; d >= 0; d-- {
			idx[d]++
			off += srcStrides[d]
			if idx[d] < shape[d] {
				break
			}
			off -= srcStrides[d] * idx[d]
			idx[d] = 0
		}
	}
}

// TransposePlan computes the output shape and source strides for transposing inShape by perm.
func TransposePlan[I ~int | ~int32 | ~int64](inShape []int, perm []I) (outShape, srcStrides []int) {
	st := Strides(inShape)
	outShape = make([]int, len(perm))
	srcStrides = make([]int, len(perm))
	for i, p := range perm {
		outShape[i] = inShape[p]
		srcStrides[i] = st[p]
	}
	return outShape, srcStrides
}

// Tile repeats src of inShape into dst of outShape (where outShape[d] == inShape[d] * reps[d]).
func Tile[T any](dst, src []T, inShape, outShape []int) {
	st := Strides(inShape)
	coord := make([]int, len(outShape))
	for k := range dst {
		off := 0
		for d := range outShape {
			off += (coord[d] % inShape[d]) * st[d]
		}
		dst[k] = src[off]
		for d := len(outShape) - 1; d >= 0; d-- {
			if coord[d]++; coord[d] < outShape[d] {
				break
			}
			coord[d] = 0
		}
	}
}

// Concat appends slices of srcs along an axis with precomputed outer = Numel(shape[:axis]) and
// inner = Numel(shape[axis+1:]). axisSizes[k] is srcs[k]'s dimension along axis.
func Concat[T any](dst []T, srcs [][]T, axisSizes []int, outer, inner int) []T {
	out := dst[:0]
	for o := range outer {
		for k, s := range srcs {
			w := axisSizes[k] * inner
			out = append(out, s[o*w:(o+1)*w]...)
		}
	}
	return out
}

// PadShape returns the output shape when inShape is padded by pads (length 2*len(inShape), pairs
// of [before, after] per axis).
func PadShape(inShape, pads []int) []int {
	out := make([]int, len(inShape))
	for i, d := range inShape {
		out[i] = d + pads[2*i] + pads[2*i+1]
	}
	return out
}

// Pad writes src (of inShape) padded by pads (length 2*len(inShape), [before, after] per axis)
// into dst (length Numel(PadShape(inShape, pads))). Supported modes are "reflect",
// "edge"/"replicate", and constant fill (any other mode, such as "" or "constant"). Negative pads
// crop along the corresponding axis.
func Pad[T any](dst, src []T, inShape, pads []int, mode string, constVal T) {
	r := len(inShape)
	outShape := PadShape(inShape, pads)
	st := Strides(inShape)
	coord := make([]int, r)
	for k := range dst {
		off, inside := 0, true
		for i := range r {
			j := coord[i] - pads[2*i]
			d := inShape[i]
			if j < 0 || j >= d {
				switch mode {
				case "reflect":
					if d == 1 {
						j = 0
						break
					}
					for j < 0 || j >= d {
						if j < 0 {
							j = -j
						}
						if j >= d {
							j = 2*(d-1) - j
						}
					}
				case "edge", "replicate":
					j = min(max(j, 0), d-1)
				default:
					inside = false
				}
			}
			off += j * st[i]
		}
		if inside {
			dst[k] = src[off]
		} else {
			dst[k] = constVal
		}
		for i := r - 1; i >= 0; i-- {
			if coord[i]++; coord[i] < outShape[i] {
				break
			}
			coord[i] = 0
		}
	}
}
