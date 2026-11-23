package f32

import "unsafe"

// Depthwise3 writes a row of a 3 x 3 convolution of one channel, stride 1, from three rows of
// its input padded by a column on each side: dst[i] = bias + the sum over k < 3 of w[3k] rk[i]
// + w[3k+1] rk[i+1] + w[3k+2] rk[i+2], then max(., 0) when relu. Each rk holds len(dst) + 2.
func Depthwise3(dst, r0, r1, r2 []float32, w *[9]float32, bias float32, relu bool) {
	m := len(dst)
	r0, r1, r2 = r0[:m+2], r1[:m+2], r2[:m+2]
	n := dw3Kernel(dst, r0, r1, r2, w, bias, relu)
	if n > 0 && n < m {
		// the last vector again, ending at m: it writes what it computes, so the overlap is
		// the same values
		dw3Kernel(dst[m-8:], r0[m-8:], r1[m-8:], r2[m-8:], w, bias, relu)
		return
	}
	for i := n; i < m; i++ {
		v := bias + w[0]*r0[i] + w[1]*r0[i+1] + w[2]*r0[i+2] +
			w[3]*r1[i] + w[4]*r1[i+1] + w[5]*r1[i+2] +
			w[6]*r2[i] + w[7]*r2[i+1] + w[8]*r2[i+2]
		if relu {
			v = max(v, 0)
		}
		dst[i] = v
	}
}

// MaxPool2 writes a row of a 2 x 2 max pool of stride 2: dst[i] the largest of a[2i], a[2i+1],
// b[2i] and b[2i+1], a and b two rows of the input.
func MaxPool2(dst, a, b []float32) {
	m := len(dst)
	a, b = a[:2*m], b[:2*m]
	n := maxpool2Kernel(dst, a, b)
	if n > 0 && n < m {
		maxpool2Kernel(dst[m-8:], a[2*(m-8):], b[2*(m-8):]) // overlapping, as Depthwise3's
		return
	}
	for i := n; i < m; i++ {
		dst[i] = max(a[2*i], a[2*i+1], b[2*i], b[2*i+1])
	}
}

// ReLU writes max(x, 0) of src into dst.
func ReLU(dst, src []float32) {
	src = src[:len(dst)]
	n := reluKernel(dst, src)
	for i := n; i < len(dst); i++ {
		dst[i] = max(src[i], 0)
	}
}

// WeightedRows writes dst[i] = bias + the sum over j of w[j] rows[j][i], then max(., 0) when
// relu: a row of a direct convolution, each of rows the start of an input row at a tap (each
// holding len(dst) values from it). rows and w are of the same length.
func WeightedRows(dst []float32, rows []*float32, w []float32, bias float32, relu bool) {
	w = w[:len(rows)]
	n := wrowsKernel(dst, rows, w, bias, relu)
	if n == len(dst) {
		return
	}
	if m := len(dst); n > 0 {
		// the last 16 again, ending at m (overlapping, as Depthwise3's): rows moved on
		var moved [64]*float32
		rs := moved[:0]
		if len(rows) > len(moved) {
			rs = make([]*float32, 0, len(rows))
		}
		for _, p := range rows {
			rs = append(rs, (*float32)(unsafe.Add(unsafe.Pointer(p), 4*(m-16))))
		}
		wrowsKernel(dst[m-16:], rs, w, bias, relu)
		return
	}
	for i := n; i < len(dst); i++ {
		dst[i] = bias
	}
	for j, p := range rows {
		r := unsafe.Slice(p, len(dst))
		a := w[j]
		for i := n; i < len(dst); i++ {
			dst[i] += a * r[i]
		}
	}
	if relu {
		for i := n; i < len(dst); i++ {
			dst[i] = max(dst[i], 0)
		}
	}
}

// WeightedRows4 is WeightedRows for four outputs of the same rows at once: for o < 4,
// dst[o][i] = bias[o] + the sum over j of w[4j+o] rows[j][i] for i < n (then max(., 0) when
// relu), each dst[o] holding n values. A row read once serves the four.
func WeightedRows4(dst *[4]*float32, n int, rows []*float32, w []float32, bias *[4]float32, relu bool) {
	w = w[:4*len(rows)]
	done := wrows4Kernel(dst, n, rows, w, bias, relu)
	if done == n {
		return
	}
	if done > 0 {
		// the last 16 again, ending at n (overlapping, as Depthwise3's)
		var moved [256]*float32
		rs := moved[:0]
		if len(rows) > len(moved) {
			rs = make([]*float32, 0, len(rows))
		}
		for _, p := range rows {
			rs = append(rs, (*float32)(unsafe.Add(unsafe.Pointer(p), 4*(n-16))))
		}
		var d [4]*float32
		for o := range 4 {
			d[o] = (*float32)(unsafe.Add(unsafe.Pointer(dst[o]), 4*(n-16)))
		}
		wrows4Kernel(&d, 16, rs, w, bias, relu)
		return
	}
	for o := range 4 {
		d := unsafe.Slice(dst[o], n)
		for i := done; i < n; i++ {
			d[i] = bias[o]
		}
		for j, p := range rows {
			r := unsafe.Slice(p, n)
			a := w[4*j+o]
			for i := done; i < n; i++ {
				d[i] += a * r[i]
			}
		}
		if relu {
			for i := done; i < n; i++ {
				d[i] = max(d[i], 0)
			}
		}
	}
}

// Deinterleave2 splits src into its even and its odd values: even[i] = src[2i], odd[i] =
// src[2i+1] (a row of a convolution of stride 2 cut into its two phases). src holds
// 2 len(even) values; odd is as long as even.
func Deinterleave2(even, odd, src []float32) {
	m := len(even)
	odd, src = odd[:m], src[:2*m]
	n := deint2Kernel(even, odd, src)
	if n > 0 && n < m {
		deint2Kernel(even[m-8:], odd[m-8:], src[2*(m-8):]) // overlapping, as Depthwise3's
		return
	}
	for i := n; i < m; i++ {
		even[i], odd[i] = src[2*i], src[2*i+1]
	}
}
