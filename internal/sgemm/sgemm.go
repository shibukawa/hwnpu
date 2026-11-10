// Package sgemm multiplies float32 matrices on the CPU, the inner loop of hwnpu's own CPU
// executor (internal/milcpu): convolutions as products with their im2col matrix, linears and
// matmuls. It is GotoBLAS's scheme: B packed in panels of NR columns, A in panels of MR rows, a
// register-blocked kernel of MR x NR (AVX2 and FMA on amd64, else Go) over a block of KC.
//
// B is read through a Source, so a convolution packs its im2col panels straight from the input
// without writing the matrix out.
package sgemm

import (
	"runtime"
	"sync"
)

// The kernel's block: up to MR rows of A (mr: 12 with AVX-512, else 6) by NR columns of B, and
// the blocking of K, M and N.
const (
	MR = 12
	NR = 16
	KC = 256
	MC = 96
	NC = 2048
)

// Source is the B of a product: Pack writes the block of rows k0 to k0+kc and columns j0 to
// j0+nc into dst as panels of NR columns (panel p at dst[p*kc*NR:], row kk's NR values at
// kk*NR), zeros past column n.
type Source interface {
	Pack(dst []float32, k0, kc, j0, nc int)
}

// Matrix is a row-major B (k x n): B[r][c] at Data[r*Stride+c].
type Matrix struct {
	Data   []float32
	Stride int
	N      int
}

// Pack writes a block of m's panels.
func (m Matrix) Pack(dst []float32, k0, kc, j0, nc int) {
	for p := 0; p*NR < nc; p++ {
		d := dst[p*kc*NR:][:kc*NR]
		c0 := j0 + p*NR
		w := min(NR, m.N-c0)
		for kk := range kc {
			row := m.Data[(k0+kk)*m.Stride+c0:]
			o := d[kk*NR : kk*NR+NR]
			copy(o, row[:w])
			clear(o[w:])
		}
	}
}

// Transposed is a B given as its transpose (n x k, row major: B[r][c] at Data[c*Stride+r]), as a
// linear's weight.
type Transposed struct {
	Data   []float32
	Stride int
	N      int
}

// Pack writes a block of t's panels.
func (t Transposed) Pack(dst []float32, k0, kc, j0, nc int) {
	for p := 0; p*NR < nc; p++ {
		d := dst[p*kc*NR:][:kc*NR]
		c0 := j0 + p*NR
		w := min(NR, t.N-c0)
		for c := range w {
			col := t.Data[(c0+c)*t.Stride+k0:][:kc]
			for kk, v := range col {
				d[kk*NR+c] = v
			}
		}
		if w < NR {
			for kk := range kc {
				clear(d[kk*NR+w : kk*NR+NR])
			}
		}
	}
}

// A is the A of a product (m x k): A[r][c] at Data[r*Stride+c], or with Trans at
// Data[c*Stride+r].
type A struct {
	Data   []float32
	Stride int
	Trans  bool
}

func (a A) at(r, c int) float32 {
	if a.Trans {
		return a.Data[c*a.Stride+r]
	}
	return a.Data[r*a.Stride+c]
}

// pack writes rows i0 to i0+mc and columns k0 to k0+kc of a as panels of mr rows (panel p at
// dst[p*kc*mr:], column kk's mr values at kk*mr), zeros past row m.
func (a A) pack(dst []float32, m, i0, mc, k0, kc int) {
	for p := 0; p*mr < mc; p++ {
		d := dst[p*kc*mr:][:kc*mr]
		r0 := i0 + p*mr
		h := min(mr, m-r0)
		if !a.Trans {
			for r := range h {
				row := a.Data[(r0+r)*a.Stride+k0:][:kc]
				for kk, v := range row {
					d[kk*mr+r] = v
				}
			}
		} else {
			for kk := range kc {
				col := a.Data[(k0+kk)*a.Stride+r0:][:h]
				copy(d[kk*mr:], col)
			}
		}
		if h < mr {
			for kk := range kc {
				clear(d[kk*mr+h : kk*mr+mr])
			}
		}
	}
}

// MulAdd computes C += A B for C of m x n (row major, C[r][c] at c[r*ldc+c]), A m x k, B k x n,
// on up to threads goroutines (0: GOMAXPROCS).
func MulAdd(m, n, k int, a A, b Source, c []float32, ldc, threads int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	if threads <= 0 {
		threads = runtime.GOMAXPROCS(0)
	}
	// split the columns into whole panels, or the rows when the columns are few
	panels := (n + NR - 1) / NR
	if threads > 1 && panels >= 2*threads {
		per := (panels + threads - 1) / threads
		var wg sync.WaitGroup
		for j := 0; j < n; j += per * NR {
			wg.Add(1)
			go func(j0, j1 int) {
				defer wg.Done()
				block(m, k, a, b, c, ldc, 0, m, j0, j1)
			}(j, min(n, j+per*NR))
		}
		wg.Wait()
		return
	}
	if threads > 1 && m >= 2*mr*threads {
		per := ((m+threads-1)/threads + mr - 1) / mr * mr
		var wg sync.WaitGroup
		for i := 0; i < m; i += per {
			wg.Add(1)
			go func(i0, i1 int) {
				defer wg.Done()
				block(m, k, a, b, c, ldc, i0, i1, 0, n)
			}(i, min(m, i+per))
		}
		wg.Wait()
		return
	}
	block(m, k, a, b, c, ldc, 0, m, 0, n)
}

var bufs = sync.Pool{New: func() any {
	return &buffers{a: make([]float32, (MC+MR)*KC), b: make([]float32, NC*KC)}
}}

type buffers struct{ a, b []float32 }

// block computes rows i0 to i1 and columns j0 to j1 of C.
func block(m, k int, a A, b Source, c []float32, ldc, i0, i1, j0, j1 int) {
	buf := bufs.Get().(*buffers)
	defer bufs.Put(buf)
	var edge [MR * NR]float32
	for jc := j0; jc < j1; jc += NC {
		nc := min(NC, j1-jc)
		for pc := 0; pc < k; pc += KC {
			kc := min(KC, k-pc)
			b.Pack(buf.b, pc, kc, jc, nc)
			for ic := i0; ic < i1; ic += MC {
				mc := min(MC, i1-ic)
				a.pack(buf.a, m, ic, mc, pc, kc)
				for jr := 0; jr < nc; jr += NR {
					bp := buf.b[jr/NR*kc*NR:]
					w := min(NR, nc-jr)
					for ir := 0; ir < mc; ir += mr {
						ap := buf.a[ir/mr*kc*mr:]
						h := min(mr, mc-ir)
						cp := c[(ic+ir)*ldc+jc+jr:]
						if h == mr && w == NR {
							kernel(kc, ap, bp, cp, ldc)
							continue
						}
						clear(edge[:])
						kernel(kc, ap, bp, edge[:], NR)
						for r := range h {
							row := cp[r*ldc : r*ldc+w]
							for j := range row {
								row[j] += edge[r*NR+j]
							}
						}
					}
				}
			}
		}
	}
}

// kernelGo is the kernel in Go: c (mr rows of NR, ldc apart) += a (kc x mr) b (kc x NR).
func kernelGo(kc int, a, b, c []float32, ldc int) {
	var acc [MR][NR]float32
	for kk := range kc {
		bp := b[kk*NR : kk*NR+NR]
		ap := a[kk*mr : kk*mr+mr]
		for r := range mr {
			av := ap[r]
			row := &acc[r]
			for j := range NR {
				row[j] += av * bp[j]
			}
		}
	}
	for r := range mr {
		row := c[r*ldc : r*ldc+NR]
		for j := range NR {
			row[j] += acc[r][j]
		}
	}
}
