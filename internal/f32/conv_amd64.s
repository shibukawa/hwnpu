#include "textflag.h"

// func dw3AVX2(dst, r0, r1, r2, w *float32, bias float32, relu, n int)
// dst[i] = bias + sum over k of w[3k] rk[i] + w[3k+1] rk[i+1] + w[3k+2] rk[i+2], then max with
// 0 when relu; n a multiple of 8. A row's three taps go to an accumulator of their own (the
// chains of FMAs three long, not nine), added at the end.
TEXT ·dw3AVX2(SB), NOSPLIT, $0-64
	MOVQ dst+0(FP), DI
	MOVQ r0+8(FP), R8
	MOVQ r1+16(FP), R9
	MOVQ r2+24(FP), R10
	MOVQ w+32(FP), SI
	VBROADCASTSS bias+40(FP), Y15
	MOVQ relu+48(FP), DX
	MOVQ n+56(FP), CX
	VBROADCASTSS 0(SI), Y0
	VBROADCASTSS 4(SI), Y1
	VBROADCASTSS 8(SI), Y2
	VBROADCASTSS 12(SI), Y3
	VBROADCASTSS 16(SI), Y4
	VBROADCASTSS 20(SI), Y5
	VBROADCASTSS 24(SI), Y6
	VBROADCASTSS 28(SI), Y7
	VBROADCASTSS 32(SI), Y8
	VXORPS Y14, Y14, Y14
loopd:
	VMULPS (R8), Y0, Y9
	VMULPS (R9), Y3, Y10
	VMULPS (R10), Y6, Y11
	VFMADD231PS 4(R8), Y1, Y9
	VFMADD231PS 4(R9), Y4, Y10
	VFMADD231PS 4(R10), Y7, Y11
	VFMADD231PS 8(R8), Y2, Y9
	VFMADD231PS 8(R9), Y5, Y10
	VFMADD231PS 8(R10), Y8, Y11
	VADDPS Y15, Y9, Y9
	VADDPS Y11, Y10, Y10
	VADDPS Y10, Y9, Y9
	TESTQ DX, DX
	JZ   stored
	VMAXPS Y14, Y9, Y9
stored:
	VMOVUPS Y9, (DI)
	ADDQ $32, DI
	ADDQ $32, R8
	ADDQ $32, R9
	ADDQ $32, R10
	SUBQ $8, CX
	JNZ  loopd
	VZEROUPPER
	RET

// func maxpool2AVX2(dst, a, b *float32, n int)
// dst[i] = max(a[2i], a[2i+1], b[2i], b[2i+1]); n a multiple of 8.
TEXT ·maxpool2AVX2(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ a+8(FP), R8
	MOVQ b+16(FP), R9
	MOVQ n+24(FP), CX
loopm:
	VMOVUPS (R8), Y0
	VMOVUPS 32(R8), Y1
	VMAXPS (R9), Y0, Y0
	VMAXPS 32(R9), Y1, Y1
	// the even and the odd columns of each 128-bit lane, then the lanes' 64-bit halves in order
	VSHUFPS $0x88, Y1, Y0, Y2
	VSHUFPS $0xDD, Y1, Y0, Y3
	VMAXPS Y3, Y2, Y2
	VPERMPD $0xD8, Y2, Y2
	VMOVUPS Y2, (DI)
	ADDQ $32, DI
	ADDQ $64, R8
	ADDQ $64, R9
	SUBQ $8, CX
	JNZ  loopm
	VZEROUPPER
	RET

// func reluAVX2(dst, src *float32, n int)
// n a multiple of 32.
TEXT ·reluAVX2(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	VXORPS Y15, Y15, Y15
loopr:
	VMAXPS (SI), Y15, Y0
	VMAXPS 32(SI), Y15, Y1
	VMAXPS 64(SI), Y15, Y2
	VMAXPS 96(SI), Y15, Y3
	VMOVUPS Y0, (DI)
	VMOVUPS Y1, 32(DI)
	VMOVUPS Y2, 64(DI)
	VMOVUPS Y3, 96(DI)
	ADDQ $128, SI
	ADDQ $128, DI
	SUBQ $32, CX
	JNZ  loopr
	VZEROUPPER
	RET

// func wrowsAVX2(dst *float32, rows **float32, w *float32, k int, bias float32, relu, n int)
// dst[i] = bias + sum over j < k of w[j] rows[j][i], then max with 0 when relu; n a multiple of
// 16. Two vectors of outputs a pass, each with two accumulators (the even and the odd rows), so
// that four chains of FMAs run side by side.
TEXT ·wrowsAVX2(SB), NOSPLIT, $0-56
	MOVQ dst+0(FP), DI
	MOVQ rows+8(FP), BX
	MOVQ w+16(FP), SI
	MOVQ k+24(FP), R13
	VBROADCASTSS bias+32(FP), Y15
	MOVQ relu+40(FP), DX
	MOVQ n+48(FP), CX
	XORQ R14, R14 // the byte offset of the outputs
	VXORPS Y14, Y14, Y14
loopw:
	VMOVAPS Y15, Y0
	VMOVAPS Y15, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	XORQ R11, R11
	MOVQ R13, R12
	SUBQ $1, R12 // pairs while two rows remain
pairw:
	CMPQ R11, R12
	JGE  lastw
	MOVQ (BX)(R11*8), R8
	MOVQ 8(BX)(R11*8), R9
	VBROADCASTSS (SI)(R11*4), Y4
	VBROADCASTSS 4(SI)(R11*4), Y5
	VFMADD231PS (R8)(R14*1), Y4, Y0
	VFMADD231PS 32(R8)(R14*1), Y4, Y1
	VFMADD231PS (R9)(R14*1), Y5, Y2
	VFMADD231PS 32(R9)(R14*1), Y5, Y3
	ADDQ $2, R11
	JMP  pairw
lastw:
	CMPQ R11, R13
	JGE  sumw
	MOVQ (BX)(R11*8), R8
	VBROADCASTSS (SI)(R11*4), Y4
	VFMADD231PS (R8)(R14*1), Y4, Y0
	VFMADD231PS 32(R8)(R14*1), Y4, Y1
sumw:
	VADDPS Y2, Y0, Y0
	VADDPS Y3, Y1, Y1
	TESTQ DX, DX
	JZ   storew
	VMAXPS Y14, Y0, Y0
	VMAXPS Y14, Y1, Y1
storew:
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 32(DI)(R14*1)
	ADDQ $64, R14
	SUBQ $16, CX
	JNZ  loopw
	VZEROUPPER
	RET

// func wrows4AVX2(dst **float32, rows **float32, w *float32, k int, bias *float32, relu, n int)
// For o < 4: dst[o][i] = bias[o] + sum over j < k of w[4j+o] rows[j][i], then max with 0 when
// relu; n a multiple of 16. Each row's two vectors go into the four outputs' eight accumulators
// (a load for four FMAs, eight chains side by side).
TEXT ·wrows4AVX2(SB), NOSPLIT, $0-56
	MOVQ dst+0(FP), AX
	MOVQ rows+8(FP), BX
	MOVQ w+16(FP), SI
	MOVQ k+24(FP), R13
	MOVQ bias+32(FP), R10
	MOVQ n+48(FP), CX
	MOVQ 0(AX), DI
	MOVQ 8(AX), R12
	MOVQ 16(AX), DX
	MOVQ 24(AX), AX
	XORQ R14, R14
	VXORPS Y14, Y14, Y14
loop4:
	VBROADCASTSS 0(R10), Y0
	VBROADCASTSS 4(R10), Y2
	VBROADCASTSS 8(R10), Y4
	VBROADCASTSS 12(R10), Y6
	VMOVAPS Y0, Y1
	VMOVAPS Y2, Y3
	VMOVAPS Y4, Y5
	VMOVAPS Y6, Y7
	XORQ R11, R11
	MOVQ SI, R9
tap4:
	MOVQ (BX)(R11*8), R8
	VMOVUPS (R8)(R14*1), Y8
	VMOVUPS 32(R8)(R14*1), Y9
	VBROADCASTSS 0(R9), Y10
	VBROADCASTSS 4(R9), Y11
	VBROADCASTSS 8(R9), Y12
	VBROADCASTSS 12(R9), Y13
	VFMADD231PS Y8, Y10, Y0
	VFMADD231PS Y9, Y10, Y1
	VFMADD231PS Y8, Y11, Y2
	VFMADD231PS Y9, Y11, Y3
	VFMADD231PS Y8, Y12, Y4
	VFMADD231PS Y9, Y12, Y5
	VFMADD231PS Y8, Y13, Y6
	VFMADD231PS Y9, Y13, Y7
	ADDQ $16, R9
	INCQ R11
	CMPQ R11, R13
	JLT  tap4
	CMPQ relu+40(FP), $0
	JEQ  store4
	VMAXPS Y14, Y0, Y0
	VMAXPS Y14, Y1, Y1
	VMAXPS Y14, Y2, Y2
	VMAXPS Y14, Y3, Y3
	VMAXPS Y14, Y4, Y4
	VMAXPS Y14, Y5, Y5
	VMAXPS Y14, Y6, Y6
	VMAXPS Y14, Y7, Y7
store4:
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 32(DI)(R14*1)
	VMOVUPS Y2, (R12)(R14*1)
	VMOVUPS Y3, 32(R12)(R14*1)
	VMOVUPS Y4, (DX)(R14*1)
	VMOVUPS Y5, 32(DX)(R14*1)
	VMOVUPS Y6, (AX)(R14*1)
	VMOVUPS Y7, 32(AX)(R14*1)
	ADDQ $64, R14
	SUBQ $16, CX
	JNZ  loop4
	VZEROUPPER
	RET

// func deint2AVX2(even, odd, src *float32, n int)
// even[i] = src[2i], odd[i] = src[2i+1]; n a multiple of 8.
TEXT ·deint2AVX2(SB), NOSPLIT, $0-32
	MOVQ even+0(FP), DI
	MOVQ odd+8(FP), R8
	MOVQ src+16(FP), SI
	MOVQ n+24(FP), CX
loopi:
	VMOVUPS (SI), Y0
	VMOVUPS 32(SI), Y1
	VSHUFPS $0x88, Y1, Y0, Y2
	VSHUFPS $0xDD, Y1, Y0, Y3
	VPERMPD $0xD8, Y2, Y2
	VPERMPD $0xD8, Y3, Y3
	VMOVUPS Y2, (DI)
	VMOVUPS Y3, (R8)
	ADDQ $64, SI
	ADDQ $32, DI
	ADDQ $32, R8
	SUBQ $8, CX
	JNZ  loopi
	VZEROUPPER
	RET
