#include "textflag.h"

// FMLAE(d, n, m, i) is FMLA Vd.4S, Vn.4S, Vm.S[i] (Vd += Vn times lane i of Vm), which Go's
// assembler has no form for.
#define FMLAE(d, n, m, i) WORD $(0x4F801000 | (((i)&1)<<21) | ((m)<<16) | (((i)>>1)<<11) | ((n)<<5) | (d))

// func dw3NEON(dst, r0, r1, r2, w *float32, bias float32, relu, n int)
// dw3AVX2 on NEON: 8 columns (two vectors) a pass, a row's taps in accumulators of their own;
// n a multiple of 8.
TEXT ·dw3NEON(SB), NOSPLIT, $0-64
	MOVD  dst+0(FP), R0
	MOVD  r0+8(FP), R1
	MOVD  r1+16(FP), R2
	MOVD  r2+24(FP), R3
	MOVD  w+32(FP), R4
	FMOVS bias+40(FP), F3
	MOVD  relu+48(FP), R5
	MOVD  n+56(FP), R6
	VLD1  (R4), [V0.S4, V1.S4] // w[0:8]
	FMOVS 32(R4), F2           // w[8]
	VDUP  V3.S[0], V3.S4
	VEOR  V4.B16, V4.B16, V4.B16
loopd:
	VLD1  (R1), [V16.S4, V17.S4]
	ADD   $4, R1, R9
	VLD1  (R9), [V18.S4, V19.S4]
	ADD   $8, R1, R9
	VLD1  (R9), [V20.S4, V21.S4]
	ADD   $32, R1
	VORR  V3.B16, V3.B16, V5.B16
	VORR  V3.B16, V3.B16, V6.B16
	FMLAE(5, 16, 0, 0)
	FMLAE(6, 17, 0, 0)
	FMLAE(5, 18, 0, 1)
	FMLAE(6, 19, 0, 1)
	FMLAE(5, 20, 0, 2)
	FMLAE(6, 21, 0, 2)
	VLD1  (R2), [V16.S4, V17.S4]
	ADD   $4, R2, R9
	VLD1  (R9), [V18.S4, V19.S4]
	ADD   $8, R2, R9
	VLD1  (R9), [V20.S4, V21.S4]
	ADD   $32, R2
	VEOR  V7.B16, V7.B16, V7.B16
	VEOR  V8.B16, V8.B16, V8.B16
	FMLAE(7, 16, 0, 3)
	FMLAE(8, 17, 0, 3)
	FMLAE(7, 18, 1, 0)
	FMLAE(8, 19, 1, 0)
	FMLAE(7, 20, 1, 1)
	FMLAE(8, 21, 1, 1)
	VLD1  (R3), [V16.S4, V17.S4]
	ADD   $4, R3, R9
	VLD1  (R9), [V18.S4, V19.S4]
	ADD   $8, R3, R9
	VLD1  (R9), [V20.S4, V21.S4]
	ADD   $32, R3
	VEOR  V9.B16, V9.B16, V9.B16
	VEOR  V10.B16, V10.B16, V10.B16
	FMLAE(9, 16, 1, 2)
	FMLAE(10, 17, 1, 2)
	FMLAE(9, 18, 1, 3)
	FMLAE(10, 19, 1, 3)
	FMLAE(9, 20, 2, 0)
	FMLAE(10, 21, 2, 0)
	VFADD V7.S4, V5.S4, V5.S4
	VFADD V8.S4, V6.S4, V6.S4
	VFADD V9.S4, V5.S4, V5.S4
	VFADD V10.S4, V6.S4, V6.S4
	CBZ   R5, stored
	VFMAX V4.S4, V5.S4, V5.S4
	VFMAX V4.S4, V6.S4, V6.S4
stored:
	VST1.P [V5.S4, V6.S4], 32(R0)
	SUBS  $8, R6, R6
	BNE   loopd
	RET

// func maxpool2NEON(dst, a, b *float32, n int)
// dst[i] = max(a[2i], a[2i+1], b[2i], b[2i+1]); n a multiple of 8. VLD2 splits the even and
// the odd columns.
TEXT ·maxpool2NEON(SB), NOSPLIT, $0-32
	MOVD dst+0(FP), R0
	MOVD a+8(FP), R1
	MOVD b+16(FP), R2
	MOVD n+24(FP), R3
loopm:
	VLD2.P 32(R1), [V0.S4, V1.S4]
	VLD2.P 32(R1), [V2.S4, V3.S4]
	VLD2.P 32(R2), [V4.S4, V5.S4]
	VLD2.P 32(R2), [V6.S4, V7.S4]
	VFMAX  V1.S4, V0.S4, V0.S4
	VFMAX  V5.S4, V4.S4, V4.S4
	VFMAX  V4.S4, V0.S4, V16.S4
	VFMAX  V3.S4, V2.S4, V2.S4
	VFMAX  V7.S4, V6.S4, V6.S4
	VFMAX  V6.S4, V2.S4, V17.S4
	VST1.P [V16.S4, V17.S4], 32(R0)
	SUBS   $8, R3, R3
	BNE    loopm
	RET

// func reluNEON(dst, src *float32, n int)
// n a multiple of 16.
TEXT ·reluNEON(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD n+16(FP), R2
	VEOR V31.B16, V31.B16, V31.B16
loopr:
	VLD1.P 64(R1), [V0.S4, V1.S4, V2.S4, V3.S4]
	VFMAX  V31.S4, V0.S4, V0.S4
	VFMAX  V31.S4, V1.S4, V1.S4
	VFMAX  V31.S4, V2.S4, V2.S4
	VFMAX  V31.S4, V3.S4, V3.S4
	VST1.P [V0.S4, V1.S4, V2.S4, V3.S4], 64(R0)
	SUBS   $16, R2, R2
	BNE    loopr
	RET

// func axpyNEON(dst *float32, a float32, x *float32, n int)
// dst[i] += a x[i]; n a multiple of 16.
TEXT ·axpyNEON(SB), NOSPLIT, $0-32
	MOVD  dst+0(FP), R0
	FMOVS a+8(FP), F31
	MOVD  x+16(FP), R1
	MOVD  n+24(FP), R2
loopa:
	VLD1   (R0), [V0.S4, V1.S4, V2.S4, V3.S4]
	VLD1.P 64(R1), [V4.S4, V5.S4, V6.S4, V7.S4]
	FMLAE(0, 4, 31, 0)
	FMLAE(1, 5, 31, 0)
	FMLAE(2, 6, 31, 0)
	FMLAE(3, 7, 31, 0)
	VST1.P [V0.S4, V1.S4, V2.S4, V3.S4], 64(R0)
	SUBS   $16, R2, R2
	BNE    loopa
	RET

// func wrowsNEON(dst *float32, rows **float32, w *float32, k int, bias float32, relu, n int)
// wrowsAVX2 on NEON: 16 columns (four vectors) a pass, each with two accumulators (the even and
// the odd rows); n a multiple of 16.
TEXT ·wrowsNEON(SB), NOSPLIT, $0-56
	MOVD  dst+0(FP), R0
	MOVD  rows+8(FP), R1
	MOVD  w+16(FP), R2
	MOVD  k+24(FP), R3
	FMOVS bias+32(FP), F30
	MOVD  relu+40(FP), R4
	MOVD  n+48(FP), R5
	VDUP  V30.S[0], V30.S4
	VEOR  V31.B16, V31.B16, V31.B16
	MOVD  ZR, R6 // the byte offset of the outputs
loopw:
	VORR  V30.B16, V30.B16, V0.B16
	VORR  V30.B16, V30.B16, V1.B16
	VORR  V30.B16, V30.B16, V2.B16
	VORR  V30.B16, V30.B16, V3.B16
	VEOR  V4.B16, V4.B16, V4.B16
	VEOR  V5.B16, V5.B16, V5.B16
	VEOR  V6.B16, V6.B16, V6.B16
	VEOR  V7.B16, V7.B16, V7.B16
	MOVD  R1, R7
	MOVD  R2, R8
	MOVD  R3, R9
pairw:
	CMP   $2, R9
	BLT   lastw
	MOVD.P 8(R7), R10
	MOVD.P 8(R7), R11
	ADD   R6, R10
	ADD   R6, R11
	VLD1  (R10), [V16.S4, V17.S4, V18.S4, V19.S4]
	VLD1  (R11), [V20.S4, V21.S4, V22.S4, V23.S4]
	FMOVS.P 4(R8), F24
	FMOVS.P 4(R8), F25
	FMLAE(0, 16, 24, 0)
	FMLAE(1, 17, 24, 0)
	FMLAE(2, 18, 24, 0)
	FMLAE(3, 19, 24, 0)
	FMLAE(4, 20, 25, 0)
	FMLAE(5, 21, 25, 0)
	FMLAE(6, 22, 25, 0)
	FMLAE(7, 23, 25, 0)
	SUB   $2, R9
	B     pairw
lastw:
	CBZ   R9, sumw
	MOVD  (R7), R10
	ADD   R6, R10
	VLD1  (R10), [V16.S4, V17.S4, V18.S4, V19.S4]
	FMOVS (R8), F24
	FMLAE(0, 16, 24, 0)
	FMLAE(1, 17, 24, 0)
	FMLAE(2, 18, 24, 0)
	FMLAE(3, 19, 24, 0)
sumw:
	VFADD V4.S4, V0.S4, V0.S4
	VFADD V5.S4, V1.S4, V1.S4
	VFADD V6.S4, V2.S4, V2.S4
	VFADD V7.S4, V3.S4, V3.S4
	CBZ   R4, storew
	VFMAX V31.S4, V0.S4, V0.S4
	VFMAX V31.S4, V1.S4, V1.S4
	VFMAX V31.S4, V2.S4, V2.S4
	VFMAX V31.S4, V3.S4, V3.S4
storew:
	ADD   R6, R0, R10
	VST1  [V0.S4, V1.S4, V2.S4, V3.S4], (R10)
	ADD   $64, R6
	SUBS  $16, R5, R5
	BNE   loopw
	RET

// func wrows4NEON(dst **float32, rows **float32, w *float32, k int, bias *float32, relu, n, ws int)
// wrows4AVX2 on NEON: the four outputs of 16 columns in 16 accumulators, a row's four vectors
// read once for 16 FMAs, the tap's four weights one vector (lanes); n a multiple of 16, ws the
// bytes from a tap's weights to the next's.
TEXT ·wrows4NEON(SB), NOSPLIT, $0-64
	MOVD dst+0(FP), R0
	MOVD rows+8(FP), R1
	MOVD w+16(FP), R2
	MOVD k+24(FP), R3
	MOVD bias+32(FP), R4
	MOVD relu+40(FP), R5
	MOVD n+48(FP), R6
	MOVD ws+56(FP), R7
	VLD1 (R4), [V28.S4]
	VEOR V31.B16, V31.B16, V31.B16
	MOVD 0(R0), R12
	MOVD 8(R0), R13
	MOVD 16(R0), R14
	MOVD 24(R0), R15
	MOVD ZR, R8
loop4:
	VDUP V28.S[0], V0.S4
	VDUP V28.S[0], V1.S4
	VDUP V28.S[0], V2.S4
	VDUP V28.S[0], V3.S4
	VDUP V28.S[1], V4.S4
	VDUP V28.S[1], V5.S4
	VDUP V28.S[1], V6.S4
	VDUP V28.S[1], V7.S4
	VDUP V28.S[2], V8.S4
	VDUP V28.S[2], V9.S4
	VDUP V28.S[2], V10.S4
	VDUP V28.S[2], V11.S4
	VDUP V28.S[3], V12.S4
	VDUP V28.S[3], V13.S4
	VDUP V28.S[3], V14.S4
	VDUP V28.S[3], V15.S4
	MOVD R1, R9
	MOVD R2, R10
	MOVD R3, R11
tap4:
	MOVD.P 8(R9), R16
	ADD  R8, R16
	VLD1 (R16), [V16.S4, V17.S4, V18.S4, V19.S4]
	VLD1 (R10), [V20.S4]
	ADD  R7, R10
	FMLAE(0, 16, 20, 0)
	FMLAE(1, 17, 20, 0)
	FMLAE(2, 18, 20, 0)
	FMLAE(3, 19, 20, 0)
	FMLAE(4, 16, 20, 1)
	FMLAE(5, 17, 20, 1)
	FMLAE(6, 18, 20, 1)
	FMLAE(7, 19, 20, 1)
	FMLAE(8, 16, 20, 2)
	FMLAE(9, 17, 20, 2)
	FMLAE(10, 18, 20, 2)
	FMLAE(11, 19, 20, 2)
	FMLAE(12, 16, 20, 3)
	FMLAE(13, 17, 20, 3)
	FMLAE(14, 18, 20, 3)
	FMLAE(15, 19, 20, 3)
	SUBS $1, R11, R11
	BNE  tap4
	CBZ  R5, store4
	VFMAX V31.S4, V0.S4, V0.S4
	VFMAX V31.S4, V1.S4, V1.S4
	VFMAX V31.S4, V2.S4, V2.S4
	VFMAX V31.S4, V3.S4, V3.S4
	VFMAX V31.S4, V4.S4, V4.S4
	VFMAX V31.S4, V5.S4, V5.S4
	VFMAX V31.S4, V6.S4, V6.S4
	VFMAX V31.S4, V7.S4, V7.S4
	VFMAX V31.S4, V8.S4, V8.S4
	VFMAX V31.S4, V9.S4, V9.S4
	VFMAX V31.S4, V10.S4, V10.S4
	VFMAX V31.S4, V11.S4, V11.S4
	VFMAX V31.S4, V12.S4, V12.S4
	VFMAX V31.S4, V13.S4, V13.S4
	VFMAX V31.S4, V14.S4, V14.S4
	VFMAX V31.S4, V15.S4, V15.S4
store4:
	ADD  R8, R12, R16
	VST1 [V0.S4, V1.S4, V2.S4, V3.S4], (R16)
	ADD  R8, R13, R16
	VST1 [V4.S4, V5.S4, V6.S4, V7.S4], (R16)
	ADD  R8, R14, R16
	VST1 [V8.S4, V9.S4, V10.S4, V11.S4], (R16)
	ADD  R8, R15, R16
	VST1 [V12.S4, V13.S4, V14.S4, V15.S4], (R16)
	ADD  $64, R8
	SUBS $16, R6, R6
	BNE  loop4
	RET

// func wrows8NEON(dst **float32, rows **float32, w *float32, k int, bias *float32, relu, n int)
// For o < 8: dst[o][i] = bias[o] + sum over j < k of w[8j+o] rows[j][i], then max with 0 when
// relu; n a multiple of 8. The eight outputs of 8 columns in 16 accumulators: a row's two
// vectors and the tap's eight weights (two vectors, lanes) read for 16 FMAs.
TEXT ·wrows8NEON(SB), NOSPLIT, $0-56
	MOVD dst+0(FP), R0
	MOVD rows+8(FP), R1
	MOVD w+16(FP), R2
	MOVD k+24(FP), R3
	MOVD bias+32(FP), R4
	MOVD relu+40(FP), R5
	MOVD n+48(FP), R6
	VLD1 (R4), [V28.S4, V29.S4]
	VEOR V31.B16, V31.B16, V31.B16
	MOVD ZR, R8
loop8:
	VDUP V28.S[0], V0.S4
	VDUP V28.S[0], V1.S4
	VDUP V28.S[1], V2.S4
	VDUP V28.S[1], V3.S4
	VDUP V28.S[2], V4.S4
	VDUP V28.S[2], V5.S4
	VDUP V28.S[3], V6.S4
	VDUP V28.S[3], V7.S4
	VDUP V29.S[0], V8.S4
	VDUP V29.S[0], V9.S4
	VDUP V29.S[1], V10.S4
	VDUP V29.S[1], V11.S4
	VDUP V29.S[2], V12.S4
	VDUP V29.S[2], V13.S4
	VDUP V29.S[3], V14.S4
	VDUP V29.S[3], V15.S4
	MOVD R1, R9
	MOVD R2, R10
	MOVD R3, R11
tap8:
	MOVD.P 8(R9), R16
	ADD  R8, R16
	VLD1 (R16), [V16.S4, V17.S4]
	VLD1.P 32(R10), [V20.S4, V21.S4]
	FMLAE(0, 16, 20, 0)
	FMLAE(1, 17, 20, 0)
	FMLAE(2, 16, 20, 1)
	FMLAE(3, 17, 20, 1)
	FMLAE(4, 16, 20, 2)
	FMLAE(5, 17, 20, 2)
	FMLAE(6, 16, 20, 3)
	FMLAE(7, 17, 20, 3)
	FMLAE(8, 16, 21, 0)
	FMLAE(9, 17, 21, 0)
	FMLAE(10, 16, 21, 1)
	FMLAE(11, 17, 21, 1)
	FMLAE(12, 16, 21, 2)
	FMLAE(13, 17, 21, 2)
	FMLAE(14, 16, 21, 3)
	FMLAE(15, 17, 21, 3)
	SUBS $1, R11, R11
	BNE  tap8
	CBZ  R5, store8
	VFMAX V31.S4, V0.S4, V0.S4
	VFMAX V31.S4, V1.S4, V1.S4
	VFMAX V31.S4, V2.S4, V2.S4
	VFMAX V31.S4, V3.S4, V3.S4
	VFMAX V31.S4, V4.S4, V4.S4
	VFMAX V31.S4, V5.S4, V5.S4
	VFMAX V31.S4, V6.S4, V6.S4
	VFMAX V31.S4, V7.S4, V7.S4
	VFMAX V31.S4, V8.S4, V8.S4
	VFMAX V31.S4, V9.S4, V9.S4
	VFMAX V31.S4, V10.S4, V10.S4
	VFMAX V31.S4, V11.S4, V11.S4
	VFMAX V31.S4, V12.S4, V12.S4
	VFMAX V31.S4, V13.S4, V13.S4
	VFMAX V31.S4, V14.S4, V14.S4
	VFMAX V31.S4, V15.S4, V15.S4
store8:
	MOVD 0(R0), R16
	ADD  R8, R16
	VST1 [V0.S4, V1.S4], (R16)
	MOVD 8(R0), R16
	ADD  R8, R16
	VST1 [V2.S4, V3.S4], (R16)
	MOVD 16(R0), R16
	ADD  R8, R16
	VST1 [V4.S4, V5.S4], (R16)
	MOVD 24(R0), R16
	ADD  R8, R16
	VST1 [V6.S4, V7.S4], (R16)
	MOVD 32(R0), R16
	ADD  R8, R16
	VST1 [V8.S4, V9.S4], (R16)
	MOVD 40(R0), R16
	ADD  R8, R16
	VST1 [V10.S4, V11.S4], (R16)
	MOVD 48(R0), R16
	ADD  R8, R16
	VST1 [V12.S4, V13.S4], (R16)
	MOVD 56(R0), R16
	ADD  R8, R16
	VST1 [V14.S4, V15.S4], (R16)
	ADD  $32, R8
	SUBS $8, R6, R6
	BNE  loop8
	RET

// func deint2NEON(even, odd, src *float32, n int)
// even[i] = src[2i], odd[i] = src[2i+1]; n a multiple of 8.
TEXT ·deint2NEON(SB), NOSPLIT, $0-32
	MOVD even+0(FP), R0
	MOVD odd+8(FP), R1
	MOVD src+16(FP), R2
	MOVD n+24(FP), R3
loopi:
	VLD1.P 64(R2), [V0.S4, V1.S4, V2.S4, V3.S4]
	VUZP1  V1.S4, V0.S4, V4.S4
	VUZP1  V3.S4, V2.S4, V5.S4
	VUZP2  V1.S4, V0.S4, V6.S4
	VUZP2  V3.S4, V2.S4, V7.S4
	VST1.P [V4.S4, V5.S4], 32(R0)
	VST1.P [V6.S4, V7.S4], 32(R1)
	SUBS   $8, R3, R3
	BNE    loopi
	RET
