#include "textflag.h"

// NEON floating point instructions that Go's assembler does not know, encoded by hand for the
// .4S arrangement: d, n and m are vector register numbers.
#define FCMGT(m, n, d) WORD $(0x6EA0E400 | (m)<<16 | (n)<<5 | (d)) // d = n > m (all ones), 0 for NaN
#define FMIN(m, n, d) WORD $(0x4EA0F400 | (m)<<16 | (n)<<5 | (d))  // NaN if either is NaN
#define FMAX(m, n, d) WORD $(0x4E20F400 | (m)<<16 | (n)<<5 | (d))
#define FDIV(m, n, d) WORD $(0x6E20FC00 | (m)<<16 | (n)<<5 | (d))  // d = n / m
#define FMUL(m, n, d) WORD $(0x6E20DC00 | (m)<<16 | (n)<<5 | (d))
#define FADD(m, n, d) WORD $(0x4E20D400 | (m)<<16 | (n)<<5 | (d))
#define FSUB(m, n, d) WORD $(0x4EA0D400 | (m)<<16 | (n)<<5 | (d))  // d = n - m
#define FABS(n, d) WORD $(0x4EA0F800 | (n)<<5 | (d))
#define FRINTN(n, d) WORD $(0x4E218800 | (n)<<5 | (d))             // to nearest even
#define FCVTZS(n, d) WORD $(0x4EA1B800 | (n)<<5 | (d))
#define AND16B(m, n, d) WORD $(0x4E201C00 | (m)<<16 | (n)<<5 | (d)) // .16B
#define FMINV(n, d) WORD $(0x6EB0F800 | (n)<<5 | (d))              // across the lanes into Sd
#define FMAXV(n, d) WORD $(0x6E30F800 | (n)<<5 | (d))
#define FCMEQ(m, n, d) WORD $(0x4E20E400 | (m)<<16 | (n)<<5 | (d)) // all ones where n == m: not for a NaN
#define FCVTN(n, d) WORD $(0x0E216800 | (n)<<5 | (d))               // .4S to the low .4H, to nearest even
#define FCVTN2(n, d) WORD $(0x4E216800 | (n)<<5 | (d))              // .4S to the high .4H
#define FCVTL(n, d) WORD $(0x0E217800 | (n)<<5 | (d))               // the low .4H to .4S
#define FCVTL2(n, d) WORD $(0x4E217800 | (n)<<5 | (d))              // the high .4H to .4S
#define UMINVS(n, d) WORD $(0x6EB1A800 | (n)<<5 | (d))              // .4S, across the lanes
#define CMHIH(m, n, d) WORD $(0x6E603400 | (m)<<16 | (n)<<5 | (d))  // .8H: all ones where n > m, unsigned
#define UMAXVH(n, d) WORD $(0x6E70A800 | (n)<<5 | (d))              // .8H, across the lanes

// One threshold, broadcast in register m, against the 16 values in V0-V3: the counts in V4-V7
// grow by one where the value is above it.
#define COUNT(m) \
	FCMGT(m, 0, 8) \
	FCMGT(m, 1, 9) \
	FCMGT(m, 2, 10) \
	FCMGT(m, 3, 11) \
	VSUB V8.S4, V4.S4, V4.S4 \
	VSUB V9.S4, V5.S4, V5.S4 \
	VSUB V10.S4, V6.S4, V6.S4 \
	VSUB V11.S4, V7.S4, V7.S4

// func assign15NEON(dst *uint8, vals *float32, n int, mids *float32)
// n is a positive multiple of 16, mids holds 15 thresholds: each value's count of thresholds
// below it.
TEXT ·assign15NEON(SB), NOSPLIT, $0-32
	MOVD dst+0(FP), R0
	MOVD vals+8(FP), R1
	MOVD n+16(FP), R2
	MOVD mids+24(FP), R3
	VLD1R.P 4(R3), [V16.S4]
	VLD1R.P 4(R3), [V17.S4]
	VLD1R.P 4(R3), [V18.S4]
	VLD1R.P 4(R3), [V19.S4]
	VLD1R.P 4(R3), [V20.S4]
	VLD1R.P 4(R3), [V21.S4]
	VLD1R.P 4(R3), [V22.S4]
	VLD1R.P 4(R3), [V23.S4]
	VLD1R.P 4(R3), [V24.S4]
	VLD1R.P 4(R3), [V25.S4]
	VLD1R.P 4(R3), [V26.S4]
	VLD1R.P 4(R3), [V27.S4]
	VLD1R.P 4(R3), [V28.S4]
	VLD1R.P 4(R3), [V29.S4]
	VLD1R.P 4(R3), [V30.S4]

a15loop:
	VLD1.P 64(R1), [V0.S4, V1.S4, V2.S4, V3.S4]
	VEOR   V4.B16, V4.B16, V4.B16
	VEOR   V5.B16, V5.B16, V5.B16
	VEOR   V6.B16, V6.B16, V6.B16
	VEOR   V7.B16, V7.B16, V7.B16
	COUNT(16)
	COUNT(17)
	COUNT(18)
	COUNT(19)
	COUNT(20)
	COUNT(21)
	COUNT(22)
	COUNT(23)
	COUNT(24)
	COUNT(25)
	COUNT(26)
	COUNT(27)
	COUNT(28)
	COUNT(29)
	COUNT(30)
	VUZP1  V5.H8, V4.H8, V4.H8 // the low halves of the counts
	VUZP1  V7.H8, V6.H8, V6.H8
	VUZP1  V6.B16, V4.B16, V4.B16 // and their low bytes
	VST1.P [V4.B16], 16(R0)
	SUBS   $16, R2, R2
	BNE    a15loop
	RET

// One level of the search for value Fv with node index Rj, the level starting at R9:
// j becomes 2j+1 where the value is above the node's threshold (GT is false for a NaN), 2j
// otherwise.
#define NODE(v, j, t) \
	FMOVS (R9)(j<<2), t \
	FCMPS t, v \
	CSET  GT, R16 \
	ADD   j<<1, R16, j

// func assignTreeARM64(dst *uint8, vals *float32, n int, tree *float32, levels int)
// n is a positive multiple of 8, tree the thresholds in breadth first order, levels from 1 to 8.
// Eight values go down the tree side by side.
TEXT ·assignTreeARM64(SB), NOSPLIT, $0-40
	MOVD dst+0(FP), R0
	MOVD vals+8(FP), R1
	MOVD n+16(FP), R2
	MOVD tree+24(FP), R3
	MOVD levels+32(FP), R4

atloop:
	FMOVS 0(R1), F0
	FMOVS 4(R1), F1
	FMOVS 8(R1), F2
	FMOVS 12(R1), F3
	FMOVS 16(R1), F4
	FMOVS 20(R1), F5
	FMOVS 24(R1), F6
	FMOVS 28(R1), F7
	MOVD  ZR, R5
	MOVD  ZR, R6
	MOVD  ZR, R7
	MOVD  ZR, R8
	MOVD  ZR, R12
	MOVD  ZR, R13
	MOVD  ZR, R14
	MOVD  ZR, R15
	MOVD  R3, R9  // the level's first node
	MOVD  $1, R10 // and its number of nodes
	MOVD  R4, R11

atlevel:
	NODE(F0, R5, F8)
	NODE(F1, R6, F9)
	NODE(F2, R7, F10)
	NODE(F3, R8, F11)
	NODE(F4, R12, F12)
	NODE(F5, R13, F13)
	NODE(F6, R14, F14)
	NODE(F7, R15, F15)
	ADD  R10<<2, R9, R9
	LSL  $1, R10, R10
	SUBS $1, R11, R11
	BNE  atlevel

	MOVB R5, 0(R0)
	MOVB R6, 1(R0)
	MOVB R7, 2(R0)
	MOVB R8, 3(R0)
	MOVB R12, 4(R0)
	MOVB R13, 5(R0)
	MOVB R14, 6(R0)
	MOVB R15, 7(R0)
	ADD  $32, R1, R1
	ADD  $8, R0, R0
	SUBS $8, R2, R2
	BNE  atloop
	RET

// func minMaxNEON(vals *float32, n int) (lo, hi float32)
// n is a positive multiple of 16; a NaN in vals makes both NaN.
TEXT ·minMaxNEON(SB), NOSPLIT, $0-24
	MOVD vals+0(FP), R0
	MOVD n+8(FP), R1
	MOVW $0x7f7fffff, R2
	VDUP R2, V4.S4
	VDUP R2, V5.S4
	MOVW $0xff7fffff, R2
	VDUP R2, V6.S4
	VDUP R2, V7.S4

mmloop:
	VLD1.P 64(R0), [V0.S4, V1.S4, V2.S4, V3.S4]
	FMIN(0, 4, 4)
	FMIN(1, 5, 5)
	FMIN(2, 4, 4)
	FMIN(3, 5, 5)
	FMAX(0, 6, 6)
	FMAX(1, 7, 7)
	FMAX(2, 6, 6)
	FMAX(3, 7, 7)
	SUBS $16, R1, R1
	BNE  mmloop

	FMIN(5, 4, 4)
	FMAX(7, 6, 6)
	FMINV(4, 4)
	FMAXV(6, 6)
	FMOVS F4, lo+16(FP)
	FMOVS F6, hi+20(FP)
	RET

// func absMaxNEON(vals *float32, n int) float32
// n is a positive multiple of 16; a NaN in vals makes the result NaN.
TEXT ·absMaxNEON(SB), NOSPLIT, $0-20
	MOVD vals+0(FP), R0
	MOVD n+8(FP), R1
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16

amloop:
	VLD1.P 64(R0), [V0.S4, V1.S4, V2.S4, V3.S4]
	FABS(0, 0)
	FABS(1, 1)
	FABS(2, 2)
	FABS(3, 3)
	FMAX(0, 4, 4)
	FMAX(1, 5, 5)
	FMAX(2, 4, 4)
	FMAX(3, 5, 5)
	SUBS $16, R1, R1
	BNE  amloop

	FMAX(5, 4, 4)
	FMAXV(4, 4)
	FMOVS F4, ret+16(FP)
	RET

// The error of one register of four scales (Vs, their nonzero masks Vz, sums Ve) for the value
// broadcast in V0, through Vt.
#define CLIP(s, z, e, t) \
	FDIV(s, 0, t) \
	FRINTN(t, t) \
	FMAX(22, t, t) \
	FMIN(23, t, t) \
	AND16B(z, t, t) \
	FMUL(s, t, t) \
	FSUB(t, 0, t) \
	FMUL(t, t, t) \
	FADD(t, e, e)

// func clipErrorsNEON(errs *[16]float32, vals *float32, n int, scales *[16]float32, nz *[16]uint32, qmin, qmax float32)
// n is positive; the first 12 scales. A lane per scale: each lane sums its own errors in the
// order of vals.
TEXT ·clipErrorsNEON(SB), NOSPLIT, $0-48
	MOVD   errs+0(FP), R0
	MOVD   vals+8(FP), R1
	MOVD   n+16(FP), R2
	MOVD   scales+24(FP), R3
	MOVD   nz+32(FP), R4
	VLD1   (R3), [V16.S4, V17.S4, V18.S4]
	VLD1   (R4), [V19.S4, V20.S4, V21.S4]
	MOVD   $qmin+40(FP), R5
	VLD1R  (R5), [V22.S4]
	MOVD   $qmax+44(FP), R5
	VLD1R  (R5), [V23.S4]
	VEOR   V24.B16, V24.B16, V24.B16
	VEOR   V25.B16, V25.B16, V25.B16
	VEOR   V26.B16, V26.B16, V26.B16

celoop:
	VLD1R.P 4(R1), [V0.S4]
	CLIP(16, 19, 24, 1)
	CLIP(17, 20, 25, 2)
	CLIP(18, 21, 26, 3)
	SUBS $1, R2, R2
	BNE  celoop

	VST1 [V24.S4, V25.S4, V26.S4], (R0)
	RET

#define QUANT(r) \
	FDIV(16, r, r) \
	FRINTN(r, r) \
	FMAX(17, r, r) \
	FMIN(18, r, r) \
	FCVTZS(r, r)

// func quantizeSymNEON(dst *int8, vals *float32, n int, s, qmin, qmax float32)
// n is a positive multiple of 16, s is not 0.
TEXT ·quantizeSymNEON(SB), NOSPLIT, $0-36
	MOVD  dst+0(FP), R0
	MOVD  vals+8(FP), R1
	MOVD  n+16(FP), R2
	MOVD  $s+24(FP), R5
	VLD1R (R5), [V16.S4]
	MOVD  $qmin+28(FP), R5
	VLD1R (R5), [V17.S4]
	MOVD  $qmax+32(FP), R5
	VLD1R (R5), [V18.S4]

qsloop:
	VLD1.P 64(R1), [V0.S4, V1.S4, V2.S4, V3.S4]
	QUANT(0)
	QUANT(1)
	QUANT(2)
	QUANT(3)
	VUZP1  V1.H8, V0.H8, V0.H8
	VUZP1  V3.H8, V2.H8, V2.H8
	VUZP1  V2.B16, V0.B16, V0.B16
	VST1.P [V0.B16], 16(R0)
	SUBS   $16, R2, R2
	BNE    qsloop
	RET

// func bf16NEON(dst *float32, src *byte, n int)
// n is a positive multiple of 16: each half word becomes the high half of a word.
TEXT ·bf16NEON(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD n+16(FP), R2
	VEOR V31.B16, V31.B16, V31.B16

bfloop:
	VLD1.P 32(R1), [V0.H8, V1.H8]
	VZIP1  V0.H8, V31.H8, V2.H8
	VZIP2  V0.H8, V31.H8, V3.H8
	VZIP1  V1.H8, V31.H8, V4.H8
	VZIP2  V1.H8, V31.H8, V5.H8
	VST1.P [V2.S4, V3.S4, V4.S4, V5.S4], 64(R0)
	SUBS   $16, R2, R2
	BNE    bfloop
	RET

// func pack4NEON(dst *byte, idx *uint8, n int)
// n is a positive multiple of 32: LD2 splits even and odd entries, and each byte is the even
// one's low nibble below the odd one's.
TEXT ·pack4NEON(SB), NOSPLIT, $0-24
	MOVD  dst+0(FP), R0
	MOVD  idx+8(FP), R1
	MOVD  n+16(FP), R2
	VMOVI $0x0f, V31.B16

p4loop:
	VLD2.P 32(R1), [V0.B16, V1.B16]
	VAND   V31.B16, V0.B16, V0.B16
	VSHL   $4, V1.B16, V1.B16
	VORR   V1.B16, V0.B16, V0.B16
	VST1.P [V0.B16], 16(R0)
	SUBS   $32, R2, R2
	BNE    p4loop
	RET

// func pack6NEON(dst *byte, idx *uint8, n int)
// n is a positive multiple of 64: LD4 splits the entries a b c d of every four, which take the
// 24 bits of three bytes a|b<<6, b>>2|c<<4, c>>4|d<<2, and ST3 interleaves those back.
TEXT ·pack6NEON(SB), NOSPLIT, $0-24
	MOVD  dst+0(FP), R0
	MOVD  idx+8(FP), R1
	MOVD  n+16(FP), R2
	VMOVI $0x3f, V31.B16

p6loop:
	VLD4.P 64(R1), [V0.B16, V1.B16, V2.B16, V3.B16]
	VAND   V31.B16, V0.B16, V0.B16
	VAND   V31.B16, V1.B16, V1.B16
	VAND   V31.B16, V2.B16, V2.B16
	VSHL   $6, V1.B16, V4.B16
	VORR   V4.B16, V0.B16, V4.B16
	VUSHR  $2, V1.B16, V5.B16
	VSHL   $4, V2.B16, V6.B16
	VORR   V6.B16, V5.B16, V5.B16
	VUSHR  $4, V2.B16, V6.B16
	VSHL   $2, V3.B16, V7.B16
	VORR   V7.B16, V6.B16, V6.B16
	VST3.P [V4.B16, V5.B16, V6.B16], 48(R0)
	SUBS   $64, R2, R2
	BNE    p6loop
	RET

// func f32ToF16NEON(dst *byte, src *float32, n int) (nan bool)
// n is a positive multiple of 16: FCVTN rounds to nearest even (the FPCR's mode, which Go leaves
// at its default). V20 keeps the lanes that were equal to themselves so far: no NaN.
TEXT ·f32ToF16NEON(SB), NOSPLIT, $0-25
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD n+16(FP), R2
	VMOVI $0xff, V20.B16

hloop:
	VLD1.P 64(R1), [V0.S4, V1.S4, V2.S4, V3.S4]
	FCMEQ(0, 0, 8)
	FCMEQ(1, 1, 9)
	FCMEQ(2, 2, 10)
	FCMEQ(3, 3, 11)
	VAND   V8.B16, V20.B16, V20.B16
	VAND   V9.B16, V20.B16, V20.B16
	VAND   V10.B16, V20.B16, V20.B16
	VAND   V11.B16, V20.B16, V20.B16
	FCVTN(0, 4)
	FCVTN2(1, 4)
	FCVTN(2, 5)
	FCVTN2(3, 5)
	VST1.P [V4.H8, V5.H8], 32(R0)
	SUBS   $16, R2, R2
	BNE    hloop

	UMINVS(20, 20)
	VMOV  V20.S[0], R3
	CMPW  $0, R3
	CSET  EQ, R3
	MOVB  R3, nan+24(FP)
	RET

// func f16ToF32NEON(dst *float32, src *byte, n int) (nan bool)
// n is a positive multiple of 16: FCVTL is exact. V20 collects the halves above 0x7c00 once their
// sign is cleared: the NaNs.
TEXT ·f16ToF32NEON(SB), NOSPLIT, $0-25
	MOVD  dst+0(FP), R0
	MOVD  src+8(FP), R1
	MOVD  n+16(FP), R2
	MOVW  $0x7fff, R3
	VDUP  R3, V30.H8
	MOVW  $0x7c00, R3
	VDUP  R3, V29.H8
	VEOR  V20.B16, V20.B16, V20.B16

floop:
	VLD1.P 32(R1), [V0.H8, V1.H8]
	VAND   V30.B16, V0.B16, V6.B16
	VAND   V30.B16, V1.B16, V7.B16
	CMHIH(29, 6, 6)
	CMHIH(29, 7, 7)
	VORR   V6.B16, V20.B16, V20.B16
	VORR   V7.B16, V20.B16, V20.B16
	FCVTL(0, 2)
	FCVTL2(0, 3)
	FCVTL(1, 4)
	FCVTL2(1, 5)
	VST1.P [V2.S4, V3.S4, V4.S4, V5.S4], 64(R0)
	SUBS   $16, R2, R2
	BNE    floop

	UMAXVH(20, 20)
	VMOV  V20.H[0], R3
	CMPW  $0, R3
	CSET  NE, R3
	MOVB  R3, nan+24(FP)
	RET
