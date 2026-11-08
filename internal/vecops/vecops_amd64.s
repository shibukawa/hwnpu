#include "textflag.h"

// VPERMD indices that undo the lane interleaving of VPACKSSDW and VPACKUSWB/VPACKSSWB on four
// vectors of eight int32: the bytes come out in dwords a0-3 b0-3 c0-3 d0-3 a4-7 b4-7 c4-7 d4-7.
DATA packperm<>+0(SB)/4, $0
DATA packperm<>+4(SB)/4, $4
DATA packperm<>+8(SB)/4, $1
DATA packperm<>+12(SB)/4, $5
DATA packperm<>+16(SB)/4, $2
DATA packperm<>+20(SB)/4, $6
DATA packperm<>+24(SB)/4, $3
DATA packperm<>+28(SB)/4, $7
GLOBL packperm<>(SB), RODATA|NOPTR, $32

DATA maxf32<>+0(SB)/4, $0x7f7fffff
GLOBL maxf32<>(SB), RODATA|NOPTR, $4
DATA negmaxf32<>+0(SB)/4, $0xff7fffff
GLOBL negmaxf32<>(SB), RODATA|NOPTR, $4
DATA absmask<>+0(SB)/4, $0x7fffffff
GLOBL absmask<>(SB), RODATA|NOPTR, $4

// func hasAVX2() bool
TEXT ·hasAVX2(SB), NOSPLIT, $0-1
	MOVL $1, AX
	XORL CX, CX
	CPUID
	ANDL $0x18000000, CX // OSXSAVE and AVX
	CMPL CX, $0x18000000
	JNE  no
	XORL CX, CX
	XGETBV
	ANDL $6, AX // XMM and YMM state saved by the OS
	CMPL AX, $6
	JNE  no
	MOVL $7, AX
	XORL CX, CX
	CPUID
	BTL  $5, BX // AVX2
	JCC  no
	MOVB $1, ret+0(FP)
	RET
no:
	MOVB $0, ret+0(FP)
	RET

// One level of the search for the 32 values in Y0-Y3 with node indices Y4-Y7: the thresholds
// are in Y8-Y11; each index j becomes 2j+1 where the value is above (GT_OQ: never for a NaN),
// 2j otherwise.
#define STEP \
	VCMPPS $0x1e, Y8, Y0, Y8 \
	VCMPPS $0x1e, Y9, Y1, Y9 \
	VCMPPS $0x1e, Y10, Y2, Y10 \
	VCMPPS $0x1e, Y11, Y3, Y11 \
	VPADDD Y4, Y4, Y4 \
	VPADDD Y5, Y5, Y5 \
	VPADDD Y6, Y6, Y6 \
	VPADDD Y7, Y7, Y7 \
	VPSUBD Y8, Y4, Y4 \
	VPSUBD Y9, Y5, Y5 \
	VPSUBD Y10, Y6, Y6 \
	VPSUBD Y11, Y7, Y7

// A level of at most eight nodes, at off bytes into the tree: the thresholds by VPERMPS. Levels 4
// and 5 take two and four VPERMPS (more cost as much as a gather), levels 6 and 7 VGATHERDPS.
#define PERMLEVEL(off) \
	VPERMPS off(R8), Y4, Y8 \
	VPERMPS off(R8), Y5, Y9 \
	VPERMPS off(R8), Y6, Y10 \
	VPERMPS off(R8), Y7, Y11 \
	STEP

// Level 4 (16 nodes at tree[15]) for the values in v with indices j: two VPERMPS, and bit 3 of
// j picks between them.
#define LEVEL16(v, j) \
	VPSLLD    $28, j, Y12 \
	VPERMPS   60(R8), j, Y13 \
	VPERMPS   92(R8), j, Y14 \
	VBLENDVPS Y12, Y14, Y13, Y13 \
	VCMPPS    $0x1e, Y13, v, Y13 \
	VPADDD    j, j, j \
	VPSUBD    Y13, j, j

// Level 5 (32 nodes at tree[31]): four VPERMPS, and bits 3 and 4 of j pick among them.
#define LEVEL32(v, j) \
	VPSLLD    $28, j, Y12 \
	VPSLLD    $27, j, Y15 \
	VPERMPS   124(R8), j, Y13 \
	VPERMPS   156(R8), j, Y14 \
	VBLENDVPS Y12, Y14, Y13, Y13 \
	VPERMPS   188(R8), j, Y14 \
	VPERMPS   220(R8), j, Y8 \
	VBLENDVPS Y12, Y8, Y14, Y14 \
	VBLENDVPS Y15, Y14, Y13, Y13 \
	VCMPPS    $0x1e, Y13, v, Y13 \
	VPADDD    j, j, j \
	VPSUBD    Y13, j, j

// func assignAVX2(dst *uint8, vals *float32, n int, tree *float32, levels int)
// n is a positive multiple of 32, levels from 4 to 8.
TEXT ·assignAVX2(SB), NOSPLIT, $0-40
	MOVQ dst+0(FP), DI
	MOVQ vals+8(FP), SI
	MOVQ n+16(FP), CX
	MOVQ tree+24(FP), R8
	MOVQ levels+32(FP), R9
	SUBQ $4, R9 // levels to gather

loop:
	VMOVUPS 0(SI), Y0
	VMOVUPS 32(SI), Y1
	VMOVUPS 64(SI), Y2
	VMOVUPS 96(SI), Y3
	VPXOR   Y4, Y4, Y4
	VPXOR   Y5, Y5, Y5
	VPXOR   Y6, Y6, Y6
	VPXOR   Y7, Y7, Y7
	PERMLEVEL(0)
	PERMLEVEL(4)
	PERMLEVEL(12)
	PERMLEVEL(28)
	CMPQ R9, $0
	JEQ  pack
	LEVEL16(Y0, Y4)
	LEVEL16(Y1, Y5)
	LEVEL16(Y2, Y6)
	LEVEL16(Y3, Y7)
	CMPQ R9, $1
	JEQ  pack
	LEVEL32(Y0, Y4)
	LEVEL32(Y1, Y5)
	LEVEL32(Y2, Y6)
	LEVEL32(Y3, Y7)
	MOVQ R9, R10
	SUBQ $2, R10
	JZ   pack
	LEAQ 252(R8), R11 // level 6 starts at tree[63]
	MOVQ $256, R12    // and holds 64 nodes

gather:
	VPCMPEQD   Y12, Y12, Y12
	VPCMPEQD   Y13, Y13, Y13
	VPCMPEQD   Y14, Y14, Y14
	VPCMPEQD   Y15, Y15, Y15
	VGATHERDPS Y12, (R11)(Y4*4), Y8
	VGATHERDPS Y13, (R11)(Y5*4), Y9
	VGATHERDPS Y14, (R11)(Y6*4), Y10
	VGATHERDPS Y15, (R11)(Y7*4), Y11
	STEP
	ADDQ R12, R11
	SHLQ $1, R12
	DECQ R10
	JNZ  gather

pack:
	VPACKSSDW Y5, Y4, Y4
	VPACKSSDW Y7, Y6, Y6
	VPACKUSWB Y6, Y4, Y4
	VMOVDQU   packperm<>(SB), Y8
	VPERMD    Y4, Y8, Y4
	VMOVDQU   Y4, (DI)
	ADDQ $128, SI
	ADDQ $32, DI
	SUBQ $32, CX
	JNZ  loop
	VZEROUPPER
	RET

// Horizontal minimum (op VMINPS) or maximum of Y reg into X reg's lowest lane, through X tmp.
#define REDUCE(op, reg, xreg, tmp) \
	VEXTRACTF128 $1, reg, tmp \
	op tmp, xreg, xreg \
	VPERMILPS $0x4e, xreg, tmp \
	op tmp, xreg, xreg \
	VPERMILPS $0xb1, xreg, tmp \
	op tmp, xreg, xreg

// func minMaxAVX2(vals *float32, n int) (lo, hi float32, nan bool)
// n is a positive multiple of 32.
TEXT ·minMaxAVX2(SB), NOSPLIT, $0-25
	MOVQ vals+0(FP), SI
	MOVQ n+8(FP), CX
	VBROADCASTSS maxf32<>(SB), Y4
	VMOVAPS      Y4, Y5
	VBROADCASTSS negmaxf32<>(SB), Y6
	VMOVAPS      Y6, Y7
	VXORPS       Y8, Y8, Y8

mmloop:
	VMOVUPS 0(SI), Y0
	VMOVUPS 32(SI), Y1
	VMOVUPS 64(SI), Y2
	VMOVUPS 96(SI), Y3
	VMINPS  Y0, Y4, Y4
	VMINPS  Y1, Y5, Y5
	VMINPS  Y2, Y4, Y4
	VMINPS  Y3, Y5, Y5
	VMAXPS  Y0, Y6, Y6
	VMAXPS  Y1, Y7, Y7
	VMAXPS  Y2, Y6, Y6
	VMAXPS  Y3, Y7, Y7
	VCMPPS  $3, Y1, Y0, Y9 // unordered
	VCMPPS  $3, Y3, Y2, Y10
	VORPS   Y9, Y8, Y8
	VORPS   Y10, Y8, Y8
	ADDQ $128, SI
	SUBQ $32, CX
	JNZ  mmloop

	VMINPS Y5, Y4, Y4
	VMAXPS Y7, Y6, Y6
	REDUCE(VMINPS, Y4, X4, X5)
	REDUCE(VMAXPS, Y6, X6, X7)
	MOVSS     X4, lo+16(FP)
	MOVSS     X6, hi+20(FP)
	VMOVMSKPS Y8, AX
	TESTL     AX, AX
	SETNE     nan+24(FP)
	VZEROUPPER
	RET

// func absMaxAVX2(vals *float32, n int) (top float32, nan bool)
// n is a positive multiple of 32.
TEXT ·absMaxAVX2(SB), NOSPLIT, $0-21
	MOVQ vals+0(FP), SI
	MOVQ n+8(FP), CX
	VBROADCASTSS absmask<>(SB), Y11
	VXORPS       Y4, Y4, Y4
	VXORPS       Y5, Y5, Y5
	VXORPS       Y8, Y8, Y8

amloop:
	VMOVUPS 0(SI), Y0
	VMOVUPS 32(SI), Y1
	VMOVUPS 64(SI), Y2
	VMOVUPS 96(SI), Y3
	VCMPPS  $3, Y1, Y0, Y9
	VCMPPS  $3, Y3, Y2, Y10
	VORPS   Y9, Y8, Y8
	VORPS   Y10, Y8, Y8
	VANDPS  Y11, Y0, Y0
	VANDPS  Y11, Y1, Y1
	VANDPS  Y11, Y2, Y2
	VANDPS  Y11, Y3, Y3
	VMAXPS  Y0, Y4, Y4
	VMAXPS  Y1, Y5, Y5
	VMAXPS  Y2, Y4, Y4
	VMAXPS  Y3, Y5, Y5
	ADDQ $128, SI
	SUBQ $32, CX
	JNZ  amloop

	VMAXPS Y5, Y4, Y4
	REDUCE(VMAXPS, Y4, X4, X5)
	MOVSS     X4, top+16(FP)
	VMOVMSKPS Y8, AX
	TESTL     AX, AX
	SETNE     nan+20(FP)
	VZEROUPPER
	RET

// func clipErrorsAVX2(errs *[16]float32, vals *float32, n int, scales *[16]float32, nz *[16]uint32, qmin, qmax float32)
// n is positive. A lane per scale: each lane sums its own errors in the order of vals.
TEXT ·clipErrorsAVX2(SB), NOSPLIT, $0-48
	MOVQ errs+0(FP), DI
	MOVQ vals+8(FP), SI
	MOVQ n+16(FP), CX
	MOVQ scales+24(FP), R8
	MOVQ nz+32(FP), R9
	VMOVUPS      0(R8), Y8
	VMOVUPS      32(R8), Y9
	VMOVUPS      0(R9), Y10
	VMOVUPS      32(R9), Y11
	VBROADCASTSS qmin+40(FP), Y12
	VBROADCASTSS qmax+44(FP), Y13
	VXORPS       Y14, Y14, Y14
	VXORPS       Y15, Y15, Y15

celoop:
	VBROADCASTSS (SI), Y0
	VDIVPS   Y8, Y0, Y1
	VDIVPS   Y9, Y0, Y2
	VROUNDPS $8, Y1, Y1 // to nearest even
	VROUNDPS $8, Y2, Y2
	VMAXPS   Y12, Y1, Y1
	VMAXPS   Y12, Y2, Y2
	VMINPS   Y13, Y1, Y1
	VMINPS   Y13, Y2, Y2
	VANDPS   Y10, Y1, Y1 // 0 where the scale is 0
	VANDPS   Y11, Y2, Y2
	VMULPS   Y8, Y1, Y1
	VMULPS   Y9, Y2, Y2
	VSUBPS   Y1, Y0, Y1
	VSUBPS   Y2, Y0, Y2
	VMULPS   Y1, Y1, Y1
	VMULPS   Y2, Y2, Y2
	VADDPS   Y1, Y14, Y14
	VADDPS   Y2, Y15, Y15
	ADDQ $4, SI
	DECQ CX
	JNZ  celoop

	VMOVUPS Y14, 0(DI)
	VMOVUPS Y15, 32(DI)
	VZEROUPPER
	RET

#define QUANT(off, reg) \
	VMOVUPS   off(SI), reg \
	VDIVPS    Y8, reg, reg \
	VROUNDPS  $8, reg, reg \
	VMAXPS    Y9, reg, reg \
	VMINPS    Y10, reg, reg \
	VCVTPS2DQ reg, reg

// func quantizeSymAVX2(dst *int8, vals *float32, n int, s, qmin, qmax float32)
// n is a positive multiple of 32, s is not 0.
TEXT ·quantizeSymAVX2(SB), NOSPLIT, $0-36
	MOVQ dst+0(FP), DI
	MOVQ vals+8(FP), SI
	MOVQ n+16(FP), CX
	VBROADCASTSS s+24(FP), Y8
	VBROADCASTSS qmin+28(FP), Y9
	VBROADCASTSS qmax+32(FP), Y10
	VMOVDQU      packperm<>(SB), Y11

qsloop:
	QUANT(0, Y0)
	QUANT(32, Y1)
	QUANT(64, Y2)
	QUANT(96, Y3)
	VPACKSSDW Y1, Y0, Y0
	VPACKSSDW Y3, Y2, Y2
	VPACKSSWB Y2, Y0, Y0
	VPERMD    Y0, Y11, Y0
	VMOVDQU   Y0, (DI)
	ADDQ $128, SI
	ADDQ $32, DI
	SUBQ $32, CX
	JNZ  qsloop
	VZEROUPPER
	RET

// func bf16AVX2(dst *float32, src *byte, n int)
// n is a positive multiple of 32.
TEXT ·bf16AVX2(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX

bfloop:
	VPMOVZXWD 0(SI), Y0
	VPMOVZXWD 16(SI), Y1
	VPMOVZXWD 32(SI), Y2
	VPMOVZXWD 48(SI), Y3
	VPSLLD    $16, Y0, Y0
	VPSLLD    $16, Y1, Y1
	VPSLLD    $16, Y2, Y2
	VPSLLD    $16, Y3, Y3
	VMOVDQU   Y0, 0(DI)
	VMOVDQU   Y1, 32(DI)
	VMOVDQU   Y2, 64(DI)
	VMOVDQU   Y3, 96(DI)
	ADDQ $64, SI
	ADDQ $128, DI
	SUBQ $32, CX
	JNZ  bfloop
	VZEROUPPER
	RET

// Constants of the packing kernels.
DATA low4<>+0(SB)/1, $0x0f
GLOBL low4<>(SB), RODATA|NOPTR, $1
DATA low6<>+0(SB)/1, $0x3f
GLOBL low6<>(SB), RODATA|NOPTR, $1
DATA mul16<>+0(SB)/2, $0x1001 // bytes 1, 16: an even entry plus 16 times the odd one
GLOBL mul16<>(SB), RODATA|NOPTR, $2
DATA mul64<>+0(SB)/2, $0x4001 // bytes 1, 64
GLOBL mul64<>(SB), RODATA|NOPTR, $2
DATA mul4096<>+0(SB)/4, $0x10000001 // words 1, 4096
GLOBL mul4096<>(SB), RODATA|NOPTR, $4

// VPSHUFB indices keeping the low three bytes of each dword of a lane, then four zeros.
DATA shuf24<>+0(SB)/8, $0x0908060504020100
DATA shuf24<>+8(SB)/8, $0x808080800e0d0c0a
DATA shuf24<>+16(SB)/8, $0x0908060504020100
DATA shuf24<>+24(SB)/8, $0x808080800e0d0c0a
GLOBL shuf24<>(SB), RODATA|NOPTR, $32

// VPERMD indices joining the three dwords of each lane that shuf24 filled.
DATA perm24<>+0(SB)/4, $0
DATA perm24<>+4(SB)/4, $1
DATA perm24<>+8(SB)/4, $2
DATA perm24<>+12(SB)/4, $4
DATA perm24<>+16(SB)/4, $5
DATA perm24<>+20(SB)/4, $6
DATA perm24<>+24(SB)/4, $3
DATA perm24<>+28(SB)/4, $7
GLOBL perm24<>(SB), RODATA|NOPTR, $32

// func pack4AVX2(dst *byte, idx *uint8, n int)
// n is a positive multiple of 32: VPMADDUBSW makes each pair a word e+16o, VPACKUSWB takes the
// low bytes, and VPERMQ joins the two lanes' halves.
TEXT ·pack4AVX2(SB), NOSPLIT, $0-24
	MOVQ         dst+0(FP), DI
	MOVQ         idx+8(FP), SI
	MOVQ         n+16(FP), CX
	VPBROADCASTB low4<>(SB), Y15
	VPBROADCASTW mul16<>(SB), Y14

p4loop:
	VMOVDQU    0(SI), Y0
	VPAND      Y15, Y0, Y0
	VPMADDUBSW Y14, Y0, Y0
	VPACKUSWB  Y0, Y0, Y0
	VPERMQ     $0x08, Y0, Y0
	VMOVDQU    X0, 0(DI)
	ADDQ       $32, SI
	ADDQ       $16, DI
	SUBQ       $32, CX
	JNZ        p4loop
	VZEROUPPER
	RET

// func pack6AVX2(dst *byte, idx *uint8, n int)
// n is a positive multiple of 32: VPMADDUBSW makes each pair a word a+64b, VPMADDWD each four a
// dword of 24 bits, and VPSHUFB and VPERMD gather their low three bytes into 24.
TEXT ·pack6AVX2(SB), NOSPLIT, $0-24
	MOVQ         dst+0(FP), DI
	MOVQ         idx+8(FP), SI
	MOVQ         n+16(FP), CX
	VPBROADCASTB low6<>(SB), Y15
	VPBROADCASTW mul64<>(SB), Y14
	VPBROADCASTD mul4096<>(SB), Y13
	VMOVDQU      shuf24<>(SB), Y12
	VMOVDQU      perm24<>(SB), Y11

p6loop:
	VMOVDQU      0(SI), Y0
	VPAND        Y15, Y0, Y0
	VPMADDUBSW   Y14, Y0, Y0
	VPMADDWD     Y13, Y0, Y0
	VPSHUFB      Y12, Y0, Y0
	VPERMD       Y0, Y11, Y0
	VMOVDQU      X0, 0(DI)
	VEXTRACTI128 $1, Y0, X1
	MOVQ         X1, 16(DI)
	ADDQ         $32, SI
	ADDQ         $24, DI
	SUBQ         $32, CX
	JNZ          p6loop
	VZEROUPPER
	RET

// func hasF16C() bool
TEXT ·hasF16C(SB), NOSPLIT, $0-1
	MOVL $1, AX
	XORL CX, CX
	CPUID
	SHRL $29, CX
	ANDL $1, CX
	MOVB CX, ret+0(FP)
	RET

DATA sign16<>+0(SB)/2, $0x7fff
GLOBL sign16<>(SB), RODATA|NOPTR, $2
DATA inf16<>+0(SB)/2, $0x7c00
GLOBL inf16<>(SB), RODATA|NOPTR, $2

// func f32ToF16F16C(dst *byte, src *float32, n int) (nan bool)
// n is a positive multiple of 32: VCVTPS2PH with immediate 0 rounds to nearest even. Y8 collects
// the unordered compares of each value with itself: the NaNs.
TEXT ·f32ToF16F16C(SB), NOSPLIT, $0-25
	MOVQ   dst+0(FP), DI
	MOVQ   src+8(FP), SI
	MOVQ   n+16(FP), CX
	VXORPS Y8, Y8, Y8

hloop:
	VMOVUPS   0(SI), Y0
	VMOVUPS   32(SI), Y1
	VMOVUPS   64(SI), Y2
	VMOVUPS   96(SI), Y3
	VCMPPS    $3, Y0, Y0, Y4
	VCMPPS    $3, Y1, Y1, Y5
	VCMPPS    $3, Y2, Y2, Y6
	VCMPPS    $3, Y3, Y3, Y7
	VORPS     Y4, Y8, Y8
	VORPS     Y5, Y8, Y8
	VORPS     Y6, Y8, Y8
	VORPS     Y7, Y8, Y8
	VCVTPS2PH $0, Y0, 0(DI)
	VCVTPS2PH $0, Y1, 16(DI)
	VCVTPS2PH $0, Y2, 32(DI)
	VCVTPS2PH $0, Y3, 48(DI)
	ADDQ      $128, SI
	ADDQ      $64, DI
	SUBQ      $32, CX
	JNZ       hloop

	VPTEST Y8, Y8
	SETNE  nan+24(FP)
	VZEROUPPER
	RET

// func f16ToF32F16C(dst *float32, src *byte, n int) (nan bool)
// n is a positive multiple of 32: VCVTPH2PS is exact. Y8 collects the halves above 0x7c00 once
// their sign is cleared: the NaNs.
TEXT ·f16ToF32F16C(SB), NOSPLIT, $0-25
	MOVQ         dst+0(FP), DI
	MOVQ         src+8(FP), SI
	MOVQ         n+16(FP), CX
	VPBROADCASTW sign16<>(SB), Y14
	VPBROADCASTW inf16<>(SB), Y15
	VPXOR        Y8, Y8, Y8

floop:
	VMOVDQU   0(SI), Y0
	VMOVDQU   32(SI), Y1
	VPAND     Y14, Y0, Y2
	VPAND     Y14, Y1, Y3
	VPCMPGTW  Y15, Y2, Y2
	VPCMPGTW  Y15, Y3, Y3
	VPOR      Y2, Y8, Y8
	VPOR      Y3, Y8, Y8
	VCVTPH2PS 0(SI), Y4
	VCVTPH2PS 16(SI), Y5
	VCVTPH2PS 32(SI), Y6
	VCVTPH2PS 48(SI), Y7
	VMOVUPS   Y4, 0(DI)
	VMOVUPS   Y5, 32(DI)
	VMOVUPS   Y6, 64(DI)
	VMOVUPS   Y7, 96(DI)
	ADDQ      $64, SI
	ADDQ      $128, DI
	SUBQ      $32, CX
	JNZ       floop

	VPTEST Y8, Y8
	SETNE  nan+24(FP)
	VZEROUPPER
	RET
