#include "textflag.h"

// NEON instructions that Go's assembler does not know, encoded by hand for the .4S arrangement:
// d, n and m are vector register numbers.
#define FMLAE(i, m, n, d) WORD $(0x4F801000 | ((i)&1)<<21 | (m)<<16 | ((i)>>1)<<11 | (n)<<5 | (d)) // d += n * m.S[i]
#define FADD(m, n, d) WORD $(0x4E20D400 | (m)<<16 | (n)<<5 | (d))

// One row of C at R3: c += the accumulators V(r), V(r+1), V(r+2), V(r+3); R3 to the next row.
#define ROW(r) \
	VLD1 (R3), [V24.S4, V25.S4, V26.S4, V27.S4] \
	FADD(r, 24, 24) \
	FADD(r+1, 25, 25) \
	FADD(r+2, 26, 26) \
	FADD(r+3, 27, 27) \
	VST1 [V24.S4, V25.S4, V26.S4, V27.S4], (R3) \
	ADD R4, R3

// func kernelNEON(kc int, a, b, c *float32, ldc int)
// c (6 rows of 16, ldc apart) += a (kc x 6) b (kc x 16): row r's sums in V(4r) to V(4r+3), a
// step's 16 values of b in V24 to V27 and its 6 of a in V28 and V29's low half.
TEXT ·kernelNEON(SB), NOSPLIT, $0-40
	MOVD kc+0(FP), R0
	MOVD a+8(FP), R1
	MOVD b+16(FP), R2
	MOVD c+24(FP), R3
	MOVD ldc+32(FP), R4
	LSL $2, R4
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
	VEOR V12.B16, V12.B16, V12.B16
	VEOR V13.B16, V13.B16, V13.B16
	VEOR V14.B16, V14.B16, V14.B16
	VEOR V15.B16, V15.B16, V15.B16
	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	VEOR V18.B16, V18.B16, V18.B16
	VEOR V19.B16, V19.B16, V19.B16
	VEOR V20.B16, V20.B16, V20.B16
	VEOR V21.B16, V21.B16, V21.B16
	VEOR V22.B16, V22.B16, V22.B16
	VEOR V23.B16, V23.B16, V23.B16
	CBZ R0, store
loop:
	VLD1.P 64(R2), [V24.S4, V25.S4, V26.S4, V27.S4]
	VLD1 (R1), [V28.S4]
	FMOVD 16(R1), F29
	ADD $24, R1
	FMLAE(0, 28, 24, 0)
	FMLAE(0, 28, 25, 1)
	FMLAE(0, 28, 26, 2)
	FMLAE(0, 28, 27, 3)
	FMLAE(1, 28, 24, 4)
	FMLAE(1, 28, 25, 5)
	FMLAE(1, 28, 26, 6)
	FMLAE(1, 28, 27, 7)
	FMLAE(2, 28, 24, 8)
	FMLAE(2, 28, 25, 9)
	FMLAE(2, 28, 26, 10)
	FMLAE(2, 28, 27, 11)
	FMLAE(3, 28, 24, 12)
	FMLAE(3, 28, 25, 13)
	FMLAE(3, 28, 26, 14)
	FMLAE(3, 28, 27, 15)
	FMLAE(0, 29, 24, 16)
	FMLAE(0, 29, 25, 17)
	FMLAE(0, 29, 26, 18)
	FMLAE(0, 29, 27, 19)
	FMLAE(1, 29, 24, 20)
	FMLAE(1, 29, 25, 21)
	FMLAE(1, 29, 26, 22)
	FMLAE(1, 29, 27, 23)
	SUBS $1, R0
	BNE loop
store:
	ROW(0)
	ROW(4)
	ROW(8)
	ROW(12)
	ROW(16)
	ROW(20)
	RET
