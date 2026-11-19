#include "textflag.h"

// NEON instructions Go's assembler does not know: d, n and m are vector register numbers.
#define SDOT(m, n, d) WORD $(0x4E809400 | (m)<<16 | (n)<<5 | (d))   // d.4S += n.16B · m.16B by 4
#define SCVTF4S(n, d) WORD $(0x4E21D800 | (n)<<5 | (d))             // .4S int32 to float32
#define FMLAE(m, n, d) WORD $(0x4F801000 | (m)<<16 | (n)<<5 | (d))  // d.4S += n.4S * m.S[0]
#define FADDP4S(m, n, d) WORD $(0x6E20D400 | (m)<<16 | (n)<<5 | (d)) // pairwise
#define FADD4S(m, n, d) WORD $(0x4E20D400 | (m)<<16 | (n)<<5 | (d))

// func dotNEON(w, e *byte, q *int8, s *float32, nb int, lut *int8) float32
// nb blocks: w 16 bytes a block, e a byte, q 32 int8 (evens then odds), s a float32. The sum
// of the doubled values' products, each block's times 2^(e-127) and s.
TEXT ·dotNEON(SB), NOSPLIT, $0-52
	MOVD w+0(FP), R0
	MOVD e+8(FP), R1
	MOVD q+16(FP), R2
	MOVD s+24(FP), R3
	MOVD nb+32(FP), R4
	MOVD lut+40(FP), R5
	VLD1 (R5), [V30.B16]
	VMOVI $15, V31.B16
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	TBZ $0, R4, pairs
	// an odd block first
	VLD1.P 16(R0), [V0.B16]
	VLD1.P 32(R2), [V1.B16, V2.B16]
	MOVBU.P 1(R1), R6
	FMOVS.P 4(R3), F7
	LSL $23, R6, R6
	FMOVS R6, F6
	FMULS F7, F6, F6
	VAND V31.B16, V0.B16, V3.B16
	VUSHR $4, V0.B16, V4.B16
	VTBL V3.B16, [V30.B16], V3.B16
	VTBL V4.B16, [V30.B16], V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	SDOT(1, 3, 5)
	SDOT(2, 4, 5)
	SCVTF4S(5, 5)
	FMLAE(6, 5, 8)
	SUBS $1, R4, R4
	BEQ done
pairs:
	VLD1.P 32(R0), [V20.B16, V21.B16]
	VLD1.P 64(R2), [V1.B16, V2.B16, V3.B16, V4.B16]
	MOVHU.P 2(R1), R6
	FLDPS.P 8(R3), (F7, F17)
	UBFX $8, R6, $8, R7
	AND $255, R6, R6
	LSL $23, R6, R6
	LSL $23, R7, R7
	FMOVS R6, F6
	FMOVS R7, F16
	FMULS F7, F6, F6
	FMULS F17, F16, F16
	VAND V31.B16, V20.B16, V18.B16
	VUSHR $4, V20.B16, V19.B16
	VAND V31.B16, V21.B16, V13.B16
	VUSHR $4, V21.B16, V14.B16
	VTBL V18.B16, [V30.B16], V18.B16
	VTBL V19.B16, [V30.B16], V19.B16
	VTBL V13.B16, [V30.B16], V13.B16
	VTBL V14.B16, [V30.B16], V14.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V15.B16, V15.B16, V15.B16
	SDOT(1, 18, 5)
	SDOT(3, 13, 15)
	SDOT(2, 19, 5)
	SDOT(4, 14, 15)
	SCVTF4S(5, 5)
	SCVTF4S(15, 15)
	FMLAE(6, 5, 8)
	FMLAE(16, 15, 9)
	SUBS $2, R4, R4
	BNE pairs
done:
	FADD4S(9, 8, 8)
	FADDP4S(8, 8, 8)
	FADDP4S(8, 8, 8)
	FMOVS F8, ret+48(FP)
	RET

#define FMLAE1(m, n, d) WORD $(0x4FA01000 | (m)<<16 | (n)<<5 | (d)) // d.4S += n.4S * m.S[1]
#define FMUL2S(m, n, d) WORD $(0x2E20DC00 | (m)<<16 | (n)<<5 | (d))  // d.2S = n.2S * m.2S

// func dot4NEON(w, e *byte, q *[4]*int8, s *[4]*float32, nb int, lut *int8, out *[4]float32)
// dotNEON of one row with 4 vectors, the row's blocks decoded once for the 4. nb even.
TEXT ·dot4NEON(SB), NOSPLIT, $0-56
	MOVD w+0(FP), R0
	MOVD e+8(FP), R1
	MOVD q+16(FP), R2
	MOVD s+24(FP), R3
	MOVD nb+32(FP), R4
	MOVD lut+40(FP), R5
	MOVD 0(R2), R8
	MOVD 8(R2), R9
	MOVD 16(R2), R10
	MOVD 24(R2), R11
	MOVD 0(R3), R12
	MOVD 8(R3), R13
	MOVD 16(R3), R14
	MOVD 24(R3), R15
	VLD1 (R5), [V30.B16]
	VMOVI $15, V31.B16
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
loop4:
	VLD1.P 32(R0), [V20.B16, V21.B16]
	MOVHU.P 2(R1), R6
	UBFX $8, R6, $8, R7
	AND $255, R6, R6
	LSL $23, R6, R6
	LSL $23, R7, R7
	VMOV R6, V6.S[0]
	VMOV R7, V6.S[1]
	VAND V31.B16, V20.B16, V16.B16
	VUSHR $4, V20.B16, V17.B16
	VAND V31.B16, V21.B16, V18.B16
	VUSHR $4, V21.B16, V19.B16
	VTBL V16.B16, [V30.B16], V16.B16
	VTBL V17.B16, [V30.B16], V17.B16
	VTBL V18.B16, [V30.B16], V18.B16
	VTBL V19.B16, [V30.B16], V19.B16
	// vector 0
	VLD1.P 64(R8), [V0.B16, V1.B16, V2.B16, V3.B16]
	VLD1.P 8(R12), [V7.S2]
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	SDOT(0, 16, 4)
	SDOT(2, 18, 5)
	SDOT(1, 17, 4)
	SDOT(3, 19, 5)
	FMUL2S(6, 7, 7)
	SCVTF4S(4, 4)
	SCVTF4S(5, 5)
	FMLAE(7, 4, 8)
	FMLAE1(7, 5, 8)
	// vector 1
	VLD1.P 64(R9), [V22.B16, V23.B16, V24.B16, V25.B16]
	VLD1.P 8(R13), [V26.S2]
	VEOR V12.B16, V12.B16, V12.B16
	VEOR V13.B16, V13.B16, V13.B16
	SDOT(22, 16, 12)
	SDOT(24, 18, 13)
	SDOT(23, 17, 12)
	SDOT(25, 19, 13)
	FMUL2S(6, 26, 26)
	SCVTF4S(12, 12)
	SCVTF4S(13, 13)
	FMLAE(26, 12, 9)
	FMLAE1(26, 13, 9)
	// vector 2
	VLD1.P 64(R10), [V0.B16, V1.B16, V2.B16, V3.B16]
	VLD1.P 8(R14), [V7.S2]
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	SDOT(0, 16, 4)
	SDOT(2, 18, 5)
	SDOT(1, 17, 4)
	SDOT(3, 19, 5)
	FMUL2S(6, 7, 7)
	SCVTF4S(4, 4)
	SCVTF4S(5, 5)
	FMLAE(7, 4, 10)
	FMLAE1(7, 5, 10)
	// vector 3
	VLD1.P 64(R11), [V22.B16, V23.B16, V24.B16, V25.B16]
	VLD1.P 8(R15), [V26.S2]
	VEOR V12.B16, V12.B16, V12.B16
	VEOR V13.B16, V13.B16, V13.B16
	SDOT(22, 16, 12)
	SDOT(24, 18, 13)
	SDOT(23, 17, 12)
	SDOT(25, 19, 13)
	FMUL2S(6, 26, 26)
	SCVTF4S(12, 12)
	SCVTF4S(13, 13)
	FMLAE(26, 12, 11)
	FMLAE1(26, 13, 11)
	SUBS $2, R4, R4
	BNE loop4
	MOVD out+48(FP), R6
	FADDP4S(9, 8, 8)
	FADDP4S(11, 10, 10)
	FADDP4S(10, 8, 8)
	VST1 [V8.S4], (R6)
	RET
