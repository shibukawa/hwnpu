#include "textflag.h"

// The accumulators are zeroed by MOVI, not by EOR of a register with itself: on the M3 that EOR
// waits for the register's last value, so the sums that reuse a register run one after another
// (SDOT and SMMLA take 3 and 6 cycles). With MOVI every kernel doubled: a row with one vector
// 65 → 32 µs, an expert's gate_up with 16 vectors on a core 4.9 → 2.4 ms by SMMLA.
//
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
	VMOVI $0, V8.B16
	VMOVI $0, V9.B16
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
	VMOVI $0, V5.B16
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
	VMOVI $0, V5.B16
	VMOVI $0, V15.B16
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
	VMOVI $0, V8.B16
	VMOVI $0, V9.B16
	VMOVI $0, V10.B16
	VMOVI $0, V11.B16
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
	VMOVI $0, V4.B16
	VMOVI $0, V5.B16
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
	VMOVI $0, V12.B16
	VMOVI $0, V13.B16
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
	VMOVI $0, V4.B16
	VMOVI $0, V5.B16
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
	VMOVI $0, V12.B16
	VMOVI $0, V13.B16
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

#define SMMLA(m, n, d) WORD $(0x4E80A400 | (m)<<16 | (n)<<5 | (d))  // d.4S += n (2x8 int8) · m (2x8)ᵀ
#define ZIP1D(m, n, d) WORD $(0x4EC03800 | (m)<<16 | (n)<<5 | (d))  // d = n.D[0], m.D[0]
#define ZIP2D(m, n, d) WORD $(0x4EC07800 | (m)<<16 | (n)<<5 | (d))  // d = n.D[1], m.D[1]
#define FMUL4S(m, n, d) WORD $(0x6E20DC00 | (m)<<16 | (n)<<5 | (d)) // d.4S = n.4S * m.4S
#define FMLA4S(m, n, d) WORD $(0x4E20CC00 | (m)<<16 | (n)<<5 | (d)) // d.4S += n.4S * m.4S

#define ADD4S(m, n, d) WORD $(0x4EA08400 | (m)<<16 | (n)<<5 | (d))  // d.4S = n.4S + m.4S
#define UXTL8H(n, d) WORD $(0x2F08A400 | (n)<<5 | (d))              // d.8H = n.8B zero extended
#define UXTL4S(n, d) WORD $(0x2F10A400 | (n)<<5 | (d))              // d.4S = n.4H zero extended
#define TRN1S(m, n, d) WORD $(0x4E802800 | (m)<<16 | (n)<<5 | (d))  // d = n0 m0 n2 m2
#define TRN2S(m, n, d) WORD $(0x4E806800 | (m)<<16 | (n)<<5 | (d))  // d = n1 m1 n3 m3

// func mm28NEON(w0, w1, e0, e1 *byte, q *[4]*int8, s *[4]*float32, nb int, lut *int8, out *[16]float32)
// Two rows (w0, e0 and w1, e1) with four pairs of vectors (q[k], s[k]: pairVec), by SMMLA
// (FEAT_I8MM): a block's 2 rows x 2 vectors of 8 int8 products each in one instruction, 4 for
// the block's 32, the rows' blocks decoded once for the 8 vectors. out[4k..4k+3] is pair k's
// r0a r0b r1a r1b: the doubled values' sums as dotNEON's. nb even. A block's 4 SMMLA go into one
// accumulator (fewer instructions: 2.40 → 2.26 ms against two summed).
TEXT ·mm28NEON(SB), NOSPLIT, $0-72
	MOVD w0+0(FP), R0
	MOVD w1+8(FP), R1
	MOVD e0+16(FP), R2
	MOVD e1+24(FP), R3
	MOVD q+32(FP), R16
	MOVD s+40(FP), R17
	MOVD nb+48(FP), R12
	MOVD lut+56(FP), R13
	MOVD 0(R16), R4
	MOVD 8(R16), R5
	MOVD 16(R16), R6
	MOVD 24(R16), R7
	MOVD 0(R17), R8
	MOVD 8(R17), R9
	MOVD 16(R17), R10
	MOVD 24(R17), R11
	VLD1 (R13), [V30.B16]
	VMOVI $15, V31.B16
	VMOVI $0, V24.B16
	VMOVI $0, V25.B16
	VMOVI $0, V26.B16
	VMOVI $0, V27.B16
mm8:
	VLD1.P 32(R0), [V0.B16, V1.B16]
	VLD1.P 32(R1), [V2.B16, V3.B16]
	MOVHU.P 2(R2), R14
	MOVHU.P 2(R3), R15
	ORR R15<<16, R14, R14
	VMOV R14, V6.S[0]
	UXTL8H(6, 6)
	UXTL4S(6, 6)
	VSHL $23, V6.S4, V6.S4
	TRN1S(6, 6, 7)
	TRN2S(6, 6, 6)
	VAND V31.B16, V0.B16, V4.B16
	VUSHR $4, V0.B16, V5.B16
	VAND V31.B16, V2.B16, V8.B16
	VUSHR $4, V2.B16, V9.B16
	VTBL V4.B16, [V30.B16], V4.B16
	VTBL V5.B16, [V30.B16], V5.B16
	VTBL V8.B16, [V30.B16], V8.B16
	VTBL V9.B16, [V30.B16], V9.B16
	ZIP1D(8, 4, 16)
	ZIP2D(8, 4, 17)
	ZIP1D(9, 5, 18)
	ZIP2D(9, 5, 19)
	VAND V31.B16, V1.B16, V4.B16
	VUSHR $4, V1.B16, V5.B16
	VAND V31.B16, V3.B16, V8.B16
	VUSHR $4, V3.B16, V9.B16
	VTBL V4.B16, [V30.B16], V4.B16
	VTBL V5.B16, [V30.B16], V5.B16
	VTBL V8.B16, [V30.B16], V8.B16
	VTBL V9.B16, [V30.B16], V9.B16
	ZIP1D(8, 4, 20)
	ZIP2D(8, 4, 21)
	ZIP1D(9, 5, 22)
	ZIP2D(9, 5, 23)
	VLD1.P 64(R4), [V0.B16, V1.B16, V2.B16, V3.B16]
	VLD1.P 16(R8), [V4.S4]
	VMOVI $0, V5.B16
	SMMLA(0, 16, 5)
	SMMLA(2, 18, 5)
	SMMLA(1, 17, 5)
	SMMLA(3, 19, 5)
	FMUL4S(7, 4, 4)
	SCVTF4S(5, 5)
	FMLA4S(4, 5, 24)
	VLD1.P 64(R5), [V10.B16, V11.B16, V12.B16, V13.B16]
	VLD1.P 16(R9), [V14.S4]
	VMOVI $0, V15.B16
	SMMLA(10, 16, 15)
	SMMLA(12, 18, 15)
	SMMLA(11, 17, 15)
	SMMLA(13, 19, 15)
	FMUL4S(7, 14, 14)
	SCVTF4S(15, 15)
	FMLA4S(14, 15, 25)
	VLD1.P 64(R6), [V0.B16, V1.B16, V2.B16, V3.B16]
	VLD1.P 16(R10), [V4.S4]
	VMOVI $0, V5.B16
	SMMLA(0, 16, 5)
	SMMLA(2, 18, 5)
	SMMLA(1, 17, 5)
	SMMLA(3, 19, 5)
	FMUL4S(7, 4, 4)
	SCVTF4S(5, 5)
	FMLA4S(4, 5, 26)
	VLD1.P 64(R7), [V10.B16, V11.B16, V12.B16, V13.B16]
	VLD1.P 16(R11), [V14.S4]
	VMOVI $0, V15.B16
	SMMLA(10, 16, 15)
	SMMLA(12, 18, 15)
	SMMLA(11, 17, 15)
	SMMLA(13, 19, 15)
	FMUL4S(7, 14, 14)
	SCVTF4S(15, 15)
	FMLA4S(14, 15, 27)
	VLD1.P 64(R4), [V0.B16, V1.B16, V2.B16, V3.B16]
	VLD1.P 16(R8), [V4.S4]
	VMOVI $0, V5.B16
	SMMLA(0, 20, 5)
	SMMLA(2, 22, 5)
	SMMLA(1, 21, 5)
	SMMLA(3, 23, 5)
	FMUL4S(6, 4, 4)
	SCVTF4S(5, 5)
	FMLA4S(4, 5, 24)
	VLD1.P 64(R5), [V10.B16, V11.B16, V12.B16, V13.B16]
	VLD1.P 16(R9), [V14.S4]
	VMOVI $0, V15.B16
	SMMLA(10, 20, 15)
	SMMLA(12, 22, 15)
	SMMLA(11, 21, 15)
	SMMLA(13, 23, 15)
	FMUL4S(6, 14, 14)
	SCVTF4S(15, 15)
	FMLA4S(14, 15, 25)
	VLD1.P 64(R6), [V0.B16, V1.B16, V2.B16, V3.B16]
	VLD1.P 16(R10), [V4.S4]
	VMOVI $0, V5.B16
	SMMLA(0, 20, 5)
	SMMLA(2, 22, 5)
	SMMLA(1, 21, 5)
	SMMLA(3, 23, 5)
	FMUL4S(6, 4, 4)
	SCVTF4S(5, 5)
	FMLA4S(4, 5, 26)
	VLD1.P 64(R7), [V10.B16, V11.B16, V12.B16, V13.B16]
	VLD1.P 16(R11), [V14.S4]
	VMOVI $0, V15.B16
	SMMLA(10, 20, 15)
	SMMLA(12, 22, 15)
	SMMLA(11, 21, 15)
	SMMLA(13, 23, 15)
	FMUL4S(6, 14, 14)
	SCVTF4S(15, 15)
	FMLA4S(14, 15, 27)
	SUBS $2, R12, R12
	BNE mm8
	MOVD out+64(FP), R14
	VST1 [V24.S4, V25.S4, V26.S4, V27.S4], (R14)
	RET

#define FMLAE2(m, n, d) WORD $(0x4F801800 | (m)<<16 | (n)<<5 | (d)) // d.4S += n.4S * m.S[2]
#define FMLAE3(m, n, d) WORD $(0x4FA01800 | (m)<<16 | (n)<<5 | (d)) // d.4S += n.4S * m.S[3]

// func mvNEON(w, e *byte, q *int8, s *float32, nb, rows int, lut *int8, out *float32)
// dotNEON of rows rows one after another with one vector (out[r] is row r's, halved): a
// generated token's products, where a call a row cost as much as the row (a down product of
// Qwen3.6-35B-A3B is 16 blocks). nb even. The sums are dotNEON's, bit for bit: the blocks in
// order, the even ones into one accumulator and the odd ones into the other. Four blocks at a
// time, their scales computed together (FMOVS into the register, not an insert: that would
// wait for the last iteration's product).
TEXT ·mvNEON(SB), NOSPLIT, $0-64
	MOVD w+0(FP), R0
	MOVD e+8(FP), R1
	MOVD q+16(FP), R10
	MOVD s+24(FP), R11
	MOVD nb+32(FP), R12
	MOVD rows+40(FP), R7
	MOVD lut+48(FP), R5
	MOVD out+56(FP), R8
	VLD1 (R5), [V30.B16]
	VMOVI $15, V31.B16
	FMOVS $0.5, F29
mvrow:
	MOVD R10, R2
	MOVD R11, R3
	MOVD R12, R4
	VMOVI $0, V8.B16
	VMOVI $0, V9.B16
	CMP $4, R4
	BLT mvtail
mvquad:
	VLD1.P 64(R0), [V20.B16, V21.B16, V22.B16, V23.B16]
	VLD1.P 64(R2), [V0.B16, V1.B16, V2.B16, V3.B16]
	VLD1.P 64(R2), [V4.B16, V5.B16, V6.B16, V7.B16]
	MOVWU.P 4(R1), R6
	FMOVS R6, F24
	UXTL8H(24, 24)
	UXTL4S(24, 24)
	VSHL $23, V24.S4, V24.S4
	VLD1.P 16(R3), [V25.S4]
	FMUL4S(25, 24, 24)
	VAND V31.B16, V20.B16, V16.B16
	VUSHR $4, V20.B16, V17.B16
	VAND V31.B16, V21.B16, V18.B16
	VUSHR $4, V21.B16, V19.B16
	VAND V31.B16, V22.B16, V12.B16
	VUSHR $4, V22.B16, V13.B16
	VAND V31.B16, V23.B16, V14.B16
	VUSHR $4, V23.B16, V15.B16
	VTBL V16.B16, [V30.B16], V16.B16
	VTBL V17.B16, [V30.B16], V17.B16
	VTBL V18.B16, [V30.B16], V18.B16
	VTBL V19.B16, [V30.B16], V19.B16
	VTBL V12.B16, [V30.B16], V12.B16
	VTBL V13.B16, [V30.B16], V13.B16
	VTBL V14.B16, [V30.B16], V14.B16
	VTBL V15.B16, [V30.B16], V15.B16
	VMOVI $0, V26.B16
	VMOVI $0, V27.B16
	VMOVI $0, V28.B16
	VMOVI $0, V10.B16
	SDOT(0, 16, 26)
	SDOT(2, 18, 27)
	SDOT(4, 12, 28)
	SDOT(6, 14, 10)
	SDOT(1, 17, 26)
	SDOT(3, 19, 27)
	SDOT(5, 13, 28)
	SDOT(7, 15, 10)
	SCVTF4S(26, 26)
	SCVTF4S(27, 27)
	SCVTF4S(28, 28)
	SCVTF4S(10, 10)
	FMLAE(24, 26, 8)
	FMLAE1(24, 27, 9)
	FMLAE2(24, 28, 8)
	FMLAE3(24, 10, 9)
	SUB $4, R4, R4
	CMP $4, R4
	BGE mvquad
mvtail:
	CBZ R4, mvdone
	VLD1.P 32(R0), [V20.B16, V21.B16]
	VLD1.P 64(R2), [V1.B16, V2.B16, V3.B16, V4.B16]
	MOVHU.P 2(R1), R6
	FLDPS.P 8(R3), (F7, F17)
	UBFX $8, R6, $8, R13
	AND $255, R6, R6
	LSL $23, R6, R6
	LSL $23, R13, R13
	FMOVS R6, F6
	FMOVS R13, F16
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
	VMOVI $0, V5.B16
	VMOVI $0, V15.B16
	SDOT(1, 18, 5)
	SDOT(3, 13, 15)
	SDOT(2, 19, 5)
	SDOT(4, 14, 15)
	SCVTF4S(5, 5)
	SCVTF4S(15, 15)
	FMLAE(6, 5, 8)
	FMLAE(16, 15, 9)
mvdone:
	FADD4S(9, 8, 8)
	FADDP4S(8, 8, 8)
	FADDP4S(8, 8, 8)
	FMULS F29, F8, F8
	FMOVS.P F8, 4(R8)
	SUBS $1, R7, R7
	BNE mvrow
	RET

#define DUP16B(rn, d) WORD $(0x4E010C00 | (rn)<<5 | (d))             // d.16B = dup(Rn)
#define ZIP1B(m, n, d) WORD $(0x4E003800 | (m)<<16 | (n)<<5 | (d))   // d.16B = zip1(n.16B, m.16B)
#define ZIP2B(m, n, d) WORD $(0x4E007800 | (m)<<16 | (n)<<5 | (d))   // d.16B = zip2(n.16B, m.16B)
#define CMTST16B(m, n, d) WORD $(0x4E208C00 | (m)<<16 | (n)<<5 | (d)) // d.16B = (n.16B & m.16B) != 0
#define ADD16B(m, n, d) WORD $(0x4E208400 | (m)<<16 | (n)<<5 | (d))  // d.16B = n.16B + m.16B

// func dequantF16NEON(dst, w, e, lut *byte, nb int) int
// Unpacks normal-range MXFP4 blocks (114 <= e <= 140) directly to little-endian IEEE-754 fp16.
// Stops and returns blocks processed if an out-of-range exponent is encountered.
TEXT ·dequantF16NEON(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R0
	MOVD w+8(FP), R1
	MOVD e+16(FP), R2
	MOVD lut+24(FP), R3
	MOVD nb+32(FP), R4
	MOVD $0, R7
	CBZ R4, df16done
	VLD1 (R3), [V30.B16]
	VMOVI $15, V31.B16
	VMOVI $0, V28.B16
df16loop:
	MOVBU (R2), R5
	SUB $114, R5, R6
	CMP $26, R6
	BHI df16done
	ADD $1, R2, R2
	SUB $127, R5, R5
	LSL $2, R5, R5
	DUP16B(5, 6)
	VLD1.P 16(R1), [V0.B16]
	VAND V31.B16, V0.B16, V1.B16
	VUSHR $4, V0.B16, V2.B16
	VTBL V1.B16, [V30.B16], V1.B16
	VTBL V2.B16, [V30.B16], V2.B16
	ZIP1B(2, 1, 3)
	ZIP2B(2, 1, 4)
	CMTST16B(3, 3, 16)
	CMTST16B(4, 4, 17)
	VAND V6.B16, V16.B16, V16.B16
	VAND V6.B16, V17.B16, V17.B16
	ADD16B(16, 3, 3)
	ADD16B(17, 4, 4)
	ZIP1B(3, 28, 20)
	ZIP2B(3, 28, 21)
	ZIP1B(4, 28, 22)
	ZIP2B(4, 28, 23)
	VST1.P [V20.B16, V21.B16, V22.B16, V23.B16], 64(R0)
	ADD $1, R7, R7
	CMP R4, R7
	BLT df16loop
df16done:
	MOVD R7, ret+40(FP)
	RET
