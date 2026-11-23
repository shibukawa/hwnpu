#include "textflag.h"

// CONST puts a float32 constant's bits in every lane of V.
#define CONST(bits, V) \
	MOVW $bits, R10 \
	VDUP R10, V.S4

// EXP is e^A in place for one vector (Cephes' expf, as f32_amd64.s's expAVX2): N, P, T and Q
// are temporaries. The constants are in V16 to V29 (see expNEON).
#define EXP(A, N, P, T, Q) \
	VFMIN V21.S4, A.S4, A.S4 \
	VFMAX V22.S4, A.S4, A.S4 \
	VORR V19.B16, V19.B16, N.B16 \
	VFMLA V16.S4, A.S4, N.S4 \
	VFRINTM N.S4, N.S4 \
	VFMLS V17.S4, N.S4, A.S4 \
	VFMLS V18.S4, N.S4, A.S4 \
	VORR V24.B16, V24.B16, P.B16 \
	VFMLA V23.S4, A.S4, P.S4 \
	VORR V25.B16, V25.B16, T.B16 \
	VFMLA P.S4, A.S4, T.S4 \
	VORR V26.B16, V26.B16, P.B16 \
	VFMLA T.S4, A.S4, P.S4 \
	VORR V27.B16, V27.B16, T.B16 \
	VFMLA P.S4, A.S4, T.S4 \
	VORR V28.B16, V28.B16, P.B16 \
	VFMLA T.S4, A.S4, P.S4 \
	VFMUL A.S4, A.S4, Q.S4 \
	VFMLA P.S4, Q.S4, A.S4 \
	VFADD V20.S4, A.S4, A.S4 \
	VFCVTZS N.S4, N.S4 \
	VADD V29.S4, N.S4, N.S4 \
	VSHL $23, N.S4, N.S4 \
	VFADD A.S4, A.S4, A.S4 \
	VFMUL N.S4, A.S4, A.S4

// func expNEON(dst, src *float32, n, mode int)
// mode 0: e^x; 1: 1 / (1 + e^-x); 2: x / (1 + e^-x). n is a multiple of 8: two vectors a pass,
// side by side.
TEXT ·expNEON(SB), NOSPLIT, $0-32
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD n+16(FP), R2
	MOVD mode+24(FP), R3
	CONST(0x3fb8aa3b, V16) // log2 e
	CONST(0x3f318000, V17) // c1
	CONST(0xb95e8083, V18) // c2
	CONST(0x3f000000, V19) // 0.5
	CONST(0x3f800000, V20) // 1
	CONST(0x42b17217, V21) // 88.72283
	CONST(0xc2ad3333, V22) // -86.6
	CONST(0x39506967, V23) // p0
	CONST(0x3ab743ce, V24) // p1
	CONST(0x3c088908, V25) // p2
	CONST(0x3d2aa9c1, V26) // p3
	CONST(0x3e2aaaaa, V27) // p4
	CONST(0x3f000000, V28) // p5
	CONST(0x0000007e, V29) // 126
loop:
	VLD1.P 32(R1), [V0.S4, V1.S4] // x
	CBZ R3, plus
	VFNEG V0.S4, V2.S4
	VFNEG V1.S4, V3.S4
	B    exp
plus:
	VORR V0.B16, V0.B16, V2.B16
	VORR V1.B16, V1.B16, V3.B16
exp:
	EXP(V2, V4, V5, V6, V7)
	EXP(V3, V8, V9, V10, V11)
	CBZ R3, store
	VFADD V20.S4, V2.S4, V2.S4 // 1 + e^-x
	VFADD V20.S4, V3.S4, V3.S4
	CMP  $1, R3
	BEQ  sigmoid
	VFDIV V2.S4, V0.S4, V2.S4 // x / (1 + e^-x)
	VFDIV V3.S4, V1.S4, V3.S4
	B    store
sigmoid:
	VFDIV V2.S4, V20.S4, V2.S4 // 1 / (1 + e^-x)
	VFDIV V3.S4, V20.S4, V3.S4
store:
	VST1.P [V2.S4, V3.S4], 32(R0)
	SUBS $8, R2
	BNE  loop
	RET

// func gluNEON(dst, gu *float32, n int, alpha, limit, add float32)
// dst[j] = g / (1 + e^(-alpha g)) (u + add), g = min(gu[2j], limit), u = gu[2j+1] within
// [-limit, limit]; n a multiple of 8. LD2 splits the gates and the ups.
TEXT ·gluNEON(SB), NOSPLIT, $0-36
	MOVD  dst+0(FP), R0
	MOVD  gu+8(FP), R1
	MOVD  n+16(FP), R2
	FMOVS alpha+24(FP), F30
	VDUP  V30.S[0], V30.S4
	FMOVS limit+28(FP), F31
	VDUP  V31.S[0], V31.S4
	FMOVS add+32(FP), F15
	VDUP  V15.S[0], V15.S4
	VFNEG V31.S4, V14.S4 // -limit
	CONST(0x3fb8aa3b, V16)
	CONST(0x3f318000, V17)
	CONST(0xb95e8083, V18)
	CONST(0x3f000000, V19)
	CONST(0x3f800000, V20)
	CONST(0x42b17217, V21)
	CONST(0xc2ad3333, V22)
	CONST(0x39506967, V23)
	CONST(0x3ab743ce, V24)
	CONST(0x3c088908, V25)
	CONST(0x3d2aa9c1, V26)
	CONST(0x3e2aaaaa, V27)
	CONST(0x3f000000, V28)
	CONST(0x0000007e, V29)
loopg:
	VLD2.P 32(R1), [V0.S4, V1.S4] // gates, ups
	VLD2.P 32(R1), [V2.S4, V3.S4]
	VFMIN  V31.S4, V0.S4, V0.S4
	VFMIN  V31.S4, V2.S4, V2.S4
	VFMIN  V31.S4, V1.S4, V1.S4
	VFMAX  V14.S4, V1.S4, V1.S4
	VFMIN  V31.S4, V3.S4, V3.S4
	VFMAX  V14.S4, V3.S4, V3.S4
	VFADD  V15.S4, V1.S4, V1.S4
	VFADD  V15.S4, V3.S4, V3.S4
	VFMUL  V30.S4, V0.S4, V4.S4
	VFNEG  V4.S4, V4.S4
	VFMUL  V30.S4, V2.S4, V5.S4
	VFNEG  V5.S4, V5.S4
	EXP(V4, V6, V7, V8, V9)
	EXP(V5, V10, V11, V12, V13)
	VFADD  V20.S4, V4.S4, V4.S4
	VFADD  V20.S4, V5.S4, V5.S4
	VFDIV  V4.S4, V0.S4, V0.S4
	VFDIV  V5.S4, V2.S4, V2.S4
	VFMUL  V1.S4, V0.S4, V0.S4
	VFMUL  V3.S4, V2.S4, V1.S4
	VST1.P [V0.S4, V1.S4], 32(R0)
	SUBS   $8, R2
	BNE    loopg
	RET
