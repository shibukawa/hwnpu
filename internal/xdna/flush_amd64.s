#include "textflag.h"

// func clflushRange(p unsafe.Pointer, n uintptr, line uintptr)
//
// Flushes (writes back and invalidates) every cache line of [p, p+n), fenced on both sides,
// as XRT's shim does for the NPU's buffers on x86 (pcidev.cpp clflush_data).
TEXT ·clflushRange(SB), NOSPLIT, $0-24
	MOVQ p+0(FP), AX
	MOVQ n+8(FP), BX
	MOVQ line+16(FP), CX
	TESTQ BX, BX
	JZ   done
	LEAQ -1(AX)(BX*1), DX // the last byte
	MOVQ CX, SI
	DECQ SI
	NOTQ SI
	ANDQ SI, AX // down to the start of its line
	MFENCE
loop:
	CLFLUSH (AX)
	ADDQ CX, AX
	CMPQ AX, DX
	JLS  loop
	MFENCE
done:
	RET
