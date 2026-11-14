// Trampolines for the C functions darwin_dl.go imports (cgo_import_dynamic), as in
// golang.org/x/sys/unix.

#include "textflag.h"

TEXT npuinfo_dlopen_trampoline<>(SB),NOSPLIT,$0-0
	JMP	npuinfo_dlopen(SB)
GLOBL	·dlopenTrampolineAddr(SB), RODATA, $8
DATA	·dlopenTrampolineAddr(SB)/8, $npuinfo_dlopen_trampoline<>(SB)

TEXT npuinfo_dlsym_trampoline<>(SB),NOSPLIT,$0-0
	JMP	npuinfo_dlsym(SB)
GLOBL	·dlsymTrampolineAddr(SB), RODATA, $8
DATA	·dlsymTrampolineAddr(SB)/8, $npuinfo_dlsym_trampoline<>(SB)

TEXT npuinfo_dlerror_trampoline<>(SB),NOSPLIT,$0-0
	JMP	npuinfo_dlerror(SB)
GLOBL	·dlerrorTrampolineAddr(SB), RODATA, $8
DATA	·dlerrorTrampolineAddr(SB)/8, $npuinfo_dlerror_trampoline<>(SB)
