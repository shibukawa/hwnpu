package xdnasim

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/shibukawa/hwnpu/internal/xclbin"
	"github.com/shibukawa/hwnpu/internal/xdna"
)

// A simulated IRON style passthrough design: kernel "MLIR_AIE" with the argument layout of
// the xclbin mlir-aie v1.4.4 builds (opcode, instr, ninstr, bo0 ... bo4), copying bo0 to bo2.
// Host code leaves the unused bo3 and bo4 unset, which XRT sends as zero.
const (
	PassthroughKernel = "MLIR_AIE"
	PassthroughOpcode = 3
	passthroughID     = 0x901
	instrMagic        = 0xC0FFEE01
)

// PassthroughPDI is the PDI the simulated compute unit expects.
var PassthroughPDI = []byte("SIMULATED-PDI:passthrough")

// PassthroughXclbin builds the xclbin of the simulated design.
func PassthroughXclbin() []byte {
	return xclbin.Build(xclbin.Spec{
		UUID: [16]byte{0x68, 0x77, 0x6e, 0x70, 0x75},
		Kernels: []xclbin.Kernel{{Name: PassthroughKernel, Type: "dpu", KernelID: passthroughID, Args: []xclbin.Arg{
			{Index: 0, Name: "opcode", Type: "uint64_t", Offset: 0x00, Size: 8},
			{Index: 1, Name: "instr", Type: "char *", Offset: 0x08, Size: 8, AddressQualifier: 1},
			{Index: 2, Name: "ninstr", Type: "uint32_t", Offset: 0x10, Size: 4},
			{Index: 3, Name: "bo0", Type: "void*", Offset: 0x14, Size: 8, AddressQualifier: 1},
			{Index: 4, Name: "bo1", Type: "void*", Offset: 0x1c, Size: 8, AddressQualifier: 1},
			{Index: 5, Name: "bo2", Type: "void*", Offset: 0x24, Size: 8, AddressQualifier: 1},
			{Index: 6, Name: "bo3", Type: "void*", Offset: 0x2c, Size: 8, AddressQualifier: 1},
			{Index: 7, Name: "bo4", Type: "void*", Offset: 0x34, Size: 8, AddressQualifier: 1},
		}}},
		CUs:         []xclbin.CU{{Name: PassthroughKernel + ":MLIRAIE", KernelID: passthroughID}},
		Columns:     1,
		StartCols:   []uint16{0, 1, 2, 3},
		OpsPerCycle: 2048,
		PDIs: []xclbin.PDI{{UUID: [16]byte{7}, Image: PassthroughPDI,
			Groups: []xclbin.CDOGroup{{Name: "DPU", Type: 1, KernelIDs: []uint64{passthroughID}}}}},
	})
}

// PassthroughInstructions returns the instruction stream for copying n bytes.
func PassthroughInstructions(n int) []byte {
	b := binary.LittleEndian.AppendUint32(nil, instrMagic)
	return binary.LittleEndian.AppendUint32(b, uint32(n))
}

// RunPassthrough is a Sim.Run implementation for the passthrough design.
func RunPassthrough(e Exec) error {
	if !bytes.Equal(e.PDI[:len(PassthroughPDI)], PassthroughPDI) {
		return errors.New("unexpected PDI")
	}
	le := binary.LittleEndian
	if len(e.Regmap) < 0x3c {
		return fmt.Errorf("register map too short: %d", len(e.Regmap))
	}
	if op := le.Uint64(e.Regmap[0x00:]); op != PassthroughOpcode {
		return fmt.Errorf("opcode %d", op)
	}
	instrAddr, ninstr := le.Uint64(e.Regmap[0x08:]), le.Uint32(e.Regmap[0x10:])
	if instrAddr < xdna.DevMemBase || instrAddr >= xdna.DevMemBase+xdna.DevHeapSize {
		return fmt.Errorf("instructions at %#x are not in the device heap", instrAddr)
	}
	instr, err := e.Mem(instrAddr, int(ninstr)*4)
	if err != nil {
		return err
	}
	if ninstr != 2 || le.Uint32(instr) != instrMagic {
		return errors.New("bad instruction stream")
	}
	n := int(le.Uint32(instr[4:]))
	src, err := e.Mem(le.Uint64(e.Regmap[0x14:]), n)
	if err != nil {
		return err
	}
	dst, err := e.Mem(le.Uint64(e.Regmap[0x24:]), n)
	if err != nil {
		return err
	}
	if le.Uint64(e.Regmap[0x2c:]) != 0 || le.Uint64(e.Regmap[0x34:]) != 0 {
		return errors.New("unset arguments bo3 and bo4 are not zero")
	}
	copy(dst, src)
	return nil
}
