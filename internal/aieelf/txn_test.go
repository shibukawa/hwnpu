package aieelf

import (
	"encoding/binary"
	"testing"
)

// TestDecodeTxn decodes the start of the control code aiecc wrote for IRON's bounded ReLU:
// create the scratchpad, load a PDI, write a core's parameter and add the scratchpad's slot 0
// to it, then a shim buffer descriptor and the add of slot 0 times 32 to its length.
func TestDecodeTxn(t *testing.T) {
	words := []uint32{
		0x06040100, 0x00000108, 6, 0, // header: 6 operations, size patched below
		0x0000000a, 4, 0, 0, // CREATE_SCRATCHPAD of 4 bytes
		0x00010008, 0, 0, 0, // LOADPDI 1
		0, 0, 0x00200600, 0, 0, 24, // WRITE 0x200600 = 0
		0x0001000c, 0, 0x00200600, // UPDATE_REG slot 0, incr
		0x00000001, 0, 0x0001d000, 48, 0, 0, 0, 0, 0, 0xc0000000, 0x02000000, 0, // BLOCKWRITE BD 0
		0x0000000c, 0x20, 0x0001d000, // UPDATE_REG slot 0, mul 32
	}
	b := make([]byte, 4*len(words))
	for i, w := range words {
		binary.LittleEndian.PutUint32(b[4*i:], w)
	}
	binary.LittleEndian.PutUint32(b[12:], uint32(len(b)))
	h, ops, err := DecodeTxn(b)
	if err != nil {
		t.Fatal(err)
	}
	if h.NumCols != 8 || h.NumRows != 6 || len(ops) != 6 {
		t.Fatalf("%+v %d", h, len(ops))
	}
	want := []struct {
		code   uint8
		size   int
		reg    uint64
		idx    uint8
		fn     uint8
		arg    uint32
		offset int
	}{
		{OpCreateScratchpad, 16, 0, 0, 0, 0, 16},
		{OpLoadPDI, 16, 0, 0, 0, 0, 32},
		{OpWrite, 24, 0x200600, 0, 0, 0, 48},
		{OpUpdateReg, 12, 0x200600, 0, FuncIncr, 0, 72},
		{OpBlockWrite, 48, 0x1d000, 0, 0, 0, 84},
		{OpUpdateReg, 12, 0x1d000, 0, FuncMul, 0x20, 132},
	}
	for i, w := range want {
		o := ops[i]
		if o.Code != w.code || o.Size != w.size || o.RegOff != w.reg || o.Idx != w.idx || o.Func != w.fn || o.Arg != w.arg || o.Offset != w.offset {
			t.Errorf("op %d: %+v, want %+v", i, o, w)
		}
	}
	binary.LittleEndian.PutUint32(b[8:], 7) // one operation too many
	if _, _, err := DecodeTxn(b); err == nil {
		t.Error("a transaction shorter than its operation count was accepted")
	}
}
