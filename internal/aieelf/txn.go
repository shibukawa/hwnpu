package aieelf

import (
	"encoding/binary"
	"fmt"
)

// Transaction (TXN) opcodes of the control code (aie-rt's XAie_TxnOpcode).
const (
	OpWrite            = 0
	OpBlockWrite       = 1
	OpBlockSet         = 2
	OpMaskWrite        = 3
	OpMaskPoll         = 4
	OpNoop             = 5
	OpPreempt          = 6
	OpMaskPollBusy     = 7
	OpLoadPDI          = 8
	OpLoadPMStart      = 9
	OpCreateScratchpad = 10
	OpUpdateStateTable = 11
	OpUpdateReg        = 12
	OpUpdateScratch    = 13
	OpCustomTCT        = 0x80
	OpCustomDDRPatch   = 0x81
	OpCustomReadRegs   = 0x82
	OpCustomRecTimer   = 0x83
	OpCustomMergeSync  = 0x84
)

// State table functions of OpUpdateReg and OpUpdateStateTable (XAie_StateTableFuncType).
const (
	FuncMul  = 0
	FuncIncr = 1
	FuncDecr = 2
)

// TxnHeader is the header of a transaction (XAie_TxnHeader), 16 bytes.
type TxnHeader struct {
	Major, Minor, DevGen, NumRows, NumCols, NumMemTileRows uint8
	NumOps, Size                                           uint32
}

// Op is one decoded operation.
type Op struct {
	Offset int // byte offset in the control code
	Size   int // bytes
	Code   uint8
	// Fields, by opcode: RegOff for writes, polls and OpUpdateReg; Value for writes and polls;
	// Idx, Func and Arg for the state table operations.
	RegOff uint64
	Value  uint32
	Idx    uint8
	Func   uint8
	Arg    uint32
}

// DecodeTxn decodes the control code of an instance: the transaction header and its
// operations, in the unoptimized layout (major version 0) aiecc writes for AIE2P.
func DecodeTxn(code []byte) (TxnHeader, []Op, error) {
	le := binary.LittleEndian
	if len(code) < 16 {
		return TxnHeader{}, nil, fmt.Errorf("aieelf: control code of %d bytes", len(code))
	}
	h := TxnHeader{Major: code[0], Minor: code[1], DevGen: code[2], NumRows: code[3], NumCols: code[4], NumMemTileRows: code[5],
		NumOps: le.Uint32(code[8:]), Size: le.Uint32(code[12:])}
	if h.Major != 0 {
		return h, nil, fmt.Errorf("aieelf: transaction version %d.%d is not supported", h.Major, h.Minor)
	}
	if int(h.Size) > len(code) {
		return h, nil, fmt.Errorf("aieelf: transaction of %d bytes in %d bytes of control code", h.Size, len(code))
	}
	var ops []Op
	o := 16
	for i := uint32(0); i < h.NumOps; i++ {
		if o+4 > int(h.Size) {
			return h, ops, fmt.Errorf("aieelf: operation %d at %#x runs past the transaction", i, o)
		}
		op := Op{Offset: o, Code: code[o]}
		need := func(n int) error {
			if o+n > int(h.Size) {
				return fmt.Errorf("aieelf: operation %d (%d) at %#x runs past the transaction", i, op.Code, o)
			}
			return nil
		}
		switch op.Code {
		case OpWrite: // XAie_Write32Hdr: header, pad to 8, u64 RegOff, Value, Size
			if err := need(24); err != nil {
				return h, ops, err
			}
			op.RegOff, op.Value, op.Size = le.Uint64(code[o+8:]), le.Uint32(code[o+16:]), int(le.Uint32(code[o+20:]))
		case OpMaskWrite, OpMaskPoll, OpMaskPollBusy: // + Mask
			if err := need(28); err != nil {
				return h, ops, err
			}
			op.RegOff, op.Value, op.Size = le.Uint64(code[o+8:]), le.Uint32(code[o+16:]), int(le.Uint32(code[o+24:]))
		case OpBlockWrite: // header, Col, Row, pad, u32 RegOff, u32 Size (bytes, with the data)
			if err := need(16); err != nil {
				return h, ops, err
			}
			op.RegOff, op.Size = uint64(le.Uint32(code[o+8:])), int(le.Uint32(code[o+12:]))
		case OpNoop, OpPreempt, OpUpdateScratch:
			op.Size = 4
		case OpLoadPDI, OpCreateScratchpad:
			op.Size = 16
		case OpLoadPMStart:
			op.Size = 8
		case OpUpdateStateTable:
			if err := need(8); err != nil {
				return h, ops, err
			}
			op.Size, op.Idx, op.Func, op.Arg = 8, code[o+1], code[o+2], le.Uint32(code[o+4:])
		case OpUpdateReg:
			if err := need(12); err != nil {
				return h, ops, err
			}
			op.Size, op.Idx, op.Func, op.Arg, op.RegOff = 12, code[o+1], code[o+2], le.Uint32(code[o+4:]), uint64(le.Uint32(code[o+8:]))
		default:
			if op.Code >= OpCustomTCT { // XAie_CustomOpHdr: header, pad, u32 Size
				if err := need(8); err != nil {
					return h, ops, err
				}
				op.Size = int(le.Uint32(code[o+4:]))
			} else {
				return h, ops, fmt.Errorf("aieelf: unknown operation %d at %#x", op.Code, o)
			}
		}
		if op.Size <= 0 || o+op.Size > int(h.Size) {
			return h, ops, fmt.Errorf("aieelf: operation %d (%d) at %#x has size %d", i, op.Code, o, op.Size)
		}
		ops = append(ops, op)
		o += op.Size
	}
	return h, ops, nil
}
