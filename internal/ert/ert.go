// Package ert builds XRT Embedded Runtime (ERT) command packets, the format the amdxdna
// driver expects in command buffer objects (xrt/detail/ert.h).
package ert

import (
	"encoding/binary"
	"errors"
	"fmt"
)

// Opcodes (enum ert_cmd_opcode).
const (
	OpStartCU         = 0
	OpStartDPU        = 18
	OpCmdChain        = 19
	OpStartNPU        = 20
	OpStartNPUPreempt = 21
)

// Command types (enum ert_cmd_type).
const TypeCU = 3

// State is enum ert_cmd_state.
type State uint32

const (
	StateNew        State = 1
	StateQueued     State = 2
	StateRunning    State = 3
	StateCompleted  State = 4
	StateError      State = 5
	StateAbort      State = 6
	StateSubmitted  State = 7
	StateTimeout    State = 8
	StateNoResponse State = 9
	StateSKError    State = 10
	StateSKCrashed  State = 11
)

var stateNames = map[State]string{
	StateNew: "NEW", StateQueued: "QUEUED", StateRunning: "RUNNING", StateCompleted: "COMPLETED",
	StateError: "ERROR", StateAbort: "ABORT", StateSubmitted: "SUBMITTED", StateTimeout: "TIMEOUT",
	StateNoResponse: "NORESPONSE", StateSKError: "SKERROR", StateSKCrashed: "SKCRASHED",
}

func (s State) String() string {
	if n, ok := stateNames[s]; ok {
		return "ERT_CMD_STATE_" + n
	}
	return fmt.Sprintf("ert_cmd_state(%d)", uint32(s))
}

// Terminal reports whether the command finished, successfully or not.
func (s State) Terminal() bool { return s >= StateCompleted && s != StateSubmitted }

// Header packs the first word of struct ert_start_kernel_cmd:
// state[3:0] stat_enabled[4] unused[9:5] extra_cu_masks[11:10] count[22:12] opcode[27:23] type[31:28].
func Header(state State, extraCUMasks, count, opcode, typ uint32) uint32 {
	return uint32(state)&0xf | (extraCUMasks&0x3)<<10 | (count&0x7ff)<<12 | (opcode&0x1f)<<23 | (typ&0xf)<<28
}

// StateOf extracts the state from a header word.
func StateOf(header uint32) State { return State(header & 0xf) }

// MaxPayloadWords is the largest count a header can express (11 bits), including the CU mask.
const MaxPayloadWords = 0x7ff

// StartCU writes an ERT_START_CU packet into dst: header, one CU mask and the register map
// (kernel arguments at their metadata offsets), as XRT does for xclbin based NPU kernels.
// It returns the packet length in bytes.
func StartCU(dst []byte, cuIndex int, regmap []byte) (int, error) {
	if cuIndex < 0 || cuIndex >= 32 {
		return 0, fmt.Errorf("ert: CU index %d needs extra CU masks, which are not supported", cuIndex)
	}
	if len(regmap)%4 != 0 {
		return 0, errors.New("ert: register map size must be a multiple of 4")
	}
	count := 1 + len(regmap)/4
	if count > MaxPayloadWords {
		return 0, fmt.Errorf("ert: register map of %d bytes is too large", len(regmap))
	}
	n := 8 + len(regmap)
	if len(dst) < n {
		return 0, fmt.Errorf("ert: command buffer of %d bytes is too small for %d", len(dst), n)
	}
	binary.LittleEndian.PutUint32(dst[0:], Header(StateNew, 0, uint32(count), OpStartCU, TypeCU))
	binary.LittleEndian.PutUint32(dst[4:], 1<<uint(cuIndex))
	copy(dst[8:], regmap)
	return n, nil
}
