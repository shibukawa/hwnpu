// Package ze binds the parts of oneAPI Level Zero and the Intel NPU graph extension that hwnpu
// uses, through purego. Struct layouts follow ze_api.h and ze_graph_ext.h for 64-bit targets and
// are checked against values generated from the C headers (see testdata/abi).
package ze

import (
	"fmt"
	"structs"
)

// Result is ze_result_t.
type Result uint32

const (
	ResultSuccess  Result = 0
	ResultNotReady Result = 1
)

// resultNames is generated from the ze_result_t enum in ze_api.h.
var resultNames = map[Result]string{
	0:          "SUCCESS",
	1:          "NOT_READY",
	0x70000001: "ERROR_DEVICE_LOST",
	0x70000002: "ERROR_OUT_OF_HOST_MEMORY",
	0x70000003: "ERROR_OUT_OF_DEVICE_MEMORY",
	0x70000004: "ERROR_MODULE_BUILD_FAILURE",
	0x70000005: "ERROR_MODULE_LINK_FAILURE",
	0x70000006: "ERROR_DEVICE_REQUIRES_RESET",
	0x70000007: "ERROR_DEVICE_IN_LOW_POWER_STATE",
	0x7ff00001: "EXP_ERROR_DEVICE_IS_NOT_VERTEX",
	0x7ff00002: "EXP_ERROR_VERTEX_IS_NOT_DEVICE",
	0x7ff00003: "EXP_ERROR_REMOTE_DEVICE",
	0x7ff00004: "EXP_ERROR_OPERANDS_INCOMPATIBLE",
	0x7ff00005: "EXP_RTAS_BUILD_RETRY",
	0x7ff00006: "EXP_RTAS_BUILD_DEFERRED",
	0x70010000: "ERROR_INSUFFICIENT_PERMISSIONS",
	0x70010001: "ERROR_NOT_AVAILABLE",
	0x70020000: "ERROR_DEPENDENCY_UNAVAILABLE",
	0x70020001: "WARNING_DROPPED_DATA",
	0x78000001: "ERROR_UNINITIALIZED",
	0x78000002: "ERROR_UNSUPPORTED_VERSION",
	0x78000003: "ERROR_UNSUPPORTED_FEATURE",
	0x78000004: "ERROR_INVALID_ARGUMENT",
	0x78000005: "ERROR_INVALID_NULL_HANDLE",
	0x78000006: "ERROR_HANDLE_OBJECT_IN_USE",
	0x78000007: "ERROR_INVALID_NULL_POINTER",
	0x78000008: "ERROR_INVALID_SIZE",
	0x78000009: "ERROR_UNSUPPORTED_SIZE",
	0x7800000a: "ERROR_UNSUPPORTED_ALIGNMENT",
	0x7800000b: "ERROR_INVALID_SYNCHRONIZATION_OBJECT",
	0x7800000c: "ERROR_INVALID_ENUMERATION",
	0x7800000d: "ERROR_UNSUPPORTED_ENUMERATION",
	0x7800000e: "ERROR_UNSUPPORTED_IMAGE_FORMAT",
	0x7800000f: "ERROR_INVALID_NATIVE_BINARY",
	0x78000010: "ERROR_INVALID_GLOBAL_NAME",
	0x78000011: "ERROR_INVALID_KERNEL_NAME",
	0x78000012: "ERROR_INVALID_FUNCTION_NAME",
	0x78000013: "ERROR_INVALID_GROUP_SIZE_DIMENSION",
	0x78000014: "ERROR_INVALID_GLOBAL_WIDTH_DIMENSION",
	0x78000015: "ERROR_INVALID_KERNEL_ARGUMENT_INDEX",
	0x78000016: "ERROR_INVALID_KERNEL_ARGUMENT_SIZE",
	0x78000017: "ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE",
	0x78000018: "ERROR_INVALID_MODULE_UNLINKED",
	0x78000019: "ERROR_INVALID_COMMAND_LIST_TYPE",
	0x7800001a: "ERROR_OVERLAPPING_REGIONS",
	0x7800001b: "WARNING_ACTION_REQUIRED",
	0x7800001c: "ERROR_INVALID_KERNEL_HANDLE",
	0x7800001d: "EXT_RTAS_BUILD_RETRY",
	0x7800001e: "EXT_RTAS_BUILD_DEFERRED",
	0x7800001f: "EXT_ERROR_OPERANDS_INCOMPATIBLE",
	0x78000020: "ERROR_SURVIVABILITY_MODE_DETECTED",
	0x78000021: "ERROR_ADDRESS_NOT_FOUND",
	0x78000022: "QUERY_TRUE",
	0x78000023: "QUERY_FALSE",
	0x78000024: "ERROR_INVALID_GRAPH",
	0x78000025: "ERROR_GRAPH_CAPTURE_UNSUPPORTED",
	0x78000026: "ERROR_GRAPH_CAPTURE_INVALIDATED",
	0x78000027: "ERROR_GRAPH_CAPTURE_MERGE_ATTEMPT",
	0x78000028: "ERROR_COMMAND_LIST_NOT_CAPTURING",
	0x78000029: "ERROR_GRAPH_UNJOINED_FORKS",
	0x7800002a: "ERROR_GRAPH_INTERNAL_EVENT",
	0x7ffffffe: "ERROR_UNKNOWN",
}

func (r Result) String() string {
	if n, ok := resultNames[r]; ok {
		return "ZE_RESULT_" + n
	}
	return fmt.Sprintf("ze_result_t(%#x)", uint32(r))
}

// Error is a failed Level Zero call.
type Error struct {
	Op     string
	Result Result
	Log    string // build log, when the driver provided one
}

func (e *Error) Error() string {
	if e.Log != "" {
		return fmt.Sprintf("%s: %v: %s", e.Op, e.Result, e.Log)
	}
	return fmt.Sprintf("%s: %v", e.Op, e.Result)
}

func check(op string, r Result) error {
	if r == ResultSuccess {
		return nil
	}
	return &Error{Op: op, Result: r}
}

// Handle is any ze_*_handle_t.
type Handle uintptr

// MakeVersion is ZE_MAKE_VERSION.
func MakeVersion(major, minor uint32) uint32 { return major<<16 | minor&0xffff }

// VersionString formats a ZE_MAKE_VERSION value.
func VersionString(v uint32) string { return fmt.Sprintf("%d.%d", v>>16, v&0xffff) }

// Structure types (ze_structure_type_t) and other constants.
const (
	stypeDriverProperties   = 0x1
	stypeDeviceProperties   = 0x3
	stypeContextDesc        = 0xd
	stypeCommandQueueDesc   = 0xe
	stypeCommandListDesc    = 0xf
	stypeFenceDesc          = 0x12
	stypeHostMemAllocDesc   = 0x16
	initFlagVPUOnly         = 1 << 1
	DeviceTypeVPU           = 5
	maxExtensionName        = 256
	maxDeviceName           = 256
	commandQueueModeDefault = 0
)

// NPUDriverUUID is ze_intel_npu_driver_uuid.
var NPUDriverUUID = [16]byte{0x01, 0x7d, 0xe9, 0x31, 0x6b, 0x4d, 0x4f, 0xd4, 0xaa, 0x9b, 0x5b, 0xed, 0x77, 0xfc, 0x8e, 0x89}

// DriverProperties is ze_driver_properties_t.
type DriverProperties struct {
	_             structs.HostLayout
	SType         uint32
	PNext         uintptr
	UUID          [16]byte
	DriverVersion uint32
}

// DriverExtensionProperties is ze_driver_extension_properties_t.
type DriverExtensionProperties struct {
	_       structs.HostLayout
	Name    [maxExtensionName]byte
	Version uint32
}

// DeviceProperties is ze_device_properties_t.
type DeviceProperties struct {
	_                        structs.HostLayout
	SType                    uint32
	PNext                    uintptr
	Type                     uint32
	VendorID                 uint32
	DeviceID                 uint32
	Flags                    uint32
	SubdeviceID              uint32
	CoreClockRate            uint32
	MaxMemAllocSize          uint64
	MaxHardwareContexts      uint32
	MaxCommandQueuePriority  uint32
	NumThreadsPerEU          uint32
	PhysicalEUSimdWidth      uint32
	NumEUsPerSubslice        uint32
	NumSubslicesPerSlice     uint32
	NumSlices                uint32
	TimerResolution          uint64
	TimestampValidBits       uint32
	KernelTimestampValidBits uint32
	UUID                     [16]byte
	Name                     [maxDeviceName]byte
}

// simpleDesc covers ze_context_desc_t, ze_host_mem_alloc_desc_t and ze_fence_desc_t.
type simpleDesc struct {
	_     structs.HostLayout
	SType uint32
	PNext uintptr
	Flags uint32
}

// CommandQueueDesc is ze_command_queue_desc_t.
type CommandQueueDesc struct {
	_        structs.HostLayout
	SType    uint32
	PNext    uintptr
	Ordinal  uint32
	Index    uint32
	Flags    uint32
	Mode     uint32
	Priority uint32
}

// Command queue priorities (ze_command_queue_priority_t) and group flags.
const (
	QueuePriorityNormal         = 0
	QueuePriorityLow            = 1
	QueuePriorityHigh           = 2
	QueueGroupFlagCompute       = 1 << 0
	stypeCommandQueueGroupProps = 0x6
)

// CommandQueueGroupProperties is ze_command_queue_group_properties_t.
type CommandQueueGroupProperties struct {
	_                        structs.HostLayout
	SType                    uint32
	PNext                    uintptr
	Flags                    uint32
	MaxMemoryFillPatternSize uintptr
	NumQueues                uint32
}

// CommandListDesc is ze_command_list_desc_t.
type CommandListDesc struct {
	_                        structs.HostLayout
	SType                    uint32
	PNext                    uintptr
	CommandQueueGroupOrdinal uint32
	Flags                    uint32
}

// CString returns the NUL terminated prefix of b.
func CString(b []byte) string {
	for i, c := range b {
		if c == 0 {
			return string(b[:i])
		}
	}
	return string(b)
}
