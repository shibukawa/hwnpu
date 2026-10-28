// Pointer members are declared as unsafe.Pointer (same size as the C __u64 on 64-bit targets) so
// that Go keeps them valid if the referenced memory is on a goroutine stack that moves.
//
// Package xdna talks to the Linux amdxdna driver (drivers/accel/amdxdna) through ioctl and mmap.
// Struct layouts and ioctl numbers follow include/uapi/drm/amdxdna_accel.h and drm.h, and are
// checked against values compiled from those headers (testdata/abi/xdna_abi.json).
package xdna

import (
	"structs"
	"unsafe"
)

// Buffer object types (enum amdxdna_bo_type).
const (
	BOShare   = 1
	BODevHeap = 2
	BODev     = 3
	BOCmd     = 4
)

// Sync directions.
const (
	SyncToDevice   = 0
	SyncFromDevice = 1
)

// Misc constants.
const (
	InvalidAddr          = ^uint64(0)
	CmdSubmitExecBuf     = 0
	ConfigCU             = 0 // DRM_AMDXDNA_HWCTX_CONFIG_CU
	QueryAIEMetadata     = 1
	QuerySensors         = 4
	QueryFirmwareVersion = 8
	SensorTypePower      = 0
	SyncobjWaitForSubmit = 1 << 1 // DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT

	QoSRealtime = 0x100
	QoSHigh     = 0x180
	QoSNormal   = 0x200
	QoSLow      = 0x280

	// DevHeapSize is AIE2_DEVM_SIZE: heaps are created in multiples of it.
	DevHeapSize = 64 << 20
	// DevMemBase is AIE2_DEVM_BASE, the device address of the first heap.
	DevMemBase = 0x4000000
)

type QoSInfo struct {
	_             structs.HostLayout
	GOPS          uint32
	FPS           uint32
	DMABandwidth  uint32
	Latency       uint32
	FrameExecTime uint32
	Priority      uint32
}

type CreateHwctx struct {
	_             structs.HostLayout
	Ext           uint64
	ExtFlags      uint64
	QoS           unsafe.Pointer // *QoSInfo
	UMQBO         uint32
	LogBufBO      uint32
	MaxOPC        uint32
	NumTiles      uint32
	MemSize       uint32
	UMQDoorbell   uint32
	Handle        uint32
	SyncobjHandle uint32
}

type DestroyHwctx struct {
	_      structs.HostLayout
	Handle uint32
	Pad    uint32
}

type CUConfig struct {
	_      structs.HostLayout
	CUBO   uint32
	CUFunc uint8
	Pad    [3]uint8
}

// ConfigCUHeader is struct amdxdna_hwctx_param_config_cu without the flexible array.
type ConfigCUHeader struct {
	_      structs.HostLayout
	NumCUs uint16
	Pad    [3]uint16
}

type ConfigHwctx struct {
	_            structs.HostLayout
	Handle       uint32
	ParamType    uint32
	ParamVal     unsafe.Pointer
	ParamValSize uint32
	Pad          uint32
}

type CreateBO struct {
	_      structs.HostLayout
	Flags  uint64
	VAddr  uint64
	Size   uint64
	Type   uint32
	Handle uint32
}

type GetBOInfo struct {
	_         structs.HostLayout
	Ext       uint64
	ExtFlags  uint64
	Handle    uint32
	Pad       uint32
	MapOffset uint64
	VAddr     uint64
	XdnaAddr  uint64
}

type SyncBO struct {
	_         structs.HostLayout
	Handle    uint32
	Direction uint32
	Offset    uint64
	Size      uint64
}

type ExecCmd struct {
	_          structs.HostLayout
	Ext        uint64
	ExtFlags   uint64
	Hwctx      uint32
	Type       uint32
	CmdHandles uint64         // with one command: the command BO handle itself, not a pointer
	Args       unsafe.Pointer // *uint32 BO handles
	CmdCount   uint32
	ArgCount   uint32
	Seq        uint64
}

type TileMetadata struct {
	_               structs.HostLayout
	RowCount        uint16
	RowStart        uint16
	DMAChannelCount uint16
	LockCount       uint16
	EventRegCount   uint16
	Pad             [3]uint16
}

type AIEMetadata struct {
	_            structs.HostLayout
	ColSize      uint32
	Cols         uint16
	Rows         uint16
	VersionMajor uint32
	VersionMinor uint32
	Core         TileMetadata
	Mem          TileMetadata
	Shim         TileMetadata
}

type Sensor struct {
	_       structs.HostLayout
	Label   [64]byte
	Input   uint32
	Max     uint32
	Average uint32
	Highest uint32
	Status  [64]byte
	Units   [16]byte
	UnitM   int8
	Type    uint8
	Pad     [6]uint8
}

type FirmwareVersion struct {
	_     structs.HostLayout
	Major uint32
	Minor uint32
	Patch uint32
	Build uint32
}

type GetInfo struct {
	_          structs.HostLayout
	Param      uint32
	BufferSize uint32
	Buffer     unsafe.Pointer
}

type GemClose struct {
	_      structs.HostLayout
	Handle uint32
	Pad    uint32
}

type SyncobjTimelineWait struct {
	_             structs.HostLayout
	Handles       unsafe.Pointer // *uint32
	Points        unsafe.Pointer // *uint64
	TimeoutNsec   int64
	CountHandles  uint32
	Flags         uint32
	FirstSignaled uint32
	Pad           uint32
	DeadlineNsec  uint64
}

// ioctl number encoding of asm-generic/ioctl.h (x86 and arm64).
const (
	iocWrite     = 1
	iocRead      = 2
	drmIoctlBase = 'd'
	drmCmdBase   = 0x40
)

func ioc(dir, nr, size uintptr) uintptr { return dir<<30 | size<<16 | drmIoctlBase<<8 | nr }
func iowr(nr, size uintptr) uintptr     { return ioc(iocRead|iocWrite, nr, size) }
func iow(nr, size uintptr) uintptr      { return ioc(iocWrite, nr, size) }

// Ioctl request numbers.
var (
	IoctlCreateHwctx  = iowr(drmCmdBase+0, unsafe.Sizeof(CreateHwctx{}))
	IoctlDestroyHwctx = iowr(drmCmdBase+1, unsafe.Sizeof(DestroyHwctx{}))
	IoctlConfigHwctx  = iowr(drmCmdBase+2, unsafe.Sizeof(ConfigHwctx{}))
	IoctlCreateBO     = iowr(drmCmdBase+3, unsafe.Sizeof(CreateBO{}))
	IoctlGetBOInfo    = iowr(drmCmdBase+4, unsafe.Sizeof(GetBOInfo{}))
	IoctlSyncBO       = iowr(drmCmdBase+5, unsafe.Sizeof(SyncBO{}))
	IoctlExecCmd      = iowr(drmCmdBase+6, unsafe.Sizeof(ExecCmd{}))
	IoctlGetInfo      = iowr(drmCmdBase+7, unsafe.Sizeof(GetInfo{}))
	IoctlGemClose     = iow(0x09, unsafe.Sizeof(GemClose{}))
	IoctlSyncobjWait  = iowr(0xCA, unsafe.Sizeof(SyncobjTimelineWait{}))
)
