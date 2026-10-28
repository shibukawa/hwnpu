package xdna

import (
	"errors"
	"fmt"
	"runtime"
	"syscall"
	"time"
	"unsafe"
)

// Driver is the raw interface to one opened amdxdna device node. The Linux implementation
// issues system calls; tests substitute a simulated driver.
type Driver interface {
	// Ioctl performs request with arg pointing at the matching struct.
	Ioctl(request uintptr, arg unsafe.Pointer) error
	// Mmap maps length bytes of a buffer object at its map offset.
	Mmap(offset uint64, length int) ([]byte, error)
	Munmap(b []byte) error
	Close() error
}

// Dev wraps a Driver with typed helpers.
type Dev struct{ D Driver }

func (d Dev) ioctl(name string, req uintptr, arg unsafe.Pointer) error {
	if err := d.D.Ioctl(req, arg); err != nil {
		return &Error{Op: name, Err: err}
	}
	return nil
}

// Error is a failed ioctl.
type Error struct {
	Op  string
	Err error
}

func (e *Error) Error() string { return fmt.Sprintf("%s: %v", e.Op, e.Err) }
func (e *Error) Unwrap() error { return e.Err }

// Errno returns the errno of a failed call, or 0.
func Errno(err error) syscall.Errno {
	var en syscall.Errno
	if errors.As(err, &en) {
		return en
	}
	return 0
}

// Metadata queries the AIE array geometry.
func (d Dev) Metadata() (AIEMetadata, error) {
	var m AIEMetadata
	arg := GetInfo{Param: QueryAIEMetadata, BufferSize: uint32(unsafe.Sizeof(m)), Buffer: unsafe.Pointer(&m)}
	err := d.ioctl("GET_INFO(QUERY_AIE_METADATA)", IoctlGetInfo, unsafe.Pointer(&arg))
	runtime.KeepAlive(&m)
	return m, err
}

// Firmware queries the firmware version.
func (d Dev) Firmware() (FirmwareVersion, error) {
	var v FirmwareVersion
	arg := GetInfo{Param: QueryFirmwareVersion, BufferSize: uint32(unsafe.Sizeof(v)), Buffer: unsafe.Pointer(&v)}
	err := d.ioctl("GET_INFO(QUERY_FIRMWARE_VERSION)", IoctlGetInfo, unsafe.Pointer(&arg))
	runtime.KeepAlive(&v)
	return v, err
}

// Sensors queries the power and utilization sensors.
func (d Dev) Sensors() ([]Sensor, error) {
	buf := make([]Sensor, 16)
	arg := GetInfo{Param: QuerySensors, BufferSize: uint32(len(buf)) * uint32(unsafe.Sizeof(Sensor{})), Buffer: unsafe.Pointer(&buf[0])}
	err := d.ioctl("GET_INFO(QUERY_SENSORS)", IoctlGetInfo, unsafe.Pointer(&arg))
	runtime.KeepAlive(buf)
	if err != nil {
		return nil, err
	}
	n := int(arg.BufferSize / uint32(unsafe.Sizeof(Sensor{})))
	return buf[:min(n, len(buf))], nil
}

// BO is a created buffer object.
type BO struct {
	Handle    uint32
	Type      uint32
	Size      int
	MapOffset uint64
	XdnaAddr  uint64
}

// CreateBO creates a buffer object of the given type.
func (d Dev) CreateBO(typ uint32, size int) (BO, error) {
	arg := CreateBO{Type: typ, Size: uint64(size)}
	if err := d.ioctl("CREATE_BO", IoctlCreateBO, unsafe.Pointer(&arg)); err != nil {
		return BO{}, err
	}
	bo := BO{Handle: arg.Handle, Type: typ, Size: size}
	return bo, d.Info(&bo)
}

// Info refreshes MapOffset and XdnaAddr. With PASID the device address of a host buffer is
// its user address, so call it again after mapping.
func (d Dev) Info(bo *BO) error {
	arg := GetBOInfo{Handle: bo.Handle}
	if err := d.ioctl("GET_BO_INFO", IoctlGetBOInfo, unsafe.Pointer(&arg)); err != nil {
		return err
	}
	bo.MapOffset, bo.XdnaAddr = arg.MapOffset, arg.XdnaAddr
	return nil
}

// Sync flushes or invalidates CPU caches for a range of a buffer object.
func (d Dev) Sync(handle uint32, dir uint32, off, n int) error {
	arg := SyncBO{Handle: handle, Direction: dir, Offset: uint64(off), Size: uint64(n)}
	return d.ioctl("SYNC_BO", IoctlSyncBO, unsafe.Pointer(&arg))
}

// CloseBO releases a GEM handle.
func (d Dev) CloseBO(handle uint32) error {
	arg := GemClose{Handle: handle}
	return d.ioctl("GEM_CLOSE", IoctlGemClose, unsafe.Pointer(&arg))
}

// CreateHwctx creates a hardware context and returns its handle and timeline syncobj.
func (d Dev) CreateHwctx(qos QoSInfo, maxOPC, numTiles uint32) (handle, syncobj uint32, err error) {
	arg := CreateHwctx{QoS: unsafe.Pointer(&qos), MaxOPC: maxOPC, NumTiles: numTiles}
	err = d.ioctl("CREATE_HWCTX", IoctlCreateHwctx, unsafe.Pointer(&arg))
	runtime.KeepAlive(&qos)
	return arg.Handle, arg.SyncobjHandle, err
}

func (d Dev) DestroyHwctx(handle uint32) error {
	arg := DestroyHwctx{Handle: handle}
	return d.ioctl("DESTROY_HWCTX", IoctlDestroyHwctx, unsafe.Pointer(&arg))
}

// ConfigCUs loads PDIs (in BO_DEV buffer objects) as the context's compute units.
func (d Dev) ConfigCUs(hwctx uint32, cus []CUConfig) error {
	hdr := unsafe.Sizeof(ConfigCUHeader{})
	buf := make([]byte, hdr+uintptr(len(cus))*unsafe.Sizeof(CUConfig{}))
	(*ConfigCUHeader)(unsafe.Pointer(&buf[0])).NumCUs = uint16(len(cus))
	for i, cu := range cus {
		*(*CUConfig)(unsafe.Pointer(&buf[hdr+uintptr(i)*unsafe.Sizeof(CUConfig{})])) = cu
	}
	arg := ConfigHwctx{Handle: hwctx, ParamType: ConfigCU, ParamVal: unsafe.Pointer(&buf[0]), ParamValSize: uint32(len(buf))}
	err := d.ioctl("CONFIG_HWCTX(CONFIG_CU)", IoctlConfigHwctx, unsafe.Pointer(&arg))
	runtime.KeepAlive(buf)
	return err
}

// Exec submits one command buffer object and returns its sequence number. args lists every
// buffer object the command references.
func (d Dev) Exec(hwctx, cmdBO uint32, args []uint32) (uint64, error) {
	arg := ExecCmd{Hwctx: hwctx, Type: CmdSubmitExecBuf, CmdHandles: uint64(cmdBO), CmdCount: 1, ArgCount: uint32(len(args))}
	if len(args) > 0 {
		arg.Args = unsafe.Pointer(&args[0])
	}
	err := d.ioctl("EXEC_CMD", IoctlExecCmd, unsafe.Pointer(&arg))
	runtime.KeepAlive(args)
	return arg.Seq, err
}

// Wait waits until the timeline point seq of syncobj is signalled or timeout elapses. It
// reports false without error on timeout.
func (d Dev) Wait(syncobj uint32, seq uint64, timeout time.Duration) (bool, error) {
	handle, point := syncobj, seq
	arg := SyncobjTimelineWait{
		Handles: unsafe.Pointer(&handle), Points: unsafe.Pointer(&point),
		TimeoutNsec: MonotonicNow() + int64(timeout), CountHandles: 1, Flags: SyncobjWaitForSubmit,
	}
	err := d.D.Ioctl(IoctlSyncobjWait, unsafe.Pointer(&arg))
	runtime.KeepAlive(&handle)
	runtime.KeepAlive(&point)
	if errors.Is(err, syscall.ETIME) || errors.Is(err, syscall.ETIMEDOUT) {
		return false, nil
	}
	if err != nil {
		return false, &Error{Op: "SYNCOBJ_TIMELINE_WAIT", Err: err}
	}
	return true, nil
}
