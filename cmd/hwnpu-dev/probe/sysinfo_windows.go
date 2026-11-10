package probe

import (
	"os"
	"strconv"
	"syscall"
	"time"
	"unsafe"
)

type memoryStatusEx struct {
	length               uint32
	memoryLoad           uint32
	totalPhys, availPhys uint64
	totalPageFile        uint64
	availPageFile        uint64
	totalVirtual         uint64
	availVirtual         uint64
	availExtendedVirtual uint64
}

func machine() (cpu, model, osVersion string, ram int64) {
	var m memoryStatusEx
	m.length = uint32(unsafe.Sizeof(m))
	proc := syscall.NewLazyDLL("kernel32.dll").NewProc("GlobalMemoryStatusEx")
	if r, _, _ := proc.Call(uintptr(unsafe.Pointer(&m))); r != 0 {
		ram = int64(m.totalPhys)
	}
	osVersion = "Windows"
	if v, err := syscall.GetVersion(); err == nil {
		osVersion = "Windows " + strconv.Itoa(int(byte(v))) + "." + strconv.Itoa(int(uint8(v>>8))) + " build " + strconv.Itoa(int(v>>16))
	}
	return os.Getenv("PROCESSOR_IDENTIFIER"), "", osVersion, ram
}

// processMemoryCountersEx is PROCESS_MEMORY_COUNTERS_EX.
type processMemoryCountersEx struct {
	cb                         uint32
	pageFaultCount             uint32
	peakWorkingSetSize         uintptr
	workingSetSize             uintptr
	quotaPeakPagedPoolUsage    uintptr
	quotaPagedPoolUsage        uintptr
	quotaPeakNonPagedPoolUsage uintptr
	quotaNonPagedPoolUsage     uintptr
	pagefileUsage              uintptr
	peakPagefileUsage          uintptr
	privateUsage               uintptr
}

func processMemory() (processMemoryCountersEx, bool) {
	var c processMemoryCountersEx
	c.cb = uint32(unsafe.Sizeof(c))
	h, err := syscall.GetCurrentProcess()
	if err != nil {
		return c, false
	}
	proc := syscall.NewLazyDLL("kernel32.dll").NewProc("K32GetProcessMemoryInfo")
	r, _, _ := proc.Call(uintptr(h), uintptr(unsafe.Pointer(&c)), uintptr(c.cb))
	return c, r != 0
}

// maxRSS is the peak working set on Windows.
func maxRSS() int64 {
	if c, ok := processMemory(); ok {
		return int64(c.peakWorkingSetSize)
	}
	return 0
}

// privateBytes is the process's committed private memory now (where the NPU driver's copies of
// the graphs show on Windows).
func privateBytes() int64 {
	if c, ok := processMemory(); ok {
		return int64(c.privateUsage)
	}
	return 0
}

func cpuTime() time.Duration {
	var c, e, k, u syscall.Filetime
	h, err := syscall.GetCurrentProcess()
	if err != nil || syscall.GetProcessTimes(h, &c, &e, &k, &u) != nil {
		return 0
	}
	ft := func(f syscall.Filetime) int64 { return int64(f.HighDateTime)<<32 | int64(f.LowDateTime) }
	return time.Duration((ft(k) + ft(u)) * 100)
}
