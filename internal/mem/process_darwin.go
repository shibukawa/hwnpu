package mem

import (
	"encoding/binary"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// ProcessMem holds per-process memory counters from Mach task_vm_info (macOS), including the
// lifetime peak footprint (captured by the kernel even between polls) and Neural Engine memory
// accounted to the process outside phys_footprint (ledger_tag_neural_nofootprint).
type ProcessMem struct {
	PhysFootprint         int64
	PeakFootprint         int64
	External              int64 // file-backed (mmap) pages
	NeuralFootprint       int64
	NeuralNoFootprint     int64
	NeuralNoFootprintPeak int64
}

// Offsets and count for struct task_vm_info (<mach/task_info.h>, 4-byte packed, rev7).
const (
	taskVMInfoFlavor = 22
	taskVMInfoSize   = 372
	taskVMInfoCount  = taskVMInfoSize / 4

	offExternal              = 64
	offPhysFootprint         = 144
	offPeakFootprint         = 168
	offNeuralFootprint       = 304
	offNeuralNoFootprint     = 320
	offNeuralNoFootprintTot  = 356
	offNeuralNoFootprintPeak = 364
)

var taskVM struct {
	once sync.Once
	ok   bool
	task uint32
	info func(task uint32, flavor int32, out unsafe.Pointer, count *uint32) int32
}

func loadTaskVM() {
	lib, err := purego.Dlopen("/usr/lib/libSystem.B.dylib", purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return
	}
	ti, err := purego.Dlsym(lib, "task_info")
	if err != nil || ti == 0 {
		return
	}
	ts, err := purego.Dlsym(lib, "mach_task_self_")
	if err != nil || ts == 0 {
		return
	}
	purego.RegisterFunc(&taskVM.info, ti)
	taskVM.task = **(**uint32)(unsafe.Pointer(&struct{ p uintptr }{ts}))
	taskVM.ok = true
}

// ProcessMemory returns this process's Mach task_vm_info memory counters without root or cgo.
func ProcessMemory() (ProcessMem, bool) {
	taskVM.once.Do(loadTaskVM)
	if !taskVM.ok {
		return ProcessMem{}, false
	}
	var buf [taskVMInfoSize]byte
	count := uint32(taskVMInfoCount)
	if taskVM.info(taskVM.task, taskVMInfoFlavor, unsafe.Pointer(&buf[0]), &count) != 0 {
		return ProcessMem{}, false
	}
	u64 := func(off int) int64 { return int64(binary.LittleEndian.Uint64(buf[off:])) }
	gotBytes := int(count) * 4
	m := ProcessMem{
		External:          u64(offExternal),
		PhysFootprint:     u64(offPhysFootprint),
		PeakFootprint:     u64(offPeakFootprint),
		NeuralFootprint:   u64(offNeuralFootprint),
		NeuralNoFootprint: u64(offNeuralNoFootprint),
	}
	if gotBytes >= offNeuralNoFootprintPeak+8 {
		if tot := u64(offNeuralNoFootprintTot); tot > m.NeuralNoFootprint {
			m.NeuralNoFootprint = tot
		}
		m.NeuralNoFootprintPeak = u64(offNeuralNoFootprintPeak)
	}
	return m, true
}
