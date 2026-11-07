package mem

import (
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// aneSite is the name mach_memory_info gives the kernel memory of the Neural Engine's driver.
const aneSite = "com.apple.driver.AppleH11ANEInterface"

// memoryInfo is mach_memory_info_t of <mach_debug/zone_info.h> (176 bytes).
type memoryInfo struct {
	Flags, Site, Size, Free, Largest, Collectable, Mapped, Peak uint64
	Tag, Zone                                                   uint16
	_                                                           [2]uint16
	_                                                           [3]uint64
	Name                                                        [80]byte
}

var npu struct {
	once       sync.Once
	ok         bool
	host, task uint32
	info       func(host uint32, names *unsafe.Pointer, nn *uint32, zones *unsafe.Pointer, zn *uint32, mi *unsafe.Pointer, mn *uint32) int32
	free       func(task uint32, addr unsafe.Pointer, size uintptr) int32
}

func loadNPU() {
	lib, err := purego.Dlopen("/usr/lib/libSystem.B.dylib", purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return
	}
	var hostSelf func() uint32
	syms := map[string]uintptr{}
	for _, name := range []string{"mach_host_self", "mach_memory_info", "vm_deallocate", "mach_task_self_"} {
		p, err := purego.Dlsym(lib, name)
		if err != nil || p == 0 {
			return
		}
		syms[name] = p
	}
	purego.RegisterFunc(&hostSelf, syms["mach_host_self"])
	purego.RegisterFunc(&npu.info, syms["mach_memory_info"])
	purego.RegisterFunc(&npu.free, syms["vm_deallocate"])
	// mach_task_self() is a macro reading the variable mach_task_self_
	npu.task = **(**uint32)(unsafe.Pointer(&struct{ p uintptr }{syms["mach_task_self_"]}))
	npu.host = hostSelf()
	npu.ok = true
}

// NPUWired is the memory the kernel holds wired for the Neural Engine's driver
// (com.apple.driver.AppleH11ANEInterface in zprint's wired memory, read with mach_memory_info,
// which needs no root): where the weights of the programs loaded on the Neural Engine live. It is
// in neither the RSS nor the footprint of the process that loaded them, nor of aned. It is for
// the whole machine, and the system keeps some of it whatever runs, so a model's share is the
// difference between before it is loaded and after. ok is false where it cannot be read.
func NPUWired() (bytes int64, ok bool) {
	npu.once.Do(loadNPU)
	if !npu.ok {
		return 0, false
	}
	var names, zones, infos unsafe.Pointer
	var nn, zn, mn uint32
	if npu.info(npu.host, &names, &nn, &zones, &zn, &infos, &mn) != 0 {
		return 0, false
	}
	defer func() {
		npu.free(npu.task, names, uintptr(nn)*80)
		npu.free(npu.task, zones, uintptr(zn)*64)
		npu.free(npu.task, infos, uintptr(mn)*unsafe.Sizeof(memoryInfo{}))
	}()
	found := false
	for _, m := range unsafe.Slice((*memoryInfo)(infos), mn) {
		name := string(m.Name[:])
		if i := strings.IndexByte(name, 0); i >= 0 {
			name = name[:i]
		}
		if name == aneSite {
			bytes += int64(m.Size)
			found = true
		}
	}
	return bytes, found
}
