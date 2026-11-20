package sysload

import (
	"context"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/ebitengine/purego"

	"github.com/shibukawa/hwnpu/internal/mem"
)

// The flavors of host_statistics (<mach/host_info.h>) and their sizes in 32-bit words.
const (
	hostCPULoadInfo = 3 // host_cpu_load_info: cpu_ticks[user, system, idle, nice]
	hostVMInfo64    = 4 // vm_statistics64
	cpuLoadCount    = 4
	vmInfo64Count   = 38
)

// vmStatistics64 is struct vm_statistics64 of <mach/vm_statistics.h> (152 bytes).
type vmStatistics64 struct {
	Free, Active, Inactive, Wire                            uint32
	ZeroFill, Reactivations, Pageins, Pageouts, Faults, COW uint64
	Lookups, Hits, Purges                                   uint64
	Purgeable, Speculative                                  uint32
	Decompressions, Compressions, Swapins, Swapouts         uint64
	Compressor, Throttled, External, Internal               uint32
	UncompressedInCompressor                                uint64
}

var mach struct {
	once   sync.Once
	ok     bool
	host   uint32
	page   uint64
	stats  func(host uint32, flavor int32, info unsafe.Pointer, count *uint32) int32
	stats6 func(host uint32, flavor int32, info unsafe.Pointer, count *uint32) int32
}

func loadMach() {
	lib, err := purego.Dlopen("/usr/lib/libSystem.B.dylib", purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return
	}
	var hostSelf func() uint32
	var pageSize func(host uint32, size *uintptr) int32
	for name, fn := range map[string]any{
		"mach_host_self": &hostSelf, "host_statistics": &mach.stats,
		"host_statistics64": &mach.stats6, "host_page_size": &pageSize,
	} {
		p, err := purego.Dlsym(lib, name)
		if err != nil || p == 0 {
			return
		}
		purego.RegisterFunc(fn, p)
	}
	mach.host = hostSelf()
	var ps uintptr
	if pageSize(mach.host, &ps) != 0 || ps == 0 {
		return
	}
	mach.page = uint64(ps)
	mach.ok = true
}

func readCPU() (cpuTicks, bool) {
	mach.once.Do(loadMach)
	if !mach.ok {
		return cpuTicks{}, false
	}
	var t [cpuLoadCount]uint32
	n := uint32(cpuLoadCount)
	if mach.stats(mach.host, hostCPULoadInfo, unsafe.Pointer(&t), &n) != 0 {
		return cpuTicks{}, false
	}
	return cpuTicks{busy: uint64(t[0]) + uint64(t[1]) + uint64(t[3]), total: uint64(t[0]) + uint64(t[1]) + uint64(t[2]) + uint64(t[3])}, true
}

// readMemory is Activity Monitor's "Memory Used": app memory (anonymous pages less the
// purgeable ones), wired and compressed.
func readMemory() (used, total int64) {
	total = mem.Physical()
	mach.once.Do(loadMach)
	if !mach.ok {
		return 0, total
	}
	var v vmStatistics64
	n := uint32(vmInfo64Count)
	if mach.stats6(mach.host, hostVMInfo64, unsafe.Pointer(&v), &n) != 0 {
		return 0, total
	}
	pages := uint64(v.Internal) - uint64(min(v.Purgeable, v.Internal)) + uint64(v.Wire) + uint64(v.Compressor)
	return int64(pages * mach.page), total
}

var (
	ioregObject = regexp.MustCompile(`\+-o (\S+)`)
	ioregModel  = regexp.MustCompile(`"model" = "([^"]*)"`)
	ioregBusy   = regexp.MustCompile(`"Device Utilization %"=([0-9]+)`)
	ioregInUse  = regexp.MustCompile(`"In use system memory"=([0-9]+)`)
)

// readGPUs reads the PerformanceStatistics of the IOAccelerator objects in the I/O Registry
// (ioreg, no root): the GPU's load and the system memory it holds (the GPU has no memory of its
// own on Apple silicon).
func readGPUs() []GPU {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator").Output()
	if err != nil {
		return nil
	}
	return parseIOAccelerators(string(out))
}

func parseIOAccelerators(text string) []GPU {
	var gpus []GPU
	locs := ioregObject.FindAllStringSubmatchIndex(text, -1)
	for i, l := range locs {
		end := len(text)
		if i+1 < len(locs) {
			end = locs[i+1][0]
		}
		body := text[l[1]:end]
		g := GPU{Name: text[l[2]:l[3]]}
		if m := ioregModel.FindStringSubmatch(body); m != nil {
			g.Name = m[1]
		}
		found := false
		if m := ioregBusy.FindStringSubmatch(body); m != nil {
			p, _ := strconv.ParseFloat(m[1], 64)
			g.BusyPercent, found = &p, true
		}
		if m := ioregInUse.FindStringSubmatch(body); m != nil {
			g.SharedUsed, _ = strconv.ParseInt(m[1], 10, 64)
			found = true
		}
		if found {
			g.Name = strings.TrimSpace(g.Name)
			gpus = append(gpus, g)
		}
	}
	return gpus
}
