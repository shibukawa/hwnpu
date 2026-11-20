package sysload

import (
	"context"
	"os/exec"
	"regexp"
	"strconv"
	"sync"
	"time"
	"unsafe"

	"github.com/ebitengine/purego"
)

// rusageInfoV2 is the start of struct rusage_info_v2 of <sys/resource.h>: the times are in
// mach absolute time units (mach_timebase_info), the sizes in bytes.
type rusageInfoV2 struct {
	UUID                   [16]byte
	UserTime, SystemTime   uint64
	PkgIdleWkups, IntrWkps uint64
	Pageins                uint64
	WiredSize, Resident    uint64
	PhysFootprint          uint64
	_                      [10]uint64 // proc_start_abstime … ri_diskio_byteswritten
}

const rusageInfoV2Flavor = 2

var proc struct {
	once         sync.Once
	ok           bool
	numer, denom uint64
	rusage       func(pid int32, flavor int32, buf unsafe.Pointer) int32
}

func loadProc() {
	lib, err := purego.Dlopen("/usr/lib/libSystem.B.dylib", purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return
	}
	var timebase func(tb unsafe.Pointer) int32
	for name, fn := range map[string]any{"proc_pid_rusage": &proc.rusage, "mach_timebase_info": &timebase} {
		p, err := purego.Dlsym(lib, name)
		if err != nil || p == 0 {
			return
		}
		purego.RegisterFunc(fn, p)
	}
	var tb struct{ numer, denom uint32 }
	if timebase(unsafe.Pointer(&tb)) != 0 || tb.denom == 0 {
		return
	}
	proc.numer, proc.denom = uint64(tb.numer), uint64(tb.denom)
	proc.ok = true
}

// readProcs reads proc_pid_rusage (the CPU's time, the footprint Activity Monitor shows and the
// wired memory, which the footprint leaves out for a file's pages; only
// the user's own processes without root) and the GPU time of the processes' user clients of
// the GPU. The ANE has no time of its own per process.
func readProcs(pids []int) map[int]procTimes {
	proc.once.Do(loadProc)
	gpu := gpuTimes()
	out := make(map[int]procTimes, len(pids))
	for _, pid := range pids {
		var t procTimes
		if proc.ok {
			var ri rusageInfoV2
			if proc.rusage(int32(pid), rusageInfoV2Flavor, unsafe.Pointer(&ri)) == 0 {
				ticks := ri.UserTime + ri.SystemTime
				t.cpu, t.hasCPU = time.Duration(ticks*proc.numer/proc.denom), true
				// wired pages of a mapped file are in neither the footprint nor Activity
				// Monitor's column; wired anonymous pages would count twice (rare: mlock)
				t.memory, t.wired = int64(ri.PhysFootprint+ri.WiredSize), int64(ri.WiredSize)
			}
		}
		if g, ok := gpu[pid]; ok {
			t.gpu, t.hasGPU = g, true
		}
		out[pid] = t
	}
	return out
}

var (
	ioregCreator = regexp.MustCompile(`"IOUserClientCreator" = "pid ([0-9]+),`)
	ioregGPUTime = regexp.MustCompile(`"accumulatedGPUTime"=([0-9]+)`)
)

// gpuTimes reads the user clients under the IOAccelerator objects (ioreg -l, no root): the
// AppUsage of each has the GPU time (ns) its process used through it.
func gpuTimes() map[int]time.Duration {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "ioreg", "-l", "-w", "0", "-r", "-c", "IOAccelerator").Output()
	if err != nil {
		return nil
	}
	return parseGPUTimes(string(out))
}

func parseGPUTimes(text string) map[int]time.Duration {
	times := map[int]time.Duration{}
	locs := ioregObject.FindAllStringIndex(text, -1)
	for i, l := range locs {
		end := len(text)
		if i+1 < len(locs) {
			end = locs[i+1][0]
		}
		body := text[l[1]:end]
		m := ioregCreator.FindStringSubmatch(body)
		if m == nil {
			continue
		}
		pid, _ := strconv.Atoi(m[1])
		for _, g := range ioregGPUTime.FindAllStringSubmatch(body, -1) {
			n, _ := strconv.ParseInt(g[1], 10, 64)
			times[pid] += time.Duration(n)
		}
	}
	return times
}
