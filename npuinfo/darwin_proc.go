//go:build darwin

package npuinfo

import (
	"errors"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"syscall"
	"unsafe"
)

// rusageInfoV6 is struct rusage_info_v6 (<sys/resource.h>, macOS 14 and later). The kernel
// accounts the Neural Engine's memory of each process in ri_neural_footprint: buffers the ANE
// maps, which phys_footprint (Activity Monitor's "Memory") leaves out.
type rusageInfoV6 struct {
	UUID                       [16]byte
	UserTime, SystemTime       uint64
	PkgIdleWkups               uint64
	InterruptWkups             uint64
	Pageins                    uint64
	WiredSize                  uint64
	ResidentSize               uint64
	PhysFootprint              uint64
	ProcStartAbstime           uint64
	ProcExitAbstime            uint64
	ChildUserTime              uint64
	ChildSystemTime            uint64
	ChildPkgIdleWkups          uint64
	ChildInterruptWkups        uint64
	ChildPageins               uint64
	ChildElapsedAbstime        uint64
	DiskioBytesread            uint64
	DiskioByteswritten         uint64
	CPUTimeQOS                 [7]uint64
	BilledSystemTime           uint64
	ServicedSystemTime         uint64
	LogicalWrites              uint64
	LifetimeMaxPhysFootprint   uint64
	Instructions               uint64
	Cycles                     uint64
	BilledEnergy               uint64
	ServicedEnergy             uint64
	IntervalMaxPhysFootprint   uint64
	RunnableTime               uint64
	Flags                      uint64
	UserPtime, SystemPtime     uint64
	Pinstructions, Pcycles     uint64
	EnergyNJ, PenergyNJ        uint64
	SecureTimeInSystem         uint64
	SecurePtimeInSystem        uint64
	NeuralFootprint            uint64
	LifetimeMaxNeuralFootprint uint64
	IntervalMaxNeuralFootprint uint64
	ConclaveFootprint          uint64
	PageWaitTimeMach           uint64
	PageCacheHits              uint64
	Reserved                   [6]uint64
}

const rusageInfoV6Flavor = 6

type procLib struct{ listAllPIDs, pidRusage, pidPath, name uintptr }

var loadProc = sync.OnceValues(func() (*procLib, error) {
	l, err := libSystem()
	if err != nil {
		return nil, err
	}
	p := &procLib{}
	if err := l.syms([]string{"proc_listallpids", "proc_pid_rusage", "proc_pidpath", "proc_name"},
		&p.listAllPIDs, &p.pidRusage, &p.pidPath, &p.name); err != nil {
		return nil, err
	}
	return p, nil
})

func (p *procLib) pids() ([]int32, error) {
	n := int(int32(ccall(p.listAllPIDs, 0, 0)))
	if n <= 0 {
		return nil, errors.New("proc_listallpids failed")
	}
	buf := make([]int32, n+64)
	n = int(int32(ccall(p.listAllPIDs, uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)*4))))
	runtime.KeepAlive(buf)
	if n <= 0 {
		return nil, errors.New("proc_listallpids failed")
	}
	return buf[:min(n, len(buf))], nil
}

func (p *procLib) rusage(pid int32) (*rusageInfoV6, error) {
	ri := new(rusageInfoV6)
	r := int32(ccall(p.pidRusage, uintptr(pid), rusageInfoV6Flavor, uintptr(unsafe.Pointer(ri))))
	runtime.KeepAlive(ri)
	if r != 0 {
		return nil, errors.New("proc_pid_rusage failed")
	}
	return ri, nil
}

func (p *procLib) path(pid int32) string {
	buf := make([]byte, 4*1024) // PROC_PIDPATHINFO_MAXSIZE
	if int32(ccall(p.pidPath, uintptr(pid), uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)))) <= 0 {
		return ""
	}
	return gostringN(buf)
}

func (p *procLib) procName(pid int32) string {
	buf := make([]byte, 2*16+1) // 2*MAXCOMLEN+1
	if int32(ccall(p.name, uintptr(pid), uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)))) <= 0 {
		return ""
	}
	return gostringN(buf)
}

// darwinProcesses lists the processes that hold the ANE open (clients) or have Neural Engine
// memory (ri_neural_footprint), which also finds those that reach the ANE through aned.
// Processes of other users can be read only as root; without it they show only when they are
// clients, without memory.
func darwinProcesses(clients []DarwinANEClient) ([]DarwinANEProcess, error) {
	p, err := loadProc()
	if err != nil {
		return nil, err
	}
	pids, err := p.pids()
	if err != nil {
		return nil, err
	}
	byPID := map[int32]*DarwinANEProcess{}
	get := func(pid int32) *DarwinANEProcess {
		if e := byPID[pid]; e != nil {
			return e
		}
		e := &DarwinANEProcess{PID: int(pid)}
		byPID[pid] = e
		return e
	}
	clientName := map[int32]string{}
	for _, c := range clients {
		get(int32(c.PID)).Clients++
		clientName[int32(c.PID)] = c.Process
	}
	self := int32(syscall.Getpid())
	for _, pid := range pids {
		if pid <= 0 {
			continue
		}
		ri, err := p.rusage(pid)
		if err != nil {
			continue // another user's process without root
		}
		if ri.NeuralFootprint == 0 && ri.LifetimeMaxNeuralFootprint == 0 && byPID[pid] == nil {
			continue
		}
		if pid == self && ri.NeuralFootprint == 0 && byPID[pid] == nil {
			continue
		}
		e := get(pid)
		e.NeuralFootprint = ri.NeuralFootprint
		e.LifetimeMaxNeuralFootprint = ri.LifetimeMaxNeuralFootprint
		e.PhysFootprint = ri.PhysFootprint
		e.Read = true
	}
	out := make([]DarwinANEProcess, 0, len(byPID))
	for pid, e := range byPID {
		e.Name = p.procName(pid)
		e.Path = p.path(pid)
		if e.Name == "" { // proc_name is refused for other users' processes; the path is not
			e.Name = clientName[pid]
			if e.Name == "" && e.Path != "" {
				e.Name = filepath.Base(e.Path)
			}
		}
		out = append(out, *e)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].NeuralFootprint != out[j].NeuralFootprint {
			return out[i].NeuralFootprint > out[j].NeuralFootprint
		}
		return out[i].PID < out[j].PID
	})
	return out, nil
}
