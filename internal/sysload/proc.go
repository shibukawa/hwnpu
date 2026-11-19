package sysload

import "time"

// Proc is one process's load and memory. A nil percent or a 0 size is not known (another
// user's process without root, a driver that does not tell).
type Proc struct {
	CPUPercent *float64 `json:"cpu_percent,omitempty"` // of one core: 200 is two cores busy
	// Memory is what the process holds: on macOS its footprint and its wired memory, on Linux
	// its resident memory.
	Memory int64 `json:"memory_bytes,omitempty"`
	// Wired is the memory it wired (mlock: macOS), counted in Memory. A file's pages wired
	// (gpt-oss's experts, mapped from the checkpoint) are not in the footprint Activity Monitor
	// shows: an M3 running gpt-oss-20b, footprint 0.53 GB and 6.41 GB wired.
	Wired      int64    `json:"wired_bytes,omitempty"`
	GPUPercent *float64 `json:"gpu_percent,omitempty"`
	GPUMemory  int64    `json:"gpu_memory_bytes,omitempty"`
	NPUPercent *float64 `json:"npu_percent,omitempty"`
}

// procTimes are what a process's load is worked out from: its time on the CPU, the GPU and
// the NPU (has* false when not known), and its memory.
type procTimes struct {
	cpu, gpu, npu          time.Duration
	hasCPU, hasGPU, hasNPU bool
	memory, gpuMemory      int64
	wired                  int64
}

// ProcSampler reads the load of processes; it keeps their times between reads.
type ProcSampler struct {
	prev map[int]procTimes
	at   time.Time
}

// Read reads the processes pids now. The loads are since the previous Read (for a process
// seen then); the first waits firstWait for them.
func (s *ProcSampler) Read(pids []int) map[int]Proc {
	if len(pids) == 0 {
		return nil
	}
	now := readProcs(pids)
	at := time.Now()
	if s.prev == nil {
		s.prev, s.at = now, at
		time.Sleep(firstWait)
		now, at = readProcs(pids), time.Now()
	}
	dt := at.Sub(s.at)
	out := make(map[int]Proc, len(now))
	for pid, t := range now {
		p := Proc{Memory: t.memory, Wired: t.wired, GPUMemory: t.gpuMemory}
		if old, ok := s.prev[pid]; ok && dt > 0 {
			p.CPUPercent = rate(t.hasCPU && old.hasCPU, t.cpu-old.cpu, dt)
			p.GPUPercent = rate(t.hasGPU && old.hasGPU, t.gpu-old.gpu, dt)
			p.NPUPercent = rate(t.hasNPU && old.hasNPU, t.npu-old.npu, dt)
		}
		out[pid] = p
	}
	s.prev, s.at = now, at
	return out
}

func rate(ok bool, d, over time.Duration) *float64 {
	if !ok || d < 0 {
		return nil
	}
	p := 100 * d.Seconds() / over.Seconds()
	return &p
}
