// Package sysload reads how busy the machine is for hwnpu status: the CPU's load and the memory in
// use, and each GPU's load and memory, as the OS tells them without root.
package sysload

import "time"

// Sample is the machine at one time.
type Sample struct {
	// CPUPercent is the CPU's load since the previous Read (all cores, 0–100); nil when not known.
	CPUPercent *float64 `json:"cpu_percent,omitempty"`
	// MemoryUsed is the memory in use: Linux's MemTotal less MemAvailable, macOS' app memory,
	// wired and compressed (as Activity Monitor's "Memory Used"). 0 when not known.
	MemoryUsed  int64 `json:"memory_used_bytes,omitempty"`
	MemoryTotal int64 `json:"memory_total_bytes,omitempty"`
	GPUs        []GPU `json:"gpus,omitempty"`
}

// GPU is one GPU's load and memory.
type GPU struct {
	Name string `json:"name"`
	// BusyPercent is the driver's load (amdgpu's gpu_busy_percent, macOS' "Device Utilization
	// %"); nil when not known.
	BusyPercent *float64 `json:"busy_percent,omitempty"`
	// VRAM is the GPU's own memory (on an APU, the carve-out the firmware sets aside).
	VRAMUsed  int64 `json:"vram_used_bytes,omitempty"`
	VRAMTotal int64 `json:"vram_total_bytes,omitempty"`
	// Shared is the system memory the GPU maps (amdgpu's GTT, macOS' "In use system memory").
	SharedUsed  int64 `json:"shared_used_bytes,omitempty"`
	SharedTotal int64 `json:"shared_total_bytes,omitempty"`
}

// cpuTicks are the CPU's time counters: busy and all, in any unit.
type cpuTicks struct{ busy, total uint64 }

// Sampler reads Samples; it keeps the CPU's counters between reads for the load.
type Sampler struct {
	prev cpuTicks
	have bool
}

// firstWait is how long the first Read waits between two reads of the CPU's counters.
const firstWait = 200 * time.Millisecond

// Read reads the machine now. The CPU's load is since the previous Read; the first waits
// firstWait for it.
func (s *Sampler) Read() Sample {
	var out Sample
	t, ok := readCPU()
	if ok && !s.have {
		s.prev, s.have = t, true
		time.Sleep(firstWait)
		t, ok = readCPU()
	}
	if ok {
		if db, dt := t.busy-s.prev.busy, t.total-s.prev.total; t.total > s.prev.total && t.busy >= s.prev.busy {
			p := 100 * float64(db) / float64(dt)
			out.CPUPercent = &p
		}
		s.prev = t
	}
	out.MemoryUsed, out.MemoryTotal = readMemory()
	out.GPUs = readGPUs()
	return out
}
