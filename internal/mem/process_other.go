//go:build !darwin

package mem

// ProcessMem holds per-process memory counters (populated on macOS via task_vm_info).
type ProcessMem struct {
	PhysFootprint         int64
	PeakFootprint         int64
	External              int64
	NeuralFootprint       int64
	NeuralNoFootprint     int64
	NeuralNoFootprintPeak int64
}

// ProcessMemory is unavailable on non-Darwin platforms.
func ProcessMemory() (ProcessMem, bool) { return ProcessMem{}, false }
