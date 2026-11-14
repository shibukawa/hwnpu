package npuinfo

import "time"

// DarwinInfo is what macOS gives about the Neural Engine. Activity Monitor shows nothing of it;
// the I/O Registry has the ANE driver's objects, and the user clients under them say which
// processes opened the ANE ("IOUserClientCreator"). The kernel accounts each process's ANE
// memory (proc_pid_rusage), and IOReport, the private library behind powermetrics, has the
// ANE's energy and memory traffic without root.
type DarwinInfo struct {
	Model   string `json:"model,omitempty"`   // hw.model
	CPU     string `json:"cpu,omitempty"`     // machdep.cpu.brand_string
	Version string `json:"version,omitempty"` // kern.osproductversion
	// Device is the ANE as the driver describes it (architecture, cores, power state).
	Device *DarwinANEDevice `json:"device,omitempty"`
	// Processes are the processes that hold the ANE open or have ANE memory.
	Processes []DarwinANEProcess `json:"processes,omitempty"`
	// IOReport has the ANE's power, memory traffic and interrupts over Options.SampleInterval.
	IOReport *DarwinIOReport `json:"ioreport,omitempty"`
	// ANEClasses are the I/O Registry classes found whose name contains "ANE".
	ANEClasses []string `json:"ane_classes,omitempty"`
	// ANE are the I/O Registry objects of those classes, with properties and children.
	ANE []DarwinIONode `json:"ane,omitempty"`
	// Clients are the user clients under those objects: the processes holding the ANE open.
	Clients []DarwinANEClient `json:"clients,omitempty"`
	// Caches are the compiled-program caches of the ANE (com.apple.e5rt.e5bundlecache).
	Caches []DarwinCache `json:"caches,omitempty"`
	// PowerMetrics is powermetrics' plist (cpu_power sampler, which has ane_power); only as root.
	PowerMetrics map[string]any `json:"powermetrics,omitempty"`
}

type DarwinIONode struct {
	Class      string         `json:"class"`
	Name       string         `json:"name,omitempty"`
	Properties map[string]any `json:"properties,omitempty"`
	Children   []DarwinIONode `json:"children,omitempty"`
}

type DarwinANEClient struct {
	Class string `json:"class"`
	PID   int    `json:"pid"`
	// Process is the name in IOUserClientCreator (truncated by the kernel).
	Process string `json:"process"`
	// Creator is IOUserClientCreator as it is ("pid 123, python3.12").
	Creator string `json:"creator"`
}

type DarwinCache struct {
	Path string `json:"path"`
	// Owner is the directory above the cache: the executable or bundle id it belongs to.
	Owner   string    `json:"owner"`
	Bytes   int64     `json:"bytes"`
	Files   int       `json:"files"`
	Newest  time.Time `json:"newest,omitzero"`
	Entries []string  `json:"entries,omitempty"` // first-level directories (OS builds)
}

// DarwinANEDevice is the ANE driver's DeviceProperties and power state (H11ANEIn).
type DarwinANEDevice struct {
	Class string `json:"class"` // H11ANEIn
	// Compatible is the device tree name ("ane,t8020" on M1).
	Compatible string `json:"compatible,omitempty"`
	// Architecture is ANEDevicePropertyTypeANEArchitectureTypeStr ("h13g" on M1).
	Architecture string `json:"architecture,omitempty"`
	Cores        int    `json:"cores,omitempty"`
	Version      int    `json:"version,omitempty"`
	MinorVersion int    `json:"minor_version,omitempty"`
	BoardType    int    `json:"board_type,omitempty"`
	BoardSubType int    `json:"board_sub_type,omitempty"`
	CPUSubType   int    `json:"cpu_sub_type,omitempty"`
	// FirmwareLoaded is false until something has used the ANE since boot.
	FirmwareLoaded bool `json:"firmware_loaded"`
	// PowerState and MaxPowerState are the driver's IOPowerManagement states.
	PowerState    int `json:"power_state"`
	MaxPowerState int `json:"max_power_state"`
}

// DarwinANEProcess is one process that uses the ANE.
type DarwinANEProcess struct {
	PID  int    `json:"pid"`
	Name string `json:"name,omitempty"`
	Path string `json:"path,omitempty"`
	// Clients is how many ANE user clients the process holds (0 when it goes through aned).
	Clients int `json:"clients"`
	// Read is false when proc_pid_rusage was refused (another user's process, not root); the
	// sizes below are then unknown.
	Read bool `json:"read"`
	// NeuralFootprint is the ANE memory the kernel accounts to the process now, in bytes; it
	// is not in PhysFootprint.
	NeuralFootprint            uint64 `json:"neural_footprint"`
	LifetimeMaxNeuralFootprint uint64 `json:"lifetime_max_neural_footprint"`
	// PhysFootprint is the memory Activity Monitor shows.
	PhysFootprint uint64 `json:"phys_footprint"`
}

// DarwinIOReport is the change of the ANE's IOReport channels over Seconds. The fields above
// Channels are worked out from them; a zero one is a channel the chip does not have.
type DarwinIOReport struct {
	Seconds float64 `json:"seconds"`
	// ANEPowerW is the ANE's average power (from "PMP"/"Energy Counters"/"ANE" on M1, "Energy
	// Model"/"ANE…" on later chips).
	ANEPowerW float64 `json:"ane_power_w"`
	// ANEReadBytesPerSec and ANEWriteBytesPerSec are the ANE's DRAM traffic (memory
	// controller counters "AMC Stats"/"Perf Counters"/"ANE RD", "ANE WR").
	ANEReadBytesPerSec  float64 `json:"ane_read_bytes_per_sec"`
	ANEWriteBytesPerSec float64 `json:"ane_write_bytes_per_sec"`
	// ANEInterruptsPerSec is the ANE driver's interrupts ("Interrupt Statistics (by index)"/
	// "ane 0"); a program that finishes raises one.
	ANEInterruptsPerSec float64                 `json:"ane_interrupts_per_sec"`
	Channels            []DarwinIOReportChannel `json:"channels,omitempty"`
}

type DarwinIOReportChannel struct {
	Group    string `json:"group"`
	SubGroup string `json:"subgroup,omitempty"`
	Name     string `json:"name"`
	Unit     string `json:"unit,omitempty"`
	Format   string `json:"format"` // simple, state, histogram, simple_array
	// Value is the change of a simple channel.
	Value *int64 `json:"value,omitempty"`
	// States are the time (in Unit) a state channel spent in each state.
	States []DarwinIOReportState `json:"states,omitempty"`
}

type DarwinIOReportState struct {
	Name      string `json:"name"`
	Residency int64  `json:"residency"`
}
