package npuinfo

// WindowsInfo is what Windows gives. NPUs run under MCDM (the compute-only form of WDDM), so
// they show as DirectX kernel adapters, as "compute accelerator" PnP devices, and in the GPU
// performance counters that Task Manager reads. Which counter sets carry NPU engines differs
// between Windows builds, so every counter set whose name looks related is dumped as it is.
type WindowsInfo struct {
	// Devices are PnP devices of the ComputeAccelerator class, or whose name looks like an NPU.
	Devices []WindowsDevice `json:"devices,omitempty"`
	// Adapters are the DirectX kernel (D3DKMT) adapters, compute-only ones included.
	Adapters []WindowsAdapter `json:"adapters,omitempty"`
	// CounterObjects lists the performance counter sets read, with their counters and how many
	// instances each has.
	CounterObjects []WindowsCounterObject `json:"counter_objects,omitempty"`
	// Counters are the values of those counters, one per instance.
	Counters []WindowsCounter `json:"counters,omitempty"`
	// Processes names the processes the counters mention (pid_N in instance names).
	Processes []WindowsProcess `json:"processes,omitempty"`
}

type WindowsDevice struct {
	InstanceID string `json:"instance_id"`
	// Properties are the device's PnP properties by name (FriendlyName, DeviceDesc, Class,
	// Manufacturer, HardwareIds, DriverVersion, DriverDate, ...).
	Properties map[string]any `json:"properties"`
	Status     uint32         `json:"status"`
	Problem    uint32         `json:"problem"`
}

type WindowsAdapter struct {
	// LUID is written as the performance counters write it: 0xHIGH_0xLOW.
	LUID          string `json:"luid"`
	NumSources    uint32 `json:"num_sources"`
	AdapterString string `json:"adapter_string,omitempty"`
	BiosString    string `json:"bios_string,omitempty"`
	DacType       string `json:"dac_type,omitempty"`
	ChipType      string `json:"chip_type,omitempty"`
	// TypeFlags is D3DKMT_ADAPTERTYPE; TypeNames names its set bits (ComputeOnly, ...).
	TypeFlags   uint32   `json:"type_flags"`
	TypeNames   []string `json:"type_names,omitempty"`
	WDDMVersion uint32   `json:"wddm_version,omitempty"`
	VendorID    uint32   `json:"vendor_id,omitempty"`
	DeviceID    uint32   `json:"device_id,omitempty"`
	SubVendorID uint32   `json:"sub_vendor_id,omitempty"`
	SubSystemID uint32   `json:"sub_system_id,omitempty"`
	RevisionID  uint32   `json:"revision_id,omitempty"`
	BusType     uint32   `json:"bus_type,omitempty"`
	Bus         uint32   `json:"bus"`
	Device      uint32   `json:"device"`
	Function    uint32   `json:"function"`
	// Segment sizes (D3DKMT_SEGMENTSIZEINFO), in bytes.
	DedicatedVideoMemory  uint64 `json:"dedicated_video_memory"`
	DedicatedSystemMemory uint64 `json:"dedicated_system_memory"`
	SharedSystemMemory    uint64 `json:"shared_system_memory"`
	// Nodes are the adapter's engines (D3DKMT_NODEMETADATA and D3DKMT_NODE_PERFDATA).
	Nodes []WindowsNode       `json:"nodes,omitempty"`
	Perf  *WindowsAdapterPerf `json:"perf,omitempty"`
	// Errors is the status of each query that failed, by query name.
	Errors map[string]string `json:"errors,omitempty"`
}

type WindowsNode struct {
	Ordinal        uint32 `json:"ordinal"`
	EngineType     uint32 `json:"engine_type"`
	EngineTypeName string `json:"engine_type_name"`
	FriendlyName   string `json:"friendly_name"`
	Flags          uint32 `json:"flags"`
	GpuMmu         bool   `json:"gpu_mmu"`
	IoMmu          bool   `json:"io_mmu"`
	// From D3DKMT_NODE_PERFDATA (0 when the driver does not report them).
	Frequency    uint64 `json:"frequency,omitempty"`
	MaxFrequency uint64 `json:"max_frequency,omitempty"`
	Voltage      uint32 `json:"voltage,omitempty"`
}

type WindowsAdapterPerf struct {
	MemoryFrequency    uint64 `json:"memory_frequency"`
	MaxMemoryFrequency uint64 `json:"max_memory_frequency"`
	MemoryBandwidth    uint64 `json:"memory_bandwidth"`
	PCIEBandwidth      uint64 `json:"pcie_bandwidth"`
	FanRPM             uint32 `json:"fan_rpm"`
	Power              uint32 `json:"power"`
	Temperature        uint32 `json:"temperature"`
}

type WindowsCounterObject struct {
	Name      string   `json:"name"`
	Counters  []string `json:"counters"`
	Instances int      `json:"instances"`
}

type WindowsCounter struct {
	Object   string  `json:"object"`
	Counter  string  `json:"counter"`
	Instance string  `json:"instance,omitempty"`
	Value    float64 `json:"value"`
	// Status is the PDH status when the value could not be read.
	Status uint32 `json:"status,omitempty"`
	// The parts of a GPU-style instance name (pid_N_luid_0xH_0xL_phys_N_eng_N_engtype_X).
	PID     int    `json:"pid,omitempty"`
	LUID    string `json:"luid,omitempty"`
	Phys    *int   `json:"phys,omitempty"`
	Engine  *int   `json:"engine,omitempty"`
	EngType string `json:"engtype,omitempty"`
}

type WindowsProcess struct {
	PID   int    `json:"pid"`
	Image string `json:"image,omitempty"`
}
