package npuinfo

// LinuxInfo is what Linux gives: the accel devices (/dev/accel/accel*), the processes that hold
// them open, and, on AMD APUs, the SoC metrics amdgpu reports (the NPU's activity and power).
type LinuxInfo struct {
	Devices   []LinuxAccelDevice `json:"devices"`
	Processes []LinuxProcess     `json:"processes,omitempty"`
	// UnreadableProcesses counts processes whose open files could not be read (other users'
	// processes without root), so some NPU users may be missing.
	UnreadableProcesses int             `json:"unreadable_processes,omitempty"`
	AMDGPUMetrics       []AMDGPUMetrics `json:"amdgpu_metrics,omitempty"`
}

// LinuxAccelDevice is one /sys/class/accel/accelN.
type LinuxAccelDevice struct {
	Name    string `json:"name"`     // accel0
	DevPath string `json:"dev_path"` // /dev/accel/accel0
	Driver  string `json:"driver"`   // amdxdna, intel_vpu, ...
	// PCISlot is the PCI address (0000:c4:00.1), when the device is on PCI.
	PCISlot string `json:"pci_slot,omitempty"`
	// RuntimeStatus is the runtime power state (active, suspended, ...) from
	// device/power/runtime_status, read without waking the device.
	RuntimeStatus string `json:"runtime_status,omitempty"`
	// Sysfs holds the readable text attributes of the device directory and its power/ and
	// freq/ subdirectories (PCI ids, fw_version, power_state, intel_vpu's npu_busy_time_us and
	// npu_current_frequency_mhz, ...), keyed by path relative to the device directory.
	Sysfs map[string]string `json:"sysfs,omitempty"`
	XDNA  *XDNAInfo         `json:"xdna,omitempty"`
}

// LinuxProcess is a process that holds an accel device open.
type LinuxProcess struct {
	PID     int              `json:"pid"`
	Comm    string           `json:"comm"`
	Cmdline []string         `json:"cmdline,omitempty"`
	UID     int              `json:"uid"`
	Files   []LinuxOpenAccel `json:"files"`
	// Mapped is the bytes of the device's buffers mapped into the process (the sum of the
	// /proc/PID/maps entries of the device node), by device path. NPU buffers are pinned and
	// do not show in RSS, so this is where their size shows.
	Mapped map[string]uint64 `json:"mapped,omitempty"`
	// Mappings counts those entries, by device path.
	Mappings map[string]int `json:"mappings,omitempty"`
}

// LinuxOpenAccel is one open file descriptor of an accel device.
type LinuxOpenAccel struct {
	FD      int    `json:"fd"`
	DevPath string `json:"dev_path"`
	// FDInfo is /proc/PID/fdinfo/FD as key-value pairs. Drivers that implement DRM usage stats
	// add drm-* keys (drm-engine-*, drm-total-*, drm-resident-*); amdxdna in Linux 7.0 adds
	// none.
	FDInfo map[string]string `json:"fdinfo,omitempty"`
}

// XDNAInfo is what amdxdna's DRM_IOCTL_AMDXDNA_GET_INFO and GET_ARRAY return.
type XDNAInfo struct {
	// Skipped says why the queries did not run (the NPU was asleep and Options.Wake was not
	// set). The other fields are then empty.
	Skipped        string            `json:"skipped,omitempty"`
	AIEVersion     *XDNAVersion      `json:"aie_version,omitempty"`
	AIEMetadata    *XDNAAIEMetadata  `json:"aie_metadata,omitempty"`
	Firmware       *XDNAFirmware     `json:"firmware,omitempty"`
	Clocks         []XDNAClock       `json:"clocks,omitempty"`
	PowerMode      *int              `json:"power_mode,omitempty"`
	PowerModeName  string            `json:"power_mode_name,omitempty"`
	Resource       *XDNAResourceInfo `json:"resource,omitempty"`
	Sensors        []XDNASensor      `json:"sensors,omitempty"`
	Contexts       []XDNAContext     `json:"contexts"`
	AIEStatus      *XDNAAIEStatus    `json:"aie_status,omitempty"`
	Telemetry      *XDNATelemetry    `json:"telemetry,omitempty"`
	LastAsyncError *XDNAAsyncError   `json:"last_async_error,omitempty"`
	// Errors is the error of each query that failed, by query name.
	Errors map[string]string `json:"errors,omitempty"`
}

type XDNAVersion struct {
	Major uint32 `json:"major"`
	Minor uint32 `json:"minor"`
}

type XDNAFirmware struct {
	Major uint32 `json:"major"`
	Minor uint32 `json:"minor"`
	Patch uint32 `json:"patch"`
	Build uint32 `json:"build"`
}

type XDNATileMetadata struct {
	RowCount        uint16 `json:"row_count"`
	RowStart        uint16 `json:"row_start"`
	DMAChannelCount uint16 `json:"dma_channel_count"`
	LockCount       uint16 `json:"lock_count"`
	EventRegCount   uint16 `json:"event_reg_count"`
}

type XDNAAIEMetadata struct {
	ColSize uint32           `json:"col_size"`
	Cols    uint16           `json:"cols"`
	Rows    uint16           `json:"rows"`
	Version XDNAVersion      `json:"version"`
	Core    XDNATileMetadata `json:"core"`
	Mem     XDNATileMetadata `json:"mem"`
	Shim    XDNATileMetadata `json:"shim"`
}

type XDNAClock struct {
	Name string `json:"name"`
	MHz  uint32 `json:"mhz"`
}

// XDNAResourceInfo is DRM_AMDXDNA_QUERY_RESOURCE_INFO: the NPU's capacity and what is in use.
type XDNAResourceInfo struct {
	ClockMaxMHz  uint64 `json:"clock_max_mhz"`
	TOPSMax      uint64 `json:"tops_max"`
	TasksMax     uint64 `json:"tasks_max"`
	TOPSCurrent  uint64 `json:"tops_current"`
	TasksCurrent uint64 `json:"tasks_current"`
}

type XDNASensor struct {
	Label   string `json:"label"`
	Input   uint32 `json:"input"`
	Max     uint32 `json:"max"`
	Average uint32 `json:"average"`
	Highest uint32 `json:"highest"`
	Status  string `json:"status"`
	Units   string `json:"units"`
	// UnitM scales the values: value × 10^UnitM.
	UnitM int8  `json:"unitm"`
	Type  uint8 `json:"type"` // 0 power, 1 column utilization
}

// XDNAContext is one hardware context (amdxdna_drm_hwctx_entry): which process made it, which
// columns of the array it holds, and how many commands it has run.
type XDNAContext struct {
	ContextID               uint32 `json:"context_id"`
	StartCol                uint32 `json:"start_col"`
	NumCol                  uint32 `json:"num_col"`
	HWContextID             uint32 `json:"hw_context_id"`
	PID                     int64  `json:"pid"`
	CommandSubmissions      uint64 `json:"command_submissions"`
	CommandCompletions      uint64 `json:"command_completions"`
	Migrations              uint64 `json:"migrations"`
	Preemptions             uint64 `json:"preemptions"`
	Errors                  uint64 `json:"errors"`
	Priority                uint64 `json:"priority"`
	HeapUsage               uint64 `json:"heap_usage"`
	Suspensions             uint64 `json:"suspensions"`
	State                   uint32 `json:"state"` // 0 idle, 1 active
	PASID                   uint32 `json:"pasid"`
	GOPS                    uint32 `json:"gops"`
	FPS                     uint32 `json:"fps"`
	DMABandwidth            uint32 `json:"dma_bandwidth"`
	Latency                 uint32 `json:"latency"`
	FrameExecTime           uint32 `json:"frame_exec_time"`
	TxnOpIdx                uint32 `json:"txn_op_idx"`
	ContextPC               uint32 `json:"ctx_pc"`
	FatalErrorType          uint32 `json:"fatal_error_type"`
	FatalErrorExceptionType uint32 `json:"fatal_error_exception_type"`
	FatalErrorExceptionPC   uint32 `json:"fatal_error_exception_pc"`
	FatalErrorAppModule     uint32 `json:"fatal_error_app_module"`
}

// XDNAAIEStatus is the firmware's dump of the AIE array's status registers
// (DRM_AMDXDNA_QUERY_AIE_STATUS), kept raw for now: ColSize bytes per filled column.
type XDNAAIEStatus struct {
	ColsFilled uint32 `json:"cols_filled"` // bitmap of columns
	ColSize    int    `json:"col_size"`
	Raw        []byte `json:"raw"`
}

// XDNATelemetry is DRM_AMDXDNA_QUERY_TELEMETRY, kept raw: the header, the map from firmware
// context slots to driver context ids, and the data after it.
type XDNATelemetry struct {
	Type  uint32   `json:"type"`
	Major uint32   `json:"major"`
	Minor uint32   `json:"minor"`
	Map   []uint32 `json:"map,omitempty"`
	Data  []byte   `json:"data,omitempty"`
}

type XDNAAsyncError struct {
	ErrCode   uint64 `json:"err_code"`
	TimeUS    uint64 `json:"ts_us"`
	ExErrCode uint64 `json:"ex_err_code"`
}

// AMDGPUMetrics is one amdgpu card's gpu_metrics. On AMD APUs (format 3) it carries the NPU's
// ("IPU") activity, power, memory traffic and clock, readable without root.
type AMDGPUMetrics struct {
	Card    string        `json:"card"` // card1
	Path    string        `json:"path"`
	Size    uint16        `json:"size"`
	Format  uint8         `json:"format"`
	Content uint8         `json:"content"`
	V3      *GPUMetricsV3 `json:"v3,omitempty"`
	// Raw is set for formats this package does not decode.
	Raw []byte `json:"raw,omitempty"`
}

// GPUMetricsV3 is struct gpu_metrics_v3_0 (drivers/gpu/drm/amd/include/kgd_pp_interface.h).
// Temperatures are in centidegrees C, activities in percent, powers in mW, clocks in MHz,
// reads and writes in MB/s (as the SMU reports them). 0xFFFF / 0xFFFFFFFF mean not reported.
type GPUMetricsV3 struct {
	TemperatureGfx           uint16     `json:"temperature_gfx"`
	TemperatureSoc           uint16     `json:"temperature_soc"`
	TemperatureCore          [16]uint16 `json:"temperature_core"`
	TemperatureSkin          uint16     `json:"temperature_skin"`
	AverageGfxActivity       uint16     `json:"average_gfx_activity"`
	AverageVcnActivity       uint16     `json:"average_vcn_activity"`
	AverageIPUActivity       [8]uint16  `json:"average_ipu_activity"`
	AverageCoreC0Activity    [16]uint16 `json:"average_core_c0_activity"`
	AverageDRAMReads         uint16     `json:"average_dram_reads"`
	AverageDRAMWrites        uint16     `json:"average_dram_writes"`
	AverageIPUReads          uint16     `json:"average_ipu_reads"`
	AverageIPUWrites         uint16     `json:"average_ipu_writes"`
	SystemClockCounter       uint64     `json:"system_clock_counter"`
	AverageSocketPower       uint32     `json:"average_socket_power"`
	AverageIPUPower          uint16     `json:"average_ipu_power"`
	AverageAPUPower          uint32     `json:"average_apu_power"`
	AverageGfxPower          uint32     `json:"average_gfx_power"`
	AverageDGPUPower         uint32     `json:"average_dgpu_power"`
	AverageAllCorePower      uint32     `json:"average_all_core_power"`
	AverageCorePower         [16]uint16 `json:"average_core_power"`
	AverageSysPower          uint16     `json:"average_sys_power"`
	StapmPowerLimit          uint16     `json:"stapm_power_limit"`
	CurrentStapmPowerLimit   uint16     `json:"current_stapm_power_limit"`
	AverageGfxclkFrequency   uint16     `json:"average_gfxclk_frequency"`
	AverageSocclkFrequency   uint16     `json:"average_socclk_frequency"`
	AverageVpeclkFrequency   uint16     `json:"average_vpeclk_frequency"`
	AverageIPUclkFrequency   uint16     `json:"average_ipuclk_frequency"`
	AverageFclkFrequency     uint16     `json:"average_fclk_frequency"`
	AverageVclkFrequency     uint16     `json:"average_vclk_frequency"`
	AverageUclkFrequency     uint16     `json:"average_uclk_frequency"`
	AverageMPIPUFrequency    uint16     `json:"average_mpipu_frequency"`
	CurrentCoreclk           [16]uint16 `json:"current_coreclk"`
	CurrentCoreMaxfreq       uint16     `json:"current_core_maxfreq"`
	CurrentGfxMaxfreq        uint16     `json:"current_gfx_maxfreq"`
	ThrottleResidencyProchot uint32     `json:"throttle_residency_prochot"`
	ThrottleResidencySPL     uint32     `json:"throttle_residency_spl"`
	ThrottleResidencyFPPT    uint32     `json:"throttle_residency_fppt"`
	ThrottleResidencySPPT    uint32     `json:"throttle_residency_sppt"`
	ThrottleResidencyThmCore uint32     `json:"throttle_residency_thm_core"`
	ThrottleResidencyThmGfx  uint32     `json:"throttle_residency_thm_gfx"`
	ThrottleResidencyThmSoc  uint32     `json:"throttle_residency_thm_soc"`
	TimeFilterAlphaValue     uint32     `json:"time_filter_alphavalue"`
}
