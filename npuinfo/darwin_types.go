package npuinfo

import "time"

// DarwinInfo is what macOS gives about the Neural Engine. Activity Monitor shows nothing of it;
// the I/O Registry has the ANE driver's objects, and the user clients under them say which
// processes opened the ANE ("IOUserClientCreator"). The ANE's power needs root (powermetrics)
// or a private framework (IOReport), so only the first is read for now.
type DarwinInfo struct {
	Model   string `json:"model,omitempty"`   // hw.model
	CPU     string `json:"cpu,omitempty"`     // machdep.cpu.brand_string
	Version string `json:"version,omitempty"` // kern.osproductversion
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
