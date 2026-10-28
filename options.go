package hwnpu

// Placement selects how Compile treats graphs that do not run entirely on the NPU.
type Placement uint8

const (
	// PlacementStrict rejects a graph with a *PlacementError when any operation is placed off
	// the NPU or when the placement cannot be verified. This is the default.
	PlacementStrict Placement = iota
	// PlacementPrefer accepts the graph and records the result in Graph.Placement.
	PlacementPrefer
)

type options struct {
	vendor    Vendor
	backend   string
	index     int
	placement Placement
	power     PowerPolicy
	cacheDir  string
}

func defaultOptions() options { return options{index: -1} }

// Option configures Open and OpenDevice.
type Option func(*options)

// WithVendor restricts Open to devices of the vendor.
func WithVendor(v Vendor) Option { return func(o *options) { o.vendor = v } }

// WithBackend restricts Open to the named backend ("coreml", "levelzero", "xdna").
func WithBackend(name string) Option { return func(o *options) { o.backend = name } }

// WithIndex selects the n-th device among those that match the other options.
func WithIndex(n int) Option { return func(o *options) { o.index = n } }

// WithPlacement sets the placement policy used by Compile. The default is PlacementStrict.
func WithPlacement(p Placement) Option { return func(o *options) { o.placement = p } }

// WithPowerPolicy sets the power policy of the device's command queues. Devices without
// CapPowerControl ignore it. On Intel NPUs:
//
//	PowerLow:         low queue priority and the "background" workload type
//	PowerBalanced:    normal priority
//	PowerPerformance: high priority and turbo
func WithPowerPolicy(p PowerPolicy) Option { return func(o *options) { o.power = p } }

// WithCacheDir stores compiled graphs in dir and reuses them, keyed by the model, the build
// flags and the driver/compiler (or OS) version. Graph.Cached reports a hit.
func WithCacheDir(dir string) Option { return func(o *options) { o.cacheDir = dir } }
