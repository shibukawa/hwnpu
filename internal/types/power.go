package types

import "time"

// PowerPolicy is a vendor neutral power setting. Backends map it to their own controls; a
// device without CapPowerControl ignores it.
type PowerPolicy uint8

const (
	PowerDefault     PowerPolicy = iota // driver defaults
	PowerLow                            // prefer low power over latency
	PowerBalanced                       // vendor's balanced setting
	PowerPerformance                    // prefer latency over power
)

func (p PowerPolicy) String() string {
	switch p {
	case PowerLow:
		return "low"
	case PowerBalanced:
		return "balanced"
	case PowerPerformance:
		return "performance"
	}
	return "default"
}

// PowerScope tells what a power sample covers.
type PowerScope string

const (
	ScopeNPU     PowerScope = "npu"     // the NPU alone
	ScopePackage PowerScope = "package" // the whole CPU package (cores, GPU and NPU)
)

// PowerSample is a reading of an energy counter. Average power over an interval is
// (b.EnergyJ - a.EnergyJ) / b.Time.Sub(a.Time); see AveragePower.
type PowerSample struct {
	Time    time.Time
	EnergyJ float64 // cumulative energy in joules, monotonic for one meter
	// Watts is the instantaneous power when the meter reports it (AMD sensors), else 0. Meters
	// without an energy counter integrate Watts between samples to fill EnergyJ.
	Watts  float64
	Scope  PowerScope
	Source string // e.g. "rapl:intel-rapl:0"
}
