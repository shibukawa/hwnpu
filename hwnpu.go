// Package hwnpu runs compiled models on the NPU built into the CPU package, without cgo.
//
// Backends:
//   - Apple (macOS): Core ML through purego/objc. Core ML cannot be restricted to the Neural
//     Engine, so hwnpu disables the GPU and checks placement with MLComputePlan before running.
//   - Intel (Linux, Windows): Level Zero and the NPU graph extension through purego.
//   - AMD Ryzen AI (Linux): the amdxdna driver through ioctl, running xclbin kernels.
//
// By default hwnpu refuses graphs that do not run entirely on the NPU (PlacementStrict), so that
// a model never silently falls back to the CPU or GPU.
package hwnpu

import (
	"errors"

	"github.com/shibukawa/hwnpu/internal/types"
)

type (
	Vendor          = types.Vendor
	Capability      = types.Capability
	DeviceInfo      = types.DeviceInfo
	DType           = types.DType
	ArgKind         = types.ArgKind
	ArgInfo         = types.ArgInfo
	GraphFormat     = types.GraphFormat
	GraphSource     = types.GraphSource
	OpPlacement     = types.OpPlacement
	PlacementReport = types.PlacementReport
	PowerPolicy     = types.PowerPolicy
	PowerScope      = types.PowerScope
	PowerSample     = types.PowerSample
)

const (
	VendorUnknown = types.VendorUnknown
	VendorIntel   = types.VendorIntel
	VendorAMD     = types.VendorAMD
	VendorApple   = types.VendorApple

	CapGraph        = types.CapGraph
	CapCompile      = types.CapCompile
	CapKernel       = types.CapKernel
	CapHostCoherent = types.CapHostCoherent
	CapStrictVerify = types.CapStrictVerify
	CapPowerControl = types.CapPowerControl
	CapPowerMeter   = types.CapPowerMeter

	DTypeUnknown = types.DTypeUnknown
	F16          = types.F16
	F32          = types.F32
	F64          = types.F64
	BF16         = types.BF16
	I8           = types.I8
	U8           = types.U8
	I16          = types.I16
	I32          = types.I32
	I64          = types.I64

	ArgInput  = types.ArgInput
	ArgOutput = types.ArgOutput

	GraphFormatNative     = types.GraphFormatNative
	GraphFormatOpenVINOIR = types.GraphFormatOpenVINOIR

	PowerDefault     = types.PowerDefault
	PowerLow         = types.PowerLow
	PowerBalanced    = types.PowerBalanced
	PowerPerformance = types.PowerPerformance

	ScopeNPU     = types.ScopeNPU
	ScopePackage = types.ScopePackage
)

// AveragePower returns the mean power in watts between two samples of the same meter.
func AveragePower(a, b PowerSample) float64 {
	dt := b.Time.Sub(a.Time).Seconds()
	if dt <= 0 {
		return 0
	}
	return (b.EnergyJ - a.EnergyJ) / dt
}

type (
	PlacementError = types.PlacementError
	DriverError    = types.DriverError
	BuildError     = types.BuildError
	VersionError   = types.VersionError
)

var (
	ErrNoDevice    = errors.New("hwnpu: no NPU device found")
	ErrUnsupported = types.ErrUnsupported
	ErrLibrary     = types.ErrLibrary
	ErrNotOnNPU    = types.ErrNotOnNPU
	ErrClosed      = errors.New("hwnpu: already closed")
)
