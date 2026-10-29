// Package types holds the value types shared by the public hwnpu API and its backends.
// The public package re-exports them as aliases.
package types

import "fmt"

// Vendor identifies the NPU vendor behind a device.
type Vendor uint8

const (
	VendorUnknown Vendor = iota
	VendorIntel          // Level Zero
	VendorAMD            // amdxdna
	VendorApple          // Core ML
)

func (v Vendor) String() string {
	switch v {
	case VendorIntel:
		return "intel"
	case VendorAMD:
		return "amd"
	case VendorApple:
		return "apple"
	}
	return "unknown"
}

// Capability is a bit set describing what a device supports.
type Capability uint32

const (
	CapGraph        Capability = 1 << iota // GraphRunner is available
	CapCompile                             // the driver can compile an intermediate representation
	CapKernel                              // KernelRunner is available
	CapHostCoherent                        // Buffer.Sync* are no-ops
	CapStrictVerify                        // NPU placement can be verified before running
	CapPowerControl                        // PowerPolicy has an effect
	CapPowerMeter                          // NPU power can be sampled
)

func (c Capability) String() string {
	names := []string{"graph", "compile", "kernel", "host-coherent", "strict-verify", "power-control", "power-meter"}
	s := ""
	for i, n := range names {
		if c&(1<<i) != 0 {
			if s != "" {
				s += ","
			}
			s += n
		}
	}
	return s
}

// DeviceInfo describes an enumerated device.
type DeviceInfo struct {
	Index    int
	Vendor   Vendor
	Backend  string // "levelzero" | "xdna" | "coreml"
	ID       string // backend specific identifier used by OpenDevice
	Name     string
	Path     string // device node, empty when there is none
	Caps     Capability
	Driver   string
	Firmware string
}

// DType is the element type of a graph argument.
type DType uint8

const (
	DTypeUnknown DType = iota
	F16
	F32
	F64
	BF16
	I8
	U8
	I16
	I32
	I64
)

// Size returns the element size in bytes, or 0 for DTypeUnknown.
func (d DType) Size() int {
	switch d {
	case I8, U8:
		return 1
	case F16, BF16, I16:
		return 2
	case F32, I32:
		return 4
	case F64, I64:
		return 8
	}
	return 0
}

func (d DType) String() string {
	switch d {
	case F16:
		return "f16"
	case F32:
		return "f32"
	case F64:
		return "f64"
	case BF16:
		return "bf16"
	case I8:
		return "i8"
	case U8:
		return "u8"
	case I16:
		return "i16"
	case I32:
		return "i32"
	case I64:
		return "i64"
	}
	return "unknown"
}

// ArgKind tells whether a graph argument is an input or an output.
type ArgKind uint8

const (
	ArgInput ArgKind = iota
	ArgOutput
)

func (k ArgKind) String() string {
	if k == ArgOutput {
		return "output"
	}
	return "input"
}

// ArgInfo describes one graph argument. Bytes is the buffer size Bind requires.
type ArgInfo struct {
	Index int
	Kind  ArgKind
	Name  string
	Dims  []int
	// DType is the element type of the bound buffer.
	DType DType
	// ModelDType is the element type declared by the model. It differs from DType when the
	// device converts precision, e.g. an fp32 model argument executed as fp16 on Intel NPUs.
	ModelDType DType
	Layout     string
	Bytes      int
	// PaddedDims is the shape the device actually runs when the model was padded to reach the
	// NPU (Core ML models processed by tools/coreml/ane_pad.py); nil otherwise. Callers always
	// use Dims and Bytes: hwnpu copies to and from the padded buffers.
	PaddedDims []int
}

func (a ArgInfo) String() string {
	return fmt.Sprintf("%s %d %q %s%v (%d bytes)", a.Kind, a.Index, a.Name, a.DType, a.Dims, a.Bytes)
}

// GraphFormat selects how GraphSource is interpreted.
type GraphFormat uint8

const (
	GraphFormatNative     GraphFormat = iota // vendor specific compiled or source artifact
	GraphFormatOpenVINOIR                    // xml + bin, needs a driver compiler
)

// GraphSource is the input of GraphRunner.Compile.
type GraphSource struct {
	Format     GraphFormat
	Path       string // file or directory (Core ML: .mlpackage, .mlmodel or .mlmodelc)
	Blob       []byte // in-memory native blob
	XML, Bin   []byte // OpenVINO IR
	BuildFlags string
	// Function selects one function of a Core ML ML Program with several (macOS 15 and later);
	// empty means the default function. The functions share the package's weights, and each is
	// loaded and placed as a model of its own.
	Function string
}

// OpPlacement is one operation that does not run on the NPU.
type OpPlacement struct {
	Name   string // output name of the operation
	Type   string // operator type, e.g. "ios16.linear"
	Device string // "cpu", "gpu" or "unknown"
}

// PlacementReport is the result of checking where a graph's operations run.
type PlacementReport struct {
	// Verified is true when the backend could check the placement. For Core ML the check is
	// based on MLComputePlan, which is an estimate and not a guarantee of runtime placement.
	Verified bool
	// Method names how the placement was checked, e.g. "coreml-compute-plan".
	Method string
	// Reason explains why the placement could not be verified when Verified is false.
	Reason string
	// NPUOps is the number of operations placed on the NPU.
	NPUOps int
	// Ops lists operations that are not placed on the NPU.
	Ops []OpPlacement
}

// OnNPU reports whether the whole graph was verified to run on the NPU.
func (r PlacementReport) OnNPU() bool { return r.Verified && len(r.Ops) == 0 }
