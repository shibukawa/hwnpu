package ze

import (
	"fmt"
	"strings"
	"structs"
	"unsafe"

	"github.com/shibukawa/hwnpu/internal/dl"
)

// GraphExtName is ZE_GRAPH_EXT_NAME. Older drivers publish versioned names such as
// "ZE_extension_graph_1_2".
const GraphExtName = "ZE_extension_graph"

// Graph extension versions.
var (
	GraphExtMinVersion    = MakeVersion(1, 5)  // pfnCreate2 and ze_graph_desc_2_t
	GraphExtTargetVersion = MakeVersion(1, 20) // newest version this binding knows
)

// Graph extension constants.
const (
	GraphFormatNative     = 0x1
	GraphFormatNGraphLite = 0x2
	stypeDeviceGraphProps = 0x1
	stypeGraphDesc        = 0x2
	stypeGraphProperties  = 0x3
	stypeGraphDesc2       = 0xE
	stypeGraphProperties2 = 0x10
	stypeGraphArgProps3   = 0xD
	ArgTypeInput          = 0
	ArgTypeOutput         = 1
	StageCommandListInit  = 0x1
	StageInitialize       = 0x2
	GraphFlagNone         = 0x0
	GraphFlagDisableCache = 1 << 0
	maxGraphArgName       = 256
	maxGraphArgDims       = 5
	maxGraphTensorNames   = 32
)

// Argument precisions (ze_graph_argument_precision_t).
const (
	PrecisionUnknown = 0x00
	PrecisionFP32    = 0x01
	PrecisionFP16    = 0x02
	PrecisionUINT16  = 0x03
	PrecisionUINT8   = 0x04
	PrecisionINT32   = 0x05
	PrecisionINT16   = 0x06
	PrecisionINT8    = 0x07
	PrecisionBIN     = 0x08
	PrecisionBF16    = 0x09
	PrecisionUINT32  = 0x0A
	PrecisionUINT4   = 0x0B
	PrecisionINT4    = 0x0C
	PrecisionFP64    = 0x0F
	PrecisionUINT64  = 0x10
	PrecisionINT64   = 0x11
)

// CompilerVersion is ze_graph_compiler_version_info_t.
type CompilerVersion struct {
	_     structs.HostLayout
	Major uint16
	Minor uint16
}

func (v CompilerVersion) String() string { return fmt.Sprintf("%d.%d", v.Major, v.Minor) }

// DeviceGraphProperties is ze_device_graph_properties_t.
type DeviceGraphProperties struct {
	_                     structs.HostLayout
	SType                 uint32
	PNext                 uintptr
	GraphExtensionVersion uint32
	CompilerVersion       CompilerVersion
	GraphFormatsSupported uint32
	MaxOVOpsetVersion     uint32
}

// GraphDesc2 is ze_graph_desc_2_t.
type GraphDesc2 struct {
	_          structs.HostLayout
	SType      uint32
	PNext      uintptr
	Format     uint32
	InputSize  uintptr
	Input      unsafe.Pointer
	BuildFlags unsafe.Pointer
	Flags      uint32
}

// GraphProperties2 is ze_graph_properties_2_t (ze_graph_properties_t is its prefix).
type GraphProperties2 struct {
	_                 structs.HostLayout
	SType             uint32
	PNext             uintptr
	NumGraphArgs      uint32
	InitStageRequired uint32
}

// GraphArgumentProperties3 is ze_graph_argument_properties_3_t.
type GraphArgumentProperties3 struct {
	_                          structs.HostLayout
	SType                      uint32
	PNext                      uintptr
	Name                       [maxGraphArgName]byte
	Type                       uint32
	Dims                       [maxGraphArgDims]uint32
	NetworkPrecision           uint32
	NetworkLayout              uint32
	DevicePrecision            uint32
	DeviceLayout               uint32
	QuantReverseScale          float32
	QuantZeroPoint             uint8
	DimsCount                  uint32
	DebugFriendlyName          [maxGraphArgName]byte
	AssociatedTensorNames      [maxGraphTensorNames][maxGraphArgName]byte
	AssociatedTensorNamesCount uint32
}

// graphDDICount maps an extension version to the number of entries in ze_graph_dditable_ext_t.
var graphDDICount = []struct{ version, count uint32 }{
	{MakeVersion(1, 20), 33}, {MakeVersion(1, 16), 31}, {MakeVersion(1, 15), 30},
	{MakeVersion(1, 12), 29}, {MakeVersion(1, 11), 25}, {MakeVersion(1, 8), 23},
	{MakeVersion(1, 7), 21}, {MakeVersion(1, 6), 20}, {MakeVersion(1, 5), 19},
	{MakeVersion(1, 4), 16}, {MakeVersion(1, 3), 15}, {MakeVersion(1, 2), 12},
	{MakeVersion(1, 1), 11}, {MakeVersion(1, 0), 9},
}

// Indices into ze_graph_dditable_ext_t.
const (
	ddiCreate                   = 0
	ddiDestroy                  = 1
	ddiGetProperties            = 2
	ddiSetArgumentValue         = 4
	ddiAppendGraphInitialize    = 5
	ddiAppendGraphExecute       = 6
	ddiGetNativeBinary          = 7
	ddiDeviceGetGraphProperties = 8
	ddiGetArgumentProperties3   = 11
	ddiQueryNetworkDestroy      = 13
	ddiQueryNetworkGetSupported = 14
	ddiBuildLogGetString        = 15
	ddiCreate2                  = 16
	ddiQueryNetworkCreate2      = 17
	ddiGetProperties2           = 21
	ddiGraphInitialize          = 22
)

// GraphExt is a bound graph extension table.
type GraphExt struct {
	Name    string // extension name as published by the driver
	Version uint32 // negotiated version

	pfnCreate2                func(ctx, dev Handle, desc *GraphDesc2, graph *Handle) Result
	pfnDestroy                func(graph Handle) Result
	pfnGetProperties          func(graph Handle, props *GraphProperties2) Result
	pfnGetArgumentProperties3 func(graph Handle, index uint32, props *GraphArgumentProperties3) Result
	pfnSetArgumentValue       func(graph Handle, index uint32, value unsafe.Pointer) Result
	pfnAppendGraphInitialize  func(list, graph, signal Handle, nWait uint32, wait *Handle) Result
	pfnAppendGraphExecute     func(list, graph, profiling, signal Handle, nWait uint32, wait *Handle) Result
	pfnGetNativeBinary        func(graph Handle, size *uintptr, data *byte) Result
	pfnDeviceGetGraphProps    func(dev Handle, props *DeviceGraphProperties) Result
	pfnQueryNetworkCreate2    func(ctx, dev Handle, desc *GraphDesc2, query *Handle) Result
	pfnQueryNetworkDestroy    func(query Handle) Result
	pfnQueryNetworkGetSupport func(query Handle, size *uintptr, layers *byte) Result
	pfnBuildLogGetString      func(graph Handle, size *uint32, log *byte) Result
	pfnGetProperties2         func(graph Handle, props *GraphProperties2) Result
	pfnGraphInitialize        func(graph Handle) Result
}

// matchesGraphExt implements the name matching of the OpenVINO NPU plugin: the exact name, or
// the name followed by "_<digit>" for older drivers.
func matchesGraphExt(name string) bool {
	if name == GraphExtName {
		return true
	}
	rest, ok := strings.CutPrefix(name, GraphExtName+"_")
	return ok && rest != "" && rest[0] >= '0' && rest[0] <= '9'
}

// SelectGraphExt picks the extension name and version to use, following OpenVINO: the first
// entry at or above the target version wins with the target version, otherwise the highest.
func SelectGraphExt(props []DriverExtensionProperties) (name string, version uint32) {
	for _, p := range props {
		n := CString(p.Name[:])
		if !matchesGraphExt(n) {
			continue
		}
		if p.Version >= GraphExtTargetVersion {
			return n, GraphExtTargetVersion
		}
		if p.Version > version {
			name, version = n, p.Version
		}
	}
	return name, version
}

// LoadGraphExt negotiates and binds the graph extension of a driver.
func (a *API) LoadGraphExt(driver Handle) (*GraphExt, error) {
	props, err := a.Extensions(driver)
	if err != nil {
		return nil, err
	}
	name, version := SelectGraphExt(props)
	if name == "" {
		return nil, fmt.Errorf("driver does not publish %s", GraphExtName)
	}
	if version>>16 != GraphExtTargetVersion>>16 || version < GraphExtMinVersion {
		return nil, fmt.Errorf("%s version %s is not supported (need %s or later within major version %d)",
			name, VersionString(version), VersionString(GraphExtMinVersion), GraphExtTargetVersion>>16)
	}
	table, err := a.ExtensionFunction(driver, name)
	if err != nil {
		return nil, err
	}
	count := uint32(0)
	for _, c := range graphDDICount {
		if version >= c.version {
			count = c.count
			break
		}
	}
	entries := unsafe.Slice((*uintptr)(table), count)
	g := &GraphExt{Name: name, Version: version}
	bind := func(fp any, i int) error {
		if i >= len(entries) {
			return nil // not available in this version
		}
		if entries[i] == 0 {
			return nil
		}
		return dl.BindAddr(fp, entries[i])
	}
	for _, b := range []struct {
		fp any
		i  int
	}{
		{&g.pfnCreate2, ddiCreate2}, {&g.pfnDestroy, ddiDestroy}, {&g.pfnGetProperties, ddiGetProperties},
		{&g.pfnGetArgumentProperties3, ddiGetArgumentProperties3}, {&g.pfnSetArgumentValue, ddiSetArgumentValue},
		{&g.pfnAppendGraphInitialize, ddiAppendGraphInitialize}, {&g.pfnAppendGraphExecute, ddiAppendGraphExecute},
		{&g.pfnGetNativeBinary, ddiGetNativeBinary}, {&g.pfnDeviceGetGraphProps, ddiDeviceGetGraphProperties},
		{&g.pfnQueryNetworkCreate2, ddiQueryNetworkCreate2}, {&g.pfnQueryNetworkDestroy, ddiQueryNetworkDestroy},
		{&g.pfnQueryNetworkGetSupport, ddiQueryNetworkGetSupported}, {&g.pfnBuildLogGetString, ddiBuildLogGetString},
		{&g.pfnGetProperties2, ddiGetProperties2}, {&g.pfnGraphInitialize, ddiGraphInitialize},
	} {
		if err := bind(b.fp, b.i); err != nil {
			return nil, fmt.Errorf("%s entry %d: %w", name, b.i, err)
		}
	}
	for _, req := range []struct {
		ok   bool
		name string
	}{
		{g.pfnCreate2 != nil, "pfnCreate2"}, {g.pfnDestroy != nil, "pfnDestroy"},
		{g.pfnGetArgumentProperties3 != nil, "pfnGetArgumentProperties3"},
		{g.pfnSetArgumentValue != nil, "pfnSetArgumentValue"},
		{g.pfnAppendGraphExecute != nil, "pfnAppendGraphExecute"},
		{g.pfnDeviceGetGraphProps != nil, "pfnDeviceGetGraphProperties"},
	} {
		if !req.ok {
			return nil, fmt.Errorf("%s %s: %s is missing", name, VersionString(version), req.name)
		}
	}
	return g, nil
}

// DeviceGraphProperties queries compiler and format support of a device.
func (g *GraphExt) DeviceGraphProperties(dev Handle) (DeviceGraphProperties, error) {
	p := DeviceGraphProperties{SType: stypeDeviceGraphProps}
	err := check("pfnDeviceGetGraphProperties", g.pfnDeviceGetGraphProps(dev, &p))
	return p, err
}

// Create compiles or loads a graph. input must stay alive until Create returns.
// On failure the error carries the driver build log when available.
func (g *GraphExt) Create(ctx, dev Handle, format uint32, input []byte, buildFlags string, flags uint32) (Handle, error) {
	desc, keep := graphDesc(format, input, buildFlags, flags)
	var h Handle
	r := g.pfnCreate2(ctx, dev, &desc, &h)
	runtimeKeepAlive(keep)
	if r != ResultSuccess {
		return 0, &Error{Op: "pfnCreate2", Result: r, Log: g.buildLog(0)}
	}
	return h, nil
}

func graphDesc(format uint32, input []byte, buildFlags string, flags uint32) (GraphDesc2, [][]byte) {
	cflags := append([]byte(buildFlags), 0)
	desc := GraphDesc2{SType: stypeGraphDesc2, Format: format, InputSize: uintptr(len(input)), BuildFlags: unsafe.Pointer(&cflags[0]), Flags: flags}
	if len(input) > 0 {
		desc.Input = unsafe.Pointer(&input[0])
	}
	return desc, [][]byte{cflags, input}
}

// buildLog returns the driver's latest build log (graph may be 0).
func (g *GraphExt) buildLog(graph Handle) string {
	if g.pfnBuildLogGetString == nil {
		return ""
	}
	var n uint32
	if g.pfnBuildLogGetString(graph, &n, nil) != ResultSuccess || n == 0 {
		return ""
	}
	buf := make([]byte, n)
	if g.pfnBuildLogGetString(graph, &n, &buf[0]) != ResultSuccess {
		return ""
	}
	return strings.TrimSpace(CString(buf))
}

func (g *GraphExt) Destroy(h Handle) error { return check("pfnDestroy", g.pfnDestroy(h)) }

// Properties returns the argument count and the initialization stages a graph requires.
// Before version 1.8 only pfnAppendGraphInitialize exists.
func (g *GraphExt) Properties(h Handle) (args uint32, stages uint32, err error) {
	if g.pfnGetProperties2 != nil {
		p := GraphProperties2{SType: stypeGraphProperties2}
		err = check("pfnGetProperties2", g.pfnGetProperties2(h, &p))
		return p.NumGraphArgs, p.InitStageRequired, err
	}
	p := GraphProperties2{SType: stypeGraphProperties}
	err = check("pfnGetProperties", g.pfnGetProperties(h, &p))
	return p.NumGraphArgs, StageCommandListInit, err
}

// ArgumentProperties returns ze_graph_argument_properties_3_t of one argument.
func (g *GraphExt) ArgumentProperties(h Handle, i uint32) (*GraphArgumentProperties3, error) {
	p := &GraphArgumentProperties3{SType: stypeGraphArgProps3}
	return p, check("pfnGetArgumentProperties3", g.pfnGetArgumentProperties3(h, i, p))
}

// SetArgument records the device visible address of an argument. The value is captured when
// the graph is appended to a command list.
func (g *GraphExt) SetArgument(h Handle, i uint32, p unsafe.Pointer) error {
	return check("pfnSetArgumentValue", g.pfnSetArgumentValue(h, i, p))
}

// Initialize runs pfnGraphInitialize (version 1.8+).
func (g *GraphExt) Initialize(h Handle) error {
	if g.pfnGraphInitialize == nil {
		return fmt.Errorf("pfnGraphInitialize is not available in %s", VersionString(g.Version))
	}
	return check("pfnGraphInitialize", g.pfnGraphInitialize(h))
}

func (g *GraphExt) AppendInitialize(list, h Handle) error {
	return check("pfnAppendGraphInitialize", g.pfnAppendGraphInitialize(list, h, 0, 0, nil))
}

func (g *GraphExt) AppendExecute(list, h Handle) error {
	return check("pfnAppendGraphExecute", g.pfnAppendGraphExecute(list, h, 0, 0, 0, nil))
}

// NativeBinary returns a copy of the compiled blob.
func (g *GraphExt) NativeBinary(h Handle) ([]byte, error) {
	var n uintptr
	if err := check("pfnGetNativeBinary", g.pfnGetNativeBinary(h, &n, nil)); err != nil {
		return nil, err
	}
	if n == 0 {
		return nil, nil
	}
	buf := make([]byte, n)
	return buf, check("pfnGetNativeBinary", g.pfnGetNativeBinary(h, &n, &buf[0]))
}

// SupportedLayers asks the compiler which layers of an IR it can compile. The driver answers
// "<name0><name1>...".
func (g *GraphExt) SupportedLayers(ctx, dev Handle, input []byte, buildFlags string) (map[string]bool, error) {
	if g.pfnQueryNetworkCreate2 == nil {
		return nil, fmt.Errorf("pfnQueryNetworkCreate2 is not available in %s", VersionString(g.Version))
	}
	desc, keep := graphDesc(GraphFormatNGraphLite, input, buildFlags, GraphFlagNone)
	if g.Version < MakeVersion(1, 16) {
		desc.SType = stypeGraphDesc // as the OpenVINO plugin does for older drivers
	}
	var q Handle
	r := g.pfnQueryNetworkCreate2(ctx, dev, &desc, &q)
	runtimeKeepAlive(keep)
	if r != ResultSuccess {
		return nil, &Error{Op: "pfnQueryNetworkCreate2", Result: r, Log: g.buildLog(0)}
	}
	defer g.pfnQueryNetworkDestroy(q)
	var n uintptr
	if err := check("pfnQueryNetworkGetSupportedLayers", g.pfnQueryNetworkGetSupport(q, &n, nil)); err != nil {
		return nil, err
	}
	buf := make([]byte, n+1)
	if n > 0 {
		if err := check("pfnQueryNetworkGetSupportedLayers", g.pfnQueryNetworkGetSupport(q, &n, &buf[0])); err != nil {
			return nil, err
		}
	}
	return ParseSupportedLayers(CString(buf[:n])), nil
}

// ParseSupportedLayers parses "<a><b>" into a set.
func ParseSupportedLayers(s string) map[string]bool {
	out := map[string]bool{}
	for {
		i := strings.IndexByte(s, '<')
		if i < 0 {
			return out
		}
		j := strings.IndexByte(s[i+1:], '>')
		if j < 0 {
			return out
		}
		out[s[i+1:i+1+j]] = true
		s = s[i+2+j:]
	}
}
