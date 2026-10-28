package ze

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
	"unsafe"
)

type cStruct struct {
	Size   uintptr            `json:"size"`
	Fields map[string]uintptr `json:"fields"`
}

// Acceptance requirement:cgo-free/abi-golden. The golden file is generated from the C headers by
// testdata/abi/levelzero_abi.c.
func TestABIMatchesCHeaders(t *testing.T) {
	if unsafe.Sizeof(uintptr(0)) != 8 {
		t.Skip("golden values are for 64-bit targets")
	}
	raw, err := os.ReadFile("../../testdata/abi/levelzero_abi.json")
	if err != nil {
		t.Fatal(err)
	}
	var golden map[string]cStruct
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		c      string
		v      any
		fields map[string]string // C member -> Go field
	}{
		{"ze_driver_properties_t", DriverProperties{}, map[string]string{"stype": "SType", "pNext": "PNext", "uuid": "UUID", "driverVersion": "DriverVersion"}},
		{"ze_driver_extension_properties_t", DriverExtensionProperties{}, map[string]string{"name": "Name", "version": "Version"}},
		{"ze_device_properties_t", DeviceProperties{}, map[string]string{
			"stype": "SType", "pNext": "PNext", "type": "Type", "vendorId": "VendorID", "deviceId": "DeviceID", "flags": "Flags",
			"subdeviceId": "SubdeviceID", "coreClockRate": "CoreClockRate", "maxMemAllocSize": "MaxMemAllocSize",
			"maxHardwareContexts": "MaxHardwareContexts", "maxCommandQueuePriority": "MaxCommandQueuePriority",
			"numThreadsPerEU": "NumThreadsPerEU", "physicalEUSimdWidth": "PhysicalEUSimdWidth", "numEUsPerSubslice": "NumEUsPerSubslice",
			"numSubslicesPerSlice": "NumSubslicesPerSlice", "numSlices": "NumSlices", "timerResolution": "TimerResolution",
			"timestampValidBits": "TimestampValidBits", "kernelTimestampValidBits": "KernelTimestampValidBits", "uuid": "UUID", "name": "Name"}},
		{"ze_context_desc_t", simpleDesc{}, map[string]string{"stype": "SType", "pNext": "PNext", "flags": "Flags"}},
		{"ze_host_mem_alloc_desc_t", simpleDesc{}, map[string]string{"stype": "SType", "pNext": "PNext", "flags": "Flags"}},
		{"ze_fence_desc_t", simpleDesc{}, map[string]string{"stype": "SType", "pNext": "PNext", "flags": "Flags"}},
		{"ze_command_queue_desc_t", CommandQueueDesc{}, map[string]string{"stype": "SType", "pNext": "PNext", "ordinal": "Ordinal", "index": "Index", "flags": "Flags", "mode": "Mode", "priority": "Priority"}},
		{"ze_command_list_desc_t", CommandListDesc{}, map[string]string{"stype": "SType", "pNext": "PNext", "commandQueueGroupOrdinal": "CommandQueueGroupOrdinal", "flags": "Flags"}},
		{"ze_command_queue_group_properties_t", CommandQueueGroupProperties{}, map[string]string{"stype": "SType", "pNext": "PNext", "flags": "Flags", "maxMemoryFillPatternSize": "MaxMemoryFillPatternSize", "numQueues": "NumQueues"}},
		{"ze_command_queue_desc_npu_ext_t", QueueDescNPUExt{}, map[string]string{"stype": "SType", "pNext": "PNext", "turbo": "Turbo"}},
		{"ze_command_queue_desc_npu_ext_2_t", QueueDescNPUExt2{}, map[string]string{"stype": "SType", "pNext": "PNext", "options": "Options"}},
		{"ze_graph_compiler_version_info_t", CompilerVersion{}, map[string]string{"major": "Major", "minor": "Minor"}},
		{"ze_device_graph_properties_t", DeviceGraphProperties{}, map[string]string{"stype": "SType", "pNext": "PNext", "graphExtensionVersion": "GraphExtensionVersion", "compilerVersion": "CompilerVersion", "graphFormatsSupported": "GraphFormatsSupported", "maxOVOpsetVersionSupported": "MaxOVOpsetVersion"}},
		{"ze_graph_desc_2_t", GraphDesc2{}, map[string]string{"stype": "SType", "pNext": "PNext", "format": "Format", "inputSize": "InputSize", "pInput": "Input", "pBuildFlags": "BuildFlags", "flags": "Flags"}},
		{"ze_graph_properties_2_t", GraphProperties2{}, map[string]string{"stype": "SType", "pNext": "PNext", "numGraphArgs": "NumGraphArgs", "initStageRequired": "InitStageRequired"}},
		{"ze_graph_argument_properties_3_t", GraphArgumentProperties3{}, map[string]string{
			"stype": "SType", "pNext": "PNext", "name": "Name", "type": "Type", "dims": "Dims", "networkPrecision": "NetworkPrecision",
			"networkLayout": "NetworkLayout", "devicePrecision": "DevicePrecision", "deviceLayout": "DeviceLayout",
			"quantReverseScale": "QuantReverseScale", "quantZeroPoint": "QuantZeroPoint", "dims_count": "DimsCount",
			"debug_friendly_name": "DebugFriendlyName", "associated_tensor_names": "AssociatedTensorNames",
			"associated_tensor_names_count": "AssociatedTensorNamesCount"}},
	}
	for _, c := range cases {
		g, ok := golden[c.c]
		if !ok {
			t.Errorf("%s missing from golden file", c.c)
			continue
		}
		typ := reflect.TypeOf(c.v)
		if typ.Size() != g.Size {
			t.Errorf("%s: Go size %d, C size %d", c.c, typ.Size(), g.Size)
		}
		for cName, off := range g.Fields {
			goName, ok := c.fields[cName]
			if !ok {
				t.Errorf("%s.%s has no Go mapping", c.c, cName)
				continue
			}
			f, ok := typ.FieldByName(goName)
			if !ok {
				t.Errorf("%s: Go field %s not found", c.c, goName)
				continue
			}
			if f.Offset != off {
				t.Errorf("%s.%s: Go offset %d, C offset %d", c.c, cName, f.Offset, off)
			}
		}
	}
	// ze_graph_properties_t is the prefix of ze_graph_properties_2_t.
	if golden["ze_graph_properties_t"].Fields["numGraphArgs"] != golden["ze_graph_properties_2_t"].Fields["numGraphArgs"] {
		t.Error("ze_graph_properties_t is not a prefix of ze_graph_properties_2_t")
	}

	// Dispatch table indices and per-version entry counts.
	ddi := golden["ze_graph_dditable_ext_t"]
	idx := map[string]int{
		"pfnCreate": ddiCreate, "pfnDestroy": ddiDestroy, "pfnGetProperties": ddiGetProperties,
		"pfnSetArgumentValue": ddiSetArgumentValue, "pfnAppendGraphInitialize": ddiAppendGraphInitialize,
		"pfnAppendGraphExecute": ddiAppendGraphExecute, "pfnGetNativeBinary": ddiGetNativeBinary,
		"pfnDeviceGetGraphProperties": ddiDeviceGetGraphProperties, "pfnGetArgumentProperties3": ddiGetArgumentProperties3,
		"pfnQueryNetworkDestroy": ddiQueryNetworkDestroy, "pfnQueryNetworkGetSupportedLayers": ddiQueryNetworkGetSupported,
		"pfnBuildLogGetString": ddiBuildLogGetString, "pfnCreate2": ddiCreate2, "pfnQueryNetworkCreate2": ddiQueryNetworkCreate2,
		"pfnGetProperties2": ddiGetProperties2, "pfnGraphInitialize": ddiGraphInitialize,
	}
	for name, i := range idx {
		if ddi.Fields[name] != uintptr(i)*8 {
			t.Errorf("%s: index %d, C offset %d", name, i, ddi.Fields[name])
		}
	}
	// The first entry added by a version must be the last counted entry of that version.
	firstOfVersion := map[string]uint32{
		"pfnCreate2": MakeVersion(1, 5), "pfnGraphInitialize": MakeVersion(1, 8), "pfnCreate3": MakeVersion(1, 12),
		"pfnSetArgumentValue2": MakeVersion(1, 15), "pfnEvict": MakeVersion(1, 16), "pfnGetArgumentNames": MakeVersion(1, 20),
	}
	for name, v := range firstOfVersion {
		var count uint32
		for _, c := range graphDDICount {
			if v >= c.version {
				count = c.count
				break
			}
		}
		if uintptr(count) <= ddi.Fields[name]/8 {
			t.Errorf("version %s counts %d entries but %s is entry %d", VersionString(v), count, name, ddi.Fields[name]/8)
		}
	}
	if ddi.Size/8 != uintptr(graphDDICount[0].count) {
		t.Errorf("table has %d entries, newest version counts %d", ddi.Size/8, graphDDICount[0].count)
	}
}

func TestSelectGraphExt(t *testing.T) {
	mk := func(name string, v uint32) DriverExtensionProperties {
		var p DriverExtensionProperties
		copy(p.Name[:], name)
		p.Version = v
		return p
	}
	name, v := SelectGraphExt([]DriverExtensionProperties{mk("ZE_extension_other", MakeVersion(9, 9)), mk("ZE_extension_graph_1_2", MakeVersion(1, 2)), mk("ZE_extension_graph_1_7", MakeVersion(1, 7))})
	if name != "ZE_extension_graph_1_7" || v != MakeVersion(1, 7) {
		t.Fatalf("old naming: %s %s", name, VersionString(v))
	}
	name, v = SelectGraphExt([]DriverExtensionProperties{mk("ZE_extension_graph", MakeVersion(1, 30))})
	if name != "ZE_extension_graph" || v != GraphExtTargetVersion {
		t.Fatalf("newer driver: %s %s", name, VersionString(v))
	}
	if name, _ := SelectGraphExt([]DriverExtensionProperties{mk("ZE_extension_graphx", 1)}); name != "" {
		t.Fatalf("matched %q", name)
	}
}

func TestParseSupportedLayers(t *testing.T) {
	got := ParseSupportedLayers("<x><mul><y/sink_port_0>")
	if len(got) != 3 || !got["mul"] || !got["y/sink_port_0"] {
		t.Fatalf("%v", got)
	}
	if len(ParseSupportedLayers("")) != 0 {
		t.Fatal("empty")
	}
}

func TestSelectComputeOrdinal(t *testing.T) {
	g := func(flags ...uint32) []CommandQueueGroupProperties {
		ps := make([]CommandQueueGroupProperties, len(flags))
		for i, f := range flags {
			ps[i].Flags = f
		}
		return ps
	}
	if got := SelectComputeOrdinal(g(2, 3, 1)); got != 2 {
		t.Fatalf("compute-only group: %d", got)
	}
	if got := SelectComputeOrdinal(g(2, 3)); got != 1 {
		t.Fatalf("group with compute: %d", got)
	}
	if got := SelectComputeOrdinal(g(2)); got != 0 {
		t.Fatalf("fallback: %d", got)
	}
}
