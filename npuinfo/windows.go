//go:build windows

package npuinfo

import (
	"context"
	"encoding/binary"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"syscall"
	"time"
	"unsafe"
)

var (
	modPDH      = syscall.NewLazyDLL("pdh.dll")
	modCfgMgr   = syscall.NewLazyDLL("cfgmgr32.dll")
	modGDI      = syscall.NewLazyDLL("gdi32.dll")
	modKernel32 = syscall.NewLazyDLL("kernel32.dll")

	procPdhEnumObjectsW             = modPDH.NewProc("PdhEnumObjectsW")
	procPdhEnumObjectItemsW         = modPDH.NewProc("PdhEnumObjectItemsW")
	procPdhOpenQueryW               = modPDH.NewProc("PdhOpenQueryW")
	procPdhAddCounterW              = modPDH.NewProc("PdhAddCounterW")
	procPdhCollectQueryData         = modPDH.NewProc("PdhCollectQueryData")
	procPdhGetFormattedCounterArray = modPDH.NewProc("PdhGetFormattedCounterArrayW")
	procPdhCloseQuery               = modPDH.NewProc("PdhCloseQuery")

	procCMGetDeviceIDListSize = modCfgMgr.NewProc("CM_Get_Device_ID_List_SizeW")
	procCMGetDeviceIDList     = modCfgMgr.NewProc("CM_Get_Device_ID_ListW")
	procCMLocateDevNode       = modCfgMgr.NewProc("CM_Locate_DevNodeW")
	procCMGetDevNodeProperty  = modCfgMgr.NewProc("CM_Get_DevNode_PropertyW")
	procCMGetDevNodeStatus    = modCfgMgr.NewProc("CM_Get_DevNode_Status")

	procD3DKMTEnumAdapters2    = modGDI.NewProc("D3DKMTEnumAdapters2")
	procD3DKMTEnumAdapters3    = modGDI.NewProc("D3DKMTEnumAdapters3")
	procD3DKMTQueryAdapterInfo = modGDI.NewProc("D3DKMTQueryAdapterInfo")
	procD3DKMTCloseAdapter     = modGDI.NewProc("D3DKMTCloseAdapter")

	procOpenProcess                = modKernel32.NewProc("OpenProcess")
	procQueryFullProcessImageNameW = modKernel32.NewProc("QueryFullProcessImageNameW")
	procCloseHandle                = modKernel32.NewProc("CloseHandle")
)

func collect(ctx context.Context, opts Options, info *Info) error {
	wi := &WindowsInfo{}
	info.Windows = wi
	guard("pnp devices", info, func() error {
		d, err := windowsDevices()
		wi.Devices = d
		return err
	})
	guard("d3dkmt adapters", info, func() error {
		a, err := windowsAdapters()
		wi.Adapters = a
		return err
	})
	guard("performance counters", info, func() error {
		interval := opts.SampleInterval
		if interval <= 0 {
			interval = 500 * time.Millisecond
		}
		objs, counters, err := windowsCounters(ctx, interval)
		wi.CounterObjects, wi.Counters = objs, counters
		return err
	})
	guard("processes", info, func() error {
		wi.Processes = windowsProcesses(wi.Counters)
		return nil
	})
	return nil
}

// ---- PnP devices (cfgmgr32) ----

type devPropKey struct {
	fmtid syscall.GUID
	pid   uint32
}

var (
	guidDevice = syscall.GUID{Data1: 0xa45c254e, Data2: 0xdf1c, Data3: 0x4efd, Data4: [8]byte{0x80, 0x20, 0x67, 0xd1, 0x46, 0xa8, 0x50, 0xe0}}
	guidDriver = syscall.GUID{Data1: 0xa8b865dd, Data2: 0x2e3d, Data3: 0x4094, Data4: [8]byte{0xad, 0x97, 0xe5, 0x93, 0xa7, 0x0c, 0x75, 0xd6}}

	devProps = []struct {
		name string
		key  devPropKey
	}{
		{"DeviceDesc", devPropKey{guidDevice, 2}},
		{"HardwareIds", devPropKey{guidDevice, 3}},
		{"Service", devPropKey{guidDevice, 6}},
		{"Class", devPropKey{guidDevice, 9}},
		{"ClassGuid", devPropKey{guidDevice, 10}},
		{"Manufacturer", devPropKey{guidDevice, 13}},
		{"FriendlyName", devPropKey{guidDevice, 14}},
		{"LocationInfo", devPropKey{guidDevice, 15}},
		{"DriverDate", devPropKey{guidDriver, 2}},
		{"DriverVersion", devPropKey{guidDriver, 3}},
		{"DriverDesc", devPropKey{guidDriver, 4}},
		{"DriverProvider", devPropKey{guidDriver, 9}},
	}
)

const (
	crSuccess                = 0
	crBufferSmall            = 0x1A
	cmGetIDListFilterPresent = 0x100
)

var npuName = regexp.MustCompile(`(?i)npu|neural|ai boost|\bipu\b|hexagon|compute accelerator`)

func windowsDevices() ([]WindowsDevice, error) {
	if err := procCMGetDeviceIDList.Find(); err != nil {
		return nil, err
	}
	var size uint32
	if r, _, _ := procCMGetDeviceIDListSize.Call(uintptr(unsafe.Pointer(&size)), 0, cmGetIDListFilterPresent); r != crSuccess {
		return nil, fmt.Errorf("CM_Get_Device_ID_List_Size: CR %#x", r)
	}
	buf := make([]uint16, size+2)
	if r, _, _ := procCMGetDeviceIDList.Call(0, uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)), cmGetIDListFilterPresent); r != crSuccess {
		return nil, fmt.Errorf("CM_Get_Device_ID_List: CR %#x", r)
	}
	var out []WindowsDevice
	for _, id := range multiSZ(buf) {
		idp, err := syscall.UTF16PtrFromString(id)
		if err != nil {
			continue
		}
		var inst uint32
		if r, _, _ := procCMLocateDevNode.Call(uintptr(unsafe.Pointer(&inst)), uintptr(unsafe.Pointer(idp)), 0); r != crSuccess {
			continue
		}
		class, _ := devNodeProp(inst, devProps[3].key).(string)
		name, _ := devNodeProp(inst, devProps[6].key).(string)
		desc, _ := devNodeProp(inst, devProps[0].key).(string)
		if class != "ComputeAccelerator" && !npuName.MatchString(name) && !npuName.MatchString(desc) {
			continue
		}
		d := WindowsDevice{InstanceID: id, Properties: map[string]any{}}
		for _, p := range devProps {
			if v := devNodeProp(inst, p.key); v != nil {
				d.Properties[p.name] = v
			}
		}
		procCMGetDevNodeStatus.Call(uintptr(unsafe.Pointer(&d.Status)), uintptr(unsafe.Pointer(&d.Problem)), uintptr(inst), 0)
		out = append(out, d)
	}
	return out, nil
}

// devNodeProp reads one PnP property and decodes the common types; nil when absent.
func devNodeProp(inst uint32, key devPropKey) any {
	var typ, size uint32
	r, _, _ := procCMGetDevNodeProperty.Call(uintptr(inst), uintptr(unsafe.Pointer(&key)), uintptr(unsafe.Pointer(&typ)), 0, uintptr(unsafe.Pointer(&size)), 0)
	if r != crBufferSmall || size == 0 {
		return nil
	}
	buf := make([]byte, size)
	r, _, _ = procCMGetDevNodeProperty.Call(uintptr(inst), uintptr(unsafe.Pointer(&key)), uintptr(unsafe.Pointer(&typ)), uintptr(unsafe.Pointer(&buf[0])), uintptr(unsafe.Pointer(&size)), 0)
	if r != crSuccess {
		return nil
	}
	buf = buf[:size]
	u16 := func() []uint16 {
		u := make([]uint16, len(buf)/2)
		for i := range u {
			u[i] = binary.LittleEndian.Uint16(buf[2*i:])
		}
		return u
	}
	switch typ {
	case 0x12: // DEVPROP_TYPE_STRING
		return utf16z(u16())
	case 0x2012: // DEVPROP_TYPE_STRING_LIST
		return multiSZ(append(u16(), 0, 0))
	case 0x07: // DEVPROP_TYPE_UINT32
		if len(buf) >= 4 {
			return binary.LittleEndian.Uint32(buf)
		}
	case 0x10: // DEVPROP_TYPE_FILETIME
		if len(buf) >= 8 {
			ft := syscall.Filetime{LowDateTime: binary.LittleEndian.Uint32(buf), HighDateTime: binary.LittleEndian.Uint32(buf[4:])}
			return time.Unix(0, ft.Nanoseconds()).UTC()
		}
	case 0x11: // DEVPROP_TYPE_BOOLEAN
		return len(buf) > 0 && buf[0] != 0
	case 0x0D: // DEVPROP_TYPE_GUID
		if len(buf) >= 16 {
			return fmt.Sprintf("{%08x-%04x-%04x-%x-%x}", binary.LittleEndian.Uint32(buf), binary.LittleEndian.Uint16(buf[4:]),
				binary.LittleEndian.Uint16(buf[6:]), buf[8:10], buf[10:16])
		}
	}
	return fmt.Sprintf("type %#x: %x", typ, buf)
}

// ---- DirectX kernel adapters (D3DKMT) ----

// D3DKMT_ADAPTERINFO: hAdapter, LUID (low, high), NumOfSources, bPrecisePresentRegionsPreferred.
type kmtAdapterInfo struct {
	hAdapter   uint32
	luidLow    uint32
	luidHigh   int32
	numSources uint32
	precise    int32
}

type kmtEnumAdapters3 struct {
	filter      uint64
	numAdapters uint32
	adapters    *kmtAdapterInfo
}

type kmtEnumAdapters2 struct {
	numAdapters uint32
	adapters    *kmtAdapterInfo
}

type kmtQueryAdapterInfo struct {
	hAdapter uint32
	typ      uint32
	data     unsafe.Pointer
	size     uint32
}

// KMTQUERYADAPTERINFOTYPE values used here.
const (
	kmtqaiGetSegmentSize           = 3
	kmtqaiAdapterAddress           = 6
	kmtqaiAdapterRegistryInfo      = 8
	kmtqaiDriverVersion            = 13
	kmtqaiAdapterType              = 15
	kmtqaiNodeMetadata             = 25
	kmtqaiPhysicalAdapterDeviceIDs = 31
	kmtqaiNodePerfData             = 61
	kmtqaiAdapterPerfData          = 62
)

func kmtQuery(h uint32, typ uint32, buf []byte) error {
	q := kmtQueryAdapterInfo{hAdapter: h, typ: typ, data: unsafe.Pointer(&buf[0]), size: uint32(len(buf))}
	r, _, _ := procD3DKMTQueryAdapterInfo.Call(uintptr(unsafe.Pointer(&q)))
	if r != 0 {
		return fmt.Errorf("NTSTATUS %#x", uint32(r))
	}
	return nil
}

func windowsAdapters() ([]WindowsAdapter, error) {
	infos := make([]kmtAdapterInfo, 64)
	var n uint32
	if procD3DKMTEnumAdapters3.Find() == nil {
		e := kmtEnumAdapters3{filter: 1 /* IncludeComputeOnly */, numAdapters: uint32(len(infos)), adapters: &infos[0]}
		if r, _, _ := procD3DKMTEnumAdapters3.Call(uintptr(unsafe.Pointer(&e))); r != 0 {
			return nil, fmt.Errorf("D3DKMTEnumAdapters3: NTSTATUS %#x", uint32(r))
		}
		n = e.numAdapters
	} else if procD3DKMTEnumAdapters2.Find() == nil {
		e := kmtEnumAdapters2{numAdapters: uint32(len(infos)), adapters: &infos[0]}
		if r, _, _ := procD3DKMTEnumAdapters2.Call(uintptr(unsafe.Pointer(&e))); r != 0 {
			return nil, fmt.Errorf("D3DKMTEnumAdapters2: NTSTATUS %#x", uint32(r))
		}
		n = e.numAdapters
	} else {
		return nil, fmt.Errorf("no D3DKMTEnumAdapters2/3 in gdi32.dll")
	}
	if int(n) > len(infos) {
		n = uint32(len(infos))
	}
	le := binary.LittleEndian
	var out []WindowsAdapter
	for _, ai := range infos[:n] {
		a := WindowsAdapter{LUID: formatLUID(uint32(ai.luidHigh), ai.luidLow), NumSources: ai.numSources, Errors: map[string]string{}}
		h := ai.hAdapter
		fail := func(name string, err error) { a.Errors[name] = err.Error() }

		reg := make([]byte, 4*260*2)
		if err := kmtQuery(h, kmtqaiAdapterRegistryInfo, reg); err != nil {
			fail("adapter_registry_info", err)
		} else {
			field := func(i int) string {
				u := make([]uint16, 260)
				for j := range u {
					u[j] = le.Uint16(reg[(i*260+j)*2:])
				}
				return utf16z(u)
			}
			a.AdapterString, a.BiosString, a.DacType, a.ChipType = field(0), field(1), field(2), field(3)
		}
		b4 := make([]byte, 4)
		if err := kmtQuery(h, kmtqaiAdapterType, b4); err != nil {
			fail("adapter_type", err)
		} else {
			a.TypeFlags = le.Uint32(b4)
			a.TypeNames = adapterTypeNames(a.TypeFlags)
		}
		b4 = make([]byte, 4)
		if err := kmtQuery(h, kmtqaiDriverVersion, b4); err != nil {
			fail("driver_version", err)
		} else {
			a.WDDMVersion = le.Uint32(b4)
		}
		ids := make([]byte, 28) // PhysicalAdapterIndex + D3DKMT_DEVICE_IDS
		if err := kmtQuery(h, kmtqaiPhysicalAdapterDeviceIDs, ids); err != nil {
			fail("device_ids", err)
		} else {
			a.VendorID, a.DeviceID, a.SubVendorID = le.Uint32(ids[4:]), le.Uint32(ids[8:]), le.Uint32(ids[12:])
			a.SubSystemID, a.RevisionID, a.BusType = le.Uint32(ids[16:]), le.Uint32(ids[20:]), le.Uint32(ids[24:])
		}
		addr := make([]byte, 12)
		if err := kmtQuery(h, kmtqaiAdapterAddress, addr); err != nil {
			fail("adapter_address", err)
		} else {
			a.Bus, a.Device, a.Function = le.Uint32(addr), le.Uint32(addr[4:]), le.Uint32(addr[8:])
		}
		seg := make([]byte, 24)
		if err := kmtQuery(h, kmtqaiGetSegmentSize, seg); err != nil {
			fail("segment_size", err)
		} else {
			a.DedicatedVideoMemory, a.DedicatedSystemMemory, a.SharedSystemMemory = le.Uint64(seg), le.Uint64(seg[8:]), le.Uint64(seg[16:])
		}
		perf := make([]byte, 64)
		if err := kmtQuery(h, kmtqaiAdapterPerfData, perf); err != nil {
			fail("adapter_perfdata", err)
		} else {
			a.Perf = &WindowsAdapterPerf{MemoryFrequency: le.Uint64(perf[8:]), MaxMemoryFrequency: le.Uint64(perf[16:]),
				MemoryBandwidth: le.Uint64(perf[32:]), PCIEBandwidth: le.Uint64(perf[40:]),
				FanRPM: le.Uint32(perf[48:]), Power: le.Uint32(perf[52:]), Temperature: le.Uint32(perf[56:])}
		}
		// Nodes: D3DKMT_NODEMETADATA until the ordinal is out of range.
		for node := uint32(0); node < 64; node++ {
			md := make([]byte, 80)
			le.PutUint32(md, node) // physical adapter 0 in the high word
			if err := kmtQuery(h, kmtqaiNodeMetadata, md); err != nil {
				if node == 0 {
					fail("node_metadata", err)
				}
				break
			}
			name := make([]uint16, 32)
			for j := range name {
				name[j] = le.Uint16(md[8+2*j:])
			}
			nd := WindowsNode{Ordinal: node, EngineType: le.Uint32(md[4:]), FriendlyName: utf16z(name),
				Flags: le.Uint32(md[72:]), GpuMmu: md[76] != 0, IoMmu: md[77] != 0}
			nd.EngineTypeName = engineTypeName(nd.EngineType)
			np := make([]byte, 56)
			le.PutUint32(np, node)
			if err := kmtQuery(h, kmtqaiNodePerfData, np); err == nil {
				nd.Frequency, nd.MaxFrequency, nd.Voltage = le.Uint64(np[8:]), le.Uint64(np[16:]), le.Uint32(np[32:])
			}
			a.Nodes = append(a.Nodes, nd)
		}
		if len(a.Errors) == 0 {
			a.Errors = nil
		}
		out = append(out, a)
		closeArg := struct{ h uint32 }{h}
		procD3DKMTCloseAdapter.Call(uintptr(unsafe.Pointer(&closeArg)))
	}
	return out, nil
}

// ---- performance counters (PDH) ----

const (
	pdhMoreData      = 0x800007D2
	perfDetailWizard = 400
	pdhFmtDouble     = 0x00000200
	pdhFmtNoCap100   = 0x00008000
)

var counterObject = regexp.MustCompile(`(?i)gpu|npu|neural`)

// pdhFmtCounterValueItem is PDH_FMT_COUNTERVALUE_ITEM_W on 64-bit Windows.
type pdhFmtCounterValueItem struct {
	name    *uint16
	cstatus uint32
	_       uint32
	value   float64
}

func pdhEnumObjects() ([]string, error) {
	var size uint32
	r, _, _ := procPdhEnumObjectsW.Call(0, 0, 0, uintptr(unsafe.Pointer(&size)), perfDetailWizard, 1)
	if r != pdhMoreData && r != 0 {
		return nil, fmt.Errorf("PdhEnumObjects: %#x", uint32(r))
	}
	buf := make([]uint16, size+2)
	size = uint32(len(buf))
	r, _, _ = procPdhEnumObjectsW.Call(0, 0, uintptr(unsafe.Pointer(&buf[0])), uintptr(unsafe.Pointer(&size)), perfDetailWizard, 0)
	if r != 0 {
		return nil, fmt.Errorf("PdhEnumObjects: %#x", uint32(r))
	}
	return multiSZ(buf), nil
}

func pdhEnumItems(object string) (counters, instances []string, err error) {
	obj, err := syscall.UTF16PtrFromString(object)
	if err != nil {
		return nil, nil, err
	}
	var nc, ni uint32
	r, _, _ := procPdhEnumObjectItemsW.Call(0, 0, uintptr(unsafe.Pointer(obj)), 0, uintptr(unsafe.Pointer(&nc)), 0, uintptr(unsafe.Pointer(&ni)), perfDetailWizard, 0)
	if r != pdhMoreData && r != 0 {
		return nil, nil, fmt.Errorf("PdhEnumObjectItems(%s): %#x", object, uint32(r))
	}
	cb := make([]uint16, nc+2)
	ib := make([]uint16, ni+2)
	nc, ni = uint32(len(cb)), uint32(len(ib))
	r, _, _ = procPdhEnumObjectItemsW.Call(0, 0, uintptr(unsafe.Pointer(obj)), uintptr(unsafe.Pointer(&cb[0])), uintptr(unsafe.Pointer(&nc)),
		uintptr(unsafe.Pointer(&ib[0])), uintptr(unsafe.Pointer(&ni)), perfDetailWizard, 0)
	if r != 0 {
		return nil, nil, fmt.Errorf("PdhEnumObjectItems(%s): %#x", object, uint32(r))
	}
	return multiSZ(cb), multiSZ(ib), nil
}

func windowsCounters(ctx context.Context, interval time.Duration) ([]WindowsCounterObject, []WindowsCounter, error) {
	objects, err := pdhEnumObjects()
	if err != nil {
		return nil, nil, err
	}
	var query uintptr
	if r, _, _ := procPdhOpenQueryW.Call(0, 0, uintptr(unsafe.Pointer(&query))); r != 0 {
		return nil, nil, fmt.Errorf("PdhOpenQuery: %#x", uint32(r))
	}
	defer procPdhCloseQuery.Call(query)

	type added struct {
		object, counter string
		h               uintptr
	}
	var objs []WindowsCounterObject
	var handles []added
	var errs []string
	for _, o := range objects {
		if !counterObject.MatchString(o) {
			continue
		}
		counters, instances, err := pdhEnumItems(o)
		if err != nil {
			errs = append(errs, err.Error())
			continue
		}
		objs = append(objs, WindowsCounterObject{Name: o, Counters: counters, Instances: len(instances)})
		for _, c := range counters {
			path := `\` + o + `(*)\` + c
			if len(instances) == 0 {
				path = `\` + o + `\` + c
			}
			p, err := syscall.UTF16PtrFromString(path)
			if err != nil {
				continue
			}
			var h uintptr
			if r, _, _ := procPdhAddCounterW.Call(query, uintptr(unsafe.Pointer(p)), 0, uintptr(unsafe.Pointer(&h))); r != 0 {
				errs = append(errs, fmt.Sprintf("PdhAddCounter(%s): %#x", path, uint32(r)))
				continue
			}
			handles = append(handles, added{o, c, h})
		}
	}
	// Rate counters need two samples.
	procPdhCollectQueryData.Call(query)
	select {
	case <-time.After(interval):
	case <-ctx.Done():
		return objs, nil, ctx.Err()
	}
	if r, _, _ := procPdhCollectQueryData.Call(query); r != 0 {
		return objs, nil, fmt.Errorf("PdhCollectQueryData: %#x", uint32(r))
	}
	var out []WindowsCounter
	for _, a := range handles {
		var size, count uint32
		r, _, _ := procPdhGetFormattedCounterArray.Call(a.h, pdhFmtDouble|pdhFmtNoCap100, uintptr(unsafe.Pointer(&size)), uintptr(unsafe.Pointer(&count)), 0)
		if r != pdhMoreData {
			if r != 0 {
				out = append(out, WindowsCounter{Object: a.object, Counter: a.counter, Status: uint32(r)})
			}
			continue
		}
		itemSize := uint32(unsafe.Sizeof(pdhFmtCounterValueItem{}))
		items := make([]pdhFmtCounterValueItem, size/itemSize+1)
		r, _, _ = procPdhGetFormattedCounterArray.Call(a.h, pdhFmtDouble|pdhFmtNoCap100, uintptr(unsafe.Pointer(&size)), uintptr(unsafe.Pointer(&count)), uintptr(unsafe.Pointer(&items[0])))
		if r != 0 {
			out = append(out, WindowsCounter{Object: a.object, Counter: a.counter, Status: uint32(r)})
			continue
		}
		for _, it := range items[:count] {
			c := WindowsCounter{Object: a.object, Counter: a.counter, Instance: utf16PtrString(it.name), Value: it.value}
			if it.cstatus > 1 { // PDH_CSTATUS_VALID_DATA, PDH_CSTATUS_NEW_DATA
				c.Status = it.cstatus
			}
			parseGPUInstance(&c)
			out = append(out, c)
		}
	}
	var err2 error
	if len(errs) > 0 {
		err2 = fmt.Errorf("%s", strings.Join(errs, "; "))
	}
	return objs, out, err2
}

func utf16PtrString(p *uint16) string {
	if p == nil {
		return ""
	}
	var u []uint16
	for ptr := unsafe.Pointer(p); ; ptr = unsafe.Add(ptr, 2) {
		c := *(*uint16)(ptr)
		if c == 0 {
			break
		}
		u = append(u, c)
	}
	return utf16z(append(u, 0))
}

// ---- processes ----

func windowsProcesses(counters []WindowsCounter) []WindowsProcess {
	seen := map[int]bool{}
	var out []WindowsProcess
	for _, c := range counters {
		if c.PID == 0 || seen[c.PID] {
			continue
		}
		seen[c.PID] = true
		out = append(out, WindowsProcess{PID: c.PID, Image: processImage(c.PID)})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].PID < out[j].PID })
	return out
}

func processImage(pid int) string {
	const processQueryLimitedInformation = 0x1000
	h, _, _ := procOpenProcess.Call(processQueryLimitedInformation, 0, uintptr(pid))
	if h == 0 {
		return ""
	}
	defer procCloseHandle.Call(h)
	buf := make([]uint16, 1024)
	size := uint32(len(buf))
	if r, _, _ := procQueryFullProcessImageNameW.Call(h, 0, uintptr(unsafe.Pointer(&buf[0])), uintptr(unsafe.Pointer(&size))); r == 0 {
		return ""
	}
	return utf16z(buf[:size])
}
