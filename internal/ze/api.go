package ze

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"unsafe"

	"github.com/shibukawa/hwnpu/internal/dl"
)

// LoaderEnv overrides the Level Zero loader path.
const LoaderEnv = "HWNPU_ZE_LOADER"

// API holds the bound core Level Zero functions.
type API struct {
	Lib *dl.Library

	zeInit                              func(flags uint32) Result
	zeDriverGet                         func(count *uint32, drivers *Handle) Result
	zeDriverGetProperties               func(driver Handle, props *DriverProperties) Result
	zeDriverGetApiVersion               func(driver Handle, version *uint32) Result
	zeDriverGetExtensionProperties      func(driver Handle, count *uint32, props *DriverExtensionProperties) Result
	zeDriverGetExtensionFunctionAddress func(driver Handle, name *byte, fn *unsafe.Pointer) Result
	zeDeviceGet                         func(driver Handle, count *uint32, devices *Handle) Result
	zeDeviceGetProperties               func(device Handle, props *DeviceProperties) Result
	zeDeviceGetCommandQueueGroupProps   func(device Handle, count *uint32, props *CommandQueueGroupProperties) Result
	zeContextCreate                     func(driver Handle, desc *simpleDesc, ctx *Handle) Result
	zeContextDestroy                    func(ctx Handle) Result
	zeMemAllocHost                      func(ctx Handle, desc *simpleDesc, size, alignment uintptr, ptr *unsafe.Pointer) Result
	zeMemFree                           func(ctx Handle, ptr unsafe.Pointer) Result
	zeCommandQueueCreate                func(ctx, device Handle, desc *CommandQueueDesc, queue *Handle) Result
	zeCommandQueueDestroy               func(queue Handle) Result
	zeCommandQueueExecuteCommandLists   func(queue Handle, n uint32, lists *Handle, fence Handle) Result
	zeCommandListCreate                 func(ctx, device Handle, desc *CommandListDesc, list *Handle) Result
	zeCommandListDestroy                func(list Handle) Result
	zeCommandListClose                  func(list Handle) Result
	zeCommandListReset                  func(list Handle) Result
	zeFenceCreate                       func(queue Handle, desc *simpleDesc, fence *Handle) Result
	zeFenceDestroy                      func(fence Handle) Result
	zeFenceHostSynchronize              func(fence Handle, timeoutNs uint64) Result
	zeFenceQueryStatus                  func(fence Handle) Result
	zeFenceReset                        func(fence Handle) Result
}

// LoaderCandidates lists the library names tried by Load, after the LoaderEnv override.
func LoaderCandidates() []string {
	switch runtime.GOOS {
	case "windows":
		return []string{"ze_loader.dll"}
	case "darwin":
		return []string{"libze_loader.dylib"}
	}
	return []string{"libze_loader.so.1", "libze_loader.so"}
}

// Load opens the Level Zero loader and binds the core functions.
func Load() (*API, error) {
	lib, err := dl.Open(append([]string{os.Getenv(LoaderEnv)}, LoaderCandidates()...)...)
	if err != nil {
		return nil, err
	}
	a := &API{Lib: lib}
	binds := []struct {
		fp   any
		name string
	}{
		{&a.zeInit, "zeInit"},
		{&a.zeDriverGet, "zeDriverGet"},
		{&a.zeDriverGetProperties, "zeDriverGetProperties"},
		{&a.zeDriverGetApiVersion, "zeDriverGetApiVersion"},
		{&a.zeDriverGetExtensionProperties, "zeDriverGetExtensionProperties"},
		{&a.zeDriverGetExtensionFunctionAddress, "zeDriverGetExtensionFunctionAddress"},
		{&a.zeDeviceGet, "zeDeviceGet"},
		{&a.zeDeviceGetProperties, "zeDeviceGetProperties"},
		{&a.zeDeviceGetCommandQueueGroupProps, "zeDeviceGetCommandQueueGroupProperties"},
		{&a.zeContextCreate, "zeContextCreate"},
		{&a.zeContextDestroy, "zeContextDestroy"},
		{&a.zeMemAllocHost, "zeMemAllocHost"},
		{&a.zeMemFree, "zeMemFree"},
		{&a.zeCommandQueueCreate, "zeCommandQueueCreate"},
		{&a.zeCommandQueueDestroy, "zeCommandQueueDestroy"},
		{&a.zeCommandQueueExecuteCommandLists, "zeCommandQueueExecuteCommandLists"},
		{&a.zeCommandListCreate, "zeCommandListCreate"},
		{&a.zeCommandListDestroy, "zeCommandListDestroy"},
		{&a.zeCommandListClose, "zeCommandListClose"},
		{&a.zeCommandListReset, "zeCommandListReset"},
		{&a.zeFenceCreate, "zeFenceCreate"},
		{&a.zeFenceDestroy, "zeFenceDestroy"},
		{&a.zeFenceHostSynchronize, "zeFenceHostSynchronize"},
		{&a.zeFenceQueryStatus, "zeFenceQueryStatus"},
		{&a.zeFenceReset, "zeFenceReset"},
	}
	for _, b := range binds {
		if err := lib.Bind(b.fp, b.name); err != nil {
			return nil, fmt.Errorf("%s: %w", lib.Path, err)
		}
	}
	return a, nil
}

// Init calls zeInit for NPU (VPU) drivers only.
func (a *API) Init() error { return check("zeInit", a.zeInit(initFlagVPUOnly)) }

// Drivers lists driver handles.
func (a *API) Drivers() ([]Handle, error) {
	var n uint32
	if err := check("zeDriverGet", a.zeDriverGet(&n, nil)); err != nil || n == 0 {
		return nil, err
	}
	hs := make([]Handle, n)
	if err := check("zeDriverGet", a.zeDriverGet(&n, &hs[0])); err != nil {
		return nil, err
	}
	return hs[:n], nil
}

// DriverProperties returns ze_driver_properties_t.
func (a *API) DriverProperties(d Handle) (DriverProperties, error) {
	p := DriverProperties{SType: stypeDriverProperties}
	err := check("zeDriverGetProperties", a.zeDriverGetProperties(d, &p))
	return p, err
}

// APIVersion returns the API version a driver implements.
func (a *API) APIVersion(d Handle) (uint32, error) {
	var v uint32
	err := check("zeDriverGetApiVersion", a.zeDriverGetApiVersion(d, &v))
	return v, err
}

// Extensions lists the extensions a driver supports.
func (a *API) Extensions(d Handle) ([]DriverExtensionProperties, error) {
	var n uint32
	if err := check("zeDriverGetExtensionProperties", a.zeDriverGetExtensionProperties(d, &n, nil)); err != nil || n == 0 {
		return nil, err
	}
	ps := make([]DriverExtensionProperties, n)
	if err := check("zeDriverGetExtensionProperties", a.zeDriverGetExtensionProperties(d, &n, &ps[0])); err != nil {
		return nil, err
	}
	return ps[:n], nil
}

// ExtensionFunction returns the address registered for an extension name. For the graph
// extension it is the address of a ze_graph_dditable_ext_t.
func (a *API) ExtensionFunction(d Handle, name string) (unsafe.Pointer, error) {
	cname := append([]byte(name), 0)
	var p unsafe.Pointer
	if err := check("zeDriverGetExtensionFunctionAddress("+name+")", a.zeDriverGetExtensionFunctionAddress(d, &cname[0], &p)); err != nil {
		return nil, err
	}
	if p == nil {
		return nil, fmt.Errorf("zeDriverGetExtensionFunctionAddress(%s) returned NULL", name)
	}
	return p, nil
}

// Devices lists the devices of a driver.
func (a *API) Devices(d Handle) ([]Handle, error) {
	var n uint32
	if err := check("zeDeviceGet", a.zeDeviceGet(d, &n, nil)); err != nil || n == 0 {
		return nil, err
	}
	hs := make([]Handle, n)
	if err := check("zeDeviceGet", a.zeDeviceGet(d, &n, &hs[0])); err != nil {
		return nil, err
	}
	return hs[:n], nil
}

// DeviceProperties returns ze_device_properties_t.
func (a *API) DeviceProperties(dev Handle) (DeviceProperties, error) {
	p := DeviceProperties{SType: stypeDeviceProperties}
	err := check("zeDeviceGetProperties", a.zeDeviceGetProperties(dev, &p))
	return p, err
}

// CreateContext creates a context on a driver.
func (a *API) CreateContext(d Handle) (Handle, error) {
	var c Handle
	err := check("zeContextCreate", a.zeContextCreate(d, &simpleDesc{SType: stypeContextDesc}, &c))
	return c, err
}

func (a *API) DestroyContext(c Handle) error { return check("zeContextDestroy", a.zeContextDestroy(c)) }

// AllocHost allocates host memory visible to the device.
func (a *API) AllocHost(ctx Handle, size, alignment int) (unsafe.Pointer, error) {
	if size <= 0 {
		return nil, errors.New("zeMemAllocHost: size must be positive")
	}
	var p unsafe.Pointer
	if err := check("zeMemAllocHost", a.zeMemAllocHost(ctx, &simpleDesc{SType: stypeHostMemAllocDesc}, uintptr(size), uintptr(alignment), &p)); err != nil {
		return nil, err
	}
	return p, nil
}

func (a *API) Free(ctx Handle, p unsafe.Pointer) error {
	return check("zeMemFree", a.zeMemFree(ctx, p))
}

// ComputeOrdinal returns the command queue group to use for compute work, choosing like the
// OpenVINO NPU plugin: a group with only the compute flag, then any group with it, then 0.
func (a *API) ComputeOrdinal(dev Handle) (uint32, error) {
	var n uint32
	if err := check("zeDeviceGetCommandQueueGroupProperties", a.zeDeviceGetCommandQueueGroupProps(dev, &n, nil)); err != nil || n == 0 {
		return 0, err
	}
	ps := make([]CommandQueueGroupProperties, n)
	for i := range ps {
		ps[i].SType = stypeCommandQueueGroupProps
	}
	if err := check("zeDeviceGetCommandQueueGroupProperties", a.zeDeviceGetCommandQueueGroupProps(dev, &n, &ps[0])); err != nil {
		return 0, err
	}
	return SelectComputeOrdinal(ps[:n]), nil
}

// SelectComputeOrdinal implements the choice of ComputeOrdinal.
func SelectComputeOrdinal(ps []CommandQueueGroupProperties) uint32 {
	for i, p := range ps {
		if p.Flags == QueueGroupFlagCompute {
			return uint32(i)
		}
	}
	for i, p := range ps {
		if p.Flags&QueueGroupFlagCompute != 0 {
			return uint32(i)
		}
	}
	return 0
}

// QueueOptions configure CreateCommandQueue.
type QueueOptions struct {
	Ordinal  uint32
	Priority uint32  // QueuePriority*
	PNext    uintptr // extension descriptor chain, 0 for none
}

// CreateCommandQueue creates a queue with default mode.
func (a *API) CreateCommandQueue(ctx, dev Handle, o QueueOptions) (Handle, error) {
	var q Handle
	desc := CommandQueueDesc{SType: stypeCommandQueueDesc, PNext: o.PNext, Ordinal: o.Ordinal, Mode: commandQueueModeDefault, Priority: o.Priority}
	err := check("zeCommandQueueCreate", a.zeCommandQueueCreate(ctx, dev, &desc, &q))
	return q, err
}

func (a *API) DestroyCommandQueue(q Handle) error {
	return check("zeCommandQueueDestroy", a.zeCommandQueueDestroy(q))
}

// Execute submits one command list and signals fence on completion.
func (a *API) Execute(q, list, fence Handle) error {
	return check("zeCommandQueueExecuteCommandLists", a.zeCommandQueueExecuteCommandLists(q, 1, &list, fence))
}

func (a *API) CreateCommandList(ctx, dev Handle, ordinal uint32) (Handle, error) {
	var l Handle
	err := check("zeCommandListCreate", a.zeCommandListCreate(ctx, dev, &CommandListDesc{SType: stypeCommandListDesc, CommandQueueGroupOrdinal: ordinal}, &l))
	return l, err
}

func (a *API) DestroyCommandList(l Handle) error {
	return check("zeCommandListDestroy", a.zeCommandListDestroy(l))
}
func (a *API) CloseCommandList(l Handle) error {
	return check("zeCommandListClose", a.zeCommandListClose(l))
}
func (a *API) ResetCommandList(l Handle) error {
	return check("zeCommandListReset", a.zeCommandListReset(l))
}

func (a *API) CreateFence(q Handle) (Handle, error) {
	var f Handle
	err := check("zeFenceCreate", a.zeFenceCreate(q, &simpleDesc{SType: stypeFenceDesc}, &f))
	return f, err
}

func (a *API) DestroyFence(f Handle) error { return check("zeFenceDestroy", a.zeFenceDestroy(f)) }
func (a *API) ResetFence(f Handle) error   { return check("zeFenceReset", a.zeFenceReset(f)) }

// SyncFence waits up to timeoutNs. It reports false without error when the fence is not
// signalled yet.
func (a *API) SyncFence(f Handle, timeoutNs uint64) (bool, error) {
	r := a.zeFenceHostSynchronize(f, timeoutNs)
	if r == ResultNotReady {
		return false, nil
	}
	return r == ResultSuccess, check("zeFenceHostSynchronize", r)
}

// FenceDone reports without blocking whether the fence is signalled.
func (a *API) FenceDone(f Handle) (bool, error) {
	r := a.zeFenceQueryStatus(f)
	if r == ResultNotReady {
		return false, nil
	}
	return r == ResultSuccess, check("zeFenceQueryStatus", r)
}
