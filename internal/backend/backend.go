// Package backend defines the interface every NPU backend implements and a registry of the
// backends compiled into the current binary.
package backend

import (
	"context"
	"sync"

	"github.com/shibukawa/hwnpu/internal/types"
)

// Provider enumerates and opens devices of one backend.
type Provider interface {
	Name() string
	// Enumerate lists devices. An error means the backend could not look for devices;
	// an empty result means it looked and found none.
	Enumerate(ctx context.Context) ([]types.DeviceInfo, error)
	Open(ctx context.Context, info types.DeviceInfo, opts Options) (Device, error)
}

// Options are the device level settings chosen by the user.
type Options struct {
	Power    types.PowerPolicy
	CacheDir string // compiled graph cache, "" disables caching
}

// KernelDevice is implemented by devices with CapKernel.
type KernelDevice interface {
	LoadProgram(ctx context.Context, src types.ProgramSource) (Program, error)
}

// Program is a loaded kernel program (an AMD hardware context with its compute units).
type Program interface {
	Kernels() []types.KernelInfo
	NewRun(kernel string) (Run, error)
	Close() error
}

// Run is one kernel invocation with its arguments. SetArg receives little endian bytes of
// exactly the argument's size.
type Run interface {
	SetArg(index int, value []byte) error
	SetBuffer(index int, b Buffer) error
	Submit() (Fence, error)
	Close() error
}

// PowerMeter is implemented by devices with CapPowerMeter.
type PowerMeter interface {
	PowerSample() (types.PowerSample, error)
}

// Device is an opened device.
type Device interface {
	AllocBuffer(size int, kind types.BufferKind) (Buffer, error)
	// Compile returns ErrUnsupported (from the public package) when the device has no graph support.
	Compile(ctx context.Context, src types.GraphSource) (Graph, error)
	Close() error
}

// Buffer is device visible memory.
type Buffer interface {
	Bytes() []byte
	SyncToDevice(off, n int) error
	SyncFromDevice(off, n int) error
	Close() error
}

// Graph is a compiled graph.
type Graph interface {
	Args() []types.ArgInfo
	// Cached reports whether the graph was loaded from the compile cache.
	Cached() bool
	Placement() types.PlacementReport
	NewRequest() (Request, error)
	Close() error
}

// Request binds buffers to graph arguments and submits executions.
// Bind has already been checked against ArgInfo by the caller.
type Request interface {
	Bind(index int, b Buffer) error
	Submit() (Fence, error)
	Close() error
}

// Fence waits for one submission.
type Fence interface {
	Wait(ctx context.Context) error
	Done() (bool, error)
}

var (
	mu        sync.Mutex
	providers []Provider
)

// Register adds a provider. It is called from init functions of build-tagged files.
func Register(p Provider) {
	mu.Lock()
	defer mu.Unlock()
	providers = append(providers, p)
}

// Providers returns the registered providers in registration order.
func Providers() []Provider {
	mu.Lock()
	defer mu.Unlock()
	return append([]Provider(nil), providers...)
}
