// Package coreml implements the Apple backend on top of Core ML through purego/objc.
// Core ML cannot be restricted to the Neural Engine, so models run with
// MLComputeUnitsCPUAndNeuralEngine (no GPU) and placement is checked with MLComputePlan.
package coreml

import (
	"context"
	"fmt"
	"runtime"

	"github.com/ebitengine/purego"
	"github.com/ebitengine/purego/objc"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/mem"
	"github.com/shibukawa/hwnpu/internal/types"
)

// MLComputeUnits values.
const (
	computeUnitsCPUOnly            = 0
	computeUnitsCPUAndGPU          = 1
	computeUnitsAll                = 2
	computeUnitsCPUAndNeuralEngine = 3
)

// ErrUnsupported is returned for requests this backend cannot serve.
var ErrUnsupported = types.ErrUnsupported

// Provider is the Core ML backend provider.
type Provider struct{}

func (Provider) Name() string { return "coreml" }

func (Provider) Enumerate(ctx context.Context) ([]types.DeviceInfo, error) {
	if err := load(); err != nil {
		return nil, err
	}
	var infos []types.DeviceInfo
	err := withPool(func() error {
		name, ok := neuralEngineName()
		if !ok {
			return nil
		}
		caps := types.CapGraph | types.CapCompile | types.CapHostCoherent
		if class("MLComputePlan") != 0 {
			caps |= types.CapStrictVerify
		}
		infos = append(infos, types.DeviceInfo{
			Vendor:  types.VendorApple,
			Backend: "coreml",
			Name:    name,
			Caps:    caps,
			Driver:  "Core ML / " + goString(send(send(class("NSProcessInfo"), "processInfo"), "operatingSystemVersionString")),
		})
		return nil
	})
	return infos, err
}

// neuralEngineName reports whether Core ML sees a Neural Engine.
func neuralEngineName() (string, bool) {
	if mlAllComputeDevices == 0 {
		// Before macOS 14 there is no device list. Every Apple Silicon Mac has a Neural Engine.
		if runtime.GOARCH == "arm64" {
			return "Apple Neural Engine", true
		}
		return "", false
	}
	r1, _, _ := purego.SyscallN(mlAllComputeDevices)
	for _, d := range arrayItems(objc.ID(r1)) {
		if isKind(d, "MLNeuralEngineComputeDevice") {
			return fmt.Sprintf("Apple Neural Engine (%d cores)", sendInt(d, "totalCoreCount")), true
		}
	}
	return "", false
}

func (Provider) Open(ctx context.Context, info types.DeviceInfo, opts backend.Options) (backend.Device, error) {
	if err := load(); err != nil {
		return nil, err
	}
	// Core ML has no power controls (opts.Power is ignored; the device lacks CapPowerControl).
	return &device{opts: opts}, nil
}

type device struct{ opts backend.Options }

func (d *device) AllocBuffer(size int, kind types.BufferKind) (backend.Buffer, error) {
	b, err := mem.Alloc(size)
	if err != nil {
		return nil, err
	}
	return &buffer{b: b}, nil
}

func (d *device) Close() error { return nil }

// buffer is page aligned anonymous memory. Core ML wraps it in MLMultiArray without copying.
type buffer struct{ b []byte }

func (b *buffer) Bytes() []byte                   { return b.b }
func (b *buffer) SyncToDevice(off, n int) error   { return nil }
func (b *buffer) SyncFromDevice(off, n int) error { return nil }
func (b *buffer) Close() error {
	err := mem.Free(b.b)
	b.b = nil
	return err
}
