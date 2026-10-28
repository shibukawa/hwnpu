package hwnpu

import (
	"context"
	"errors"
	"fmt"

	"github.com/shibukawa/hwnpu/internal/backend"
)

// Enumerate lists the devices of every backend compiled into the binary. Backends that fail to
// load are reported through the returned error (joined), while the others are still listed.
func Enumerate(ctx context.Context) ([]DeviceInfo, error) {
	var (
		all  []DeviceInfo
		errs []error
	)
	for _, p := range backend.Providers() {
		infos, err := p.Enumerate(ctx)
		if err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", p.Name(), err))
			continue
		}
		for _, info := range infos {
			info.Index = len(all)
			all = append(all, info)
		}
	}
	return all, errors.Join(errs...)
}

// Open opens the first device that matches the options.
func Open(ctx context.Context, opts ...Option) (*Device, error) {
	o := defaultOptions()
	for _, opt := range opts {
		opt(&o)
	}
	infos, enumErr := Enumerate(ctx)
	n := 0
	for _, info := range infos {
		if o.vendor != VendorUnknown && info.Vendor != o.vendor {
			continue
		}
		if o.backend != "" && info.Backend != o.backend {
			continue
		}
		if o.index >= 0 && n != o.index {
			n++
			continue
		}
		return OpenDevice(ctx, info, opts...)
	}
	if enumErr != nil {
		return nil, errors.Join(ErrNoDevice, enumErr)
	}
	return nil, ErrNoDevice
}

// OpenDevice opens an enumerated device.
func OpenDevice(ctx context.Context, info DeviceInfo, opts ...Option) (*Device, error) {
	o := defaultOptions()
	for _, opt := range opts {
		opt(&o)
	}
	for _, p := range backend.Providers() {
		if p.Name() != info.Backend {
			continue
		}
		be, err := p.Open(ctx, info, backend.Options{Power: o.power, CacheDir: o.cacheDir})
		if err != nil {
			return nil, err
		}
		return &Device{info: info, be: be, opts: o}, nil
	}
	return nil, fmt.Errorf("%w: backend %q is not compiled in", ErrUnsupported, info.Backend)
}

// Device is an opened NPU.
type Device struct {
	info DeviceInfo
	be   backend.Device
	opts options
}

// Info returns the enumerated information of the device.
func (d *Device) Info() DeviceInfo { return d.info }

// Close releases the device. Close graphs, requests and buffers first.
func (d *Device) Close() error {
	if d.be == nil {
		return ErrClosed
	}
	err := d.be.Close()
	d.be = nil
	return err
}

// AllocBuffer allocates device visible memory of size bytes.
func (d *Device) AllocBuffer(size int, opts ...BufferOption) (*Buffer, error) {
	if d.be == nil {
		return nil, ErrClosed
	}
	var o bufferOptions
	for _, opt := range opts {
		opt(&o)
	}
	b, err := d.be.AllocBuffer(size, o.kind)
	if err != nil {
		return nil, err
	}
	return &Buffer{be: b, size: size, dev: d}, nil
}

// PowerPolicy returns the power policy the device was opened with.
func (d *Device) PowerPolicy() PowerPolicy { return d.opts.power }

// PowerSample reads the device's energy counter. Check Scope: a "package" sample covers the
// whole CPU package, not the NPU alone. Devices without CapPowerMeter return ErrUnsupported.
func (d *Device) PowerSample() (PowerSample, error) {
	if d.be == nil {
		return PowerSample{}, ErrClosed
	}
	m, ok := d.be.(backend.PowerMeter)
	if !ok || d.info.Caps&CapPowerMeter == 0 {
		return PowerSample{}, ErrUnsupported
	}
	return m.PowerSample()
}

// Graph returns the graph runner, or ErrUnsupported when the device cannot run graphs.
func (d *Device) Graph() (GraphRunner, error) {
	if d.be == nil {
		return nil, ErrClosed
	}
	if d.info.Caps&CapGraph == 0 {
		return nil, ErrUnsupported
	}
	return graphRunner{d: d}, nil
}

// Buffer is device visible memory. Bytes returns a view without copying.
type Buffer struct {
	be   backend.Buffer
	size int
	dev  *Device
}

// Bytes returns the buffer contents. The slice is invalid after Close.
func (b *Buffer) Bytes() []byte {
	if b.be == nil {
		return nil
	}
	return b.be.Bytes()
}

// Size returns the size requested in AllocBuffer.
func (b *Buffer) Size() int { return b.size }

// SyncToDevice makes CPU writes in [off, off+n) visible to the device. Call it before Submit.
// It is a no-op on devices with CapHostCoherent.
func (b *Buffer) SyncToDevice(off, n int) error {
	if b.be == nil {
		return ErrClosed
	}
	return b.be.SyncToDevice(off, n)
}

// SyncFromDevice makes device writes in [off, off+n) visible to the CPU. Call it after Wait.
func (b *Buffer) SyncFromDevice(off, n int) error {
	if b.be == nil {
		return ErrClosed
	}
	return b.be.SyncFromDevice(off, n)
}

// Close frees the buffer.
func (b *Buffer) Close() error {
	if b.be == nil {
		return ErrClosed
	}
	err := b.be.Close()
	b.be = nil
	return err
}
