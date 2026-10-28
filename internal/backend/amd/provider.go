// Package amd implements the AMD Ryzen AI (XDNA) backend by talking to the Linux amdxdna driver
// directly, following the XRT shim (github.com/amd/xdna-driver, src/shim) for the xclbin flow:
// a device heap per open device, PDIs in BO_DEV buffer objects configured as compute units of a
// hardware context, ERT_START_CU commands in BO_CMD buffer objects, and completion on the
// context's timeline syncobj.
package amd

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"

	"github.com/shibukawa/hwnpu/internal/accelfs"
	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/types"
	"github.com/shibukawa/hwnpu/internal/xdna"
)

// Node is an amdxdna device node.
type Node struct {
	Path     string // /dev/accel/accelN
	Device   string // PCI device id, e.g. "0x1502"
	Revision string // PCI revision, e.g. "0x10"
}

// Provider is the AMD backend provider. The zero value uses the real driver.
type Provider struct {
	// ProviderName overrides the backend name ("xdna").
	ProviderName string
	// Nodes lists device nodes; nil scans /sys/class/accel on Linux.
	Nodes func() ([]Node, error)
	// OpenDriver opens a node; nil uses xdna.OpenNode.
	OpenDriver func(path string) (xdna.Driver, error)
}

func (p Provider) Name() string {
	if p.ProviderName != "" {
		return p.ProviderName
	}
	return "xdna"
}

func (p Provider) nodes() ([]Node, error) {
	if p.Nodes != nil {
		return p.Nodes()
	}
	if runtime.GOOS != "linux" {
		return nil, nil
	}
	all, err := accelfs.List()
	if err != nil {
		return nil, err
	}
	var out []Node
	for _, n := range accelfs.ByDriver(all, accelfs.DriverAMD) {
		rev, _ := os.ReadFile(filepath.Join("/sys/class/accel", n.Name, "device", "revision"))
		out = append(out, Node{Path: n.Path, Device: n.Device, Revision: strings.TrimSpace(string(rev))})
	}
	return out, nil
}

func (p Provider) open(path string) (xdna.Driver, error) {
	open := p.OpenDriver
	if open == nil {
		open = xdna.OpenNode
	}
	d, err := open(path)
	if errors.Is(err, syscall.EACCES) || errors.Is(err, syscall.EPERM) {
		return nil, fmt.Errorf("open %s: %w (add the user to the render group, or allow access with udev)", path, err)
	}
	return d, err
}

// generation names the NPU from its PCI device id and revision (amdxdna_pci_drv.c).
func generation(n Node) string {
	switch strings.ToLower(n.Device) {
	case "0x1502":
		return "npu1 (Phoenix / Hawk Point)"
	case "0x17f0":
		switch strings.ToLower(n.Revision) {
		case "0x10":
			return "npu4 (Strix Point)"
		case "0x11":
			return "npu5"
		case "0x20":
			return "npu6"
		}
	}
	return "XDNA " + n.Device
}

func (p Provider) Enumerate(ctx context.Context) ([]types.DeviceInfo, error) {
	nodes, err := p.nodes()
	if err != nil {
		return nil, err
	}
	var infos []types.DeviceInfo
	var errs []error
	for _, n := range nodes {
		drv, err := p.open(n.Path)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		d := xdna.Dev{D: drv}
		meta, err := d.Metadata()
		if err != nil {
			drv.Close()
			errs = append(errs, fmt.Errorf("%s: %w", n.Path, err))
			continue
		}
		caps := types.CapKernel | types.CapPowerControl
		if _, err := d.Sensors(); err == nil {
			caps |= types.CapPowerMeter
		}
		info := types.DeviceInfo{
			Vendor: types.VendorAMD, Backend: p.Name(), ID: n.Path, Path: n.Path, Caps: caps,
			Name:   fmt.Sprintf("AMD NPU %s, %d columns", generation(n), meta.Cols),
			Driver: fmt.Sprintf("amdxdna, AIE %d.%d, %d core rows", meta.VersionMajor, meta.VersionMinor, meta.Core.RowCount),
		}
		if fw, err := d.Firmware(); err == nil {
			info.Firmware = fmt.Sprintf("%d.%d.%d.%d", fw.Major, fw.Minor, fw.Patch, fw.Build)
		}
		drv.Close()
		infos = append(infos, info)
	}
	return infos, errors.Join(errs...)
}

func (p Provider) Open(ctx context.Context, info types.DeviceInfo, opts backend.Options) (backend.Device, error) {
	drv, err := p.open(info.ID)
	if err != nil {
		return nil, err
	}
	d := &device{dev: xdna.Dev{D: drv}, opts: opts}
	if d.meta, err = d.dev.Metadata(); err != nil {
		drv.Close()
		return nil, err
	}
	// The device heap holds BO_DEV objects (PDIs and instruction streams). It must be mapped
	// before BO_DEV objects are created; their CPU addresses are offsets into this mapping.
	if d.heap, err = d.dev.CreateBO(xdna.BODevHeap, xdna.DevHeapSize); err != nil {
		drv.Close()
		return nil, fmt.Errorf("amd: create device heap: %w", err)
	}
	if d.heapMem, err = drv.Mmap(d.heap.MapOffset, xdna.DevHeapSize); err != nil {
		d.dev.CloseBO(d.heap.Handle)
		drv.Close()
		return nil, fmt.Errorf("amd: map device heap: %w", err)
	}
	return d, nil
}
