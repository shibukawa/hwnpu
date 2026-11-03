// Package amdxrt implements the AMD Ryzen AI (XDNA) backend through XRT, for Windows, where
// AMD's NPU driver is reached only through XRT's C++ library (xrt_coreutil.dll). hwnpu has no
// cgo, so it calls XRT through xrtshim, a small C library of its own (native/xrtshim), loaded
// at run time. On Linux, hwnpu talks to the amdxdna driver itself (internal/backend/amd); this
// backend runs there too, against the XRT packages, which is how it is tested, but lists a
// device only when asked (HWNPU_BACKEND=xrt).
package amdxrt

import (
	"context"
	"errors"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"sync/atomic"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/types"
)

// Name is the backend's name, and the value of HWNPU_BACKEND that turns it on outside Windows.
const Name = "xrt"

// Provider is the XRT backend provider.
type Provider struct{}

func (Provider) Name() string { return Name }

var enabled atomic.Bool

// Enable makes the backend list its devices outside Windows without HWNPU_BACKEND=xrt.
func Enable() { enabled.Store(true) }

func listing() bool {
	return runtime.GOOS == "windows" || os.Getenv("HWNPU_BACKEND") == Name || enabled.Load()
}

func (Provider) Enumerate(ctx context.Context) ([]types.DeviceInfo, error) {
	if !listing() {
		return nil, nil
	}
	s, err := loadShim()
	if err != nil {
		if runtime.GOOS == "windows" && errors.Is(err, types.ErrLibrary) {
			return nil, nil // no shim: as no device, so that other backends' devices are still found
		}
		return nil, err
	}
	var n uint32
	if err := s.err(s.devCount(&n)); err != nil {
		return nil, err
	}
	var infos []types.DeviceInfo
	var errs []error
	for i := uint32(0); i < n; i++ {
		var d uintptr
		if err := s.err(s.devOpen(i, &d)); err != nil {
			errs = append(errs, err)
			continue
		}
		name, err := s.info(d, infoName)
		if err != nil {
			errs = append(errs, err)
		}
		bdf, _ := s.info(d, infoBDF)
		ver, _ := s.info(d, infoXRTVersion)
		caps := types.CapKernel
		if _, err := s.watts(d); err == nil {
			caps |= types.CapPowerMeter
		}
		s.devFree(d)
		info := types.DeviceInfo{
			Vendor: types.VendorAMD, Backend: Name, ID: strconv.Itoa(int(i)), Path: bdf, Caps: caps,
			Name:   "AMD NPU " + name,
			Driver: "XRT " + ver + " through xrtshim",
		}
		infos = append(infos, info)
	}
	return infos, errors.Join(errs...)
}

func (Provider) Open(ctx context.Context, info types.DeviceInfo, opts backend.Options) (backend.Device, error) {
	s, err := loadShim()
	if err != nil {
		return nil, err
	}
	i, err := strconv.Atoi(info.ID)
	if err != nil {
		return nil, fmt.Errorf("amdxrt: device id %q", info.ID)
	}
	d := &device{s: s, opts: opts}
	if err := s.err(s.devOpen(uint32(i), &d.h)); err != nil {
		return nil, err
	}
	return d, nil
}
