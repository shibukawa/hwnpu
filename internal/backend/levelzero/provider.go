// Package levelzero implements the Intel NPU backend on Level Zero and the NPU graph extension.
package levelzero

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"strconv"
	"strings"
	"sync"

	"github.com/shibukawa/hwnpu/internal/accelfs"
	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/dl"
	"github.com/shibukawa/hwnpu/internal/power"
	"github.com/shibukawa/hwnpu/internal/types"
	"github.com/shibukawa/hwnpu/internal/ze"
)

// Provider is the Level Zero backend provider.
type Provider struct{}

func (Provider) Name() string { return "levelzero" }

type zeRuntime struct {
	api     *ze.API
	drivers []*driver
}

type driver struct {
	handle     ze.Handle
	props      ze.DriverProperties
	apiVersion uint32
	graph      *ze.GraphExt
	queueExt   *ze.QueueExt // nil when the driver lacks the NPU command queue extension
	devices    []*deviceEntry
}

type deviceEntry struct {
	handle  ze.Handle
	props   ze.DeviceProperties
	gprops  ze.DeviceGraphProperties
	ordinal uint32 // compute command queue group
}

var (
	rtOnce sync.Once
	rt     *zeRuntime
	rtErr  error
)

// loadRuntime loads the loader, initializes NPU drivers and binds their graph extensions once.
func loadRuntime() (*zeRuntime, error) {
	rtOnce.Do(func() {
		api, err := ze.Load()
		if err != nil {
			if errors.Is(err, dl.ErrNotFound) {
				err = fmt.Errorf("%w: Level Zero loader (%v); install the Intel NPU driver and the level-zero loader, or set %s", types.ErrLibrary, err, ze.LoaderEnv)
			}
			rtErr = err
			return
		}
		if err := api.Init(); err != nil {
			rtErr = driverErr(err)
			return
		}
		handles, err := api.Drivers()
		if err != nil {
			rtErr = driverErr(err)
			return
		}
		r := &zeRuntime{api: api}
		var errs []error
		for _, h := range handles {
			d, err := openDriver(api, h)
			if err != nil {
				errs = append(errs, err)
				continue
			}
			if d != nil {
				r.drivers = append(r.drivers, d)
			}
		}
		if len(r.drivers) == 0 && len(errs) > 0 {
			rtErr = errors.Join(errs...)
			return
		}
		rt = r
	})
	return rt, rtErr
}

// openDriver returns nil when the driver has no NPU device.
func openDriver(api *ze.API, h ze.Handle) (*driver, error) {
	d := &driver{handle: h}
	var err error
	if d.props, err = api.DriverProperties(h); err != nil {
		return nil, driverErr(err)
	}
	if d.apiVersion, err = api.APIVersion(h); err != nil {
		return nil, driverErr(err)
	}
	devs, err := api.Devices(h)
	if err != nil {
		return nil, driverErr(err)
	}
	for _, dh := range devs {
		p, err := api.DeviceProperties(dh)
		if err != nil {
			return nil, driverErr(err)
		}
		if p.Type == ze.DeviceTypeVPU {
			d.devices = append(d.devices, &deviceEntry{handle: dh, props: p})
		}
	}
	if len(d.devices) == 0 {
		return nil, nil
	}
	if d.graph, err = api.LoadGraphExt(h); err != nil {
		return nil, &types.VersionError{Component: ze.GraphExtName, Have: err.Error(), Need: ze.VersionString(ze.GraphExtMinVersion) + " or later"}
	}
	// The queue extension is optional; without it power policies only set queue priorities.
	d.queueExt, _ = api.LoadQueueExt(h)
	for _, e := range d.devices {
		if e.gprops, err = d.graph.DeviceGraphProperties(e.handle); err != nil {
			return nil, driverErr(err)
		}
		if e.ordinal, err = api.ComputeOrdinal(e.handle); err != nil {
			return nil, driverErr(err)
		}
	}
	return d, nil
}

func (Provider) Enumerate(ctx context.Context) ([]types.DeviceInfo, error) {
	var nodes []accelfs.Node
	if runtime.GOOS == "linux" {
		all, _ := accelfs.List()
		nodes = accelfs.ByDriver(all, accelfs.DriverIntel)
	}
	r, err := loadRuntime()
	if err != nil {
		if errors.Is(err, types.ErrLibrary) && len(nodes) == 0 {
			return nil, nil // no Intel NPU and no runtime: nothing to report
		}
		if len(nodes) > 0 {
			return nil, fmt.Errorf("found %s but cannot use it: %w", nodes[0].Path, err)
		}
		return nil, err
	}
	meter := openMeter() != nil
	var infos []types.DeviceInfo
	n := 0
	for di, d := range r.drivers {
		for ei, e := range d.devices {
			caps := types.CapGraph | types.CapHostCoherent | types.CapStrictVerify
			if e.gprops.GraphFormatsSupported&ze.GraphFormatNGraphLite != 0 {
				caps |= types.CapCompile
			}
			caps |= types.CapPowerControl // queue priority always; turbo and workload type with the extension
			if meter {
				caps |= types.CapPowerMeter
			}
			queue := "no " + ze.QueueExtName
			if d.queueExt != nil {
				queue = fmt.Sprintf("%s %s", d.queueExt.Name, ze.VersionString(d.queueExt.Version))
			}
			info := types.DeviceInfo{
				Vendor:  types.VendorIntel,
				Backend: "levelzero",
				ID:      fmt.Sprintf("%d:%d", di, ei),
				Name:    ze.CString(e.props.Name[:]),
				Caps:    caps,
				Driver: fmt.Sprintf("Level Zero %s, driver %d, %s %s, compiler %s, %s",
					ze.VersionString(d.apiVersion), d.props.DriverVersion, d.graph.Name,
					ze.VersionString(d.graph.Version), e.gprops.CompilerVersion, queue),
			}
			if n < len(nodes) {
				info.Path = nodes[n].Path
			}
			n++
			infos = append(infos, info)
		}
	}
	return infos, nil
}

func (Provider) Open(ctx context.Context, info types.DeviceInfo, opts backend.Options) (backend.Device, error) {
	r, err := loadRuntime()
	if err != nil {
		return nil, err
	}
	di, ei, ok := strings.Cut(info.ID, ":")
	d, err1 := strconv.Atoi(di)
	e, err2 := strconv.Atoi(ei)
	if !ok || err1 != nil || err2 != nil || d >= len(r.drivers) || e >= len(r.drivers[d].devices) {
		return nil, fmt.Errorf("levelzero: unknown device id %q", info.ID)
	}
	drv := r.drivers[d]
	ctxh, err := r.api.CreateContext(drv.handle)
	if err != nil {
		return nil, driverErr(err)
	}
	return &device{api: r.api, drv: drv, entry: drv.devices[e], ctx: ctxh, opts: opts, meter: openMeter()}, nil
}

// openMeter returns the package energy counter on Linux, or nil.
func openMeter() *power.RAPL {
	if runtime.GOOS != "linux" {
		return nil
	}
	m, err := power.OpenRAPL("/sys/class/powercap")
	if err != nil {
		return nil
	}
	return m
}

// driverErr converts a Level Zero failure to *types.DriverError.
func driverErr(err error) error {
	var ze_ *ze.Error
	if errors.As(err, &ze_) {
		return &types.DriverError{Backend: "levelzero", Op: ze_.Op, Code: int64(ze_.Result), Msg: ze_.Result.String()}
	}
	return err
}
