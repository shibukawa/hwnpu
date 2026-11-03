package amd

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/shibukawa/hwnpu/internal/power"
	"github.com/shibukawa/hwnpu/internal/types"
	"github.com/shibukawa/hwnpu/internal/xdna"
)

// packagePowerPath finds the package power the amdgpu driver reports on an APU (its hwmon's
// power1_input, PPT: the whole socket, cores, GPU and NPU, in microwatts), readable without
// root. "" when there is none.
func packagePowerPath(root string) string {
	if root == "" {
		root = "/sys/class/hwmon"
	}
	dirs, _ := filepath.Glob(filepath.Join(root, "hwmon*"))
	for _, d := range dirs {
		name, err := os.ReadFile(filepath.Join(d, "name"))
		if err != nil || strings.TrimSpace(string(name)) != "amdgpu" {
			continue
		}
		p := filepath.Join(d, "power1_input")
		if _, err := readMicrowatts(p); err == nil {
			return p
		}
	}
	return ""
}

func readMicrowatts(path string) (float64, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(string(b)), 64)
	if err != nil {
		return 0, err
	}
	return v / 1e6, nil
}

// watts reads the power now: the NPU's sensor where the driver has one (AMD's DKMS driver;
// Linux 7.0's has none), else the package power (PPT).
func (d *device) watts() (float64, types.PowerScope, string, error) {
	if sensors, err := d.dev.Sensors(); err == nil {
		for _, s := range sensors {
			if s.Type == xdna.SensorTypePower {
				return float64(s.Input) * math.Pow10(int(s.UnitM)), types.ScopeNPU, "amdxdna-sensor", nil
			}
		}
	}
	if d.pptPath != "" {
		w, err := readMicrowatts(d.pptPath)
		if err != nil {
			return 0, "", "", err
		}
		return w, types.ScopePackage, "amdgpu PPT", nil
	}
	return 0, "", "", fmt.Errorf("amd: no power sensor: %w", types.ErrUnsupported)
}

// PowerSample returns the power now and the energy integrated since the first call (both
// sources are instantaneous readings: power.Integrator).
func (d *device) PowerSample() (types.PowerSample, error) {
	d.powerMu.Lock()
	if d.meter == nil {
		d.meter = power.NewIntegrator(d.watts)
	}
	m := d.meter
	d.powerMu.Unlock()
	return m.Sample()
}

// stopPower ends the sampler.
func (d *device) stopPower() {
	d.powerMu.Lock()
	m := d.meter
	d.powerMu.Unlock()
	if m != nil {
		m.Stop()
	}
}
