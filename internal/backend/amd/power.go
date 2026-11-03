package amd

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/types"
	"github.com/shibukawa/hwnpu/internal/xdna"
)

// powerInterval is how often the meter samples the power to integrate it into energy.
const powerInterval = 50 * time.Millisecond

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

// PowerSample returns the power now and the energy integrated since the first call. The first
// call starts a sampler that integrates every powerInterval until Close: both sources are
// instantaneous readings, so energy from two samples alone would be the mean of two moments.
func (d *device) PowerSample() (types.PowerSample, error) {
	w, scope, source, err := d.watts()
	if err != nil {
		return types.PowerSample{}, err
	}
	now := time.Now()
	d.powerMu.Lock()
	defer d.powerMu.Unlock()
	d.integrate(w, now)
	if d.powerStop == nil {
		d.powerStop, d.powerDone = make(chan struct{}), make(chan struct{})
		go d.sample(d.powerStop, d.powerDone)
	}
	return types.PowerSample{Time: now, EnergyJ: d.energyJ, Watts: w, Scope: scope, Source: source + " (integrated)"}, nil
}

// integrate adds the energy since the last reading, by the trapezoid rule. powerMu is held.
func (d *device) integrate(w float64, now time.Time) {
	if !d.lastT.IsZero() {
		d.energyJ += (d.lastW + w) / 2 * now.Sub(d.lastT).Seconds()
	}
	d.lastW, d.lastT = w, now
}

func (d *device) sample(stop <-chan struct{}, done chan<- struct{}) {
	defer close(done)
	t := time.NewTicker(powerInterval)
	defer t.Stop()
	for {
		select {
		case <-stop:
			return
		case <-t.C:
			if w, _, _, err := d.watts(); err == nil {
				d.powerMu.Lock()
				d.integrate(w, time.Now())
				d.powerMu.Unlock()
			}
		}
	}
}

// stopPower ends the sampler.
func (d *device) stopPower() {
	d.powerMu.Lock()
	stop, done := d.powerStop, d.powerDone
	d.powerStop = nil
	d.powerMu.Unlock()
	if stop != nil {
		close(stop)
		<-done
	}
}
