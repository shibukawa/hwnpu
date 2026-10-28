// Package power reads energy counters.
//
// RAPL (Running Average Power Limit) is exposed by Linux under /sys/class/powercap for Intel and
// AMD processors. It measures the whole CPU package, NPU included, not the NPU alone. Reading
// energy_uj usually requires root (it is restricted since CVE-2020-8694).
package power

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu/internal/types"
)

// RAPL is a package energy counter.
type RAPL struct {
	dir       string // e.g. /sys/class/powercap/intel-rapl:0
	maxRangeJ float64

	mu      sync.Mutex
	lastRaw float64
	wraps   float64
	started bool
}

// OpenRAPL finds the first package domain under root (normally /sys/class/powercap) and checks
// that its counter is readable.
func OpenRAPL(root string) (*RAPL, error) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, err
	}
	for _, e := range entries {
		name := e.Name()
		if !strings.HasPrefix(name, "intel-rapl:") || strings.Count(name, ":") != 1 {
			continue // sub domains such as intel-rapl:0:0 (core) are skipped
		}
		dir := filepath.Join(root, name)
		if label := readString(filepath.Join(dir, "name")); !strings.HasPrefix(label, "package") {
			continue
		}
		r := &RAPL{dir: dir}
		if v, err := readFloat(filepath.Join(dir, "max_energy_range_uj")); err == nil {
			r.maxRangeJ = v / 1e6
		}
		if _, err := readFloat(filepath.Join(dir, "energy_uj")); err != nil {
			if errors.Is(err, os.ErrPermission) {
				return nil, fmt.Errorf("power: %s is not readable; run as root or grant read access (it is restricted since CVE-2020-8694): %w", filepath.Join(dir, "energy_uj"), err)
			}
			return nil, err
		}
		return r, nil
	}
	return nil, errors.New("power: no RAPL package domain under " + root)
}

// Sample reads the counter, unwrapping overflows seen between calls.
func (r *RAPL) Sample() (types.PowerSample, error) {
	v, err := readFloat(filepath.Join(r.dir, "energy_uj"))
	if err != nil {
		return types.PowerSample{}, err
	}
	now := time.Now()
	j := v / 1e6
	r.mu.Lock()
	if r.started && j < r.lastRaw && r.maxRangeJ > 0 {
		r.wraps += r.maxRangeJ
	}
	r.lastRaw, r.started = j, true
	total := j + r.wraps
	r.mu.Unlock()
	return types.PowerSample{Time: now, EnergyJ: total, Scope: types.ScopePackage, Source: "rapl:" + filepath.Base(r.dir)}, nil
}

func readString(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}

func readFloat(path string) (float64, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	return strconv.ParseFloat(strings.TrimSpace(string(b)), 64)
}
