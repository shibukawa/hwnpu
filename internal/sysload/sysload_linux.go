package sysload

import (
	"bytes"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// readCPU reads the first line of /proc/stat: user nice system idle iowait irq softirq steal.
func readCPU() (cpuTicks, bool) {
	b, err := os.ReadFile("/proc/stat")
	if err != nil {
		return cpuTicks{}, false
	}
	line, _, _ := bytes.Cut(b, []byte("\n"))
	f := strings.Fields(string(line))
	if len(f) < 5 || f[0] != "cpu" {
		return cpuTicks{}, false
	}
	var t cpuTicks
	for i, s := range f[1:min(len(f), 9)] {
		n, err := strconv.ParseUint(s, 10, 64)
		if err != nil {
			return cpuTicks{}, false
		}
		t.total += n
		if i != 3 && i != 4 { // idle, iowait
			t.busy += n
		}
	}
	return t, true
}

func readMemory() (used, total int64) {
	b, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0, 0
	}
	var avail int64
	for _, line := range strings.Split(string(b), "\n") {
		k, v, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		f := strings.Fields(v)
		if len(f) == 0 {
			continue
		}
		n, _ := strconv.ParseInt(f[0], 10, 64)
		switch k {
		case "MemTotal":
			total = n << 10
		case "MemAvailable":
			avail = n << 10
		}
	}
	if total == 0 {
		return 0, 0
	}
	return total - avail, total
}

var cardName = regexp.MustCompile(`^card[0-9]+$`)

// readGPUs reads what amdgpu puts in sysfs (/sys/class/drm/cardN/device): gpu_busy_percent,
// mem_info_vram_* and mem_info_gtt_*. Other drivers (i915, xe, nvidia) have none of them and are
// left out.
func readGPUs() []GPU {
	return gpusIn("/sys/class/drm")
}

func gpusIn(root string) []GPU {
	es, err := os.ReadDir(root)
	if err != nil {
		return nil
	}
	var out []GPU
	for _, e := range es {
		if !cardName.MatchString(e.Name()) {
			continue
		}
		dev := filepath.Join(root, e.Name(), "device")
		g := GPU{Name: e.Name()}
		if d, err := os.Readlink(filepath.Join(dev, "driver")); err == nil {
			g.Name += " " + filepath.Base(d)
		}
		found := false
		if n, ok := readInt(filepath.Join(dev, "gpu_busy_percent")); ok {
			p := float64(n)
			g.BusyPercent, found = &p, true
		}
		for _, f := range []struct {
			name string
			p    *int64
		}{
			{"mem_info_vram_used", &g.VRAMUsed}, {"mem_info_vram_total", &g.VRAMTotal},
			{"mem_info_gtt_used", &g.SharedUsed}, {"mem_info_gtt_total", &g.SharedTotal},
		} {
			if n, ok := readInt(filepath.Join(dev, f.name)); ok {
				*f.p, found = n, true
			}
		}
		if found {
			out = append(out, g)
		}
	}
	return out
}

func readInt(path string) (int64, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, false
	}
	n, err := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64)
	return n, err == nil
}
