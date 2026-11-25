package sysload

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// clockTick is USER_HZ, the unit of /proc/PID/stat's times (100 on every Linux bonsaiai runs on).
const clockTick = time.Second / 100

// readProcs reads /proc/PID/stat (utime, stime), /proc/PID/statm (resident) and the DRM
// clients in /proc/PID/fdinfo (Documentation/gpu/drm-usage-stats.rst): amdgpu's for the GPU,
// amdxdna's and intel_vpu's for the NPU.
func readProcs(pids []int) map[int]procTimes {
	out := make(map[int]procTimes, len(pids))
	page := int64(os.Getpagesize())
	for _, pid := range pids {
		dir := filepath.Join("/proc", strconv.Itoa(pid))
		var t procTimes
		if b, err := os.ReadFile(filepath.Join(dir, "stat")); err == nil {
			// the name in parentheses may hold spaces: the fields are after the last ')'
			if i := strings.LastIndexByte(string(b), ')'); i >= 0 {
				f := strings.Fields(string(b)[i+1:])
				if len(f) > 12 {
					u, e1 := strconv.ParseUint(f[11], 10, 64)
					s, e2 := strconv.ParseUint(f[12], 10, 64)
					if e1 == nil && e2 == nil {
						t.cpu, t.hasCPU = time.Duration(u+s)*clockTick, true
					}
				}
			}
		}
		if b, err := os.ReadFile(filepath.Join(dir, "statm")); err == nil {
			if f := strings.Fields(string(b)); len(f) > 1 {
				if n, err := strconv.ParseInt(f[1], 10, 64); err == nil {
					t.memory = n * page
				}
			}
		}
		readDRMClients(filepath.Join(dir, "fdinfo"), &t)
		out[pid] = t
	}
	return out
}

// drmClient is what a DRM fd's fdinfo tells of its client.
type drmClient struct {
	driver, id string
	engine     time.Duration // all its engines
	hasEngine  bool
	memory     int64 // resident memory of all its regions
}

// readDRMClients adds the DRM clients in the fdinfo directory to t, each client once (fds
// dup'ed from one have the same drm-client-id).
func readDRMClients(dir string, t *procTimes) {
	es, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	seen := map[string]bool{}
	for _, e := range es {
		b, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil || !strings.Contains(string(b), "drm-driver:") {
			continue
		}
		c := parseDRMFDInfo(string(b))
		key := c.driver + "/" + c.id
		if c.id != "" && seen[key] {
			continue
		}
		seen[key] = true
		switch c.driver {
		case "amdgpu", "i915", "xe":
			t.gpu += c.engine
			t.hasGPU = t.hasGPU || c.hasEngine
			t.gpuMemory += c.memory
		case "amdxdna", "intel_vpu":
			t.npu += c.engine
			t.hasNPU = t.hasNPU || c.hasEngine
		}
	}
}

func parseDRMFDInfo(text string) drmClient {
	var c drmClient
	resident, legacy := int64(0), int64(0)
	for _, line := range strings.Split(text, "\n") {
		k, v, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		v = strings.TrimSpace(v)
		switch {
		case k == "drm-driver":
			c.driver = v
		case k == "drm-client-id":
			c.id = v
		case strings.HasPrefix(k, "drm-engine-") && !strings.HasPrefix(k, "drm-engine-capacity-"):
			if n, ok := withUnit(v); ok {
				c.engine += time.Duration(n) // ns
				c.hasEngine = true
			}
		case strings.HasPrefix(k, "drm-resident-"):
			if n, ok := withUnit(v); ok {
				resident += n
			}
		case strings.HasPrefix(k, "drm-memory-"): // the older name of drm-resident-
			if n, ok := withUnit(v); ok {
				legacy += n
			}
		}
	}
	c.memory = resident
	if c.memory == 0 {
		c.memory = legacy
	}
	return c
}

// withUnit reads "123 ns", "456 KiB", "7 MiB", "8".
func withUnit(v string) (int64, bool) {
	num, unit, _ := strings.Cut(v, " ")
	n, err := strconv.ParseInt(num, 10, 64)
	if err != nil {
		return 0, false
	}
	switch strings.TrimSpace(unit) {
	case "KiB":
		n <<= 10
	case "MiB":
		n <<= 20
	case "GiB":
		n <<= 30
	}
	return n, true
}
