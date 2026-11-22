package mem

import (
	"bufio"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// DRMMemory is what this process's open DRM devices (AMD's NPU among them, /dev/accel) hold:
// the sum of drm-total-memory in /proc/self/fdinfo, each client (drm-client-id) once. On AMD's
// NPU every page of its buffers is pinned and counted in no process's RSS. ok is false when no
// descriptor reports it.
func DRMMemory() (bytes int64, ok bool) { return drmMemory("/proc/self/fdinfo") }

func drmMemory(dir string) (int64, bool) {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return 0, false
	}
	seen := map[string]bool{}
	var total int64
	found := false
	for _, e := range ents {
		f, err := os.Open(filepath.Join(dir, e.Name()))
		if err != nil {
			continue
		}
		client, n, has := "", int64(0), false
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			k, v, ok := strings.Cut(sc.Text(), ":")
			if !ok {
				continue
			}
			v = strings.TrimSpace(v)
			switch k {
			case "drm-client-id":
				client = v
			case "drm-total-memory":
				n, has = drmBytes(v), true
			}
		}
		f.Close()
		if !has || (client != "" && seen[client]) {
			continue
		}
		seen[client] = true
		total, found = total+n, true
	}
	return total, found
}

// drmBytes reads a value of fdinfo's memory keys: a number and an optional unit (KiB, MiB).
func drmBytes(v string) int64 {
	num, unit, _ := strings.Cut(v, " ")
	n, _ := strconv.ParseInt(num, 10, 64)
	switch strings.TrimSpace(unit) {
	case "KiB":
		n <<= 10
	case "MiB":
		n <<= 20
	case "GiB":
		n <<= 30
	}
	return n
}
