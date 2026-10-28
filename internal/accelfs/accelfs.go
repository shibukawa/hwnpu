// Package accelfs lists Linux compute accelerator nodes (/dev/accel/accelN) and their drivers.
package accelfs

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Driver names of the NPUs hwnpu knows.
const (
	DriverIntel = "intel_vpu"
	DriverAMD   = "amdxdna"
)

// Node is one accelerator node.
type Node struct {
	Name   string // "accel0"
	Path   string // "/dev/accel/accel0"
	Driver string // kernel driver name, e.g. "intel_vpu"
	Vendor string // PCI vendor id, e.g. "0x8086"
	Device string // PCI device id
}

// List scans /sys/class/accel. It returns nothing on systems without it.
func List() ([]Node, error) { return ListAt("/sys/class/accel", "/dev/accel") }

// ListAt scans sysRoot (normally /sys/class/accel) and reports device paths under devRoot.
func ListAt(sysRoot, devRoot string) ([]Node, error) {
	entries, err := os.ReadDir(sysRoot)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var nodes []Node
	for _, e := range entries {
		if !strings.HasPrefix(e.Name(), "accel") {
			continue
		}
		dev := filepath.Join(sysRoot, e.Name(), "device")
		n := Node{Name: e.Name(), Path: filepath.Join(devRoot, e.Name())}
		if target, err := os.Readlink(filepath.Join(dev, "driver")); err == nil {
			n.Driver = filepath.Base(target)
		}
		n.Vendor = readTrim(filepath.Join(dev, "vendor"))
		n.Device = readTrim(filepath.Join(dev, "device"))
		nodes = append(nodes, n)
	}
	sort.Slice(nodes, func(i, j int) bool { return nodes[i].Name < nodes[j].Name })
	return nodes, nil
}

// ByDriver filters nodes by kernel driver name.
func ByDriver(nodes []Node, driver string) []Node {
	var out []Node
	for _, n := range nodes {
		if n.Driver == driver {
			out = append(out, n)
		}
	}
	return out
}

func readTrim(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}
