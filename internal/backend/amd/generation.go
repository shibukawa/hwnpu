package amd

import "strings"

// Generation is the generation of this machine's AMD NPU as models.yaml names it ("xdna1" for the
// NPU of PCI id 0x1502, "xdna2" for 0x17f0), "" without one or for one it does not know.
func Generation() string {
	nodes, err := Provider{}.nodes()
	if err != nil || len(nodes) == 0 {
		return ""
	}
	return xdnaOf(nodes[0])
}

// xdnaOf is the generation of an NPU node by its PCI device id.
func xdnaOf(n Node) string {
	switch strings.ToLower(n.Device) {
	case "0x1502":
		return "xdna1"
	case "0x17f0":
		return "xdna2"
	}
	return ""
}
