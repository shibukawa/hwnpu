package main

import (
	"strings"
	"testing"
)

func TestMarkdownVerify(t *testing.T) {
	rep := &report{Verify: []result{
		{Format: "lut4", Layers: 16, Tokens: 1, Units: "npu", NPUOps: 1, MsPerCall: 8},
		{Format: "lut4", Layers: 16, Tokens: 4, Units: "npu", NPUOps: 1, MsPerCall: 10},
	}}
	md := markdown(rep)
	for _, want := range []string{
		"| lut4 | 1 | NPU | 8.0 | 1.00 | 8.0 | 125.0 |",
		"| lut4 | 4 | NPU | 10.0 | 1.25 | 2.5 | 400.0 |",
	} {
		if !strings.Contains(md, want) {
			t.Errorf("missing %q in\n%s", want, md)
		}
	}
}
