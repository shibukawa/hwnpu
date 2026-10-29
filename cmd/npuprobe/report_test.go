package main

import (
	"strings"
	"testing"
)

func TestMarkdownVerify(t *testing.T) {
	rep := &report{Verify: []result{
		{Format: "lut4", Layers: 16, Tokens: 1, Units: "npu", NPUOps: 1, MsPerCall: 8},
		{Format: "lut4", Layers: 16, Tokens: 4, Units: "npu", NPUOps: 1, MsPerCall: 10},
		{Format: "lut4", Layers: 16, Tokens: 64, Units: "npu", NPUOps: 1, MsPerCall: 9.1},
	}}
	md := markdown(rep)
	for _, want := range []string{
		"| lut4 | 1 | NPU | 8.0 | 1.00 | 8.0 | 125.0 |",
		"| lut4 | 4 | NPU | 10.0 | 1.25 | 2.5 | 400.0 |",
		"| lut4 | 64 | NPU | 9.1 | 1.14 | 0.142 | 7033.0 |",
	} {
		if !strings.Contains(md, want) {
			t.Errorf("missing %q in\n%s", want, md)
		}
	}
}

// The M3's lut4 run of LFM3-1.7B sized weights: flat up to 24 tokens, compute bound from 48.
func TestKnees(t *testing.T) {
	const params = 1.73e9
	row := func(layers, k int, ms float64) result {
		return result{Format: "lut4", Layers: layers, Tokens: k, Params: params, MsPerCall: ms, OnNPU: true}
	}
	rows := []result{row(16, 1, 13.7), row(16, 24, 14.0), row(16, 48, 15.2), row(16, 96, 30.0),
		row(8, 1, 13.7), row(8, 12, 13.9), row(8, 24, 14.0)}
	got := knees(rows)
	if len(got) != 2 {
		t.Fatalf("knees: %q", got)
	}
	// 2 × 1.73e9 × 96 / 30 ms = 11.1 TFLOPS; 13.7 ms at that rate computes 44 tokens.
	if !strings.Contains(got[0], "at most 11.1 TFLOPS") || !strings.Contains(got[0], "about 44 tokens") {
		t.Errorf("compute bound: %q", got[0])
	}
	if !strings.Contains(got[1], "still bound by reading the weights at K=24") {
		t.Errorf("read bound: %q", got[1])
	}
}
