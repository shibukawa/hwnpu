package main

import (
	"fmt"
	"math"
	"testing"
)

// A trimmed powermetrics -f plist sample (Apple silicon layout: rails under "processor", in mW).
const plistSample = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
<key>is_delta</key><true/>
<key>elapsed_ns</key><integer>%d</integer>
<key>hw_model</key><string>Mac15,3</string>
<key>timestamp</key><date>2026-10-28T01:02:03Z</date>
<key>processor</key>
<dict>
	<key>clusters</key>
	<array><dict><key>name</key><string>E-Cluster</string><key>freq_hz</key><real>1e9</real></dict></array>
	<key>cpu_energy</key><integer>40</integer>
	<key>cpu_power</key><real>%g</real>
	<key>gpu_power</key><real>1</real>
	<key>ane_power</key><real>%g</real>
	<key>combined_power</key><real>%g</real>
</dict>
</dict>
</plist>
`

func sample(elapsed int, cpu, ane float64) string {
	return fmt.Sprintf(plistSample, elapsed, cpu, ane, cpu+ane+1)
}

func TestParseAndAverage(t *testing.T) {
	out := sample(500e6, 100, 0) + "\x00" + sample(500e6, 300, 50) + "\x00" + sample(1000e6, 200, 20) + "\x00"
	samples, err := parseSamples([]byte(out))
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 3 {
		t.Fatalf("got %d samples", len(samples))
	}
	if samples[0]["processor.cpu_energy"] != 40 || samples[0]["is_delta"] != 1 {
		t.Fatalf("flattened sample: %v", samples[0])
	}
	if _, ok := samples[0]["processor.clusters.freq_hz"]; ok {
		t.Fatal("arrays must be skipped")
	}
	r, err := average(samples[1:])
	if err != nil {
		t.Fatal(err)
	}
	// weighted by elapsed_ns: (300*0.5 + 200*1) / 1.5
	want := Rails{CPU: 350.0 / 1.5, GPU: 1, ANE: (25 + 20) / 1.5, Combined: (351*0.5 + 221) / 1.5}
	for _, p := range [][2]float64{{r.CPU, want.CPU}, {r.GPU, want.GPU}, {r.ANE, want.ANE}, {r.Combined, want.Combined}} {
		if math.Abs(p[0]-p[1]) > 1e-9 {
			t.Fatalf("average = %+v, want %+v", r, want)
		}
	}
}

func TestAverageWithoutRails(t *testing.T) {
	samples, err := parseSamples([]byte(`<plist><dict><key>elapsed_ns</key><integer>1</integer></dict></plist>`))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := average(samples); err == nil {
		t.Fatal("want an error when no *_power value exists")
	}
}
