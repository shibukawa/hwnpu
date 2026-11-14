//go:build darwin

package npuinfo

import (
	"context"
	"testing"
	"unsafe"
)

// The size and offsets of struct rusage_info_v6 as clang gives them (macOS 27 SDK).
func TestRusageInfoV6Layout(t *testing.T) {
	var r rusageInfoV6
	if unsafe.Sizeof(r) != 464 || unsafe.Offsetof(r.PhysFootprint) != 72 ||
		unsafe.Offsetof(r.NeuralFootprint) != 368 || unsafe.Offsetof(r.LifetimeMaxNeuralFootprint) != 376 {
		t.Fatalf("size %d, phys_footprint at %d, neural_footprint at %d", unsafe.Sizeof(r),
			unsafe.Offsetof(r.PhysFootprint), unsafe.Offsetof(r.NeuralFootprint))
	}
}

// On a Mac with a Neural Engine, every reader works without root: the device, this machine's
// processes and IOReport's ANE channels.
func TestDarwinGet(t *testing.T) {
	info, err := GetWithContext(context.Background(), Options{SkipCaches: true, SampleInterval: 100e6})
	if err != nil {
		t.Fatal(err)
	}
	if len(info.Errors) > 0 {
		t.Errorf("errors: %v", info.Errors)
	}
	d := info.Darwin
	if d.Device == nil {
		t.Skip("no Neural Engine (Intel Mac?)")
	}
	if d.Device.Cores <= 0 || d.Device.Architecture == "" {
		t.Errorf("device %+v", d.Device)
	}
	if d.IOReport == nil || len(d.IOReport.Channels) == 0 {
		t.Fatalf("no IOReport channels: %+v", d.IOReport)
	}
	var energy bool
	for _, c := range d.IOReport.Channels {
		if (c.Group == "PMP" || c.Group == "Energy Model") && c.Unit != "" && c.Value != nil {
			energy = true
		}
	}
	if !energy {
		t.Errorf("no ANE energy channel in %d channels", len(d.IOReport.Channels))
	}
}
