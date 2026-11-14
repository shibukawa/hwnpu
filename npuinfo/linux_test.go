//go:build linux

package npuinfo

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReadSysfsAttrs(t *testing.T) {
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "power"), 0o755)
	os.WriteFile(filepath.Join(dir, "fw_version"), []byte("1.1.2.64\n"), 0o444)
	os.WriteFile(filepath.Join(dir, "npu_busy_time_us"), []byte("12345\n"), 0o444)
	os.WriteFile(filepath.Join(dir, "config"), []byte("x"), 0o444)
	os.WriteFile(filepath.Join(dir, "resource0"), []byte("x"), 0o444)
	os.WriteFile(filepath.Join(dir, "blob"), []byte{0, 1, 2}, 0o444)
	os.WriteFile(filepath.Join(dir, "power", "runtime_status"), []byte("suspended\n"), 0o444)
	got := readSysfsAttrs(dir)
	want := map[string]string{"fw_version": "1.1.2.64", "npu_busy_time_us": "12345", "power/runtime_status": "suspended"}
	if len(got) != len(want) {
		t.Fatalf("got %v", got)
	}
	for k, v := range want {
		if got[k] != v {
			t.Fatalf("%s = %q, want %q (all %v)", k, got[k], v, got)
		}
	}
}

// TestGetLinux reads the real machine without waking the NPU.
func TestGetLinux(t *testing.T) {
	info, err := GetWithContext(context.Background(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range info.Errors {
		if strings.Contains(e, "panic") {
			t.Fatal(e)
		}
	}
	if _, err := os.Stat(accelClass); err == nil && len(info.Linux.Devices) == 0 {
		t.Fatalf("%s exists but no devices: %v", accelClass, info.Errors)
	}
	for _, d := range info.Linux.Devices {
		if d.Driver == "amdxdna" && d.RuntimeStatus != "active" && d.XDNA.Skipped == "" {
			t.Fatalf("queried a sleeping NPU: %+v", d.XDNA)
		}
	}
	t.Log(info.String()[:min(len(info.String()), 2000)])
}

func TestQueryXDNASkipsSleepingNPU(t *testing.T) {
	x := queryXDNA("/nonexistent/accel", "suspended", Options{})
	if x.Skipped == "" || x.Errors != nil {
		t.Fatalf("got %+v", x)
	}
	x = queryXDNA("/nonexistent/accel", "suspended", Options{Wake: true})
	if x.Skipped != "" || x.Errors["open"] == "" {
		t.Fatalf("Wake should try to open: %+v", x)
	}
}
