package sysload

import (
	"os"
	"path/filepath"
	"testing"
)

func TestGPUsIn(t *testing.T) {
	root := t.TempDir()
	dev := filepath.Join(root, "card1", "device")
	os.MkdirAll(dev, 0o755)
	os.MkdirAll(filepath.Join(root, "card1-DP-1"), 0o755)
	os.MkdirAll(filepath.Join(root, "card0", "device"), 0o755) // nothing amdgpu has: left out
	for name, v := range map[string]string{
		"gpu_busy_percent": "7\n", "mem_info_vram_used": "100\n", "mem_info_vram_total": "512\n",
		"mem_info_gtt_used": "30\n", "mem_info_gtt_total": "1000\n",
	} {
		os.WriteFile(filepath.Join(dev, name), []byte(v), 0o644)
	}
	g := gpusIn(root)
	if len(g) != 1 || g[0].Name != "card1" || *g[0].BusyPercent != 7 || g[0].VRAMUsed != 100 || g[0].VRAMTotal != 512 || g[0].SharedUsed != 30 || g[0].SharedTotal != 1000 {
		t.Fatalf("got %+v", g)
	}
}

func TestReadLinux(t *testing.T) {
	var s Sampler
	got := s.Read()
	if got.CPUPercent == nil || *got.CPUPercent < 0 || *got.CPUPercent > 100 || got.MemoryUsed <= 0 || got.MemoryUsed > got.MemoryTotal {
		t.Fatalf("got %+v", got)
	}
	t.Logf("%+v cpu %.1f%%", got, *got.CPUPercent)
}
