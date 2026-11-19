package sysload

import (
	"testing"
	"unsafe"
)

func TestVMStatistics64Layout(t *testing.T) {
	if n := unsafe.Sizeof(vmStatistics64{}); n != vmInfo64Count*4 {
		t.Fatalf("vm_statistics64 is %d bytes, want %d", n, vmInfo64Count*4)
	}
}

func TestParseIOAccelerators(t *testing.T) {
	text := `+-o AGXAcceleratorG13X  <class AGXAcceleratorG13X, id 0x1000, registered>
    {
      "model" = "Apple M1"
      "PerformanceStatistics" = {"In use system memory (driver)"=0,"Device Utilization %"=12,"In use system memory"=1048576}
    }
`
	g := parseIOAccelerators(text)
	if len(g) != 1 || g[0].Name != "Apple M1" || g[0].BusyPercent == nil || *g[0].BusyPercent != 12 || g[0].SharedUsed != 1048576 {
		t.Fatalf("got %+v", g)
	}
}

func TestReadDarwin(t *testing.T) {
	var s Sampler
	got := s.Read()
	if got.CPUPercent == nil || got.MemoryUsed <= 0 || got.MemoryUsed > got.MemoryTotal {
		t.Fatalf("got %+v", got)
	}
}
