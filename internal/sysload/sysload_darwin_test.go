package sysload

import (
	"os"
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

func TestRusageInfoV2Size(t *testing.T) {
	if n := unsafe.Sizeof(rusageInfoV2{}); n != 160 {
		t.Fatalf("rusage_info_v2 is %d bytes, want 160", n)
	}
}

func TestParseGPUTimes(t *testing.T) {
	text := `+-o AGXAcceleratorG13X  <class AGXAcceleratorG13X>
  | {
  | }
  +-o AGXDeviceUserClient  <class AGXDeviceUserClient>
      {
        "IOUserClientCreator" = "pid 584, WindowServer"
        "AppUsage" = ({"API"="Metal","accumulatedGPUTime"=1000},{"API"="Metal","accumulatedGPUTime"=500})
      }
  +-o AGXDeviceUserClient  <class AGXDeviceUserClient>
      {
        "IOUserClientCreator" = "pid 584, WindowServer"
        "AppUsage" = ({"API"="Metal","accumulatedGPUTime"=250})
      }
  +-o AGXDeviceUserClient  <class AGXDeviceUserClient>
      {
        "IOUserClientCreator" = "pid 900, bonsaiai-chat"
      }
`
	g := parseGPUTimes(text)
	if len(g) != 1 || g[584] != 1750 {
		t.Fatalf("got %v", g)
	}
}

func TestReadProcsDarwin(t *testing.T) {
	var s ProcSampler
	pid := os.Getpid()
	p := s.Read([]int{pid})[pid]
	if p.CPUPercent == nil || p.Memory <= 0 {
		t.Fatalf("got %+v", p)
	}
}
