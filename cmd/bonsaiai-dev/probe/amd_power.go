package probe

import (
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/shibukawa/bonsaiai/npuinfo"
)

// amdPower is the SMU's own split of an APU's power and memory traffic (amdgpu's gpu_metrics,
// format 3), averaged over a run: what the PPT meter's one number does not tell. Read from sysfs,
// which does not wake the NPU (amdxdna's telemetry ioctls do, and have hung its firmware).
type amdPower struct {
	SocketW   float64 `json:"socket_w"`           // the whole package (PPT)
	NPUW      float64 `json:"npu_w"`              // the NPU's (IPU's) rail
	CoresW    float64 `json:"cores_w"`            // the CPU cores
	GfxW      float64 `json:"gfx_w"`              // the GPU
	RestW     float64 `json:"rest_w"`             // the rest: the fabric, the memory controller and its PHY, I/O
	DRAMRead  float64 `json:"dram_read_gbs"`      // GB/s read from DRAM, every reader
	DRAMWrite float64 `json:"dram_write_gbs"`     // and written
	NPUBusy   float64 `json:"npu_busy,omitempty"` // the NPU's activity, % (the mean of its columns)
	NPUClock  float64 `json:"npu_mhz,omitempty"`  // the NPU's clock
	FClock    float64 `json:"fclk_mhz,omitempty"` // the fabric's
	UClock    float64 `json:"uclk_mhz,omitempty"` // the memory controller's
	Samples   int     `json:"samples"`
}

// amdMetricsPath is the gpu_metrics file of format 3 (an APU's), "" when there is none.
func amdMetricsPath() string {
	paths, _ := filepath.Glob("/sys/class/drm/card*/device/gpu_metrics")
	for _, p := range paths {
		if b, err := os.ReadFile(p); err == nil {
			if m, err := npuinfo.DecodeAMDGPUMetrics(b); err == nil && m.V3 != nil {
				return p
			}
		}
	}
	return ""
}

// amdSampler reads gpu_metrics every 50 ms from a second after its start (the SMU's averages
// follow the load in about that) until stop, which returns the means.
type amdSampler struct {
	path  string
	skip  time.Time // the SMU's averages lag: samples before it are not taken
	stopc chan struct{}
	wg    sync.WaitGroup
	sum   amdPower
}

func startAMDSampler(path string) *amdSampler {
	s := &amdSampler{path: path, skip: time.Now().Add(time.Second), stopc: make(chan struct{})}
	if path == "" {
		return s
	}
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		t := time.NewTicker(50 * time.Millisecond)
		defer t.Stop()
		for {
			s.sample()
			select {
			case <-s.stopc:
				return
			case <-t.C:
			}
		}
	}()
	return s
}

func (s *amdSampler) sample() {
	if time.Now().Before(s.skip) {
		return
	}
	b, err := os.ReadFile(s.path)
	if err != nil {
		return
	}
	m, err := npuinfo.DecodeAMDGPUMetrics(b)
	if err != nil || m.V3 == nil {
		return
	}
	v := m.V3
	mw := func(x uint32) float64 {
		if x == 0xFFFFFFFF || x == 0xFFFF {
			return 0
		}
		return float64(x) / 1000
	}
	mbs := func(x uint16) float64 {
		if x == 0xFFFF {
			return 0
		}
		return float64(x) / 1000
	}
	f := func(x uint16) float64 {
		if x == 0xFFFF {
			return 0
		}
		return float64(x)
	}
	var busy float64
	for _, a := range v.AverageIPUActivity {
		busy += f(a)
	}
	socket, npu, cores, gfx := mw(v.AverageSocketPower), mw(uint32(v.AverageIPUPower)), mw(v.AverageAllCorePower), mw(v.AverageGfxPower)
	s.sum.SocketW += socket
	s.sum.NPUW += npu
	s.sum.CoresW += cores
	s.sum.GfxW += gfx
	s.sum.RestW += max(socket-npu-cores-gfx, 0)
	s.sum.DRAMRead += mbs(v.AverageDRAMReads)
	s.sum.DRAMWrite += mbs(v.AverageDRAMWrites)
	s.sum.NPUBusy += busy / float64(len(v.AverageIPUActivity))
	s.sum.NPUClock += f(v.AverageIPUclkFrequency)
	s.sum.FClock += f(v.AverageFclkFrequency)
	s.sum.UClock += f(v.AverageUclkFrequency)
	s.sum.Samples++
}

func (s *amdSampler) stop() *amdPower {
	if s.path == "" {
		return nil
	}
	close(s.stopc)
	s.wg.Wait()
	p := s.sum
	if p.Samples == 0 {
		return nil
	}
	n := float64(p.Samples)
	for _, x := range []*float64{&p.SocketW, &p.NPUW, &p.CoresW, &p.GfxW, &p.RestW, &p.DRAMRead, &p.DRAMWrite, &p.NPUBusy, &p.NPUClock, &p.FClock, &p.UClock} {
		*x /= n
	}
	return &p
}
