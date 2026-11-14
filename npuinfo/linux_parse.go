package npuinfo

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"fmt"
	"strconv"
	"strings"
)

// The decoders here have no build tag so that their tests run on every OS.

// parseFDInfo parses /proc/PID/fdinfo/FD ("key:\tvalue" lines).
func parseFDInfo(b []byte) map[string]string {
	m := map[string]string{}
	sc := bufio.NewScanner(bytes.NewReader(b))
	for sc.Scan() {
		k, v, ok := strings.Cut(sc.Text(), ":")
		if !ok {
			continue
		}
		m[strings.TrimSpace(k)] = strings.TrimSpace(v)
	}
	return m
}

// sumMaps adds up the /proc/PID/maps entries whose path is one of paths: bytes and count by path.
func sumMaps(b []byte, paths map[string]bool) (map[string]uint64, map[string]int) {
	size := map[string]uint64{}
	count := map[string]int{}
	sc := bufio.NewScanner(bytes.NewReader(b))
	sc.Buffer(make([]byte, 64*1024), 1024*1024)
	for sc.Scan() {
		f := strings.Fields(sc.Text())
		if len(f) < 6 || !paths[f[5]] {
			continue
		}
		lo, hi, ok := strings.Cut(f[0], "-")
		if !ok {
			continue
		}
		a, err1 := strconv.ParseUint(lo, 16, 64)
		z, err2 := strconv.ParseUint(hi, 16, 64)
		if err1 != nil || err2 != nil || z < a {
			continue
		}
		size[f[5]] += z - a
		count[f[5]]++
	}
	if len(size) == 0 {
		return nil, nil
	}
	return size, count
}

// cstruct reads a little-endian C struct with natural alignment.
type cstruct struct {
	b   []byte
	off int
	err error
}

func (c *cstruct) align(n int) {
	if r := c.off % n; r != 0 {
		c.off += n - r
	}
}

func (c *cstruct) need(n int) bool {
	if c.err == nil && c.off+n > len(c.b) {
		c.err = fmt.Errorf("struct needs %d bytes, have %d", c.off+n, len(c.b))
	}
	return c.err == nil
}

func (c *cstruct) u8() uint8 {
	if !c.need(1) {
		return 0
	}
	v := c.b[c.off]
	c.off++
	return v
}

func (c *cstruct) u16() uint16 {
	c.align(2)
	if !c.need(2) {
		return 0
	}
	v := binary.LittleEndian.Uint16(c.b[c.off:])
	c.off += 2
	return v
}

func (c *cstruct) u32() uint32 {
	c.align(4)
	if !c.need(4) {
		return 0
	}
	v := binary.LittleEndian.Uint32(c.b[c.off:])
	c.off += 4
	return v
}

func (c *cstruct) u64() uint64 {
	c.align(8)
	if !c.need(8) {
		return 0
	}
	v := binary.LittleEndian.Uint64(c.b[c.off:])
	c.off += 8
	return v
}

func (c *cstruct) bytes(n int) []byte {
	if !c.need(n) {
		return nil
	}
	v := c.b[c.off : c.off+n]
	c.off += n
	return v
}

func (c *cstruct) cstring(n int) string {
	b := c.bytes(n)
	if i := bytes.IndexByte(b, 0); i >= 0 {
		b = b[:i]
	}
	return string(b)
}

func (c *cstruct) u16s(dst []uint16) {
	for i := range dst {
		dst[i] = c.u16()
	}
}

// decodeGPUMetrics decodes amdgpu's gpu_metrics file: the header always, the body for format 3.
func decodeGPUMetrics(b []byte) (AMDGPUMetrics, error) {
	var m AMDGPUMetrics
	if len(b) < 4 {
		return m, fmt.Errorf("gpu_metrics: %d bytes", len(b))
	}
	c := &cstruct{b: b}
	m.Size, m.Format, m.Content = c.u16(), c.u8(), c.u8()
	if m.Format != 3 || m.Content != 0 {
		m.Raw = b
		return m, nil
	}
	var v GPUMetricsV3
	v.TemperatureGfx = c.u16()
	v.TemperatureSoc = c.u16()
	c.u16s(v.TemperatureCore[:])
	v.TemperatureSkin = c.u16()
	v.AverageGfxActivity = c.u16()
	v.AverageVcnActivity = c.u16()
	c.u16s(v.AverageIPUActivity[:])
	c.u16s(v.AverageCoreC0Activity[:])
	v.AverageDRAMReads = c.u16()
	v.AverageDRAMWrites = c.u16()
	v.AverageIPUReads = c.u16()
	v.AverageIPUWrites = c.u16()
	v.SystemClockCounter = c.u64()
	v.AverageSocketPower = c.u32()
	v.AverageIPUPower = c.u16()
	v.AverageAPUPower = c.u32()
	v.AverageGfxPower = c.u32()
	v.AverageDGPUPower = c.u32()
	v.AverageAllCorePower = c.u32()
	c.u16s(v.AverageCorePower[:])
	v.AverageSysPower = c.u16()
	v.StapmPowerLimit = c.u16()
	v.CurrentStapmPowerLimit = c.u16()
	v.AverageGfxclkFrequency = c.u16()
	v.AverageSocclkFrequency = c.u16()
	v.AverageVpeclkFrequency = c.u16()
	v.AverageIPUclkFrequency = c.u16()
	v.AverageFclkFrequency = c.u16()
	v.AverageVclkFrequency = c.u16()
	v.AverageUclkFrequency = c.u16()
	v.AverageMPIPUFrequency = c.u16()
	c.u16s(v.CurrentCoreclk[:])
	v.CurrentCoreMaxfreq = c.u16()
	v.CurrentGfxMaxfreq = c.u16()
	v.ThrottleResidencyProchot = c.u32()
	v.ThrottleResidencySPL = c.u32()
	v.ThrottleResidencyFPPT = c.u32()
	v.ThrottleResidencySPPT = c.u32()
	v.ThrottleResidencyThmCore = c.u32()
	v.ThrottleResidencyThmGfx = c.u32()
	v.ThrottleResidencyThmSoc = c.u32()
	v.TimeFilterAlphaValue = c.u32()
	if c.err != nil {
		m.Raw = b
		return m, fmt.Errorf("gpu_metrics v3: %w", c.err)
	}
	m.V3 = &v
	return m, nil
}

// xdnaHWCtxEntrySize is sizeof(struct amdxdna_drm_hwctx_entry) in the header this was written
// against (Linux 7.0); the driver reports the size it fills.
const xdnaHWCtxEntrySize = 144

func decodeXDNAContext(b []byte) (XDNAContext, error) {
	c := &cstruct{b: b}
	var x XDNAContext
	x.ContextID, x.StartCol, x.NumCol, x.HWContextID = c.u32(), c.u32(), c.u32(), c.u32()
	x.PID = int64(c.u64())
	x.CommandSubmissions, x.CommandCompletions = c.u64(), c.u64()
	x.Migrations, x.Preemptions, x.Errors = c.u64(), c.u64(), c.u64()
	x.Priority, x.HeapUsage, x.Suspensions = c.u64(), c.u64(), c.u64()
	if c.err != nil {
		return x, c.err
	}
	// The fields after suspensions came later; read what the driver filled.
	tail := []*uint32{&x.State, &x.PASID, &x.GOPS, &x.FPS, &x.DMABandwidth, &x.Latency,
		&x.FrameExecTime, &x.TxnOpIdx, &x.ContextPC, &x.FatalErrorType,
		&x.FatalErrorExceptionType, &x.FatalErrorExceptionPC, &x.FatalErrorAppModule}
	for _, p := range tail {
		if c.off+4 > len(b) {
			break
		}
		*p = c.u32()
	}
	return x, nil
}

// decodeXDNAQueryHWCtx decodes the older struct amdxdna_drm_query_hwctx (QUERY_HW_CONTEXTS).
func decodeXDNAQueryHWCtx(b []byte) (XDNAContext, error) {
	c := &cstruct{b: b}
	var x XDNAContext
	x.ContextID, x.StartCol, x.NumCol = c.u32(), c.u32(), c.u32()
	c.u32()
	x.PID = int64(c.u64())
	x.CommandSubmissions, x.CommandCompletions = c.u64(), c.u64()
	x.Migrations, x.Preemptions, x.Errors = c.u64(), c.u64(), c.u64()
	return x, c.err
}

func decodeXDNATile(c *cstruct) XDNATileMetadata {
	t := XDNATileMetadata{RowCount: c.u16(), RowStart: c.u16(), DMAChannelCount: c.u16(),
		LockCount: c.u16(), EventRegCount: c.u16()}
	c.u16()
	c.u16()
	c.u16()
	return t
}

func decodeXDNAMetadata(b []byte) (XDNAAIEMetadata, error) {
	c := &cstruct{b: b}
	var m XDNAAIEMetadata
	m.ColSize = c.u32()
	m.Cols, m.Rows = c.u16(), c.u16()
	m.Version = XDNAVersion{Major: c.u32(), Minor: c.u32()}
	m.Core = decodeXDNATile(c)
	m.Mem = decodeXDNATile(c)
	m.Shim = decodeXDNATile(c)
	return m, c.err
}

const xdnaSensorSize = 64 + 4*4 + 64 + 16 + 1 + 1 + 6

func decodeXDNASensor(b []byte) (XDNASensor, error) {
	c := &cstruct{b: b}
	var s XDNASensor
	s.Label = c.cstring(64)
	s.Input, s.Max, s.Average, s.Highest = c.u32(), c.u32(), c.u32(), c.u32()
	s.Status = c.cstring(64)
	s.Units = c.cstring(16)
	s.UnitM = int8(c.u8())
	s.Type = c.u8()
	return s, c.err
}

var xdnaPowerModes = []string{"default", "low", "medium", "high", "turbo"}

func xdnaPowerModeName(m int) string {
	if m >= 0 && m < len(xdnaPowerModes) {
		return xdnaPowerModes[m]
	}
	return strconv.Itoa(m)
}
