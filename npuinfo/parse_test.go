package npuinfo

import (
	"encoding/binary"
	"encoding/json"
	"strings"
	"testing"
)

func TestParseFDInfo(t *testing.T) {
	m := parseFDInfo([]byte("pos:\t0\nflags:\t02100002\ndrm-driver:\tamdxdna\ndrm-engine-npu:\t123 ns\n"))
	if m["drm-driver"] != "amdxdna" || m["drm-engine-npu"] != "123 ns" || m["pos"] != "0" {
		t.Fatalf("got %v", m)
	}
}

func TestSumMaps(t *testing.T) {
	maps := `5555555000-5555556000 r-xp 00000000 00:1f 123 /usr/bin/x
7f0000000000-7f0000100000 rw-s 100000000 00:06 800 /dev/accel/accel0
7f0000200000-7f0000201000 rw-s 100200000 00:06 800 /dev/accel/accel0
7f0000300000-7f0000400000 rw-p 00000000 00:00 0
`
	size, count := sumMaps([]byte(maps), map[string]bool{"/dev/accel/accel0": true})
	if size["/dev/accel/accel0"] != 0x101000 || count["/dev/accel/accel0"] != 2 {
		t.Fatalf("size %v count %v", size, count)
	}
	if s, c := sumMaps([]byte(maps), map[string]bool{"/dev/accel/accel1": true}); s != nil || c != nil {
		t.Fatalf("want nil for no entries, got %v %v", s, c)
	}
}

// Offsets of gpu_metrics_v3_0 worked out by hand from the C layout (natural alignment), so
// that the decoder's alignment is checked against something independent of it.
func TestDecodeGPUMetricsV3(t *testing.T) {
	b := make([]byte, 264)
	le := binary.LittleEndian
	le.PutUint16(b[0:], 264)
	b[2], b[3] = 3, 0
	le.PutUint16(b[6:], 4800)  // temperature_soc
	le.PutUint16(b[46:], 93)   // average_ipu_activity[0]
	le.PutUint16(b[60:], 94)   // average_ipu_activity[7]
	le.PutUint16(b[94:], 2000) // average_dram_reads
	le.PutUint16(b[98:], 1700) // average_ipu_reads
	le.PutUint64(b[104:], 0x1122334455667788)
	le.PutUint32(b[112:], 18169) // average_socket_power
	le.PutUint16(b[116:], 1026)  // average_ipu_power
	le.PutUint32(b[120:], 7)     // average_apu_power
	le.PutUint32(b[132:], 586)   // average_all_core_power
	le.PutUint16(b[180:], 1810)  // average_ipuclk_frequency
	le.PutUint16(b[188:], 1267)  // average_mpipu_frequency
	le.PutUint16(b[224:], 2900)  // current_gfx_maxfreq
	le.PutUint32(b[228:], 5)     // throttle_residency_prochot
	le.PutUint32(b[256:], 99)    // time_filter_alphavalue
	m, err := decodeGPUMetrics(b)
	if err != nil {
		t.Fatal(err)
	}
	v := m.V3
	if v == nil {
		t.Fatal("no v3")
	}
	checks := []struct {
		name      string
		got, want uint64
	}{
		{"temperature_soc", uint64(v.TemperatureSoc), 4800},
		{"ipu_activity[0]", uint64(v.AverageIPUActivity[0]), 93},
		{"ipu_activity[7]", uint64(v.AverageIPUActivity[7]), 94},
		{"dram_reads", uint64(v.AverageDRAMReads), 2000},
		{"ipu_reads", uint64(v.AverageIPUReads), 1700},
		{"system_clock_counter", v.SystemClockCounter, 0x1122334455667788},
		{"socket_power", uint64(v.AverageSocketPower), 18169},
		{"ipu_power", uint64(v.AverageIPUPower), 1026},
		{"apu_power", uint64(v.AverageAPUPower), 7},
		{"all_core_power", uint64(v.AverageAllCorePower), 586},
		{"ipuclk", uint64(v.AverageIPUclkFrequency), 1810},
		{"mpipu", uint64(v.AverageMPIPUFrequency), 1267},
		{"gfx_maxfreq", uint64(v.CurrentGfxMaxfreq), 2900},
		{"prochot", uint64(v.ThrottleResidencyProchot), 5},
		{"alpha", uint64(v.TimeFilterAlphaValue), 99},
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("%s = %d, want %d", c.name, c.got, c.want)
		}
	}
	other, err := decodeGPUMetrics([]byte{8, 0, 2, 1, 0, 0, 0, 0})
	if err != nil || other.V3 != nil || other.Format != 2 || len(other.Raw) != 8 {
		t.Fatalf("format 2: %+v %v", other, err)
	}
}

func TestDecodeXDNAContext(t *testing.T) {
	b := make([]byte, xdnaHWCtxEntrySize)
	le := binary.LittleEndian
	le.PutUint32(b[0:], 3) // context_id
	le.PutUint32(b[8:], 8) // num_col
	le.PutUint64(b[16:], 560636)
	le.PutUint64(b[24:], 181)  // command_submissions
	le.PutUint64(b[32:], 180)  // command_completions
	le.PutUint64(b[72:], 4096) // heap_usage
	le.PutUint32(b[88:], 1)    // state
	le.PutUint32(b[92:], 1)    // pasid
	le.PutUint32(b[136:], 9)   // fatal_error_app_module
	c, err := decodeXDNAContext(b)
	if err != nil {
		t.Fatal(err)
	}
	if c.ContextID != 3 || c.NumCol != 8 || c.PID != 560636 || c.CommandSubmissions != 181 ||
		c.CommandCompletions != 180 || c.HeapUsage != 4096 || c.State != 1 || c.PASID != 1 || c.FatalErrorAppModule != 9 {
		t.Fatalf("got %+v", c)
	}
	// A driver with the shorter entry (up to suspensions) still decodes.
	if c, err := decodeXDNAContext(b[:88]); err != nil || c.PID != 560636 || c.State != 0 {
		t.Fatalf("short entry: %+v %v", c, err)
	}
}

func TestDecodeXDNAMetadata(t *testing.T) {
	b := make([]byte, 64)
	le := binary.LittleEndian
	le.PutUint32(b[0:], 504)
	le.PutUint16(b[4:], 8)
	le.PutUint16(b[6:], 6)
	le.PutUint32(b[8:], 1)
	le.PutUint32(b[12:], 1)
	le.PutUint16(b[16:], 4)  // core.row_count
	le.PutUint16(b[18:], 2)  // core.row_start
	le.PutUint16(b[38:], 64) // mem.lock_count
	m, err := decodeXDNAMetadata(b)
	if err != nil {
		t.Fatal(err)
	}
	if m.ColSize != 504 || m.Cols != 8 || m.Rows != 6 || m.Core.RowCount != 4 || m.Core.RowStart != 2 || m.Mem.LockCount != 64 {
		t.Fatalf("got %+v", m)
	}
}

const ioregText = `+-o Root  <class IORegistryEntry, id 0x100000100, retain 30>
  +-o J316sAP  <class IOPlatformExpertDevice, id 0x10000020f, registered, matched, active, busy 0 (1 ms), retain 40>
    | +-o ane@8000000  <class AppleARMIODevice, id 0x1000001d4, registered, matched, active, busy 0 (0 ms), retain 9>
    | | +-o H11ANEIn  <class H11ANEIn, id 0x1000004a1, registered, matched, active, busy 0 (0 ms), retain 12>
    | | | +-o H11ANEInDirectPathClient  <class H11ANEInDirectPathClient, id 0x100002c4f, !registered, !matched, active, busy 0, retain 6>
`

const ioregPlist = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<array>
	<dict>
		<key>IOObjectClass</key>
		<string>H11ANEIn</string>
		<key>IORegistryEntryName</key>
		<string>H11ANEIn</string>
		<key>DeviceProperties</key>
		<dict>
			<key>ANEDevicePropertyNumANECores</key>
			<integer>16</integer>
			<key>Ratio</key>
			<real>0.5</real>
			<key>Bad</key>
			<real>nan</real>
			<key>Enabled</key>
			<true/>
			<key>Blob</key>
			<data>AAEC</data>
		</dict>
		<key>IORegistryEntryChildren</key>
		<array>
			<dict>
				<key>IOObjectClass</key>
				<string>H11ANEInDirectPathClient</string>
				<key>IOUserClientCreator</key>
				<string>pid 4242, python3.12</string>
			</dict>
		</array>
	</dict>
</array>
</plist>
`

func TestIOReg(t *testing.T) {
	cls := aneClasses([]byte(ioregText))
	if strings.Join(cls, ",") != "H11ANEIn,H11ANEInDirectPathClient" {
		t.Fatalf("classes %v", cls)
	}
	v, err := parsePlist(strings.NewReader(ioregPlist))
	if err != nil {
		t.Fatal(err)
	}
	nodes := ioregNodes(v)
	if len(nodes) != 1 || nodes[0].Class != "H11ANEIn" || len(nodes[0].Children) != 1 {
		t.Fatalf("nodes %+v", nodes)
	}
	dp := nodes[0].Properties["DeviceProperties"].(map[string]any)
	if dp["ANEDevicePropertyNumANECores"] != int64(16) || dp["Ratio"] != 0.5 || dp["Bad"] != "nan" || dp["Enabled"] != true {
		t.Fatalf("properties %v", dp)
	}
	if b, ok := dp["Blob"].([]byte); !ok || len(b) != 3 || b[2] != 2 {
		t.Fatalf("data %v", dp["Blob"])
	}
	clients := aneClients(nodes)
	if len(clients) != 1 || clients[0].PID != 4242 || clients[0].Process != "python3.12" {
		t.Fatalf("clients %+v", clients)
	}
	seen := map[string]bool{}
	nodeClasses(nodes, seen)
	if !seen["H11ANEInDirectPathClient"] {
		t.Fatalf("seen %v", seen)
	}
	// The result must encode as JSON (no NaN).
	if _, err := json.Marshal(nodes); err != nil {
		t.Fatal(err)
	}
}

func TestGPUInstance(t *testing.T) {
	c := WindowsCounter{Instance: "pid_1234_luid_0x00000000_0x0000d1a2_phys_0_eng_3_engtype_Compute 0"}
	parseGPUInstance(&c)
	if c.PID != 1234 || c.LUID != "0x00000000_0x0000D1A2" || c.Phys == nil || *c.Phys != 0 || c.Engine == nil || *c.Engine != 3 || c.EngType != "Compute 0" {
		t.Fatalf("got %+v", c)
	}
	m := WindowsCounter{Instance: "luid_0x00000000_0x0000D1A2_phys_0"}
	parseGPUInstance(&m)
	if m.PID != 0 || m.LUID != "0x00000000_0x0000D1A2" || m.Engine != nil {
		t.Fatalf("got %+v", m)
	}
	if formatLUID(0, 0xd1a2) != m.LUID {
		t.Fatal("formatLUID differs from the counters' form")
	}
}

func TestMultiSZ(t *testing.T) {
	u := []uint16{'a', 'b', 0, 'c', 0, 0, 'x'}
	if got := multiSZ(u); strings.Join(got, ",") != "ab,c" {
		t.Fatalf("got %v", got)
	}
	if got := adapterTypeNames(1<<11 | 1); strings.Join(got, ",") != "RenderSupported,ComputeOnly" {
		t.Fatalf("got %v", got)
	}
}

func TestDarwinIOReportSummarize(t *testing.T) {
	v := func(x int64) *int64 { return &x }
	r := &DarwinIOReport{Seconds: 0.5, Channels: []DarwinIOReportChannel{
		{Group: "PMP", SubGroup: "Energy Counters", Name: "ANE", Unit: "mJ", Value: v(1000)},
		{Group: "AMC Stats", SubGroup: "Perf Counters", Name: "ANE RD", Unit: "B", Value: v(22e9)},
		{Group: "AMC Stats", SubGroup: "Perf Counters", Name: "ANE WR", Unit: "B", Value: v(5e6)},
		{Group: "Interrupt Statistics (by index)", SubGroup: "ane 0", Name: "   First Level Interrupt Handler Count", Value: v(270)},
		{Group: "Interrupt Statistics (by index)", SubGroup: "dart-ane 0", Name: "   First Level Interrupt Handler Count", Value: v(99999)},
	}}
	r.summarize()
	if r.ANEPowerW != 2 || r.ANEReadBytesPerSec != 44e9 || r.ANEWriteBytesPerSec != 10e6 || r.ANEInterruptsPerSec != 540 {
		t.Fatalf("got %+v", r)
	}
	// Later chips: Energy Model's ANE in nJ wins over PMP.
	r = &DarwinIOReport{Seconds: 1, Channels: []DarwinIOReportChannel{
		{Group: "PMP", SubGroup: "Energy Counters", Name: "ANE", Unit: "mJ", Value: v(5)},
		{Group: "Energy Model", Name: "ANE0", Unit: "nJ", Value: v(1e9)},
		{Group: "Energy Model", Name: "ANE1", Unit: "nJ", Value: v(5e8)},
	}}
	r.summarize()
	if r.ANEPowerW != 1.5 {
		t.Fatalf("got %v W", r.ANEPowerW)
	}
}

func TestIsANEChannel(t *testing.T) {
	for _, c := range []struct {
		g, s, n string
		want    bool
	}{
		{"PMP", "Energy Counters", "ANE", true},
		{"PMP", "Energy Counters", "GPU", false},
		{"AMC Stats", "Perf Counters", "ANE RD", true},
		{"Interrupt Statistics (by index)", "ane 0", "  First Level Interrupt Handler Count", true},
		{"Interrupt Statistics (by index)", "gfx 0", "  First Level Interrupt Handler Count", false},
		{"AOP endpoint 39", "wake reasons", "aop_crane", false},
	} {
		if got := isANEChannel(c.g, c.s, c.n); got != c.want {
			t.Errorf("%q %q %q: %v", c.g, c.s, c.n, got)
		}
	}
}

func TestANEDevice(t *testing.T) {
	nodes := []DarwinIONode{{Class: "H1xANELoadBalancer"}, {Class: "H11ANEIn", Properties: map[string]any{
		"FirmwareLoaded": true,
		"IONameMatched":  "ane,t8020",
		"DeviceProperties": map[string]any{
			"ANEDevicePropertyNumANECores":                int64(16),
			"ANEDevicePropertyTypeANEArchitectureTypeStr": "h13g",
			"ANEDevicePropertyANEVersion":                 int64(64),
		},
		"IOPowerManagement": map[string]any{"CurrentPowerState": int64(1), "MaxPowerState": int64(1)},
	}}}
	d := aneDevice(nodes)
	if d == nil || d.Class != "H11ANEIn" || d.Cores != 16 || d.Architecture != "h13g" || d.Version != 64 ||
		!d.FirmwareLoaded || d.Compatible != "ane,t8020" || d.PowerState != 1 || d.MaxPowerState != 1 {
		t.Fatalf("got %+v", d)
	}
	if aneDevice(nodes[:1]) != nil {
		t.Fatal("found a device without DeviceProperties")
	}
}
