// Package npustatus is hwnpu status: this machine's NPU (the device, its driver, how busy it is,
// what uses it), what hwnpu runs on, and the cache. It is the command of hwnpu and of the apps
// made with the ai package (aicli.Status).
package npustatus

import (
	"bufio"
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"runtime"
	"runtime/debug"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/mem"
	"github.com/shibukawa/hwnpu/internal/sysload"
	"github.com/shibukawa/hwnpu/modelstore"
	"github.com/shibukawa/hwnpu/npuinfo"
)

// Status is what hwnpu status shows: this machine, its NPU and what uses it, what hwnpu runs
// on, and the cache.
type Status struct {
	Time       time.Time          `json:"time"`
	Version    string             `json:"version"` // hwnpu's: the module's version, the commit it was built from
	OS         string             `json:"os"`
	OSVersion  string             `json:"os_version,omitempty"`
	Arch       string             `json:"arch"`
	CPU        string             `json:"cpu,omitempty"`
	Memory     int64              `json:"memory_bytes,omitempty"`
	Vendor     string             `json:"vendor"`                  // the NPU the commands run on: amd, apple, intel ("" for none)
	BackendEnv string             `json:"hwnpu_backend,omitempty"` // HWNPU_BACKEND
	Devices    []hwnpu.DeviceInfo `json:"devices,omitempty"`
	// Opened says the devices were opened (--wake): Devices is what hwnpu finds then.
	Opened    bool   `json:"devices_opened"`
	EnumError string `json:"enumerate_error,omitempty"`
	// Load is how busy the CPU, the GPUs and the NPU are, and the memory each uses.
	Load Load `json:"load"`
	// NPU are lines that say how the NPU is now (its power state, how busy, its power).
	NPU []string `json:"npu,omitempty"`
	// Processes are the processes that use the NPU now.
	Processes []Process   `json:"processes,omitempty"`
	Cache     CacheStatus `json:"cache"`
	// Info is all that npuinfo read (hwnpu status --json).
	Info *npuinfo.Info `json:"npuinfo,omitempty"`
}

// Process is a process that uses the NPU.
type Process struct {
	PID  int    `json:"pid"`
	Name string `json:"name"`
	// Args are its command line (left out of an issue: it may hold a prompt).
	Args []string `json:"args,omitempty"`
	// NPUMemory is the NPU's memory it holds (bytes; 0 when not told).
	NPUMemory int64  `json:"npu_memory_bytes,omitempty"`
	Detail    string `json:"detail,omitempty"`
	// Proc is its load and memory: the CPU's, the GPU's, the NPU's (where the OS tells them).
	sysload.Proc
}

// Samplers keep the counters that loads are worked out from between collects.
type Samplers struct {
	sys   sysload.Sampler
	procs sysload.ProcSampler
}

// Load is the load part of Status: the machine's (sysload) and the NPU's.
type Load struct {
	sysload.Sample
	// NPUPercent is how busy the NPU is (amdgpu's gpu_metrics on Linux); nil when not known.
	NPUPercent *float64 `json:"npu_percent,omitempty"`
	// NPUPowerW is the NPU's power (gpu_metrics, macOS' IOReport); nil when not known.
	NPUPowerW *float64 `json:"npu_power_w,omitempty"`
	// NPUMemory is the memory the NPU holds: what its processes map (Linux), what the kernel
	// keeps wired for the ANE's driver (macOS), else the processes' ANE footprint. 0 when not
	// known.
	NPUMemory int64 `json:"npu_memory_bytes,omitempty"`
}

// CacheStatus is the cache's part of Status.
type CacheStatus struct {
	Root   string           `json:"root"`
	Bytes  int64            `json:"bytes"`
	ByKind map[string]int64 `json:"by_kind,omitempty"`
	Free   int64            `json:"disk_free"`
	Total  int64            `json:"disk_total"`
	Limits string           `json:"limits"`
	Layers []string         `json:"layers"`
}

// version is hwnpu's version: the module's, and the commit of a build from a checkout.
func version() string {
	bi, ok := debug.ReadBuildInfo()
	if !ok {
		return "unknown"
	}
	v := bi.Main.Version
	var rev, dirty string
	for _, s := range bi.Settings {
		switch s.Key {
		case "vcs.revision":
			rev = s.Value[:min(12, len(s.Value))]
		case "vcs.modified":
			if s.Value == "true" {
				dirty = "+changes"
			}
		}
	}
	if rev != "" && !strings.Contains(v, rev) {
		v += " (" + rev + dirty + ")"
	}
	return v + ", " + runtime.Version()
}

// collect reads the Status. scan walks the cache (a few seconds with many files); wake lets
// npuinfo ask a sleeping NPU (and wake it); load keeps the counters of the loads between reads.
func Collect(ctx context.Context, s StoreView, scan, wake bool, load *Samplers) Status {
	st := Status{
		Time: time.Now(), Version: version(), OS: runtime.GOOS, Arch: runtime.GOARCH,
		Memory: mem.Physical(), BackendEnv: os.Getenv(llm.BackendEnv),
	}
	st.Load.Sample = load.sys.Read()
	st.OSVersion, st.CPU = osVersion(), cpuName()
	if info, err := npuinfo.GetWithContext(ctx, npuinfo.Options{Wake: wake, SkipCaches: true}); info != nil {
		st.Info = info
		summarize(&st, info)
		if err != nil {
			st.NPU = append(st.NPU, "npuinfo: "+err.Error())
		}
	}
	pids := make([]int, len(st.Processes))
	for i, p := range st.Processes {
		pids[i] = p.PID
	}
	procs := load.procs.Read(pids)
	for i := range st.Processes {
		p := &st.Processes[i]
		npu := p.NPUPercent // put by summarize: Windows' counters, the ANE's share on macOS
		p.Proc = procs[p.PID]
		if npu != nil {
			p.NPUPercent = npu
		}
	}
	// opening the devices wakes a sleeping NPU (amdxdna): only when asked. Without it the NPU is
	// known from what the OS says of it (the driver, the PCI vendor, the ANE in the I/O Registry)
	if wake {
		st.Devices, st.EnumError = devices(ctx)
		st.Opened = true
		st.Vendor = chat.HereVendor()
	} else {
		st.Vendor = vendorOf(st.Info)
	}
	st.Cache = cacheStatus(s, scan)
	return st
}

var (
	enumerated bool
	devs       []hwnpu.DeviceInfo
	enumErr    string
)

// devices are the NPUs hwnpu finds, looked for once (status --watch: not every time).
func devices(ctx context.Context) ([]hwnpu.DeviceInfo, string) {
	if !enumerated {
		enumerated = true
		d, err := hwnpu.Enumerate(ctx)
		devs = d
		if err != nil {
			enumErr = err.Error()
		}
	}
	return devs, enumErr
}

// vendorOf is the NPU the commands run on, as the OS tells of it without opening a device: the
// vendor HWNPU_BACKEND names, else the driver's (Linux), the ANE (macOS), the vendor of a
// compute-only adapter (Windows). "" for none.
func vendorOf(info *npuinfo.Info) string {
	if os.Getenv(llm.BackendEnv) != "" {
		return chat.HereVendor() // the variable decides, nothing is opened
	}
	if info == nil {
		return ""
	}
	switch {
	case info.Linux != nil:
		for _, d := range info.Linux.Devices {
			switch d.Driver {
			case "amdxdna":
				return "amd"
			case "intel_vpu":
				return "intel"
			}
		}
	case info.Darwin != nil:
		if info.Darwin.Device != nil {
			return "apple"
		}
	case info.Windows != nil:
		for _, a := range info.Windows.Adapters {
			if !slices.Contains(a.TypeNames, "ComputeOnly") {
				continue
			}
			switch a.VendorID {
			case 0x1022:
				return "amd"
			case 0x8086:
				return "intel"
			}
		}
	}
	return ""
}

// StoreView is what Status reads of the store (modelstore.Store).
type StoreView interface {
	DiskSpace() (int64, int64, error)
	Limits() (cache.Limits, error)
	Usage() (cache.Usage, error)
	LayerNames() []string
}

func cacheStatus(s StoreView, scan bool) CacheStatus {
	c := CacheStatus{Root: cache.Root(), Layers: s.LayerNames()}
	c.Free, c.Total, _ = s.DiskSpace()
	if l, err := s.Limits(); err != nil {
		c.Limits = err.Error()
	} else {
		c.Limits = l.String()
	}
	if scan {
		if u, err := s.Usage(); err == nil {
			c.Bytes, c.ByKind = u.Bytes, u.ByKind()
		}
	}
	return c
}

// summarize puts what npuinfo read in st's NPU and Processes, as each OS tells it.
func summarize(st *Status, info *npuinfo.Info) {
	switch {
	case info.Linux != nil:
		summarizeLinux(st, info.Linux)
	case info.Darwin != nil:
		summarizeDarwin(st, info.Darwin)
	case info.Windows != nil:
		summarizeWindows(st, info.Windows)
	}
	slices.SortFunc(st.Processes, func(a, b Process) int { return cmp.Compare(a.PID, b.PID) })
}

func summarizeLinux(st *Status, l *npuinfo.LinuxInfo) {
	for _, d := range l.Devices {
		line := fmt.Sprintf("%s: %s", d.Name, d.Driver)
		if v := d.Sysfs["vbnv"]; v != "" {
			line += " " + v
		}
		if v := d.Sysfs["fw_version"]; v != "" {
			line += ", firmware " + v
		}
		if d.RuntimeStatus != "" {
			line += ", " + d.RuntimeStatus
		}
		if x := d.XDNA; x != nil {
			if x.AIEMetadata != nil {
				line += fmt.Sprintf(", %d×%d tiles", x.AIEMetadata.Cols, x.AIEMetadata.Rows)
			}
			if x.PowerModeName != "" {
				line += ", power mode " + x.PowerModeName
			}
			for _, c := range x.Clocks {
				line += fmt.Sprintf(", %s %d MHz", c.Name, c.MHz)
			}
			busy := 0
			for _, c := range x.Contexts {
				if c.State == 1 {
					busy++
				}
			}
			if len(x.Contexts) > 0 {
				line += fmt.Sprintf(", %d contexts (%d active)", len(x.Contexts), busy)
			}
		}
		st.NPU = append(st.NPU, line)
	}
	for _, g := range l.AMDGPUMetrics {
		v := g.V3
		if v == nil {
			continue
		}
		var sum, n int
		for _, a := range v.AverageIPUActivity {
			if a != 0xFFFF {
				sum += int(a)
				n++
			}
		}
		line := "NPU"
		if n > 0 {
			line += fmt.Sprintf(" %d%% busy", sum/n)
			p := float64(sum) / float64(n)
			st.Load.NPUPercent = &p
		}
		if v.AverageIPUPower != 0xFFFF {
			line += fmt.Sprintf(", %.1f W", float64(v.AverageIPUPower)/1000)
			w := float64(v.AverageIPUPower) / 1000
			st.Load.NPUPowerW = &w
		}
		if v.AverageSocketPower != 0xFFFFFFFF {
			line += fmt.Sprintf(" (the whole chip %.1f W)", float64(v.AverageSocketPower)/1000)
		}
		st.NPU = append(st.NPU, line+" ("+g.Card+" gpu_metrics)")
	}
	for _, p := range l.Processes {
		var b uint64
		for _, n := range p.Mapped {
			b += n
		}
		st.Processes = append(st.Processes, Process{PID: p.PID, Name: p.Comm, Args: p.Cmdline, NPUMemory: int64(b)})
		st.Load.NPUMemory += int64(b)
	}
}

func summarizeDarwin(st *Status, d *npuinfo.DarwinInfo) {
	if d.Version != "" {
		st.OSVersion = "macOS " + d.Version
	}
	if d.CPU != "" {
		st.CPU = d.CPU
		if d.Model != "" {
			st.CPU += " (" + d.Model + ")"
		}
	}
	if a := d.Device; a != nil {
		line := fmt.Sprintf("ANE %s", cmp.Or(a.Architecture, a.Class))
		if a.Cores > 0 {
			line += fmt.Sprintf(", %d cores", a.Cores)
		}
		state := "asleep"
		if a.PowerState > 0 {
			state = "awake"
		}
		line += ", " + state
		if a.FirmwareKnown && !a.FirmwareLoaded {
			line += ", firmware not loaded"
		}
		st.NPU = append(st.NPU, line)
	}
	if r := d.IOReport; r != nil {
		var parts []string
		if r.ANEBusyPercent != nil {
			b := *r.ANEBusyPercent
			st.Load.NPUPercent = &b
			parts = append(parts, fmt.Sprintf("reading memory %.0f%% of the time", b))
		}
		if r.ANEPowerKnown {
			w := r.ANEPowerW
			st.Load.NPUPowerW = &w
			parts = append(parts, fmt.Sprintf("%.2f W", w))
		} else {
			parts = append(parts, "power not readable without root (sudo powermetrics --Samplers ane_power)")
		}
		if r.ANEReadBytesPerSec > 0 {
			parts = append(parts, fmt.Sprintf("reads %.1f GB/s", r.ANEReadBytesPerSec/1e9))
		}
		parts = append(parts, fmt.Sprintf("%.0f calls/s", r.ANEInterruptsPerSec/3))
		st.NPU = append(st.NPU, "ANE "+strings.Join(parts, ", ")+" (IOReport)")
	}
	if d.ThermalPressure != "" {
		st.NPU = append(st.NPU, "thermal pressure "+d.ThermalPressure)
	}
	for _, p := range d.Processes {
		st.Processes = append(st.Processes, Process{PID: p.PID, Name: p.Name, NPUMemory: int64(p.NeuralFootprint)})
		st.Load.NPUMemory += int64(p.NeuralFootprint)
	}
	if b, ok := mem.NPUWired(); ok {
		st.Load.NPUMemory = b // the weights are here, not in the processes' footprint
	}
	// the ANE has no time per process: the process holding nearly all of its memory (a model,
	// beside the input method's or Siri's few MB) is given the ANE's busy share
	if i := npuHolder(st.Processes); i >= 0 && st.Load.NPUPercent != nil {
		b := *st.Load.NPUPercent
		st.Processes[i].NPUPercent = &b
	}
}

// npuHolder is the process that holds at least 90 % of the processes' NPU memory, -1 for none.
func npuHolder(ps []Process) int {
	best, sum := -1, int64(0)
	for i, p := range ps {
		sum += p.NPUMemory
		if p.NPUMemory > 0 && (best < 0 || p.NPUMemory > ps[best].NPUMemory) {
			best = i
		}
	}
	if best < 0 || 10*ps[best].NPUMemory < 9*sum {
		return -1
	}
	return best
}

func summarizeWindows(st *Status, w *npuinfo.WindowsInfo) {
	for _, a := range w.Adapters {
		if slices.Contains(a.TypeNames, "ComputeOnly") || strings.Contains(strings.ToLower(a.AdapterString), "npu") {
			st.NPU = append(st.NPU, fmt.Sprintf("%s (%s)", a.AdapterString, strings.Join(a.TypeNames, ", ")))
		}
	}
	busy := map[int]float64{}
	for _, c := range w.Counters {
		if c.PID > 0 && c.Counter == "Utilization Percentage" {
			busy[c.PID] += c.Value
		}
	}
	for _, p := range w.Processes {
		pr := Process{PID: p.PID, Name: p.Image}
		if b, ok := busy[p.PID]; ok {
			pr.NPUPercent = &b
		}
		st.Processes = append(st.Processes, pr)
	}
}

func osVersion() string {
	switch runtime.GOOS {
	case "linux":
		name := ""
		if f, err := os.Open("/etc/os-release"); err == nil {
			sc := bufio.NewScanner(f)
			for sc.Scan() {
				if v, ok := strings.CutPrefix(sc.Text(), "PRETTY_NAME="); ok {
					name = strings.Trim(v, `"`)
				}
			}
			f.Close()
		}
		if b, err := os.ReadFile("/proc/sys/kernel/osrelease"); err == nil {
			name = strings.TrimSpace(name + ", Linux " + strings.TrimSpace(string(b)))
		}
		return strings.TrimPrefix(name, ", ")
	}
	return ""
}

func cpuName() string {
	if runtime.GOOS != "linux" {
		return ""
	}
	f, err := os.Open("/proc/cpuinfo")
	if err != nil {
		return ""
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if k, v, ok := strings.Cut(sc.Text(), ":"); ok && strings.TrimSpace(k) == "model name" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

// write shows st for a person: what stays (writeFixed), then what changes (writeLive).
func (st Status) Write(w io.Writer, args bool) {
	st.writeFixed(w)
	st.writeLive(w, args)
}

// writeFixed writes what does not change while hwnpu status --watch runs: hwnpu, the machine,
// the device, the cache.
func (st Status) writeFixed(w io.Writer) {
	fmt.Fprintf(w, "hwnpu:    %s\n", st.Version)
	sys := st.OS + "/" + st.Arch
	if st.OSVersion != "" {
		sys += ", " + st.OSVersion
	}
	fmt.Fprintf(w, "machine:  %s\n", sys)
	if st.CPU != "" || st.Memory > 0 {
		fmt.Fprintf(w, "          %s", st.CPU)
		if st.Memory > 0 {
			fmt.Fprintf(w, ", %.0f GiB of memory", float64(st.Memory)/(1<<30))
		}
		fmt.Fprintln(w)
	}
	runs := cmp.Or(st.Vendor, "none (the CPU)")
	if st.BackendEnv != "" {
		runs += " (" + llm.BackendEnv + "=" + st.BackendEnv + ")"
	}
	fmt.Fprintf(w, "runs on:  %s\n", runs)
	for _, d := range st.Devices {
		fmt.Fprintf(w, "device:   %s, %s backend", d.Name, d.Backend)
		if d.Driver != "" {
			fmt.Fprintf(w, ", driver %s", d.Driver)
		}
		if d.Firmware != "" {
			fmt.Fprintf(w, ", firmware %s", d.Firmware)
		}
		fmt.Fprintln(w)
	}
	switch {
	case !st.Opened:
		// on macOS opening it tells nothing more and wakes nothing: not worth a line
		if st.OS != "darwin" {
			fmt.Fprintln(w, "device:   not opened (--wake opens it: on Linux that wakes a sleeping NPU)")
		}
	case len(st.Devices) == 0:
		fmt.Fprintf(w, "device:   no NPU found")
		if st.EnumError != "" {
			fmt.Fprintf(w, " (%s)", st.EnumError)
		}
		fmt.Fprintln(w)
	}
	c := st.Cache
	fmt.Fprintf(w, "cache:    %s", c.Root)
	if c.Bytes > 0 {
		fmt.Fprintf(w, ", %s", cache.FormatSize(c.Bytes))
	}
	if c.Total > 0 {
		fmt.Fprintf(w, "; disk %s free of %s", cache.FormatSize(c.Free), cache.FormatSize(c.Total))
	}
	fmt.Fprintf(w, "\n          %s\n", c.Limits)
	for i, l := range c.Layers {
		fmt.Fprintf(w, "layer %d:  %s\n", i, l)
	}
}

// writeLive writes what changes: the load of the CPU, the GPUs and the NPU and their memory,
// the NPU's state, the processes that use it.
func (st Status) writeLive(w io.Writer, args bool) {
	l := st.Load
	fmt.Fprintf(w, "cpu:      %s", loadBar(l.CPUPercent))
	if l.MemoryTotal > 0 {
		fmt.Fprintf(w, "  memory %s of %s", sizeOr(l.MemoryUsed), cache.FormatSize(l.MemoryTotal))
	}
	fmt.Fprintln(w)
	for _, g := range l.GPUs {
		fmt.Fprintf(w, "gpu:      %s  %s", loadBar(g.BusyPercent), g.Name)
		if g.VRAMTotal > 0 {
			fmt.Fprintf(w, ", vram %s of %s", cache.FormatSize(g.VRAMUsed), cache.FormatSize(g.VRAMTotal))
		}
		switch {
		case g.SharedTotal > 0:
			fmt.Fprintf(w, ", gtt %s of %s", cache.FormatSize(g.SharedUsed), cache.FormatSize(g.SharedTotal))
		case g.SharedUsed > 0:
			fmt.Fprintf(w, ", memory %s", cache.FormatSize(g.SharedUsed))
		}
		fmt.Fprintln(w)
	}
	if st.Vendor != "" || l.NPUPercent != nil || l.NPUPowerW != nil || l.NPUMemory > 0 {
		fmt.Fprintf(w, "npu:      %s", loadBar(l.NPUPercent))
		var parts []string
		if l.NPUPowerW != nil {
			parts = append(parts, fmt.Sprintf("%.2f W", *l.NPUPowerW))
		}
		parts = append(parts, "memory "+cache.FormatSize(l.NPUMemory))
		fmt.Fprintf(w, "  %s\n", strings.Join(parts, ", "))
	}
	for _, l := range st.NPU {
		fmt.Fprintf(w, "npu:      %s\n", l)
	}
	writeProcesses(w, st.Processes, args)
}

// procName is the width of the processes' NAME column.
const procName = 16

// writeProcesses writes the processes that use the NPU as a table: "-" is not known (another
// user's process without root, what the OS does not tell). args adds their command lines.
func writeProcesses(w io.Writer, ps []Process, args bool) {
	if len(ps) == 0 {
		fmt.Fprintln(w, "used by:  no process")
		return
	}
	fmt.Fprintln(w, "used by:")
	row := "%7s  %-*s  %6s %9s  %5s %9s  %5s %9s"
	head := fmt.Sprintf(row, "PID", procName, "NAME", "CPU%", "MEMORY", "GPU%", "GPU MEM", "NPU%", "NPU MEM")
	if args {
		head += "  COMMAND"
	}
	fmt.Fprintln(w, head)
	for _, p := range ps {
		line := fmt.Sprintf(row, strconv.Itoa(p.PID), procName, trimName(p.Name, procName),
			percent(p.CPUPercent), sizeOr(p.Memory), percent(p.GPUPercent), sizeOr(p.GPUMemory),
			percent(p.NPUPercent), sizeOr(p.NPUMemory))
		if args && len(p.Args) > 0 {
			line += "  " + strings.Join(p.Args, " ")
		}
		if p.Detail != "" {
			line += "  " + p.Detail
		}
		fmt.Fprintln(w, line)
	}
}

// trimName cuts s to n runes, its end shown by "…".
func trimName(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n-1]) + "…"
}

func percent(p *float64) string {
	if p == nil {
		return "-"
	}
	return fmt.Sprintf("%.1f", *p)
}

// loadBarWidth is the cells of loadBar's bar.
const loadBarWidth = 20

// loadBar is a load as a bar and a percentage ("[████░░…]  21%"), blank cells and "  -" when
// not known.
func loadBar(p *float64) string {
	if p == nil {
		return "[" + strings.Repeat(" ", loadBarWidth) + "]    -"
	}
	v := min(max(*p, 0), 100)
	n := int(v/100*loadBarWidth + 0.5)
	return fmt.Sprintf("[%s%s] %3.0f%%", strings.Repeat("█", n), strings.Repeat("░", loadBarWidth-n), v)
}

func sizeOr(n int64) string {
	if n <= 0 {
		return "-"
	}
	return cache.FormatSize(n)
}

// Cmd is hwnpu status.
type Cmd struct {
	JSON  bool          `help:"Write all that was read as JSON (npuinfo's too)."`
	Watch time.Duration `short:"w" help:"Show it again every this long (2s) until Ctrl-C: on a terminal, as top does (what stays on top, the load below it, redrawn in place)." placeholder:"EVERY"`
	Wake  bool          `help:"Open the NPU and ask it too (its driver's and firmware's details, what hwnpu finds): on Linux this wakes a sleeping NPU for a few seconds."`
	// Store is the store whose cache and layers are shown: the zero value is that of hwnpu's
	// commands, an app gives its own (ai.Store.Models).
	Store modelstore.Options `kong:"-"`
}

func (c Cmd) Run(ctx context.Context) error {
	o := c.Store
	if o.Log == nil {
		o.Log = os.Stderr
	}
	s, err := modelstore.Open(o)
	if err != nil {
		return err
	}
	v := View{s}
	var load Samplers
	if c.Watch <= 0 {
		st := Collect(ctx, v, true, c.Wake, &load)
		if c.JSON {
			e := json.NewEncoder(os.Stdout)
			e.SetIndent("", "  ")
			return e.Encode(st)
		}
		st.Write(os.Stdout, true)
		return nil
	}
	if !c.JSON && runtime.GOOS != "windows" && console.Terminal(os.Stdout) {
		return c.top(ctx, v, &load)
	}
	for first := true; ; first = false {
		st := Collect(ctx, v, first, c.Wake, &load) // the cache is walked once
		if c.JSON {
			json.NewEncoder(os.Stdout).Encode(st)
		} else {
			if !first {
				fmt.Println()
			}
			fmt.Println(st.Time.Format(time.TimeOnly))
			st.Write(os.Stdout, true)
		}
		select {
		case <-ctx.Done():
			return nil
		case <-time.After(c.Watch):
		}
	}
}

// The escape sequences top draws with.
const (
	altScreen   = "\x1b[?1049h\x1b[?25l" // the alternate screen, no cursor
	mainScreen  = "\x1b[?25h\x1b[?1049l"
	home        = "\x1b[H"
	clearLine   = "\x1b[K"
	clearBelow  = "\x1b[J"
	moveToLineF = "\x1b[%d;1H"
)

// top is hwnpu status --watch on a terminal: the alternate screen, with what stays written
// once at the top and the load redrawn in place below it every c.Watch (again all when the
// terminal is resized), until Ctrl-C.
func (c Cmd) top(ctx context.Context, v View, load *Samplers) error {
	out := os.Stdout
	fmt.Fprint(out, altScreen)
	defer fmt.Fprint(out, mainScreen)
	var fixed []string
	width, height := -1, -1
	for first := true; ; first = false {
		next := time.Now().Add(c.Watch) // every c.Watch, however long reading takes
		st := Collect(ctx, v, first, c.Wake, load)
		if ctx.Err() != nil {
			return nil
		}
		if first { // the cache is walked once: later Statuses keep its size
			var f strings.Builder
			st.writeFixed(&f)
			fixed = strings.Split(strings.TrimSuffix(f.String(), "\n"), "\n")
		}
		cols, rows := console.Size(out)
		if cols <= 0 || rows <= 0 {
			cols, rows = 80, 24
		}
		// a small terminal keeps the last 3 rows at least for the load
		top := min(len(fixed), max(rows-3, 0))
		var b strings.Builder
		if cols != width || rows != height {
			width, height = cols, rows
			b.WriteString(home + clearBelow)
			for i, l := range fixed[:top] {
				fmt.Fprintf(&b, moveToLineF, i+1)
				b.WriteString(fit(l, width))
			}
		}
		lines := []string{fmt.Sprintf("\x1b[7m%s\x1b[0m", fit(fmt.Sprintf(" %s  every %s, Ctrl-C to quit ", st.Time.Format(time.TimeOnly), c.Watch), width))}
		var live strings.Builder
		st.writeLive(&live, true)
		lines = append(lines, strings.Split(strings.TrimSuffix(live.String(), "\n"), "\n")...)
		// no line wraps and none goes below the screen (it would scroll what stays away)
		for i, l := range lines[:min(len(lines), rows-top)] {
			fmt.Fprintf(&b, moveToLineF, top+1+i)
			b.WriteString(fit(l, width) + clearLine)
		}
		b.WriteString(clearBelow)
		io.WriteString(out, b.String())
		select {
		case <-ctx.Done():
			return nil
		case <-time.After(time.Until(next)):
		}
	}
}

// fit cuts s to width cells (runes; the status has no wide ones), so that no line wraps and
// moves what is below it.
func fit(s string, width int) string {
	if width <= 0 {
		return s
	}
	r := []rune(s)
	// the escape sequences take no cells
	cells, i := 0, 0
	for i < len(r) && cells < width {
		if r[i] == 0x1b {
			for i < len(r) && !(r[i] >= 'A' && r[i] <= 'Z' || r[i] >= 'a' && r[i] <= 'z') {
				i++
			}
			i++
			continue
		}
		cells++
		i++
	}
	if i >= len(r) {
		return s
	}
	return string(r[:i])
}

// View is a Store as status reads it.
type View struct{ *modelstore.Store }

// LayerNames are the names of the layers, in order.
func (v View) LayerNames() []string {
	var n []string
	for _, l := range v.Layers() {
		s := l.Name
		switch {
		case l.Bundled:
			s += " (bundled, read only)"
		case l.Writeback:
			s += " (written back to)"
		}
		n = append(n, s)
	}
	return n
}
