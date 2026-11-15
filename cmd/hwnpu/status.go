package main

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
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/mem"
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
// npuinfo ask a sleeping NPU (and wake it).
func collect(ctx context.Context, s storeView, scan, wake bool) Status {
	st := Status{
		Time: time.Now(), Version: version(), OS: runtime.GOOS, Arch: runtime.GOARCH,
		Memory: mem.Physical(), BackendEnv: os.Getenv(llm.BackendEnv),
	}
	st.OSVersion, st.CPU = osVersion(), cpuName()
	if info, err := npuinfo.GetWithContext(ctx, npuinfo.Options{Wake: wake, SkipCaches: true}); info != nil {
		st.Info = info
		summarize(&st, info)
		if err != nil {
			st.NPU = append(st.NPU, "npuinfo: "+err.Error())
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

// storeView is what Status reads of the store (modelstore.Store).
type storeView interface {
	DiskSpace() (int64, int64, error)
	Limits() (cache.Limits, error)
	Usage() (cache.Usage, error)
	LayerNames() []string
}

func cacheStatus(s storeView, scan bool) CacheStatus {
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
		}
		if v.AverageIPUPower != 0xFFFF {
			line += fmt.Sprintf(", %.1f W", float64(v.AverageIPUPower)/1000)
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
		if !a.FirmwareLoaded {
			line += ", firmware not loaded"
		}
		st.NPU = append(st.NPU, line)
	}
	if r := d.IOReport; r != nil {
		st.NPU = append(st.NPU, fmt.Sprintf("ANE %.2f W, reads %.1f GB/s, %.0f calls/s (IOReport)", r.ANEPowerW, r.ANEReadBytesPerSec/1e9, r.ANEInterruptsPerSec/3))
	}
	if d.ThermalPressure != "" {
		st.NPU = append(st.NPU, "thermal pressure "+d.ThermalPressure)
	}
	for _, p := range d.Processes {
		st.Processes = append(st.Processes, Process{PID: p.PID, Name: p.Name, NPUMemory: int64(p.NeuralFootprint)})
	}
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
			pr.Detail = fmt.Sprintf("%.0f%% busy", b)
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

// write shows st for a person.
func (st Status) write(w io.Writer, args bool) {
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
		fmt.Fprintln(w, "device:   not opened (--wake opens it: on Linux that wakes a sleeping NPU)")
	case len(st.Devices) == 0:
		fmt.Fprintf(w, "device:   no NPU found")
		if st.EnumError != "" {
			fmt.Fprintf(w, " (%s)", st.EnumError)
		}
		fmt.Fprintln(w)
	}
	for _, l := range st.NPU {
		fmt.Fprintf(w, "npu:      %s\n", l)
	}
	if len(st.Processes) == 0 {
		fmt.Fprintln(w, "used by:  no process")
	}
	for _, p := range st.Processes {
		fmt.Fprintf(w, "used by:  %d %s", p.PID, p.Name)
		if p.NPUMemory > 0 {
			fmt.Fprintf(w, ", %s of NPU memory", cache.FormatSize(p.NPUMemory))
		}
		if p.Detail != "" {
			fmt.Fprintf(w, ", %s", p.Detail)
		}
		if args && len(p.Args) > 1 {
			fmt.Fprintf(w, ": %s", strings.Join(p.Args, " "))
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

// StatusCmd is hwnpu status.
type StatusCmd struct {
	JSON  bool          `help:"Write all that was read as JSON (npuinfo's too)."`
	Watch time.Duration `short:"w" help:"Show it again every this long (2s) until Ctrl-C." placeholder:"EVERY"`
	Wake  bool          `help:"Open the NPU and ask it too (its driver's and firmware's details, what hwnpu finds): on Linux this wakes a sleeping NPU for a few seconds."`
}

func (c StatusCmd) Run(ctx context.Context) error {
	s, err := open()
	if err != nil {
		return err
	}
	v := view{s}
	if c.Watch <= 0 {
		st := collect(ctx, v, true, c.Wake)
		if c.JSON {
			e := json.NewEncoder(os.Stdout)
			e.SetIndent("", "  ")
			return e.Encode(st)
		}
		st.write(os.Stdout, true)
		return nil
	}
	for first := true; ; first = false {
		st := collect(ctx, v, first, c.Wake) // the cache is walked once
		if c.JSON {
			json.NewEncoder(os.Stdout).Encode(st)
		} else {
			if !first {
				fmt.Println()
			}
			fmt.Println(st.Time.Format(time.TimeOnly))
			st.write(os.Stdout, true)
		}
		select {
		case <-ctx.Done():
			return nil
		case <-time.After(c.Watch):
		}
	}
}
