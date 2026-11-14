//go:build linux

package npuinfo

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"unicode/utf8"
)

const (
	accelClass = "/sys/class/accel"
	procRoot   = "/proc"
	drmClass   = "/sys/class/drm"
)

func collect(ctx context.Context, opts Options, info *Info) error {
	li := &LinuxInfo{}
	info.Linux = li
	guard("accel devices", info, func() error {
		devs, err := linuxAccelDevices(opts)
		li.Devices = devs
		return err
	})
	guard("processes", info, func() error {
		paths := map[string]bool{}
		for _, d := range li.Devices {
			paths[d.DevPath] = true
		}
		if len(paths) == 0 {
			return nil
		}
		procs, unreadable, err := linuxAccelProcesses(ctx, paths)
		li.Processes, li.UnreadableProcesses = procs, unreadable
		return err
	})
	guard("amdgpu gpu_metrics", info, func() error {
		m, err := linuxGPUMetrics()
		li.AMDGPUMetrics = m
		return err
	})
	return nil
}

var accelName = regexp.MustCompile(`^accel\d+$`)

func linuxAccelDevices(opts Options) ([]LinuxAccelDevice, error) {
	ents, err := os.ReadDir(accelClass)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var devs []LinuxAccelDevice
	for _, e := range ents {
		if !accelName.MatchString(e.Name()) {
			continue
		}
		devDir := filepath.Join(accelClass, e.Name(), "device")
		d := LinuxAccelDevice{Name: e.Name(), DevPath: "/dev/accel/" + e.Name()}
		if t, err := os.Readlink(filepath.Join(devDir, "driver")); err == nil {
			d.Driver = filepath.Base(t)
		}
		if t, err := filepath.EvalSymlinks(devDir); err == nil {
			if b := filepath.Base(t); strings.Count(b, ":") == 2 {
				d.PCISlot = b
			}
		}
		d.Sysfs = readSysfsAttrs(devDir)
		d.RuntimeStatus = d.Sysfs["power/runtime_status"]
		if d.Driver == "amdxdna" {
			d.XDNA = queryXDNA(d.DevPath, d.RuntimeStatus, opts)
		}
		devs = append(devs, d)
	}
	return devs, nil
}

// Attributes left out of the sysfs dump: binary files, ones that only write, and the PCI
// config space (reading it resumes a suspended device).
var sysfsSkip = map[string]bool{
	"config": true, "rom": true, "vpd": true, "remove": true, "rescan": true, "reset": true,
	"driver_override": true,
}

func readSysfsAttrs(devDir string) map[string]string {
	attrs := map[string]string{}
	for _, sub := range []string{"", "power", "freq"} {
		dir := filepath.Join(devDir, sub)
		ents, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		for _, e := range ents {
			name := e.Name()
			if sysfsSkip[name] || strings.HasPrefix(name, "resource") || !e.Type().IsRegular() {
				continue
			}
			fi, err := e.Info()
			if err != nil || fi.Mode().Perm()&0o444 == 0 {
				continue
			}
			v, ok := readSmallText(filepath.Join(dir, name))
			if !ok {
				continue
			}
			attrs[filepath.Join(sub, name)] = v
		}
	}
	return attrs
}

func readSmallText(path string) (string, bool) {
	f, err := os.Open(path)
	if err != nil {
		return "", false
	}
	defer f.Close()
	b := make([]byte, 4096)
	n, err := f.Read(b)
	if err != nil && n == 0 {
		return "", false
	}
	b = b[:n]
	if !utf8.Valid(b) || strings.IndexByte(string(b), 0) >= 0 {
		return "", false
	}
	return strings.TrimSpace(string(b)), true
}

func linuxAccelProcesses(ctx context.Context, devPaths map[string]bool) ([]LinuxProcess, int, error) {
	ents, err := os.ReadDir(procRoot)
	if err != nil {
		return nil, 0, err
	}
	var procs []LinuxProcess
	unreadable := 0
	for _, e := range ents {
		if ctx.Err() != nil {
			return procs, unreadable, ctx.Err()
		}
		pid, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}
		pdir := filepath.Join(procRoot, e.Name())
		fds, err := os.ReadDir(filepath.Join(pdir, "fd"))
		if err != nil {
			if errors.Is(err, os.ErrPermission) {
				unreadable++
			}
			continue
		}
		var files []LinuxOpenAccel
		for _, fd := range fds {
			t, err := os.Readlink(filepath.Join(pdir, "fd", fd.Name()))
			if err != nil || !devPaths[t] {
				continue
			}
			n, _ := strconv.Atoi(fd.Name())
			f := LinuxOpenAccel{FD: n, DevPath: t}
			if b, err := os.ReadFile(filepath.Join(pdir, "fdinfo", fd.Name())); err == nil {
				f.FDInfo = parseFDInfo(b)
			}
			files = append(files, f)
		}
		if len(files) == 0 {
			continue
		}
		p := LinuxProcess{PID: pid, Files: files, UID: -1}
		if b, err := os.ReadFile(filepath.Join(pdir, "comm")); err == nil {
			p.Comm = strings.TrimSpace(string(b))
		}
		if b, err := os.ReadFile(filepath.Join(pdir, "cmdline")); err == nil {
			p.Cmdline = strings.Split(strings.TrimRight(string(b), "\x00"), "\x00")
		}
		if fi, err := os.Stat(pdir); err == nil {
			if st, ok := fi.Sys().(*syscall.Stat_t); ok {
				p.UID = int(st.Uid)
			}
		}
		if b, err := os.ReadFile(filepath.Join(pdir, "maps")); err == nil {
			p.Mapped, p.Mappings = sumMaps(b, devPaths)
		}
		procs = append(procs, p)
	}
	sort.Slice(procs, func(i, j int) bool { return procs[i].PID < procs[j].PID })
	return procs, unreadable, nil
}

var cardName = regexp.MustCompile(`^card\d+$`)

func linuxGPUMetrics() ([]AMDGPUMetrics, error) {
	ents, err := os.ReadDir(drmClass)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	var out []AMDGPUMetrics
	var errs []error
	for _, e := range ents {
		if !cardName.MatchString(e.Name()) {
			continue
		}
		p := filepath.Join(drmClass, e.Name(), "device", "gpu_metrics")
		b, err := os.ReadFile(p)
		if err != nil {
			if !errors.Is(err, os.ErrNotExist) {
				errs = append(errs, err)
			}
			continue
		}
		m, err := decodeGPUMetrics(b)
		m.Card, m.Path = e.Name(), p
		if err != nil {
			errs = append(errs, err)
		}
		out = append(out, m)
	}
	return out, errors.Join(errs...)
}
