//go:build darwin

package npuinfo

import (
	"bytes"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"unsafe"
)

type notifyLib struct{ registerCheck, getState, cancel uintptr }

var loadNotify = sync.OnceValues(func() (*notifyLib, error) {
	l, err := libSystem()
	if err != nil {
		return nil, err
	}
	n := &notifyLib{}
	if err := l.syms([]string{"notify_register_check", "notify_get_state", "notify_cancel"},
		&n.registerCheck, &n.getState, &n.cancel); err != nil {
		return nil, err
	}
	return n, nil
})

// notifyState reads the state of a notify(3) name, as notifyutil -g does.
func notifyState(name string) (uint64, bool) {
	n, err := loadNotify()
	if err != nil {
		return 0, false
	}
	cs := cstring(name)
	token := new(int32)
	if ccall(n.registerCheck, uintptr(unsafe.Pointer(cs)), uintptr(unsafe.Pointer(token))) != 0 {
		return 0, false
	}
	defer ccall(n.cancel, uintptr(*token))
	state := new(uint64)
	r := ccall(n.getState, uintptr(*token), uintptr(unsafe.Pointer(state)))
	runtime.KeepAlive(cs)
	runtime.KeepAlive(token)
	runtime.KeepAlive(state)
	return *state, r == 0
}

func darwinPowerState(di *DarwinInfo) {
	if v, ok := notifyState("com.apple.system.thermalpressurelevel"); ok {
		di.ThermalPressure = thermalPressureNames[v]
		if di.ThermalPressure == "" {
			di.ThermalPressure = "unknown"
		}
	}
	if v, ok := notifyState("com.apple.system.lowpowermode"); ok {
		b := v != 0
		di.LowPowerMode = &b
	}
}

// darwinDrivers reads the loaded ANE kexts (the *ANE*.kext whose bundle id an ANE object in
// the I/O Registry has) and AppleNeuralEngine.framework's version.
func darwinDrivers(di *DarwinInfo) error {
	loaded := map[string]bool{}
	var walk func(ns []DarwinIONode)
	walk = func(ns []DarwinIONode) {
		for _, n := range ns {
			if id, ok := n.Properties["CFBundleIdentifier"].(string); ok {
				loaded[id] = true
			}
			walk(n.Children)
		}
	}
	walk(di.ANE)
	paths, err := filepath.Glob("/System/Library/Extensions/*ANE*.kext")
	if err != nil {
		return err
	}
	for _, p := range paths {
		b, err := os.ReadFile(filepath.Join(p, "Contents", "Info.plist"))
		if err != nil {
			continue
		}
		id, v, ok := aneKextVersion(bytes.NewReader(b))
		if !ok || !loaded[id] {
			continue // the others are for other chips
		}
		di.Drivers = append(di.Drivers, DarwinKext{Path: p, BundleID: id, Version: v})
	}
	b, err := os.ReadFile("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/Resources/version.plist")
	if err != nil {
		return nil
	}
	if v, err := parsePlist(bytes.NewReader(b)); err == nil {
		if m, ok := v.(map[string]any); ok {
			di.Framework = map[string]string{}
			for k, x := range m {
				if s, ok := x.(string); ok {
					di.Framework[k] = s
				}
			}
		}
	}
	return nil
}

// isANEService tells the system's ANE processes by their executable.
func isANEService(path string) bool {
	return strings.Contains(path, "/AppleNeuralEngine.framework/") ||
		strings.HasPrefix(path, "/usr/libexec/ane")
}

// darwinServices finds the system's ANE processes. proc_pidpath works for every user's process.
func darwinServices() ([]DarwinANEService, error) {
	p, err := loadProc()
	if err != nil {
		return nil, err
	}
	pids, err := p.pids()
	if err != nil {
		return nil, err
	}
	var out []DarwinANEService
	for _, pid := range pids {
		if pid <= 0 {
			continue
		}
		path := p.path(pid)
		if !isANEService(path) {
			continue
		}
		out = append(out, DarwinANEService{PID: int(pid), Name: filepath.Base(path), Path: path})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].PID < out[j].PID })
	return out, nil
}
