package amdxrt

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"unsafe"

	"github.com/shibukawa/hwnpu/internal/dl"
	"github.com/shibukawa/hwnpu/internal/types"
)

// ShimEnv names the shim library to load, before the default names.
const ShimEnv = "HWNPU_XRTSHIM"

// XRTEnv names XRT's xrt_coreutil.dll on Windows, before the places AMD's driver installs it.
const XRTEnv = "HWNPU_XRT_CORE"

// shimABI is the xrtshim ABI this package binds (XS_ABI in native/xrtshim/xrtshim.h).
const shimABI = 1

// Buffer kinds, sync directions, info keys and command states of xrtshim.h.
const (
	boHost      = 0
	boCacheable = 1

	syncToDevice   = 0
	syncFromDevice = 1

	infoName       = 0
	infoBDF        = 1
	infoPlatform   = 2
	infoElectrical = 3
	infoXRTVersion = 4

	stateCompleted = 4
	stateSubmitted = 7
	stateTimeout   = 8
)

// shim is the bound xrtshim library. Functions returning unsafe.Pointer return an error
// message (nil for success), freed with freeError.
type shim struct {
	lib *dl.Library

	abi        func() int32
	freeError  func(unsafe.Pointer)
	devCount   func(count *uint32) unsafe.Pointer
	devOpen    func(index uint32, out *uintptr) unsafe.Pointer
	devFree    func(d uintptr)
	devInfo    func(d uintptr, key int32, buf *byte, n uintptr, need *uintptr) unsafe.Pointer
	boAlloc    func(d uintptr, size uintptr, kind, group int32, out *uintptr) unsafe.Pointer
	boFree     func(b uintptr)
	boSub      func(parent uintptr, n, off uintptr, out *uintptr) unsafe.Pointer
	boMap      func(b uintptr, ptr *unsafe.Pointer) unsafe.Pointer
	boAddress  func(b uintptr) uint64
	boSync     func(b uintptr, dir int32, off, n uintptr) unsafe.Pointer
	ctxXclbin  func(d uintptr, data *byte, n uintptr, out *uintptr) unsafe.Pointer
	ctxELF     func(d uintptr, data *byte, n uintptr, out *uintptr) unsafe.Pointer
	ctxFree    func(c uintptr)
	kernelOpen func(c uintptr, name string, out *uintptr) unsafe.Pointer
	kernelFree func(k uintptr)
	kernelGrp  func(k uintptr, arg int32) int32
	runNew     func(k uintptr, out *uintptr) unsafe.Pointer
	runFree    func(r uintptr)
	runSetBO   func(r uintptr, index int32, b uintptr) unsafe.Pointer
	runSetVal  func(r uintptr, index int32, value *byte, n uintptr) unsafe.Pointer
	runScratch func(r uintptr, ptr *unsafe.Pointer, n *uintptr) unsafe.Pointer
	runStart   func(r uintptr) unsafe.Pointer
	runWait    func(r uintptr, timeoutMs uint32, state *int32) unsafe.Pointer
	runState   func(r uintptr) int32
}

// shimCandidates are the names tried after ShimEnv: next to the executable, then the loader's
// search path.
func shimCandidates() []string {
	name := "libxrtshim.so"
	if runtime.GOOS == "windows" {
		name = "xrtshim.dll"
	}
	var c []string
	if exe, err := os.Executable(); err == nil {
		c = append(c, filepath.Join(filepath.Dir(exe), name))
	}
	return append(c, name)
}

var (
	shimOnce sync.Once
	theShim  *shim
	shimErr  error
)

// loadShim loads and binds the shim once.
func loadShim() (*shim, error) {
	shimOnce.Do(func() {
		preloadXRT()
		theShim, shimErr = bindShim(append([]string{os.Getenv(ShimEnv)}, shimCandidates()...)...)
	})
	return theShim, shimErr
}

func bindShim(candidates ...string) (*shim, error) {
	lib, err := dl.Open(candidates...)
	if err != nil {
		if errors.Is(err, dl.ErrNotFound) {
			err = fmt.Errorf("%w: xrtshim (%v); build native/xrtshim (INSTALL_AMD.md) and put it next to the executable, or set %s", types.ErrLibrary, err, ShimEnv)
		}
		return nil, err
	}
	s := &shim{lib: lib}
	binds := []struct {
		f    any
		name string
	}{
		{&s.abi, "xs_abi"}, {&s.freeError, "xs_free_error"},
		{&s.devCount, "xs_device_count"}, {&s.devOpen, "xs_device_open"}, {&s.devFree, "xs_device_free"}, {&s.devInfo, "xs_device_info"},
		{&s.boAlloc, "xs_bo_alloc"}, {&s.boFree, "xs_bo_free"}, {&s.boSub, "xs_bo_sub"}, {&s.boMap, "xs_bo_map"}, {&s.boAddress, "xs_bo_address"}, {&s.boSync, "xs_bo_sync"},
		{&s.ctxXclbin, "xs_ctx_xclbin"}, {&s.ctxELF, "xs_ctx_elf"}, {&s.ctxFree, "xs_ctx_free"},
		{&s.kernelOpen, "xs_kernel_open"}, {&s.kernelFree, "xs_kernel_free"}, {&s.kernelGrp, "xs_kernel_group_id"},
		{&s.runNew, "xs_run_new"}, {&s.runFree, "xs_run_free"}, {&s.runSetBO, "xs_run_set_bo"}, {&s.runSetVal, "xs_run_set_scalar"},
		{&s.runScratch, "xs_run_scratchpad"}, {&s.runStart, "xs_run_start"}, {&s.runWait, "xs_run_wait"}, {&s.runState, "xs_run_state"},
	}
	for _, b := range binds {
		if err := lib.Bind(b.f, b.name); err != nil {
			return nil, fmt.Errorf("amdxrt: %s: %w", lib.Path, err)
		}
	}
	if v := s.abi(); v != shimABI {
		return nil, fmt.Errorf("amdxrt: %s has ABI %d, this hwnpu needs %d: rebuild native/xrtshim", lib.Path, v, shimABI)
	}
	return s, nil
}

// err turns a returned message into an error and frees it.
func (s *shim) err(p unsafe.Pointer) error {
	if p == nil {
		return nil
	}
	msg := cString(p)
	s.freeError(p)
	return errors.New("amdxrt: " + msg)
}

func cString(p unsafe.Pointer) string {
	n := 0
	for *(*byte)(unsafe.Add(p, n)) != 0 {
		n++
	}
	return string(unsafe.Slice((*byte)(p), n))
}

// info reads a device's info value.
func (s *shim) info(d uintptr, key int32) (string, error) {
	buf := make([]byte, 256)
	for {
		var need uintptr
		if err := s.err(s.devInfo(d, key, &buf[0], uintptr(len(buf)), &need)); err != nil {
			return "", err
		}
		if int(need) <= len(buf) {
			return string(buf[:need-1]), nil
		}
		buf = make([]byte, need)
	}
}

func bytePtr(b []byte) *byte {
	if len(b) == 0 {
		return nil
	}
	return &b[0]
}

// watts reads the power XRT reports in the device's electrical info (JSON): the first number
// under a key naming power in watts.
func (s *shim) watts(d uintptr) (float64, error) {
	js, err := s.info(d, infoElectrical)
	if err != nil {
		return 0, err
	}
	var v any
	if err := json.Unmarshal([]byte(js), &v); err != nil {
		return 0, fmt.Errorf("amdxrt: electrical info: %w", err)
	}
	if w, ok := findWatts(v); ok {
		return w, nil
	}
	return 0, fmt.Errorf("amdxrt: no power in the electrical info: %w", types.ErrUnsupported)
}

// findWatts looks for power_consumption_watts (XRT's key), or another key with power and
// watts in its name (not a maximum or a minimum), holding a number or a numeric string.
func findWatts(v any) (float64, bool) {
	switch t := v.(type) {
	case map[string]any:
		keys := make([]string, 0, len(t))
		for k := range t {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		if w, ok := number(t["power_consumption_watts"]); ok {
			return w, true
		}
		for _, k := range keys {
			lk := strings.ToLower(k)
			if strings.Contains(lk, "power") && strings.Contains(lk, "watts") && !strings.Contains(lk, "max") && !strings.Contains(lk, "min") {
				if w, ok := number(t[k]); ok {
					return w, true
				}
			}
		}
		for _, k := range keys {
			if w, ok := findWatts(t[k]); ok {
				return w, true
			}
		}
	case []any:
		for _, e := range t {
			if w, ok := findWatts(e); ok {
				return w, true
			}
		}
	}
	return 0, false
}

func number(v any) (float64, bool) {
	switch t := v.(type) {
	case float64:
		return t, true
	case string:
		w, err := strconv.ParseFloat(strings.TrimSpace(t), 64)
		return w, err == nil
	}
	return 0, false
}
