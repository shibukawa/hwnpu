//go:build linux

package npuinfo

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"runtime"
	"syscall"
	"unsafe"
)

// amdxdna's ioctls (include/uapi/drm/amdxdna_accel.h).
const (
	xdnaCmdGetInfo  = 7
	xdnaCmdGetArray = 10

	xdnaQueryAIEStatus       = 0
	xdnaQueryAIEMetadata     = 1
	xdnaQueryAIEVersion      = 2
	xdnaQueryClockMetadata   = 3
	xdnaQuerySensors         = 4
	xdnaQueryHWContexts      = 5
	xdnaQueryFirmwareVersion = 8
	xdnaGetPowerMode         = 9
	xdnaQueryTelemetry       = 10
	xdnaQueryResourceInfo    = 12

	xdnaArrayHWContextAll = 0
	xdnaArrayLastAsyncErr = 2
	xdnaMaxContexts       = 256
)

// drmIOWR is _IOWR('d', DRM_COMMAND_BASE + nr, size).
func drmIOWR(nr, size uintptr) uintptr {
	const commandBase = 0x40
	return 3<<30 | size<<16 | uintptr('d')<<8 | (commandBase + nr)
}

type xdnaDev struct {
	fd     int
	pinner runtime.Pinner
}

func (d *xdnaDev) ioctl(req uintptr, arg []byte) error {
	d.pinner.Pin(&arg[0])
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(d.fd), req, uintptr(unsafe.Pointer(&arg[0])))
	if errno != 0 {
		return errno
	}
	return nil
}

func (d *xdnaDev) addr(b []byte) uint64 {
	d.pinner.Pin(&b[0])
	return uint64(uintptr(unsafe.Pointer(&b[0])))
}

// getInfo runs DRM_IOCTL_AMDXDNA_GET_INFO with a buffer of size bytes and returns the bytes the
// kernel says it wrote.
func (d *xdnaDev) getInfo(param uint32, size int) ([]byte, error) {
	return d.getInfoBuf(param, make([]byte, size))
}

func (d *xdnaDev) getInfoBuf(param uint32, buf []byte) ([]byte, error) {
	arg := make([]byte, 16)
	binary.LittleEndian.PutUint32(arg[0:], param)
	binary.LittleEndian.PutUint32(arg[4:], uint32(len(buf)))
	binary.LittleEndian.PutUint64(arg[8:], d.addr(buf))
	if err := d.ioctl(drmIOWR(xdnaCmdGetInfo, 16), arg); err != nil {
		return nil, err
	}
	if n := int(binary.LittleEndian.Uint32(arg[4:])); n > 0 && n < len(buf) {
		return buf[:n], nil
	}
	return buf, nil
}

// getArray runs DRM_IOCTL_AMDXDNA_GET_ARRAY and returns the element size and the elements.
func (d *xdnaDev) getArray(param uint32, elemSize, maxElems int) (int, [][]byte, error) {
	buf := make([]byte, elemSize*maxElems)
	arg := make([]byte, 24)
	binary.LittleEndian.PutUint32(arg[0:], param)
	binary.LittleEndian.PutUint32(arg[4:], uint32(elemSize))
	binary.LittleEndian.PutUint32(arg[8:], uint32(maxElems))
	binary.LittleEndian.PutUint64(arg[16:], d.addr(buf))
	if err := d.ioctl(drmIOWR(xdnaCmdGetArray, 24), arg); err != nil {
		return 0, nil, err
	}
	es := int(binary.LittleEndian.Uint32(arg[4:]))
	n := int(binary.LittleEndian.Uint32(arg[8:]))
	if es <= 0 || es > elemSize || n > maxElems {
		return es, nil, fmt.Errorf("get_array returned element size %d, count %d", es, n)
	}
	out := make([][]byte, n)
	for i := range out {
		out[i] = buf[i*es : (i+1)*es]
	}
	return es, out, nil
}

func queryXDNA(devPath, runtimeStatus string, opts Options) *XDNAInfo {
	x := &XDNAInfo{Contexts: []XDNAContext{}, Errors: map[string]string{}}
	if runtimeStatus != "active" && !opts.Wake {
		x.Skipped = "runtime_status is " + runtimeStatus + "; the queries would wake the NPU (Options.Wake)"
		x.Errors = nil
		return x
	}
	fd, err := syscall.Open(devPath, syscall.O_RDWR|syscall.O_CLOEXEC, 0)
	if err != nil {
		x.Errors["open"] = err.Error()
		return x
	}
	d := &xdnaDev{fd: fd}
	defer func() {
		d.pinner.Unpin()
		syscall.Close(fd)
	}()
	fail := func(name string, err error) { x.Errors[name] = err.Error() }

	if b, err := d.getInfo(xdnaQueryAIEVersion, 8); err != nil {
		fail("aie_version", err)
	} else {
		x.AIEVersion = &XDNAVersion{Major: binary.LittleEndian.Uint32(b), Minor: binary.LittleEndian.Uint32(b[4:])}
	}
	if b, err := d.getInfo(xdnaQueryAIEMetadata, 64); err != nil {
		fail("aie_metadata", err)
	} else if m, err := decodeXDNAMetadata(b); err != nil {
		fail("aie_metadata", err)
	} else {
		x.AIEMetadata = &m
	}
	if b, err := d.getInfo(xdnaQueryFirmwareVersion, 16); err != nil {
		fail("firmware_version", err)
	} else {
		le := binary.LittleEndian
		x.Firmware = &XDNAFirmware{Major: le.Uint32(b), Minor: le.Uint32(b[4:]), Patch: le.Uint32(b[8:]), Build: le.Uint32(b[12:])}
	}
	if b, err := d.getInfo(xdnaQueryClockMetadata, 48); err != nil {
		fail("clock_metadata", err)
	} else {
		for i := 0; i+24 <= len(b); i += 24 {
			name := b[i : i+16]
			if j := bytes.IndexByte(name, 0); j >= 0 {
				name = name[:j]
			}
			x.Clocks = append(x.Clocks, XDNAClock{Name: string(name), MHz: binary.LittleEndian.Uint32(b[i+16:])})
		}
	}
	if b, err := d.getInfo(xdnaGetPowerMode, 8); err != nil {
		fail("power_mode", err)
	} else {
		m := int(b[0])
		x.PowerMode, x.PowerModeName = &m, xdnaPowerModeName(m)
	}
	if b, err := d.getInfo(xdnaQueryResourceInfo, 40); err != nil {
		fail("resource_info", err)
	} else {
		le := binary.LittleEndian
		x.Resource = &XDNAResourceInfo{ClockMaxMHz: le.Uint64(b), TOPSMax: le.Uint64(b[8:]), TasksMax: le.Uint64(b[16:]),
			TOPSCurrent: le.Uint64(b[24:]), TasksCurrent: le.Uint64(b[32:])}
	}
	if b, err := d.getInfo(xdnaQuerySensors, xdnaSensorSize*16); err != nil {
		fail("sensors", err)
	} else {
		for i := 0; i+xdnaSensorSize <= len(b); i += xdnaSensorSize {
			s, err := decodeXDNASensor(b[i : i+xdnaSensorSize])
			if err != nil || s.Label == "" {
				break
			}
			x.Sensors = append(x.Sensors, s)
		}
	}

	if _, elems, err := d.getArray(xdnaArrayHWContextAll, 4096, xdnaMaxContexts); err != nil {
		fail("hw_context_all", err)
		// The older query, from drivers before GET_ARRAY.
		const sz = 64
		if b, err := d.getInfo(xdnaQueryHWContexts, sz*xdnaMaxContexts); err != nil {
			fail("hw_contexts", err)
		} else {
			for i := 0; i+sz <= len(b); i += sz {
				c, err := decodeXDNAQueryHWCtx(b[i : i+sz])
				if err != nil || c.PID == 0 {
					continue
				}
				x.Contexts = append(x.Contexts, c)
			}
		}
	} else {
		for _, e := range elems {
			c, err := decodeXDNAContext(e)
			if err != nil {
				fail("hw_context_all", err)
				break
			}
			x.Contexts = append(x.Contexts, c)
		}
	}
	if _, elems, err := d.getArray(xdnaArrayLastAsyncErr, 24, 1); err != nil {
		fail("last_async_error", err)
	} else if len(elems) == 1 {
		le := binary.LittleEndian
		e := elems[0]
		x.LastAsyncError = &XDNAAsyncError{ErrCode: le.Uint64(e), TimeUS: le.Uint64(e[8:]), ExErrCode: le.Uint64(e[16:])}
	}

	if x.AIEMetadata != nil && x.AIEMetadata.Cols > 0 && x.AIEMetadata.ColSize > 0 {
		colSize := int(x.AIEMetadata.ColSize)
		out := make([]byte, colSize*int(x.AIEMetadata.Cols))
		st := make([]byte, 16)
		binary.LittleEndian.PutUint64(st, d.addr(out))
		binary.LittleEndian.PutUint32(st[8:], uint32(len(out)))
		if _, err := d.getInfoBuf(xdnaQueryAIEStatus, st); err != nil {
			fail("aie_status", err)
		} else {
			x.AIEStatus = &XDNAAIEStatus{ColsFilled: binary.LittleEndian.Uint32(st[12:]), ColSize: colSize, Raw: out}
		}
	}

	// Telemetry: a header (major, minor, type, map_num_elements, map[]) and the data after it.
	buf := make([]byte, 64*1024)
	if b, err := d.getInfoBuf(xdnaQueryTelemetry, buf); err != nil {
		fail("telemetry", err)
	} else if len(b) >= 16 {
		le := binary.LittleEndian
		t := &XDNATelemetry{Major: le.Uint32(b), Minor: le.Uint32(b[4:]), Type: le.Uint32(b[8:])}
		n := int(le.Uint32(b[12:]))
		off := 16
		for i := 0; i < n && off+4 <= len(b); i++ {
			t.Map = append(t.Map, le.Uint32(b[off:]))
			off += 4
		}
		t.Data = bytes.TrimRight(b[off:], "\x00")
		x.Telemetry = t
	}
	if len(x.Errors) == 0 {
		x.Errors = nil
	}
	return x
}
