// Package xdnasim simulates the amdxdna kernel driver behind xdna.Driver, for tests on machines
// without a Ryzen AI NPU. It enforces the rules of drivers/accel/amdxdna that hwnpu relies on:
//
//   - a device heap (BO_DEV_HEAP, a multiple of 64 MiB) must exist and be mapped before
//     BO_DEV objects can be created (EAGAIN otherwise); BO_DEV objects are not mappable
//   - CONFIG_CU takes BO_DEV handles holding PDIs; a context is configured once
//   - EXEC_CMD takes one CMD buffer object whose handle is passed by value, holding an
//     ERT_START_CU packet; the CU index comes from the CU mask
//   - completion is signalled on the context's timeline syncobj at the returned sequence number
//
// The compute units themselves are simulated by the Exec callback.
package xdnasim

import (
	"encoding/binary"
	"fmt"
	"sync"
	"syscall"
	"time"
	"unsafe"

	"github.com/shibukawa/hwnpu/internal/ert"
	"github.com/shibukawa/hwnpu/internal/xdna"
)

// Exec is the simulated work of one command. Mem resolves a device address range to memory and
// fails if no buffer object listed in the command's argument handles covers it.
type Exec struct {
	CU     int
	PDI    []byte
	Func   uint8
	Regmap []byte
	Mem    func(addr uint64, n int) ([]byte, error)
}

// Sim is a simulated device node.
type Sim struct {
	Meta    xdna.AIEMetadata
	FW      xdna.FirmwareVersion
	PowerMW uint32
	// Run simulates a compute unit. A returned error marks the command ERT_CMD_STATE_ERROR.
	Run func(e Exec) error
	// Delay postpones completion, to exercise waiting.
	Delay time.Duration

	mu        sync.Mutex
	cond      *sync.Cond
	next      uint32
	bos       map[uint32]*bo
	heap      *bo
	heapUsed  int
	ctxs      map[uint32]*hwctx
	timelines map[uint32]map[uint64]bool
	// Observations for tests.
	LastQoS    xdna.QoSInfo
	LastArgs   []uint32
	Syncs      int
	ClosedBOs  int
	LiveHwctxs int
}

type bo struct {
	handle    uint32
	typ       uint32
	mem       []byte
	mapOffset uint64
	devAddr   uint64
	mapped    bool
}

type hwctx struct {
	syncobj uint32
	seq     uint64
	cus     []cu
}

type cu struct {
	pdi []byte
	fn  uint8
}

// New returns a simulated NPU with 4 columns and 4 core rows (an npu1 / Phoenix like device).
func New() *Sim {
	s := &Sim{
		Meta: xdna.AIEMetadata{ColSize: 0x2000000, Cols: 4, Rows: 6, VersionMajor: 1, VersionMinor: 1,
			Core: xdna.TileMetadata{RowCount: 4, RowStart: 2}, Mem: xdna.TileMetadata{RowCount: 1, RowStart: 1}, Shim: xdna.TileMetadata{RowCount: 1}},
		FW:      xdna.FirmwareVersion{Major: 1, Minor: 5, Patch: 4, Build: 361},
		PowerMW: 1500,
		bos:     map[uint32]*bo{}, ctxs: map[uint32]*hwctx{}, timelines: map[uint32]map[uint64]bool{},
	}
	s.cond = sync.NewCond(&s.mu)
	return s
}

func (s *Sim) handle() uint32 { s.next++; return s.next }

// Ioctl implements xdna.Driver.
func (s *Sim) Ioctl(req uintptr, arg unsafe.Pointer) error {
	switch req {
	case xdna.IoctlSyncobjWait:
		return s.wait((*xdna.SyncobjTimelineWait)(arg))
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	switch req {
	case xdna.IoctlCreateBO:
		return s.createBO((*xdna.CreateBO)(arg))
	case xdna.IoctlGetBOInfo:
		a := (*xdna.GetBOInfo)(arg)
		if a.Ext != 0 || a.ExtFlags != 0 || a.Pad != 0 {
			return syscall.EINVAL
		}
		b, ok := s.bos[a.Handle]
		if !ok {
			return syscall.ENOENT
		}
		a.MapOffset, a.XdnaAddr, a.VAddr = b.mapOffset, b.devAddr, xdna.InvalidAddr
		if b.mapped {
			a.VAddr = uint64(uintptr(unsafe.Pointer(&b.mem[0])))
		}
		return nil
	case xdna.IoctlSyncBO:
		a := (*xdna.SyncBO)(arg)
		b, ok := s.bos[a.Handle]
		if !ok {
			return syscall.ENOENT
		}
		if a.Offset >= uint64(len(b.mem)) || a.Direction > xdna.SyncFromDevice {
			return syscall.EINVAL
		}
		s.Syncs++
		return nil
	case xdna.IoctlGemClose:
		a := (*xdna.GemClose)(arg)
		if _, ok := s.bos[a.Handle]; !ok {
			return syscall.EINVAL
		}
		delete(s.bos, a.Handle)
		s.ClosedBOs++
		return nil
	case xdna.IoctlCreateHwctx:
		a := (*xdna.CreateHwctx)(arg)
		if a.Ext != 0 || a.ExtFlags != 0 || a.QoS == nil {
			return syscall.EINVAL
		}
		s.LastQoS = *(*xdna.QoSInfo)(a.QoS)
		rows := uint32(s.Meta.Core.RowCount)
		if a.NumTiles == 0 || a.NumTiles%rows != 0 || a.NumTiles/rows > uint32(s.Meta.Cols) {
			return syscall.EINVAL
		}
		h := s.handle()
		so := s.handle()
		s.ctxs[h] = &hwctx{syncobj: so}
		s.timelines[so] = map[uint64]bool{}
		a.Handle, a.SyncobjHandle = h, so
		s.LiveHwctxs++
		return nil
	case xdna.IoctlDestroyHwctx:
		a := (*xdna.DestroyHwctx)(arg)
		if _, ok := s.ctxs[a.Handle]; !ok {
			return syscall.EINVAL
		}
		delete(s.ctxs, a.Handle)
		s.LiveHwctxs--
		return nil
	case xdna.IoctlConfigHwctx:
		return s.configCU((*xdna.ConfigHwctx)(arg))
	case xdna.IoctlExecCmd:
		return s.exec((*xdna.ExecCmd)(arg))
	case xdna.IoctlGetInfo:
		return s.getInfo((*xdna.GetInfo)(arg))
	}
	return syscall.ENOTTY
}

func (s *Sim) createBO(a *xdna.CreateBO) error {
	if a.Flags != 0 || a.Size == 0 {
		return syscall.EINVAL
	}
	size := int((a.Size + 4095) &^ 4095)
	b := &bo{typ: a.Type, mapOffset: xdna.InvalidAddr}
	switch a.Type {
	case xdna.BOShare, xdna.BOCmd:
		b.mem = make([]byte, size)
	case xdna.BODevHeap:
		if a.Size%xdna.DevHeapSize != 0 {
			return syscall.EINVAL
		}
		if s.heap != nil {
			return syscall.EOPNOTSUPP // npu1 cannot expand the heap
		}
		b.mem = make([]byte, size)
		b.devAddr = xdna.DevMemBase
	case xdna.BODev:
		if s.heap == nil || !s.heap.mapped {
			return syscall.EAGAIN
		}
		if s.heapUsed+size > len(s.heap.mem) {
			return syscall.ENOMEM
		}
		b.mem = s.heap.mem[s.heapUsed : s.heapUsed+size : s.heapUsed+size]
		b.devAddr = xdna.DevMemBase + uint64(s.heapUsed)
		s.heapUsed += size
	default:
		return syscall.EINVAL
	}
	b.handle = s.handle()
	if a.Type != xdna.BODev {
		b.mapOffset = uint64(b.handle) << 32
		if a.Type != xdna.BODevHeap {
			b.devAddr = 0x1_0000_0000_0 + uint64(b.handle)<<28 // a DMA address
		}
	}
	if a.Type == xdna.BODevHeap {
		s.heap = b
	}
	s.bos[b.handle] = b
	a.Handle = b.handle
	return nil
}

// Mmap implements xdna.Driver.
func (s *Sim) Mmap(offset uint64, length int) ([]byte, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, b := range s.bos {
		if b.mapOffset == offset && b.typ != xdna.BODev {
			if length > len(b.mem) {
				return nil, syscall.EINVAL
			}
			b.mapped = true
			return b.mem[:length:length], nil
		}
	}
	return nil, syscall.EINVAL
}

// Munmap implements xdna.Driver.
func (s *Sim) Munmap(b []byte) error { return nil }

// Close implements xdna.Driver.
func (s *Sim) Close() error { return nil }

func (s *Sim) configCU(a *xdna.ConfigHwctx) error {
	if a.Pad != 0 {
		return syscall.EINVAL
	}
	c, ok := s.ctxs[a.Handle]
	if !ok || a.ParamType != xdna.ConfigCU || a.ParamVal == nil {
		return syscall.EINVAL
	}
	if c.cus != nil {
		return syscall.EINVAL // re-config is not supported
	}
	buf := unsafe.Slice((*byte)(a.ParamVal), a.ParamValSize)
	hdr := (*xdna.ConfigCUHeader)(unsafe.Pointer(&buf[0]))
	n := int(hdr.NumCUs)
	need := int(unsafe.Sizeof(xdna.ConfigCUHeader{})) + n*int(unsafe.Sizeof(xdna.CUConfig{}))
	if n == 0 || need > len(buf) {
		return syscall.EINVAL
	}
	var cus []cu
	for i := 0; i < n; i++ {
		cfg := (*xdna.CUConfig)(unsafe.Pointer(&buf[int(unsafe.Sizeof(xdna.ConfigCUHeader{}))+i*int(unsafe.Sizeof(xdna.CUConfig{}))]))
		b, ok := s.bos[cfg.CUBO]
		if !ok || b.typ != xdna.BODev {
			return syscall.EINVAL // the PDI must be in a device buffer object
		}
		cus = append(cus, cu{pdi: append([]byte(nil), b.mem...), fn: cfg.CUFunc})
	}
	c.cus = cus
	return nil
}

func (s *Sim) exec(a *xdna.ExecCmd) error {
	if a.Ext != 0 || a.ExtFlags != 0 || a.Type != xdna.CmdSubmitExecBuf || a.CmdCount != 1 {
		return syscall.EINVAL
	}
	c, ok := s.ctxs[a.Hwctx]
	if !ok {
		return syscall.EINVAL
	}
	cmd, ok := s.bos[uint32(a.CmdHandles)]
	if !ok || cmd.typ != xdna.BOCmd {
		return syscall.EINVAL
	}
	var args []uint32
	if a.ArgCount > 0 {
		args = append(args, unsafe.Slice((*uint32)(a.Args), a.ArgCount)...)
	}
	for _, h := range args {
		if _, ok := s.bos[h]; !ok {
			return syscall.ENOENT
		}
	}
	s.LastArgs = args
	le := binary.LittleEndian
	header := le.Uint32(cmd.mem)
	count := int(header>>12) & 0x7ff
	if op := (header >> 23) & 0x1f; op != ert.OpStartCU {
		return syscall.EOPNOTSUPP
	}
	extra := int(header>>10) & 0x3
	if count <= 1+extra || 4+4*count > len(cmd.mem) {
		return syscall.EINVAL
	}
	mask := le.Uint32(cmd.mem[4:])
	cuIdx := -1
	for i := 0; i < 32; i++ {
		if mask&(1<<i) != 0 {
			cuIdx = i
			break
		}
	}
	if c.cus == nil || cuIdx < 0 || cuIdx >= len(c.cus) {
		return syscall.EINVAL
	}
	regmap := append([]byte(nil), cmd.mem[4+4*(1+extra):4+4*count]...)
	seq := c.seq
	c.seq++
	a.Seq = seq
	le.PutUint32(cmd.mem, header&^0xf|uint32(ert.StateQueued))

	e := Exec{CU: cuIdx, PDI: c.cus[cuIdx].pdi, Func: c.cus[cuIdx].fn, Regmap: regmap, Mem: s.resolver(args)}
	complete := func() {
		state := ert.StateCompleted
		if s.Run != nil {
			if err := s.Run(e); err != nil {
				state = ert.StateError
			}
		}
		s.mu.Lock()
		le.PutUint32(cmd.mem, le.Uint32(cmd.mem)&^0xf|uint32(state))
		s.timelines[c.syncobj][seq] = true
		s.cond.Broadcast()
		s.mu.Unlock()
	}
	if s.Delay > 0 {
		go func() { time.Sleep(s.Delay); complete() }()
	} else {
		s.mu.Unlock()
		complete()
		s.mu.Lock()
	}
	return nil
}

// resolver maps device addresses to buffer objects listed in args.
func (s *Sim) resolver(args []uint32) func(addr uint64, n int) ([]byte, error) {
	return func(addr uint64, n int) ([]byte, error) {
		s.mu.Lock()
		defer s.mu.Unlock()
		for _, h := range args {
			b, ok := s.bos[h]
			if !ok {
				continue
			}
			if addr >= b.devAddr && addr+uint64(n) <= b.devAddr+uint64(len(b.mem)) {
				off := addr - b.devAddr
				return b.mem[off : off+uint64(n)], nil
			}
		}
		return nil, fmt.Errorf("xdnasim: address %#x (+%d) is not inside a buffer object passed with the command", addr, n)
	}
}

func (s *Sim) wait(a *xdna.SyncobjTimelineWait) error {
	if a.CountHandles != 1 || a.Handles == nil || a.Points == nil {
		return syscall.EINVAL
	}
	h, point := *(*uint32)(a.Handles), *(*uint64)(a.Points)
	s.mu.Lock()
	defer s.mu.Unlock()
	tl, ok := s.timelines[h]
	if !ok {
		return syscall.EINVAL
	}
	for !tl[point] {
		now := xdna.MonotonicNow()
		if now >= a.TimeoutNsec {
			return syscall.ETIME
		}
		// Wake up periodically to re-check the deadline.
		timer := time.AfterFunc(time.Duration(min(a.TimeoutNsec-now, int64(time.Millisecond))), func() {
			s.mu.Lock()
			s.cond.Broadcast()
			s.mu.Unlock()
		})
		s.cond.Wait()
		timer.Stop()
	}
	return nil
}

func (s *Sim) getInfo(a *xdna.GetInfo) error {
	if a.Buffer == nil {
		return syscall.EINVAL
	}
	put := func(v unsafe.Pointer, size uintptr) error {
		if uintptr(a.BufferSize) < size {
			return syscall.EINVAL
		}
		copy(unsafe.Slice((*byte)(a.Buffer), size), unsafe.Slice((*byte)(v), size))
		return nil
	}
	switch a.Param {
	case xdna.QueryAIEMetadata:
		return put(unsafe.Pointer(&s.Meta), unsafe.Sizeof(s.Meta))
	case xdna.QueryFirmwareVersion:
		return put(unsafe.Pointer(&s.FW), unsafe.Sizeof(s.FW))
	case xdna.QuerySensors:
		sensors := []xdna.Sensor{{Input: s.PowerMW, UnitM: -3, Type: xdna.SensorTypePower}, {Input: 37, Type: 1}}
		copy(sensors[0].Label[:], "Total Power")
		copy(sensors[0].Units[:], "mW")
		size := unsafe.Sizeof(xdna.Sensor{})
		n := min(int(uintptr(a.BufferSize)/size), len(sensors))
		copy(unsafe.Slice((*xdna.Sensor)(a.Buffer), n), sensors[:n])
		a.BufferSize = uint32(uintptr(n) * size)
		return nil
	}
	return syscall.EINVAL
}

// HeapUsed reports how many heap bytes BO_DEV objects use.
func (s *Sim) HeapUsed() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.heapUsed
}

// LiveBOs reports the number of open buffer object handles.
func (s *Sim) LiveBOs() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.bos)
}
