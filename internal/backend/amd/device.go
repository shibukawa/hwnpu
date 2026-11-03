package amd

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu/internal/aieelf"
	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/ert"
	"github.com/shibukawa/hwnpu/internal/types"
	"github.com/shibukawa/hwnpu/internal/xclbin"
	"github.com/shibukawa/hwnpu/internal/xdna"
)

const cmdBOSize = 4096 // XRT's kernel_command::allocation_size

type device struct {
	dev     xdna.Dev
	meta    xdna.AIEMetadata
	opts    backend.Options
	heap    xdna.BO
	heapMem []byte

	powerMu sync.Mutex
	energyJ float64
	lastW   float64
	lastT   time.Time
}

func (d *device) Close() error {
	var errs []error
	if d.heapMem != nil {
		errs = append(errs, d.dev.D.Munmap(d.heapMem))
		errs = append(errs, d.dev.CloseBO(d.heap.Handle))
		d.heapMem = nil
	}
	errs = append(errs, d.dev.D.Close())
	return errors.Join(errs...)
}

func (d *device) Compile(ctx context.Context, src types.GraphSource) (backend.Graph, error) {
	return nil, fmt.Errorf("amd: graphs are not supported; load an xclbin with KernelRunner: %w", types.ErrUnsupported)
}

// buffer is a BO_SHARE object mapped into the process, or a BO_DEV object inside the heap.
type buffer struct {
	d      *device
	bo     xdna.BO
	mem    []byte // the caller's view
	mapped []byte // the whole mapping to unmap (BO_SHARE / BO_CMD only)
}

func (d *device) AllocBuffer(size int, kind types.BufferKind) (backend.Buffer, error) {
	var b *buffer
	var err error
	if kind == types.BufferInstructions {
		b, err = d.allocDev(size)
	} else {
		b, err = d.allocMapped(xdna.BOShare, size)
	}
	if err != nil {
		return nil, err // not a typed nil
	}
	return b, nil
}

func (d *device) allocMapped(typ uint32, size int) (*buffer, error) {
	if size <= 0 {
		return nil, errors.New("amd: buffer size must be positive")
	}
	bo, err := d.dev.CreateBO(typ, size)
	if err != nil {
		return nil, err
	}
	mapped, err := d.dev.D.Mmap(bo.MapOffset, roundPage(size))
	if err != nil {
		d.dev.CloseBO(bo.Handle)
		return nil, fmt.Errorf("amd: map buffer: %w", err)
	}
	// Start with no dirty cache lines. A new page is zeroed when it is first touched, through
	// the CPU's caches; a line still dirty from that is written back over what the NPU wrote
	// there, by the next flush or eviction. A new output buffer read back as zeros after the
	// first command on a Ryzen AI 9 365, and as the NPU's results from then on. So touch every
	// page now and flush it all once; afterwards only lines the caller writes are dirty, and
	// SyncToDevice cleans them before the NPU runs.
	clear(mapped)
	if !xdna.FlushCPU(mapped) {
		if err := d.dev.Sync(bo.Handle, xdna.SyncToDevice, 0, len(mapped)); err != nil {
			d.dev.D.Munmap(mapped)
			d.dev.CloseBO(bo.Handle)
			return nil, err
		}
	}
	// With PASID the device address of a host buffer is its mapped user address.
	if err := d.dev.Info(&bo); err != nil {
		d.dev.D.Munmap(mapped)
		d.dev.CloseBO(bo.Handle)
		return nil, err
	}
	if bo.DeviceAddr() == xdna.InvalidAddr {
		d.dev.D.Munmap(mapped)
		d.dev.CloseBO(bo.Handle)
		return nil, errors.New("amd: the driver reports no device address for a mapped buffer")
	}
	return &buffer{d: d, bo: bo, mem: mapped[:size:size], mapped: mapped}, nil
}

func (d *device) allocDev(size int) (*buffer, error) {
	if size <= 0 {
		return nil, errors.New("amd: buffer size must be positive")
	}
	bo, err := d.dev.CreateBO(xdna.BODev, size)
	if err != nil {
		return nil, err
	}
	off := bo.XdnaAddr - d.heap.XdnaAddr
	if bo.XdnaAddr < d.heap.XdnaAddr || off+uint64(size) > uint64(len(d.heapMem)) {
		d.dev.CloseBO(bo.Handle)
		return nil, fmt.Errorf("amd: device buffer at %#x is outside the heap at %#x", bo.XdnaAddr, d.heap.XdnaAddr)
	}
	return &buffer{d: d, bo: bo, mem: d.heapMem[off : off+uint64(size) : off+uint64(size)]}, nil
}

func roundPage(n int) int { return (n + 4095) &^ 4095 }

func (b *buffer) Bytes() []byte { return b.mem }

// The NPU is not cache coherent with the CPU. On x86 the CPU's cache lines are flushed in
// user space (CLFLUSH, as XRT does), elsewhere through the driver (SYNC_BO); a flush writes
// back and invalidates, so one serves both directions.
func (b *buffer) SyncToDevice(off, n int) error { return b.sync(xdna.SyncToDevice, off, n) }

func (b *buffer) SyncFromDevice(off, n int) error { return b.sync(xdna.SyncFromDevice, off, n) }

func (b *buffer) sync(dir uint32, off, n int) error {
	if off < 0 || n < 0 || off+n > len(b.mem) {
		return fmt.Errorf("amd: sync of [%d, %d) in a buffer of %d bytes", off, off+n, len(b.mem))
	}
	if xdna.FlushCPU(b.mem[off : off+n]) {
		return nil
	}
	return b.d.dev.Sync(b.bo.Handle, dir, off, n)
}

func (b *buffer) Close() error {
	if b.mem == nil {
		return nil
	}
	var errs []error
	if b.mapped != nil {
		errs = append(errs, b.d.dev.D.Munmap(b.mapped))
	}
	errs = append(errs, b.d.dev.CloseBO(b.bo.Handle))
	b.mem, b.mapped = nil, nil
	return errors.Join(errs...)
}

// PowerSample reads the NPU power sensor (milliwatts on AIE2) and integrates it into energy.
func (d *device) PowerSample() (types.PowerSample, error) {
	sensors, err := d.dev.Sensors()
	if err != nil {
		return types.PowerSample{}, err
	}
	for _, s := range sensors {
		if s.Type != xdna.SensorTypePower {
			continue
		}
		w := float64(s.Input) * math.Pow10(int(s.UnitM))
		now := time.Now()
		d.powerMu.Lock()
		if !d.lastT.IsZero() {
			d.energyJ += (d.lastW + w) / 2 * now.Sub(d.lastT).Seconds()
		}
		d.lastW, d.lastT = w, now
		e := d.energyJ
		d.powerMu.Unlock()
		return types.PowerSample{Time: now, EnergyJ: e, Watts: w, Scope: types.ScopeNPU, Source: "amdxdna-sensor (integrated)"}, nil
	}
	return types.PowerSample{}, fmt.Errorf("amd: no power sensor: %w", types.ErrUnsupported)
}

// qosPriority maps the power policy to the hardware context priority (unprivileged). The device
// wide power mode (SET_STATE) needs root and affects every user, so hwnpu leaves it alone.
func qosPriority(p types.PowerPolicy) uint32 {
	switch p {
	case types.PowerLow:
		return xdna.QoSLow
	case types.PowerBalanced:
		return xdna.QoSNormal
	case types.PowerPerformance:
		return xdna.QoSHigh
	}
	return 0 // driver default
}

type program struct {
	d       *device
	file    *xclbin.File
	hwctx   uint32
	syncobj uint32
	cus     []string // kernel name of each configured CU, in CU index order
	pdis    []*buffer
	infos   []types.KernelInfo
}

func (d *device) LoadProgram(ctx context.Context, src types.ProgramSource) (backend.Program, error) {
	if src.ELF != nil {
		return d.loadELF(src.ELF)
	}
	data := src.Xclbin
	if data == nil {
		var err error
		if data, err = os.ReadFile(src.Path); err != nil {
			return nil, err
		}
	}
	if aieelf.IsELF(data) {
		return d.loadELF(data)
	}
	f, err := xclbin.Parse(data)
	if err != nil {
		return nil, err
	}
	if f.Partition == nil || f.Partition.Columns == 0 {
		return nil, errors.New("amd: the xclbin has no AIE partition")
	}
	p := &program{d: d, file: f}
	fail := func(err error) (backend.Program, error) {
		p.Close()
		return nil, err
	}
	// Compute units: as the XRT shim, each CU of a kernel whose kernel id has a PDI.
	var cfgs []xdna.CUConfig
	for _, cu := range f.CUs {
		k, ok := f.Kernel(cu.Kernel)
		if !ok {
			continue
		}
		id, fn := k.KernelID, k.Functional
		if id == 0 {
			id, fn = uint64(cu.KernelID), cu.Functional
		}
		pdi, ok := f.PDIForKernel(id)
		if !ok {
			continue
		}
		buf, err := d.allocDev(len(pdi))
		if err != nil {
			return fail(fmt.Errorf("amd: PDI buffer: %w", err))
		}
		p.pdis = append(p.pdis, buf)
		copy(buf.mem, pdi)
		if err := buf.SyncToDevice(0, len(pdi)); err != nil {
			return fail(err)
		}
		cfgs = append(cfgs, xdna.CUConfig{CUBO: buf.bo.Handle, CUFunc: uint8(fn)})
		p.cus = append(p.cus, k.Name)
	}
	if len(cfgs) == 0 {
		return fail(errors.New("amd: no valid DPU kernel found in the xclbin"))
	}
	cols := uint32(f.Partition.Columns)
	if cols > uint32(d.meta.Cols) {
		return fail(fmt.Errorf("amd: the xclbin needs %d columns, the device has %d", cols, d.meta.Cols))
	}
	qos := xdna.QoSInfo{Priority: qosPriority(d.opts.Power)}
	p.hwctx, p.syncobj, err = d.dev.CreateHwctx(qos, f.Partition.OpsPerCycle, cols*uint32(d.meta.Core.RowCount))
	if err != nil {
		return fail(err)
	}
	if err := d.dev.ConfigCUs(p.hwctx, cfgs); err != nil {
		return fail(err)
	}
	for _, k := range f.Kernels {
		n := 0
		for _, name := range p.cus {
			if name == k.Name {
				n++
			}
		}
		if n == 0 {
			continue
		}
		info := types.KernelInfo{Name: k.Name, CUs: n}
		for _, a := range k.Args {
			info.Args = append(info.Args, types.KernelArg{Index: a.Index, Name: a.Name, Type: a.Type, Offset: a.Offset, Size: a.Size, Buffer: a.IsBuffer()})
		}
		p.infos = append(p.infos, info)
	}
	return p, nil
}

func (p *program) Kernels() []types.KernelInfo { return p.infos }

func (p *program) Close() error {
	var errs []error
	if p.hwctx != 0 {
		errs = append(errs, p.d.dev.DestroyHwctx(p.hwctx))
		p.hwctx = 0
	}
	for _, b := range p.pdis {
		errs = append(errs, b.Close())
	}
	p.pdis = nil
	return errors.Join(errs...)
}

type run struct {
	p      *program
	kernel *xclbin.Kernel
	cu     int
	regmap []byte
	bos    map[int]uint32
	cmd    *buffer
	fence  *fence
}

func (p *program) NewRun(name string) (backend.Run, error) {
	k, ok := p.file.Kernel(name)
	if !ok {
		return nil, fmt.Errorf("amd: no kernel %q", name)
	}
	cu := -1
	for i, n := range p.cus {
		if n == name {
			cu = i
			break
		}
	}
	if cu < 0 {
		return nil, fmt.Errorf("amd: kernel %q has no compute unit with a PDI", name)
	}
	size := 0
	for _, a := range k.Args {
		size = max(size, a.Offset+a.Size)
	}
	cmd, err := p.d.allocMapped(xdna.BOCmd, cmdBOSize)
	if err != nil {
		return nil, err
	}
	return &run{p: p, kernel: k, cu: cu, regmap: make([]byte, (size+3)&^3), bos: map[int]uint32{}, cmd: cmd}, nil
}

func (r *run) arg(i int) (*xclbin.Arg, error) {
	for j := range r.kernel.Args {
		if r.kernel.Args[j].Index == i {
			return &r.kernel.Args[j], nil
		}
	}
	return nil, fmt.Errorf("amd: kernel %s has no argument %d", r.kernel.Name, i)
}

func (r *run) SetArg(i int, value []byte) error {
	a, err := r.arg(i)
	if err != nil {
		return err
	}
	if a.IsBuffer() {
		return fmt.Errorf("amd: argument %d (%s) is a buffer", i, a.Name)
	}
	if len(value) != a.Size {
		return fmt.Errorf("amd: argument %d (%s) takes %d bytes, got %d", i, a.Name, a.Size, len(value))
	}
	copy(r.regmap[a.Offset:], value)
	delete(r.bos, i)
	return nil
}

func (r *run) SetBuffer(i int, b backend.Buffer) error {
	a, err := r.arg(i)
	if err != nil {
		return err
	}
	if !a.IsBuffer() {
		return fmt.Errorf("amd: argument %d (%s) is not a buffer", i, a.Name)
	}
	buf, ok := b.(*buffer)
	if !ok || buf.d != r.p.d {
		return errors.New("amd: buffer was not allocated by this device")
	}
	switch a.Size {
	case 8:
		binary.LittleEndian.PutUint64(r.regmap[a.Offset:], buf.bo.DeviceAddr())
	case 4:
		binary.LittleEndian.PutUint32(r.regmap[a.Offset:], uint32(buf.bo.DeviceAddr()))
	default:
		return fmt.Errorf("amd: buffer argument %s has size %d", a.Name, a.Size)
	}
	r.bos[i] = buf.bo.Handle
	return nil
}

func (r *run) Submit() (backend.Fence, error) {
	if r.fence != nil {
		if done, _ := r.fence.Done(); !done {
			return nil, errors.New("amd: previous submission is still running")
		}
	}
	// Arguments left unset stay zero, as in XRT. IRON lists unused buffers in the kernel
	// signature (bo3 and bo4 of a three-buffer design), and its host code never sets them.
	clear(r.cmd.mem)
	if _, err := ert.StartCU(r.cmd.mem, r.cu, r.regmap); err != nil {
		return nil, err
	}
	seen := map[uint32]bool{}
	var args []uint32
	for _, h := range r.bos {
		if !seen[h] {
			seen[h] = true
			args = append(args, h)
		}
	}
	seq, err := r.p.d.dev.Exec(r.p.hwctx, r.cmd.bo.Handle, args)
	if err != nil {
		return nil, err
	}
	r.fence = &fence{dev: r.p.d.dev, syncobj: r.p.syncobj, seq: seq, cmd: r.cmd.mem}
	return r.fence, nil
}

func (r *run) Close() error {
	if r.fence != nil {
		r.fence.Wait(context.Background())
	}
	return r.cmd.Close()
}

type fence struct {
	dev     xdna.Dev
	syncobj uint32
	seq     uint64
	cmd     []byte
}

const pollInterval = 5 * time.Millisecond

func (f *fence) result() error {
	if st := ert.StateOf(binary.LittleEndian.Uint32(f.cmd)); st != ert.StateCompleted {
		return fmt.Errorf("amd: command finished with %v", st)
	}
	return nil
}

func (f *fence) Wait(ctx context.Context) error {
	for {
		done, err := f.dev.Wait(f.syncobj, f.seq, pollInterval)
		if err != nil {
			return err
		}
		if done {
			return f.result()
		}
		if err := ctx.Err(); err != nil {
			return err
		}
	}
}

func (f *fence) Done() (bool, error) {
	done, err := f.dev.Wait(f.syncobj, f.seq, 0)
	if err != nil || !done {
		return false, err
	}
	return true, f.result()
}
