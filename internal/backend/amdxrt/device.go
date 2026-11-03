package amdxrt

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"
	"unsafe"

	"github.com/shibukawa/hwnpu/internal/aieelf"
	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/power"
	"github.com/shibukawa/hwnpu/internal/types"
	"github.com/shibukawa/hwnpu/internal/xclbin"
)

type device struct {
	s     *shim
	h     uintptr
	opts  backend.Options
	meter *power.Integrator
	once  sync.Once
}

func (d *device) Close() error {
	if d.meter != nil {
		d.meter.Stop()
	}
	if d.h != 0 {
		d.s.devFree(d.h)
		d.h = 0
	}
	return nil
}

func (d *device) Compile(ctx context.Context, src types.GraphSource) (backend.Graph, error) {
	return nil, types.ErrUnsupported
}

// PowerSample integrates the power XRT reports for the device (its electrical info).
func (d *device) PowerSample() (types.PowerSample, error) {
	d.once.Do(func() {
		d.meter = power.NewIntegrator(func() (float64, types.PowerScope, string, error) {
			w, err := d.s.watts(d.h)
			return w, types.ScopeNPU, "XRT electrical", err
		})
	})
	return d.meter.Sample()
}

// buffer is a buffer object. An instruction stream's lives in Go memory and is copied, when
// synced, into an XRT buffer per memory group it is bound in: XRT places an xclbin kernel's
// instruction argument in a group of its own (xrt::kernel::group_id) and refuses a cacheable
// buffer of another, and the group is known only once the buffer is bound.
type buffer struct {
	d     *device
	h     uintptr // the XRT buffer of a data buffer
	mem   []byte
	addr  uint64
	instr map[int32]*instrBO // an instruction stream's XRT buffers, by group
	subs  map[[2]int]uintptr // sub-buffers bound as arguments, by offset and size
}

type instrBO struct {
	h    uintptr
	mem  []byte
	subs map[[2]int]uintptr // parts bound as arguments, by offset and size
}

func (d *device) AllocBuffer(size int, kind types.BufferKind) (backend.Buffer, error) {
	if size <= 0 {
		return nil, fmt.Errorf("amdxrt: buffer of %d bytes", size)
	}
	b := &buffer{d: d}
	if kind == types.BufferInstructions {
		b.mem, b.instr = make([]byte, size), map[int32]*instrBO{}
		return b, nil
	}
	h, mem, err := d.alloc(size, boHost, 0)
	if err != nil {
		return nil, err
	}
	b.h, b.mem, b.addr = h, mem, d.s.boAddress(h)
	return b, nil
}

// alloc allocates and maps an XRT buffer.
func (d *device) alloc(size int, kind, group int32) (uintptr, []byte, error) {
	var h uintptr
	if err := d.s.err(d.s.boAlloc(d.h, uintptr(size), kind, group, &h)); err != nil {
		return 0, nil, err
	}
	var p unsafe.Pointer
	if err := d.s.err(d.s.boMap(h, &p)); err != nil {
		d.s.boFree(h)
		return 0, nil, err
	}
	return h, unsafe.Slice((*byte)(p), size), nil
}

// in returns the XRT buffer holding an instruction stream for a group, filled and synced.
func (b *buffer) in(group int32) (*instrBO, error) {
	if ib, ok := b.instr[group]; ok {
		return ib, nil
	}
	h, mem, err := b.d.alloc(len(b.mem), boCacheable, group)
	if err != nil {
		return nil, err
	}
	ib := &instrBO{h: h, mem: mem}
	copy(mem, b.mem)
	if err := b.d.s.err(b.d.s.boSync(h, syncToDevice, 0, uintptr(len(mem)))); err != nil {
		b.d.s.boFree(h)
		return nil, err
	}
	b.instr[group] = ib
	return ib, nil
}

func (b *buffer) Bytes() []byte { return b.mem }

// DeviceAddress is the address the NPU uses for the buffer (xrt::bo::address); none for an
// instruction stream.
func (b *buffer) DeviceAddress() uint64 { return b.addr }

func (b *buffer) SyncToDevice(off, n int) error { return b.sync(syncToDevice, off, n) }

func (b *buffer) SyncFromDevice(off, n int) error { return b.sync(syncFromDevice, off, n) }

func (b *buffer) sync(dir int32, off, n int) error {
	if off < 0 || n < 0 || off+n > len(b.mem) {
		return fmt.Errorf("amdxrt: sync [%d, %d) of a %d byte buffer", off, off+n, len(b.mem))
	}
	if n == 0 {
		return nil
	}
	if b.instr != nil {
		if dir == syncFromDevice {
			return nil // the NPU does not write instruction streams
		}
		for _, ib := range b.instr {
			copy(ib.mem[off:off+n], b.mem[off:off+n])
			if err := b.d.s.err(b.d.s.boSync(ib.h, syncToDevice, uintptr(off), uintptr(n))); err != nil {
				return err
			}
		}
		return nil
	}
	return b.d.s.err(b.d.s.boSync(b.h, dir, uintptr(off), uintptr(n)))
}

// sub returns the sub-buffer of n bytes from off, made once.
func (b *buffer) sub(off, n int) (uintptr, error) {
	key := [2]int{off, n}
	if h, ok := b.subs[key]; ok {
		return h, nil
	}
	var h uintptr
	if err := b.d.s.err(b.d.s.boSub(b.h, uintptr(n), uintptr(off), &h)); err != nil {
		return 0, err
	}
	if b.subs == nil {
		b.subs = map[[2]int]uintptr{}
	}
	b.subs[key] = h
	return h, nil
}

func (b *buffer) Close() error {
	for _, h := range b.subs {
		b.d.s.boFree(h)
	}
	b.subs = nil
	if b.h != 0 {
		b.d.s.boFree(b.h)
		b.h = 0
	}
	for _, ib := range b.instr {
		for _, h := range ib.subs {
			b.d.s.boFree(h)
		}
		b.d.s.boFree(ib.h)
	}
	b.instr, b.mem = nil, nil
	return nil
}

type program struct {
	d       *device
	ctx     uintptr
	infos   []types.KernelInfo
	mu      sync.Mutex
	kernels map[string]*kernel
}

// kernel is an opened kernel and the runs using it. XRT keeps a full ELF kernel's control code
// in the 64 MiB device heap while the kernel is open, so it is closed with its last run.
type kernel struct {
	h    uintptr
	runs int
}

func (d *device) LoadProgram(ctx context.Context, src types.ProgramSource) (backend.Program, error) {
	data := src.ELF
	if data == nil {
		data = src.Xclbin
	}
	if data == nil {
		var err error
		if data, err = os.ReadFile(src.Path); err != nil {
			return nil, err
		}
	}
	p := &program{d: d, kernels: map[string]*kernel{}}
	if aieelf.IsELF(data) {
		f, err := aieelf.Parse(data)
		if err != nil {
			return nil, err
		}
		for _, in := range f.Instances {
			info := types.KernelInfo{Name: in.FullName(), CUs: 1}
			for i := 0; i < in.Args; i++ {
				info.Args = append(info.Args, types.KernelArg{Index: i, Name: fmt.Sprintf("argv%d", i), Type: "char *", Offset: 8 * i, Size: 8, Buffer: true})
			}
			p.infos = append(p.infos, info)
		}
		if err := d.s.err(d.s.ctxELF(d.h, bytePtr(data), uintptr(len(data)), &p.ctx)); err != nil {
			return nil, err
		}
		return p, nil
	}
	f, err := xclbin.Parse(data)
	if err != nil {
		return nil, err
	}
	for _, k := range f.Kernels {
		n := 0
		for _, cu := range f.CUs {
			if cu.Kernel == k.Name {
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
	if len(p.infos) == 0 {
		return nil, errors.New("amdxrt: the xclbin has no kernel with a compute unit")
	}
	if err := d.s.err(d.s.ctxXclbin(d.h, bytePtr(data), uintptr(len(data)), &p.ctx)); err != nil {
		return nil, err
	}
	return p, nil
}

func (p *program) Kernels() []types.KernelInfo { return p.infos }

func (p *program) info(name string) (*types.KernelInfo, error) {
	for i := range p.infos {
		if p.infos[i].Name == name {
			return &p.infos[i], nil
		}
	}
	return nil, fmt.Errorf("amdxrt: no kernel %q", name)
}

// open opens a kernel for a run, or takes the one open.
func (p *program) open(name string) (*kernel, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if k, ok := p.kernels[name]; ok {
		k.runs++
		return k, nil
	}
	k := &kernel{runs: 1}
	if err := p.d.s.err(p.d.s.kernelOpen(p.ctx, name, &k.h)); err != nil {
		return nil, err
	}
	p.kernels[name] = k
	return k, nil
}

// release ends a run's use of its kernel, closing it after the last.
func (p *program) release(name string, k *kernel) {
	p.mu.Lock()
	defer p.mu.Unlock()
	k.runs--
	if k.runs == 0 && p.kernels[name] == k {
		delete(p.kernels, name)
		p.d.s.kernelFree(k.h)
	}
}

func (p *program) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, k := range p.kernels {
		p.d.s.kernelFree(k.h)
	}
	p.kernels = nil
	if p.ctx != 0 {
		p.d.s.ctxFree(p.ctx)
		p.ctx = 0
	}
	return nil
}

type run struct {
	s       *shim
	p       *program
	h       uintptr
	k       *kernel
	info    *types.KernelInfo
	started bool
	scratch []byte
	gotPad  bool
	bufs    map[int]*buffer // keeps the bound buffers alive
}

func (p *program) NewRun(name string) (backend.Run, error) {
	info, err := p.info(name)
	if err != nil {
		return nil, err
	}
	k, err := p.open(name)
	if err != nil {
		return nil, err
	}
	r := &run{s: p.d.s, p: p, k: k, info: info, bufs: map[int]*buffer{}}
	if err := r.s.err(r.s.runNew(k.h, &r.h)); err != nil {
		p.release(name, k)
		return nil, err
	}
	return r, nil
}

func (r *run) arg(i int) (*types.KernelArg, error) {
	for j := range r.info.Args {
		if r.info.Args[j].Index == i {
			return &r.info.Args[j], nil
		}
	}
	return nil, fmt.Errorf("amdxrt: %s has no argument %d", r.info.Name, i)
}

func (r *run) SetArg(i int, value []byte) error {
	a, err := r.arg(i)
	if err != nil {
		return err
	}
	if a.Buffer {
		return fmt.Errorf("amdxrt: argument %d (%s) is a buffer", i, a.Name)
	}
	if len(value) != a.Size {
		return fmt.Errorf("amdxrt: argument %d (%s) takes %d bytes, got %d", i, a.Name, a.Size, len(value))
	}
	return r.s.err(r.s.runSetVal(r.h, int32(i), bytePtr(value), uintptr(len(value))))
}

func (r *run) SetBuffer(i int, b backend.Buffer) error { return r.SetBufferAt(i, b, 0, len(b.Bytes())) }

// SetBufferAt sets a buffer argument to a part of a buffer, through an XRT sub-buffer.
func (r *run) SetBufferAt(i int, b backend.Buffer, off, n int) error {
	a, err := r.arg(i)
	if err != nil {
		return err
	}
	if !a.Buffer {
		return fmt.Errorf("amdxrt: argument %d (%s) is not a buffer", i, a.Name)
	}
	buf, ok := b.(*buffer)
	if !ok {
		return fmt.Errorf("amdxrt: argument %d: a buffer of another device", i)
	}
	if off < 0 || n <= 0 || off+n > len(buf.mem) {
		return fmt.Errorf("amdxrt: argument %d: [%d, %d) of a %d byte buffer", i, off, off+n, len(buf.mem))
	}
	var h uintptr
	switch {
	case buf.instr != nil:
		ib, err := buf.in(r.s.kernelGrp(r.k.h, int32(i)))
		if err != nil {
			return err
		}
		h = ib.h
		if off != 0 || n != len(buf.mem) {
			key := [2]int{off, n}
			if sh, ok := ib.subs[key]; ok {
				h = sh
			} else {
				if err := r.s.err(r.s.boSub(ib.h, uintptr(n), uintptr(off), &h)); err != nil {
					return err
				}
				if ib.subs == nil {
					ib.subs = map[[2]int]uintptr{}
				}
				ib.subs[key] = h
			}
		}
	case off == 0 && n == len(buf.mem):
		h = buf.h
	default:
		if h, err = buf.sub(off, n); err != nil {
			return err
		}
	}
	if err := r.s.err(r.s.runSetBO(r.h, int32(i), h)); err != nil {
		return err
	}
	r.bufs[i] = buf
	return nil
}

// Scratchpad is the run's ctrl scratchpad (xrt::run::get_ctrl_scratchpad_bo), mapped; nil
// for a run without one. xs_run_start syncs it.
func (r *run) Scratchpad() []byte {
	if !r.gotPad {
		r.gotPad = true
		var p unsafe.Pointer
		var n uintptr
		if err := r.s.err(r.s.runScratch(r.h, &p, &n)); err == nil && p != nil && n > 0 {
			r.scratch = unsafe.Slice((*byte)(p), int(n))
		}
	}
	return r.scratch
}

func (r *run) Submit() (backend.Fence, error) {
	if r.started {
		// XRT starts a run again only once it has been waited for, even when it is done (as
		// a command an in-order queue finished before one that was waited for).
		var st int32
		if err := r.s.err(r.s.runWait(r.h, 0, &st)); err != nil {
			return nil, err
		}
		r.started = false
		if err := result(st); err != nil {
			return nil, err
		}
	}
	if err := r.s.err(r.s.runStart(r.h)); err != nil {
		return nil, err
	}
	r.started = true
	return &fence{r: r}, nil
}

func (r *run) Close() error {
	if r.h == 0 {
		return nil
	}
	if r.started {
		var st int32
		r.s.runWait(r.h, 0, &st)
	}
	r.s.runFree(r.h)
	r.p.release(r.info.Name, r.k)
	r.h, r.bufs, r.scratch = 0, nil, nil
	return nil
}

type fence struct{ r *run }

func result(st int32) error {
	if st != stateCompleted {
		return fmt.Errorf("amdxrt: command finished in state %d", st)
	}
	return nil
}

func done(st int32) bool { return st >= stateCompleted && st != stateSubmitted && st != stateTimeout }

// Wait blocks in xrt::run::wait until the command ends. XRT 2.21's wait with a timeout leaves
// a command that outlasts it in a state its next wait refuses ("qds_device::wait() unexpected
// command state"), so the context is not seen while the command runs.
func (f *fence) Wait(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	var st int32
	if err := f.r.s.err(f.r.s.runWait(f.r.h, 0, &st)); err != nil {
		return err
	}
	f.r.started = false
	return result(st)
}

func (f *fence) Done() (bool, error) {
	st := f.r.s.runState(f.r.h)
	if !done(st) {
		return false, nil
	}
	f.r.started = false
	return true, result(st)
}
