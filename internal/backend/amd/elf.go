package amd

import (
	"context"
	"errors"
	"fmt"

	"github.com/shibukawa/hwnpu/internal/aieelf"
	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/ert"
	"github.com/shibukawa/hwnpu/internal/types"
	"github.com/shibukawa/hwnpu/internal/xdna"
)

// elfProgram is a full ELF (AIE2P): one hardware context for the whole partition and no
// compute units. Each instance ("main:sequence") is a kernel whose control code a run copies
// into the device heap, with buffer addresses patched in (as XRT's module_sram does).
//
// Submission, as recorded from XRT 2.21 on a Ryzen AI 9 365 (docs/amd-npu-models.md 3.1):
// CREATE_HWCTX with num_tiles = columns x core rows and no CONFIG_HWCTX; the PDIs and the
// control code in BO_DEV objects; ERT_START_NPU_PREEMPT_ELF; EXEC_CMD listing only the
// argument buffer objects.
type elfProgram struct {
	d       *device
	file    *aieelf.File
	hwctx   uint32
	syncobj uint32
	pdis    map[string]*buffer
	infos   []types.KernelInfo
}

func (d *device) loadELF(data []byte) (backend.Program, error) {
	f, err := aieelf.Parse(data)
	if err != nil {
		return nil, err
	}
	if f.Columns > int(d.meta.Cols) {
		return nil, fmt.Errorf("amd: the ELF needs %d columns, the device has %d", f.Columns, d.meta.Cols)
	}
	p := &elfProgram{d: d, file: f, pdis: map[string]*buffer{}}
	fail := func(err error) (backend.Program, error) {
		p.Close()
		return nil, err
	}
	// The PDIs of the image's own instances ("main", mlir-aie's composed device) are loaded now,
	// so that they sit together at the start of the device heap: loaded when a run first needs
	// them, they fell between control code buffers and split the 64 MiB heap until a 40 MiB
	// control code no longer fit. The operators' own instances, never run, load lazily.
	for _, in := range f.Instances {
		if in.Kernel == "main" {
			if err := p.loadPDIs(in); err != nil {
				return fail(err)
			}
		}
	}
	qos := xdna.QoSInfo{Priority: qosPriority(d.opts.Power)}
	p.hwctx, p.syncobj, err = d.dev.CreateHwctx(qos, 0, uint32(f.Columns)*uint32(d.meta.Core.RowCount))
	if err != nil {
		return fail(err)
	}
	for _, in := range f.Instances {
		info := types.KernelInfo{Name: in.FullName(), CUs: 1}
		for i := 0; i < in.Args; i++ {
			info.Args = append(info.Args, types.KernelArg{Index: i, Name: fmt.Sprintf("argv%d", i), Type: "char *", Offset: 8 * i, Size: 8, Buffer: true})
		}
		p.infos = append(p.infos, info)
	}
	return p, nil
}

// loadPDIs loads the PDIs an instance's control code names, once per program (XRT 2.21 caches
// them per module).
func (p *elfProgram) loadPDIs(in *aieelf.Instance) error {
	for _, name := range in.PDIs() {
		if p.pdis[name] != nil {
			continue
		}
		img := p.file.PDIs[name]
		buf, err := p.d.allocDev(len(img))
		if err != nil {
			return fmt.Errorf("amd: PDI %s: %w", name, err)
		}
		p.pdis[name] = buf
		copy(buf.mem, img)
		if err := buf.SyncToDevice(0, len(img)); err != nil {
			return err
		}
	}
	return nil
}

func (p *elfProgram) Kernels() []types.KernelInfo { return p.infos }

func (p *elfProgram) Close() error {
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

type elfRun struct {
	p     *elfProgram
	inst  *aieelf.Instance
	instr *buffer // the patched control code, in the device heap
	cmd   *buffer
	bufs  []*buffer // by argument index; nil: unset, patched as address 0 (as XRT leaves it)
	dirty bool
	fence *fence
	// scratch is the ctrl scratchpad the control code reads (scratch-pad-ctrl), or nil.
	scratch *buffer
}

func (p *elfProgram) NewRun(name string) (backend.Run, error) {
	in, ok := p.file.Instance(name)
	if !ok {
		return nil, fmt.Errorf("amd: no kernel %q in the ELF", name)
	}
	r := &elfRun{p: p, inst: in, bufs: make([]*buffer, in.Args), dirty: true}
	fail := func(err error) (backend.Run, error) {
		r.Close()
		return nil, err
	}
	var err error
	if r.instr, err = p.d.allocDev(len(in.Ctrl)); err != nil {
		return fail(fmt.Errorf("amd: control code buffer: %w", err))
	}
	copy(r.instr.mem, in.Ctrl)
	if err := p.loadPDIs(in); err != nil {
		return fail(err)
	}
	if in.Scratchpad > 0 {
		if r.scratch, err = p.d.allocDev(in.Scratchpad); err != nil {
			return fail(fmt.Errorf("amd: ctrl scratchpad: %w", err))
		}
		clear(r.scratch.mem)
	}
	for _, rel := range in.Relocs {
		switch rel.Target {
		case aieelf.TargetPDI:
			aieelf.Patch(r.instr.mem, in.Ctrl, rel, p.pdis[rel.PDI].bo.DeviceAddr())
		case aieelf.TargetScratchpad:
			aieelf.Patch(r.instr.mem, in.Ctrl, rel, r.scratch.bo.DeviceAddr())
		}
	}
	if r.cmd, err = p.d.allocMapped(xdna.BOCmd, cmdBOSize); err != nil {
		return fail(err)
	}
	return r, nil
}

func (r *elfRun) SetArg(i int, value []byte) error {
	return fmt.Errorf("amd: %s takes buffers only: %w", r.inst.FullName(), types.ErrUnsupported)
}

func (r *elfRun) SetBuffer(i int, b backend.Buffer) error {
	if i < 0 || i >= len(r.bufs) {
		return fmt.Errorf("amd: kernel %s has no argument %d", r.inst.FullName(), i)
	}
	buf, ok := b.(*buffer)
	if !ok || buf.d != r.p.d {
		return errors.New("amd: buffer was not allocated by this device")
	}
	if r.bufs[i] != buf {
		r.bufs[i], r.dirty = buf, true
	}
	return nil
}

// UseScratchpad points the control code's ctrl scratchpad at n bytes of b from off, which the
// device writes: the run's own scratchpad goes, and nothing syncs b's bytes from the host.
func (r *elfRun) UseScratchpad(b backend.Buffer, off, n int) error {
	buf, ok := b.(*buffer)
	if !ok || buf.d != r.p.d {
		return errors.New("amd: buffer was not allocated by this device")
	}
	if r.scratch == nil {
		return fmt.Errorf("amd: kernel %s reads no ctrl scratchpad", r.inst.FullName())
	}
	if off < 0 || n <= 0 || off+n > len(buf.mem) || off%4 != 0 {
		return fmt.Errorf("amd: scratchpad [%d, %d) of a %d byte buffer", off, off+n, len(buf.mem))
	}
	for _, rel := range r.inst.Relocs {
		if rel.Target == aieelf.TargetScratchpad {
			aieelf.Patch(r.instr.mem, r.inst.Ctrl, rel, buf.bo.DeviceAddr()+uint64(off))
		}
	}
	err := r.scratch.Close()
	r.scratch, r.dirty = nil, true
	return err
}

// Scratchpad is the ctrl scratchpad the control code reads per command, or nil.
func (r *elfRun) Scratchpad() []byte {
	if r.scratch == nil {
		return nil
	}
	return r.scratch.mem
}

func (r *elfRun) Submit() (backend.Fence, error) {
	if r.fence != nil {
		if done, _ := r.fence.Done(); !done {
			return nil, errors.New("amd: previous submission is still running")
		}
	}
	if r.dirty {
		for _, rel := range r.inst.Relocs {
			if rel.Target != aieelf.TargetArg {
				continue
			}
			var addr uint64
			if b := r.bufs[rel.Arg]; b != nil {
				addr = b.dmaAddr()
			}
			aieelf.Patch(r.instr.mem, r.inst.Ctrl, rel, addr)
		}
		if err := r.instr.SyncToDevice(0, len(r.instr.mem)); err != nil {
			return nil, err
		}
		r.dirty = false
	}
	if r.scratch != nil {
		if err := r.scratch.SyncToDevice(0, len(r.scratch.mem)); err != nil {
			return nil, err
		}
	}
	return r.submit()
}

func (r *elfRun) submit() (*fence, error) {
	clear(r.cmd.mem[:64])
	pkt := ert.PreemptELF{Inst: r.instr.bo.DeviceAddr(), InstSize: uint32(len(r.inst.Ctrl))}
	if len(r.bufs) > 0 && r.bufs[0] != nil {
		pkt.Arg0 = r.bufs[0].bo.DeviceAddr()
	}
	if _, err := ert.StartNPUPreemptELF(r.cmd.mem, pkt); err != nil {
		return nil, err
	}
	var args []uint32
	seen := map[uint32]bool{}
	for _, b := range r.bufs {
		if b != nil && !seen[b.bo.Handle] {
			seen[b.bo.Handle] = true
			args = append(args, b.bo.Handle)
		}
	}
	seq, err := r.p.d.dev.Exec(r.p.hwctx, r.cmd.bo.Handle, args)
	if err != nil {
		return nil, err
	}
	r.fence = &fence{dev: r.p.d.dev, syncobj: r.p.syncobj, seq: seq, cmd: r.cmd.mem}
	return r.fence, nil
}

func (r *elfRun) Close() error {
	if r.fence != nil {
		r.fence.Wait(context.Background())
	}
	var errs []error
	for _, b := range []*buffer{r.cmd, r.instr, r.scratch} {
		if b != nil {
			errs = append(errs, b.Close())
		}
	}
	return errors.Join(errs...)
}
