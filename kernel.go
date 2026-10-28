package hwnpu

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/types"
)

type (
	BufferKind    = types.BufferKind
	ProgramSource = types.ProgramSource
	KernelArg     = types.KernelArg
)

const (
	BufferDefault      = types.BufferDefault
	BufferInstructions = types.BufferInstructions
)

// BufferOption configures AllocBuffer.
type BufferOption func(*bufferOptions)

type bufferOptions struct{ kind BufferKind }

// WithBufferKind selects the memory of a buffer. Use BufferInstructions for NPU instruction
// streams passed to AMD kernels (the IRON "instr" argument).
func WithBufferKind(k BufferKind) BufferOption { return func(o *bufferOptions) { o.kind = k } }

// KernelRunner loads kernel programs (AMD xclbin files built with IRON / mlir-aie).
type KernelRunner interface {
	Load(ctx context.Context, src ProgramSource) (*Program, error)
}

// Kernel returns the kernel runner, or ErrUnsupported when the device cannot run kernels.
func (d *Device) Kernel() (KernelRunner, error) {
	if d.be == nil {
		return nil, ErrClosed
	}
	kd, ok := d.be.(backend.KernelDevice)
	if !ok || d.info.Caps&CapKernel == 0 {
		return nil, ErrUnsupported
	}
	return kernelRunner{d: d, kd: kd}, nil
}

type kernelRunner struct {
	d  *Device
	kd backend.KernelDevice
}

func (r kernelRunner) Load(ctx context.Context, src ProgramSource) (*Program, error) {
	if r.d.be == nil {
		return nil, ErrClosed
	}
	if src.Xclbin == nil && src.Path == "" {
		return nil, errors.New("hwnpu: ProgramSource needs Xclbin or Path")
	}
	p, err := r.kd.LoadProgram(ctx, src)
	if err != nil {
		return nil, err
	}
	return &Program{be: p, d: r.d}, nil
}

// Program is a loaded kernel program. On AMD NPUs it owns a hardware context whose compute
// units run the program's kernels.
type Program struct {
	be backend.Program
	d  *Device
}

// Kernels lists the kernels that have compute units.
func (p *Program) Kernels() []string {
	var names []string
	for _, k := range p.be.Kernels() {
		names = append(names, k.Name)
	}
	return names
}

// Kernel returns a kernel by name.
func (p *Program) Kernel(name string) (*Kernel, error) {
	if p.be == nil {
		return nil, ErrClosed
	}
	for _, k := range p.be.Kernels() {
		if k.Name == name {
			return &Kernel{p: p, info: k}, nil
		}
	}
	return nil, fmt.Errorf("hwnpu: no kernel %q (have %v)", name, p.Kernels())
}

// Close releases the program. Close its runs first.
func (p *Program) Close() error {
	if p.be == nil {
		return ErrClosed
	}
	err := p.be.Close()
	p.be = nil
	return err
}

// Kernel is one kernel of a program.
type Kernel struct {
	p    *Program
	info types.KernelInfo
}

func (k *Kernel) Name() string { return k.info.Name }

// Args lists the kernel arguments in index order.
func (k *Kernel) Args() []KernelArg { return append([]KernelArg(nil), k.info.Args...) }

// NewRun creates an invocation. Set every argument, then Run or Submit; a Run can be submitted
// again after it finished, with the same or new arguments.
func (k *Kernel) NewRun() (*Run, error) {
	if k.p.be == nil {
		return nil, ErrClosed
	}
	r, err := k.p.be.NewRun(k.info.Name)
	if err != nil {
		return nil, err
	}
	return &Run{be: r, k: k}, nil
}

// Run is one kernel invocation. Setters record the first error, returned by Err, Submit and Run.
type Run struct {
	be  backend.Run
	k   *Kernel
	err error
}

func (r *Run) record(err error) *Run {
	if err != nil && r.err == nil {
		r.err = err
	}
	return r
}

// SetBuffer sets a buffer argument.
func (r *Run) SetBuffer(index int, b *Buffer) *Run {
	if b == nil || b.be == nil {
		return r.record(fmt.Errorf("hwnpu: argument %d: nil or closed buffer", index))
	}
	return r.record(r.be.SetBuffer(index, b.be))
}

// SetUint32 sets a 4 byte scalar argument.
func (r *Run) SetUint32(index int, v uint32) *Run {
	return r.record(r.be.SetArg(index, binary.LittleEndian.AppendUint32(nil, v)))
}

// SetUint64 sets an 8 byte scalar argument.
func (r *Run) SetUint64(index int, v uint64) *Run {
	return r.record(r.be.SetArg(index, binary.LittleEndian.AppendUint64(nil, v)))
}

// SetBytes sets a scalar argument from its little endian bytes.
func (r *Run) SetBytes(index int, v []byte) *Run { return r.record(r.be.SetArg(index, v)) }

// Err returns the first error recorded by a setter.
func (r *Run) Err() error { return r.err }

// Submit starts the kernel. Sync input buffers to the device first.
func (r *Run) Submit() (Fence, error) {
	if r.be == nil {
		return nil, ErrClosed
	}
	if r.err != nil {
		return nil, r.err
	}
	return r.be.Submit()
}

// Run submits and waits.
func (r *Run) Run(ctx context.Context) error {
	f, err := r.Submit()
	if err != nil {
		return err
	}
	return f.Wait(ctx)
}

// Close waits for a running submission and releases the run.
func (r *Run) Close() error {
	if r.be == nil {
		return ErrClosed
	}
	err := r.be.Close()
	r.be = nil
	return err
}
