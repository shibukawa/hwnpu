package iron

import (
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu"
)

// An xclbin per step (NPU1): each step of a version is one dispatch of a kernel of an xclbin,
// which IRON's PerStepCallable makes with XRT (opcode 3, the instruction stream, its length,
// the step's buffers). The xclbins are few (Gemma 4 E2B's decode step: 1095 steps on 2), so
// each is loaded once, a hardware context, and the steps' runs are made at load.
//
// The instruction streams live in two buffers per version and xclbin, each step taking a
// slot: the static streams, written at load, and the streams built per call. One device
// buffer per step (2191 for Gemma 4 E2B, mostly under a page) ran the 64 MiB device heap out;
// one per version is not enough for XRT, which places an xclbin's instruction arguments in a
// memory group of that xclbin and would copy a buffer bound in several.

// stepWindow bounds the commands submitted on one hardware context before waiting for one.
const stepWindow = 16

// slotAlign is the alignment of a stream's slot in its buffer, in words (64 bytes).
const slotAlign = 16

type stepRun struct {
	s       *Step
	prog    *hwnpu.Program
	run     *hwnpu.Run
	program *Program // nil for a static stream
	stream  []uint32 // the stream: static, or the last one built
	off     int      // the slot's word offset in its buffer
	slot    int      // the slot's words
}

// stepStreams are a version's instruction buffers, by xclbin.
type stepStreams struct {
	static, dynamic map[string]*hwnpu.Buffer
}

func align(n int) int { return (n + slotAlign - 1) / slotAlign * slotAlign }

// loadSteps loads a version's xclbins (shared by every version, in m.xclbins), makes its
// steps' runs and writes the static streams.
func (m *Model) loadSteps(ctx context.Context, kr hwnpu.KernelRunner, dir string, v *version) error {
	var static []byte
	if v.m.Insts != "" {
		var err error
		if static, err = os.ReadFile(filepath.Join(dir, v.m.Insts)); err != nil {
			return err
		}
	}
	programs := map[string]*Program{}
	words := map[string]int{}
	for i := range v.m.Steps {
		s := &v.m.Steps[i]
		prog, ok := m.xclbins[s.Xclbin]
		if !ok {
			var err error
			if prog, err = kr.Load(ctx, hwnpu.ProgramSource{Path: filepath.Join(dir, s.Xclbin)}); err != nil {
				return fmt.Errorf("iron: %s: %w", s.Xclbin, err)
			}
			m.xclbins[s.Xclbin] = prog
		}
		r := &stepRun{s: s, prog: prog}
		v.steps = append(v.steps, r)
		if s.Program != "" {
			p, ok := programs[s.Program]
			if !ok {
				var err error
				if p, err = ReadProgram(filepath.Join(dir, s.Program)); err != nil {
					return err
				}
				programs[s.Program] = p
			}
			r.program = p
		} else {
			off, n := s.Insts[0], s.Insts[1]
			if off < 0 || n <= 0 || 4*(off+n) > len(static) {
				return fmt.Errorf("iron: step %d: stream [%d, %d) of %d words", i, off, off+n, len(static)/4)
			}
			r.stream = make([]uint32, n)
			for j := range r.stream {
				r.stream[j] = binary.LittleEndian.Uint32(static[4*(off+j):])
			}
			r.off, r.slot = words[s.Xclbin], align(n)
			words[s.Xclbin] += r.slot
		}
		kern, err := prog.Kernel(s.Kernel)
		if err != nil {
			return fmt.Errorf("iron: step %d: %w", i, err)
		}
		if r.run, err = kern.NewRun(); err != nil {
			return fmt.Errorf("iron: step %d: %w", i, err)
		}
		r.run.SetUint64(0, 3) // opcode 3: run the instruction stream (mlir-aie's host code)
		for j, b := range s.Buffers {
			buf := m.buffer(v, b.Argument)
			if buf == nil {
				return fmt.Errorf("iron: step %d: no argument %s", i, b.Argument)
			}
			r.run.SetBufferAt(3+j, buf, b.Offset, b.Bytes)
		}
		if err := r.run.Err(); err != nil {
			return fmt.Errorf("iron: step %d (%s): %w", i, s.Kernel, err)
		}
	}
	v.streams.static = map[string]*hwnpu.Buffer{}
	for x, n := range words {
		buf, err := m.dev.AllocBuffer(4*n, hwnpu.WithBufferKind(hwnpu.BufferInstructions))
		if err != nil {
			return fmt.Errorf("iron: static streams of %s (%d KiB): %w", x, n>>8, err)
		}
		v.streams.static[x] = buf
	}
	for i, r := range v.steps {
		if r.program == nil {
			if err := r.place(v.streams.static[r.s.Xclbin]); err != nil {
				return fmt.Errorf("iron: step %d: %w", i, err)
			}
		}
	}
	for _, buf := range v.streams.static {
		if err := buf.SyncToDevice(0, buf.Size()); err != nil {
			return err
		}
	}
	return nil
}

// place writes the step's stream into its slot of buf and binds the slot (not synced).
func (r *stepRun) place(buf *hwnpu.Buffer) error {
	b := buf.Bytes()[4*r.off:]
	for i, w := range r.stream {
		binary.LittleEndian.PutUint32(b[4*i:], w)
	}
	r.run.SetBufferAt(1, buf, 4*r.off, 4*r.slot)
	r.run.SetUint32(2, uint32(len(r.stream)))
	return r.run.Err()
}

// buildStreams builds the call's streams and writes them into the version's buffer of
// streams built per call, laid out again (and made larger) when one outgrows its slot.
func (m *Model) buildStreams(v *version, values map[string]int64) error {
	var scalars []int64
	relayout := v.streams.dynamic == nil
	for i, r := range v.steps {
		if r.program == nil {
			continue
		}
		scalars = scalars[:0]
		for _, name := range r.s.Scalars {
			x, err := v.m.Words[name].Eval(values)
			if err != nil {
				return fmt.Errorf("iron: step %d: %s: %w", i, name, err)
			}
			scalars = append(scalars, x)
		}
		stream, err := r.program.Run(r.stream, scalars)
		if err != nil {
			return fmt.Errorf("iron: step %d: %w", i, err)
		}
		r.stream = stream
		if len(stream) > r.slot {
			relayout = true
		}
	}
	if relayout {
		words := map[string]int{}
		for _, r := range v.steps {
			if r.program != nil {
				r.off, r.slot = words[r.s.Xclbin], align(len(r.stream)+len(r.stream)/8) // room to grow
				words[r.s.Xclbin] += r.slot
			}
		}
		for _, b := range v.streams.dynamic {
			b.Close()
		}
		v.streams.dynamic = map[string]*hwnpu.Buffer{}
		for x, n := range words {
			buf, err := m.dev.AllocBuffer(4*n, hwnpu.WithBufferKind(hwnpu.BufferInstructions))
			if err != nil {
				return fmt.Errorf("iron: per-call streams of %s (%d KiB): %w", x, n>>8, err)
			}
			v.streams.dynamic[x] = buf
		}
	}
	for i, r := range v.steps {
		if r.program != nil {
			if err := r.place(v.streams.dynamic[r.s.Xclbin]); err != nil {
				return fmt.Errorf("iron: step %d: %w", i, err)
			}
		}
	}
	for _, buf := range v.streams.dynamic {
		if err := buf.SyncToDevice(0, buf.Size()); err != nil {
			return err
		}
	}
	return nil
}

// runSteps runs every step of v with the call's values. Commands of one hardware context run
// in order, so a step is submitted without waiting while the context stays the same; before
// a step of another, the last command is waited for.
func (m *Model) runSteps(ctx context.Context, v *version, values map[string]int64) error {
	if err := m.buildStreams(v, values); err != nil {
		return err
	}
	var last hwnpu.Fence
	var ctxProg *hwnpu.Program
	n := 0
	for i, r := range v.steps {
		if last != nil && (r.prog != ctxProg || n >= stepWindow) {
			if err := last.Wait(ctx); err != nil {
				return fmt.Errorf("iron: step %d: %w", i-1, err)
			}
			n = 0
		}
		f, err := r.run.Submit()
		if err != nil {
			return fmt.Errorf("iron: step %d: %w", i, err)
		}
		last, ctxProg = f, r.prog
		n++
	}
	if last != nil {
		if err := last.Wait(ctx); err != nil {
			return fmt.Errorf("iron: step %d: %w", len(v.steps)-1, err)
		}
	}
	return nil
}

// closeSteps releases a version's runs and instruction buffers.
func (v *version) closeSteps() []error {
	var errs []error
	for _, r := range v.steps {
		if r.run != nil {
			errs = append(errs, r.run.Close())
		}
	}
	for _, bufs := range []map[string]*hwnpu.Buffer{v.streams.static, v.streams.dynamic} {
		for _, b := range bufs {
			errs = append(errs, b.Close())
		}
	}
	v.steps, v.streams = nil, stepStreams{}
	return errs
}
