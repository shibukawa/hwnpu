package iron

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu"
)

// ddrAperture is the offset the NPU's DMA adds to a host address (IRON's APERTURE).
const ddrAperture = 0x80000000

// Model is a loaded IRON language model: the arena's banks and its versions' programs.
type Model struct {
	Manifest *Manifest
	dev      *hwnpu.Device
	banks    map[string]*hwnpu.Buffer
	decode   *version
	prompt   *version
	kv       []kvCache // caches, found once
}

type version struct {
	m    *Version
	prog *hwnpu.Program
	// args holds the version's own buffers (input, output, feedback); the banks are shared.
	args map[string]*hwnpu.Buffer
	// run is the decode step's one command, kept: its control code stays in the device heap.
	// A prompt chunk's commands are made for each call instead (newRun): their control code
	// (89 MiB for Gemma 4 E2B) is larger than the device heap (64 MiB on Linux 7.0).
	run *hwnpu.Run
}

// Open loads the model exported in dir onto dev.
func Open(ctx context.Context, dev *hwnpu.Device, dir string, logf func(string, ...any)) (*Model, error) {
	if logf == nil {
		logf = func(string, ...any) {}
	}
	man, err := ReadManifest(dir)
	if err != nil {
		return nil, err
	}
	m := &Model{Manifest: man, dev: dev, banks: map[string]*hwnpu.Buffer{}}
	fail := func(err error) (*Model, error) {
		m.Close()
		return nil, err
	}
	for kind, b := range man.Banks {
		buf, err := dev.AllocBuffer(b.Bytes)
		if err != nil {
			return fail(fmt.Errorf("iron: bank %s (%d MiB): %w", kind, b.Bytes>>20, err))
		}
		m.banks[kind] = buf
		if b.Image != "" {
			if err := readInto(filepath.Join(dir, b.Image), buf.Bytes()); err != nil {
				return fail(fmt.Errorf("iron: bank %s: %w", kind, err))
			}
		}
		if err := buf.SyncToDevice(0, buf.Size()); err != nil {
			return fail(err)
		}
		logf("bank %s: %d MiB", kind, b.Bytes>>20)
	}
	kr, err := dev.Kernel()
	if err != nil {
		return fail(err)
	}
	for name, v := range man.Versions {
		ver := &version{m: v, args: map[string]*hwnpu.Buffer{}}
		if ver.prog, err = kr.Load(ctx, hwnpu.ProgramSource{Path: filepath.Join(dir, v.ELF)}); err != nil {
			return fail(fmt.Errorf("iron: %s: %w", name, err))
		}
		for _, a := range v.Arguments {
			if _, ok := m.banks[a.Kind]; ok {
				continue
			}
			buf, err := dev.AllocBuffer(max(a.Bytes, 4096))
			if err != nil {
				return fail(fmt.Errorf("iron: %s: argument %s: %w", name, a.Kind, err))
			}
			ver.args[a.Kind] = buf
		}
		switch name {
		case "decode":
			if len(v.Kernels) != 1 {
				return fail(fmt.Errorf("iron: the decode step has %d commands", len(v.Kernels)))
			}
			if ver.run, err = m.newRun(ver, 0); err != nil {
				return fail(err)
			}
			m.decode = ver
		case "prompt":
			m.prompt = ver
		default:
			return fail(fmt.Errorf("iron: unknown version %s", name))
		}
		logf("%s: %d commands", name, len(v.Kernels))
	}
	if m.decode == nil || m.prompt == nil {
		return fail(errors.New("iron: the export needs a decode step and a prompt chunk"))
	}
	if _, err := m.caches(); err != nil {
		return fail(err)
	}
	return m, nil
}

func readInto(path string, dst []byte) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := io.ReadFull(f, dst); err != nil && !errors.Is(err, io.ErrUnexpectedEOF) {
		return err
	}
	return nil
}

// Close releases the device memory.
func (m *Model) Close() error {
	var errs []error
	for _, v := range []*version{m.decode, m.prompt} {
		if v == nil {
			continue
		}
		if v.run != nil {
			errs = append(errs, v.run.Close())
		}
		for _, b := range v.args {
			errs = append(errs, b.Close())
		}
		if v.prog != nil {
			errs = append(errs, v.prog.Close())
		}
	}
	for _, b := range m.banks {
		errs = append(errs, b.Close())
	}
	return errors.Join(errs...)
}

func (m *Model) buffer(v *version, kind string) *hwnpu.Buffer {
	if b, ok := m.banks[kind]; ok {
		return b
	}
	return v.args[kind]
}

// newRun makes the run of command k of v with every argument bound.
func (m *Model) newRun(v *version, k int) (*hwnpu.Run, error) {
	kern, err := v.prog.Kernel(v.m.Kernels[k].Name)
	if err != nil {
		return nil, err
	}
	run, err := kern.NewRun()
	if err != nil {
		return nil, err
	}
	for _, a := range v.m.Arguments {
		run.SetBuffer(a.Index, m.buffer(v, a.Kind))
	}
	if err := run.Err(); err != nil {
		run.Close()
		return nil, err
	}
	return run, nil
}

// values are a call's per-call values (CausalLM's body).
func callValues(token int32, position, start, rows int) map[string]int64 {
	return map[string]int64{"token": int64(token), "position": int64(position), "start": int64(start), "rows": int64(rows), "rope_start": int64(start)}
}

// carry writes the call's carried values into the first plane of the carry buffer.
func (m *Model) carry(v *version, values map[string]int64) error {
	if v.m.Carried == nil || v.m.Carried.Carry == nil {
		return nil
	}
	c := v.m.Carried.Carry
	buf := m.buffer(v, c.Argument)
	for j, name := range c.Names {
		binary.LittleEndian.PutUint32(buf.Bytes()[c.Offset+4*j:], uint32(values[name]))
	}
	return buf.SyncToDevice(c.Offset, 4*len(c.Names))
}

// logits reads the version's logits output.
func (m *Model) logits(v *version) ([]float32, error) {
	if len(v.m.Outputs) != 1 || v.m.Outputs[0].DType != "bfloat16" {
		return nil, errors.New("iron: the version's output is not one bfloat16 logits row")
	}
	o := v.m.Outputs[0]
	buf := m.buffer(v, o.Argument)
	if err := buf.SyncFromDevice(o.Offset, o.Bytes); err != nil {
		return nil, err
	}
	n := min(o.Bytes/2, m.Manifest.Config.VocabSize)
	out := make([]float32, n)
	b := buf.Bytes()[o.Offset:]
	for i := range out {
		out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[2*i:])) << 16)
	}
	return out, nil
}

// Step runs the decode step on token at position and returns the logits after it.
func (m *Model) Step(ctx context.Context, token int32, position int) ([]float32, error) {
	if position < 0 || position >= m.Manifest.Config.MaxSeqLen {
		return nil, fmt.Errorf("iron: position %d outside the cache of %d", position, m.Manifest.Config.MaxSeqLen)
	}
	v := m.decode
	values := callValues(token, position, 0, 1)
	if err := writeWords(v.run.Scratchpad(), v.m.Kernels[0].Parameters, values); err != nil {
		return nil, err
	}
	if err := m.carry(v, values); err != nil {
		return nil, err
	}
	if err := v.run.Run(ctx); err != nil {
		return nil, err
	}
	return m.logits(v)
}

// Prompt runs tokens from position start in prompt chunks and returns the logits after the
// last one. The cache holds the tokens before start.
func (m *Model) Prompt(ctx context.Context, tokens []int32, start int) ([]float32, error) {
	C := m.Manifest.Config.PrefillChunk
	if len(tokens) == 0 || start+len(tokens) > m.Manifest.Config.MaxSeqLen {
		return nil, fmt.Errorf("iron: %d tokens from %d do not fit the cache of %d", len(tokens), start, m.Manifest.Config.MaxSeqLen)
	}
	var logits []float32
	for begin := 0; begin < len(tokens); begin += C {
		end := min(begin+C, len(tokens))
		var err error
		if logits, err = m.chunk(ctx, tokens[begin:end], start+begin); err != nil {
			return nil, err
		}
	}
	return logits, nil
}

func (m *Model) chunk(ctx context.Context, ids []int32, start int) ([]float32, error) {
	v := m.prompt
	rows := len(ids)
	values := callValues(ids[rows-1], start+rows-1, start, rows)
	for _, e := range v.m.Encoders {
		if err := m.encode(v, &e, ids, values); err != nil {
			return nil, err
		}
	}
	if err := m.carry(v, values); err != nil {
		return nil, err
	}
	timing := os.Getenv("HWNPU_IRON_TIMING") != ""
	for k := range v.m.Kernels {
		t0 := time.Now()
		run, err := m.newRun(v, k)
		t1 := time.Now()
		if err != nil {
			return nil, fmt.Errorf("iron: prompt command %d: %w", k, err)
		}
		err = writeWords(run.Scratchpad(), v.m.Kernels[k].Parameters, values)
		if err == nil {
			err = run.Run(ctx)
		}
		t2 := time.Now()
		run.Close()
		if timing {
			fmt.Fprintf(os.Stderr, "prompt command %d (%d rows from %d): setup %.1f ms, run %.1f ms, close %.1f ms\n", k, rows, start,
				t1.Sub(t0).Seconds()*1000, t2.Sub(t1).Seconds()*1000, time.Since(t2).Seconds()*1000)
		}
		if err != nil {
			return nil, fmt.Errorf("iron: prompt command %d: %w", k, err)
		}
	}
	return m.logits(v)
}

// encode writes the control words gathering ids' rows from e's table into e's input (IRON's
// Gather.control_words): each valid row a DMA descriptor's address of its table row, the
// rows past the bound a short move of the first row.
func (m *Model) encode(v *version, e *Encoder, ids []int32, values map[string]int64) error {
	var in *Handle
	for i := range v.m.Inputs {
		if v.m.Inputs[i].Name == e.Input {
			in = &v.m.Inputs[i]
		}
	}
	if in == nil {
		return fmt.Errorf("iron: no input %s", e.Input)
	}
	valid := e.Rows
	if e.Bound != nil {
		b, err := e.Bound.Eval(values)
		if err != nil {
			return err
		}
		valid = int(b)
	}
	if valid < 0 || valid > e.Rows || valid > len(ids) {
		return fmt.Errorf("iron: gather of %d valid rows from %d ids, %d rows", valid, len(ids), e.Rows)
	}
	table := m.buffer(v, e.TableArgument)
	base, ok := table.DeviceAddress()
	if !ok {
		return errors.New("iron: the device does not expose buffer addresses")
	}
	base += uint64(e.TableOffset) + ddrAperture
	words := append([]uint32(nil), e.template...)
	n := int64(e.TableRows)
	for i, slot := range e.Slots {
		addr := base
		if i < valid {
			id := int64(ids[i])
			if id < -n || id >= n {
				return fmt.Errorf("iron: token %d outside the table of %d rows", id, n)
			}
			addr += uint64(((id%n)+n)%n) * uint64(e.Stride)
		} else {
			words[slot-1] = uint32(e.PadBytes / 4)
		}
		words[slot] = uint32(addr) & 0xfffffffc
		words[slot+2] = uint32(addr>>32) & 0xffff
	}
	buf := m.buffer(v, in.Argument)
	off := in.Offset + 4*e.StartWord
	dst := buf.Bytes()[off : off+4*len(words)]
	for i, w := range words {
		binary.LittleEndian.PutUint32(dst[4*i:], w)
	}
	return buf.SyncToDevice(off, 4*len(words))
}

// Buffer returns the device memory of a named buffer of a version ("decode" or "prompt"),
// synced from the device: a KV cache, an intermediate. It is a view, valid until Close.
func (m *Model) Buffer(version, name string) ([]byte, Placement, error) {
	v := m.decode
	if version == "prompt" {
		v = m.prompt
	}
	p, ok := v.m.Buffers[name]
	if !ok {
		return nil, p, fmt.Errorf("iron: %s has no buffer %s", version, name)
	}
	buf := m.buffer(v, p.Argument)
	if buf == nil {
		return nil, p, fmt.Errorf("iron: %s: no argument %s", version, p.Argument)
	}
	if err := buf.SyncFromDevice(p.Offset, p.Bytes); err != nil {
		return nil, p, err
	}
	return buf.Bytes()[p.Offset : p.Offset+p.Bytes], p, nil
}

// Buffers lists the version's buffers.
func (m *Model) Buffers(version string) map[string]Placement {
	if version == "prompt" {
		return m.prompt.m.Buffers
	}
	return m.decode.m.Buffers
}

// Device is the device the model is loaded on.
func (m *Model) Device() *hwnpu.Device { return m.dev }
