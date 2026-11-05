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
	"slices"
	"syscall"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3vl"
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
	placed   *version  // a prompt chunk with the towers' soft tokens at their placeholders
	vision   *version  // the vision tower, or nil
	audio    *version  // the audio tower, or nil
	kv       []kvCache // caches, found once
	// xclbins are the programs of the versions packaged as an xclbin per step, by file.
	xclbins map[string]*hwnpu.Program
	dir     string            // the export's directory
	vlProc  *lfm3vl.Processor // LFM3-VL's image processor, read on the first image
	// hostTables are the tables kept on the host (HostTable), filled before each call
	hostTables []*hostTable
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
	// steps are the runs of a version packaged as an xclbin per step (steps.go).
	steps   []*stepRun
	streams stepStreams
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
	m := &Model{Manifest: man, dev: dev, banks: map[string]*hwnpu.Buffer{}, xclbins: map[string]*hwnpu.Program{}, dir: dir}
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
		if v.ELF != "" {
			if ver.prog, err = kr.Load(ctx, hwnpu.ProgramSource{Path: filepath.Join(dir, v.ELF)}); err != nil {
				return fail(fmt.Errorf("iron: %s: %w", name, err))
			}
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
		if len(v.Steps) > 0 {
			if err := m.loadSteps(ctx, kr, dir, ver); err != nil {
				return fail(fmt.Errorf("iron: %s: %w", name, err))
			}
		}
		switch name {
		case "decode":
			if v.ELF != "" && len(v.Kernels) != 1 {
				return fail(fmt.Errorf("iron: the decode step has %d commands", len(v.Kernels)))
			}
			m.decode = ver // its run is made once every program's PDIs are in the heap
		case "prompt":
			m.prompt = ver
		case "placed":
			m.placed = ver
		case "vision":
			m.vision = ver
		case "audio":
			m.audio = ver
		default:
			return fail(fmt.Errorf("iron: unknown version %s", name))
		}
		if len(v.Steps) > 0 {
			logf("%s: %d steps (an xclbin per step)", name, len(v.Steps))
		} else {
			logf("%s: %d commands", name, len(v.Kernels))
		}
	}
	if m.decode == nil || m.prompt == nil {
		return fail(errors.New("iron: the export needs a decode step and a prompt chunk"))
	}
	if _, err := m.caches(); err != nil {
		return fail(err)
	}
	if err := m.openHostTables(); err != nil {
		return fail(err)
	}
	if m.decode.m.ELF != "" {
		if m.decode.run, err = m.newRun(m.decode, 0); err != nil {
			return fail(err)
		}
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
	m.closeHostTables()
	var errs []error
	for _, v := range m.versions() {
		if v == nil {
			continue
		}
		if v.run != nil {
			errs = append(errs, v.run.Close())
		}
		errs = append(errs, v.closeSteps()...)
		for _, b := range v.args {
			errs = append(errs, b.Close())
		}
		if v.prog != nil {
			errs = append(errs, v.prog.Close())
		}
	}
	for _, p := range m.xclbins {
		errs = append(errs, p.Close())
	}
	for _, b := range m.banks {
		errs = append(errs, b.Close())
	}
	return errors.Join(errs...)
}

func (m *Model) versions() []*version {
	return []*version{m.decode, m.prompt, m.placed, m.vision, m.audio}
}

func (m *Model) version(name string) *version {
	switch name {
	case "decode":
		return m.decode
	case "prompt":
		return m.prompt
	case "placed":
		return m.placed
	case "vision":
		return m.vision
	case "audio":
		return m.audio
	}
	return nil
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

// complete adds what a call of v needs besides own: zero for the values of the parts it does
// not run (an idle tower's, a tower call's text values), and those of buffer addresses.
func (m *Model) complete(v *version, own map[string]int64) (map[string]int64, error) {
	values := make(map[string]int64, len(v.m.Values)+len(v.m.Addresses))
	for _, n := range v.m.Values {
		values[n] = 0
	}
	for k, x := range own {
		values[k] = x
	}
	for _, a := range v.m.Addresses {
		buf := m.buffer(v, a.Argument)
		base, ok := buf.DeviceAddress()
		if !ok {
			return nil, errors.New("iron: the device does not expose buffer addresses")
		}
		addr := base + uint64(a.Offset)
		switch a.Member {
		case "base_lo":
			values[a.Name] = int64(addr & (1<<a.LowBits - 1))
		case "base_hi":
			values[a.Name] = int64(addr >> a.LowBits)
		default:
			return nil, fmt.Errorf("iron: address value %s of %s", a.Name, a.Member)
		}
	}
	return values, nil
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
	values, err := m.complete(v, callValues(token, position, 0, 1))
	if err != nil {
		return nil, err
	}
	if err := m.fillHostTables([]int32{token}); err != nil {
		return nil, err
	}
	if v.steps != nil {
		if err := m.carry(v, values); err != nil {
			return nil, err
		}
		if err := m.runSteps(ctx, v, values); err != nil {
			return nil, err
		}
		return m.logits(v)
	}
	if v.run == nil { // dropped for a tower's control code (dropDecode)
		if v.run, err = m.newRun(v, 0); err != nil {
			return nil, err
		}
	}
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
	for begin := 0; begin < len(tokens); {
		end := min(begin+C, len(tokens))
		if len(m.Manifest.Checkpoints) > 0 {
			// a call ends within the chunk it starts in: its state's slot is the chunk's
			end = min(end, begin+C-(start+begin)%C)
		}
		var err error
		if logits, err = m.chunk(ctx, tokens[begin:end], start+begin); err != nil {
			return nil, err
		}
		begin = end
	}
	return logits, nil
}

func (m *Model) chunk(ctx context.Context, ids []int32, start int) ([]float32, error) {
	return m.chunkOn(ctx, m.prompt, ids, start, nil)
}

// PromptPlaced runs a chunk of up to prefill_chunk tokens holding placeholders of the towers'
// soft tokens (Multimodal.Placeholders), which the towers have run for (RunVision,
// RunAudio). before counts each placeholder in the sequence before start.
func (m *Model) PromptPlaced(ctx context.Context, ids []int32, start int, before []int32) ([]float32, error) {
	if m.placed == nil {
		return nil, errors.New("iron: the export has no placed prompt chunk (export the multimodal model)")
	}
	if len(ids) == 0 || len(ids) > m.Manifest.Config.PrefillChunk || start+len(ids) > m.Manifest.Config.MaxSeqLen {
		return nil, fmt.Errorf("iron: a placed chunk of %d tokens from %d", len(ids), start)
	}
	return m.chunkOn(ctx, m.placed, ids, start, before)
}

func (m *Model) chunkOn(ctx context.Context, v *version, ids []int32, start int, before []int32) ([]float32, error) {
	rows := len(ids)
	values, err := m.complete(v, callValues(ids[rows-1], start+rows-1, start, rows))
	if err != nil {
		return nil, err
	}
	if before != nil {
		if err := m.writeInput(v, "before", int32Bytes(before)); err != nil {
			return nil, err
		}
	}
	// An input of token ids without a host encoder takes the ids themselves (the placed chunk's
	// placed: the device copies them and gathers their rows itself), zero past the rows.
	for _, in := range v.m.Inputs {
		if in.Name == "before" || slices.ContainsFunc(v.m.Encoders, func(e Encoder) bool { return e.Input == in.Name }) {
			continue
		}
		if in.DType != "int32" {
			return nil, fmt.Errorf("iron: input %s of %s is not token ids", in.Name, in.DType)
		}
		if err := m.writeInput(v, in.Name, int32Bytes(ids)); err != nil {
			return nil, err
		}
	}
	for _, e := range v.m.Encoders {
		if err := m.encode(v, &e, ids, values); err != nil {
			return nil, err
		}
	}
	if err := m.carry(v, values); err != nil {
		return nil, err
	}
	if err := m.fillHostTables(ids); err != nil {
		return nil, err
	}
	if v.steps != nil {
		if err := m.runSteps(ctx, v, values); err != nil {
			return nil, err
		}
		return m.logits(v)
	}
	timing := os.Getenv("HWNPU_IRON_TIMING") != ""
	for k := range v.m.Kernels {
		t0 := time.Now()
		run, err := m.newRun(v, k)
		if errors.Is(err, syscall.ENOSPC) && m.decode.run != nil {
			// the device heap holds no more control code: release the decode step's (as IRON
			// releases a run), which Step makes again
			m.dropDecode()
			run, err = m.newRun(v, k)
		}
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
	v := m.version(version)
	if v == nil {
		return nil, Placement{}, fmt.Errorf("iron: no version %s", version)
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
	if v := m.version(version); v != nil {
		return v.m.Buffers
	}
	return nil
}

// Device is the device the model is loaded on.
func (m *Model) Device() *hwnpu.Device { return m.dev }

func int32Bytes(xs []int32) []byte {
	b := make([]byte, 4*len(xs))
	for i, x := range xs {
		binary.LittleEndian.PutUint32(b[4*i:], uint32(x))
	}
	return b
}

// writeInput writes data at the start of a named input of v.
func (m *Model) writeInput(v *version, name string, data []byte) error {
	for _, in := range v.m.Inputs {
		if in.Name != name {
			continue
		}
		if len(data) > in.Bytes {
			return fmt.Errorf("iron: %d bytes for input %s of %d", len(data), name, in.Bytes)
		}
		buf := m.buffer(v, in.Argument)
		dst := buf.Bytes()[in.Offset : in.Offset+in.Bytes]
		copy(dst, data)
		clear(dst[len(data):])
		return buf.SyncToDevice(in.Offset, in.Bytes)
	}
	return fmt.Errorf("iron: no input %s", name)
}

// dropDecode releases the decode step's run, whose control code (21 MiB for Gemma 4 E2B) a
// tower's (up to 40 MiB) may not fit beside in the device heap; Step makes it again.
func (m *Model) dropDecode() {
	if m.decode != nil && m.decode.run != nil {
		m.decode.run.Close()
		m.decode.run = nil
	}
}

// runTower runs every command of a tower version on its input, with the grid's words at row.
func (m *Model) runTower(ctx context.Context, v *version, input string, data []byte, own map[string]int64, row int) error {
	values, err := m.complete(v, own)
	if err != nil {
		return err
	}
	if err := m.writeInput(v, input, data); err != nil {
		return err
	}
	if v.steps != nil {
		// IRON runs the towers on NPU2 only, as full ELFs: their grid words are scratchpad slots.
		return errors.New("iron: a tower packaged as an xclbin per step is not supported")
	}
	m.dropDecode()
	for k := range v.m.Kernels {
		run, err := m.newRun(v, k)
		if err != nil {
			return fmt.Errorf("iron: tower command %d: %w", k, err)
		}
		err = writeGridWords(run.Scratchpad(), v.m.Kernels[k].Parameters, values, v.m.Grid, row)
		if err == nil {
			err = run.Run(ctx)
		}
		run.Close()
		if err != nil {
			return fmt.Errorf("iron: tower command %d: %w", k, err)
		}
	}
	return nil
}
