package iron

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"syscall"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/f32"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3vl"
	"github.com/shibukawa/hwnpu/internal/vecops"
)

// errReleased is a call to a model that Release left without its device memory.
var errReleased = errors.New("iron: the model is released (Reload it first)")

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
	verify   *version  // the check of drafts (verify.go), or nil
	vision   *version  // the vision tower, or nil
	audio    *version  // the audio tower, or nil
	kv       []kvCache // caches, found once
	joins    []kvJoin  // the layers whose decode step keeps keys and values in one array (rows.go)
	stepped  [2]int    // the positions [from, to) decode steps wrote into those arrays alone
	// xclbins are the programs of the versions packaged as an xclbin per step, by file.
	xclbins map[string]*hwnpu.Program
	dir     string            // the export's directory
	vlProc  *lfm3vl.Processor // LFM3-VL's image processor, read on the first image
	// hostTables are the tables kept on the host (HostTable), filled before each call
	hostTables []*hostTable
	logf       func(string, ...any)
	released   bool      // by Release: nothing on the device until Reload
	parts      []float32 // logits: the partial sums of a row, decoded (kept between steps)
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
	// barrier is the decode step's empty command (chunkBarrier), kept.
	barrier *hwnpu.Run
	// stepProg is the decode ELF loaded again, a hardware context of its own, and stepRun its
	// empty command (stepBarrier).
	stepProg *hwnpu.Program
	stepRun  *hwnpu.Run
	// runs are a decode step's commands when it is several (Qwen3.6's: two a layer), each
	// kept as run is; runs[0] is run.
	runs []*hwnpu.Run
	// perCall: the check of drafts' commands made for each call (verify.go), their control
	// code not fitting the device heap beside the decode step's.
	perCall bool
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
	m := &Model{Manifest: man, dir: dir, logf: logf}
	if err := m.load(ctx, dev); err != nil {
		m.Close()
		return nil, err
	}
	if err := m.openHostTables(); err != nil {
		m.Close()
		return nil, err
	}
	return m, nil
}

// load allocates the arena's banks and the versions' buffers on dev, reads the weights into
// them and loads the programs: what Open does, and Reload again after Release.
func (m *Model) load(ctx context.Context, dev *hwnpu.Device) error {
	man, dir, logf := m.Manifest, m.dir, m.logf
	m.dev, m.banks, m.xclbins, m.kv = dev, map[string]*hwnpu.Buffer{}, map[string]*hwnpu.Program{}, nil
	if err := m.loadBanks(); err != nil {
		return err
	}
	kr, err := dev.Kernel()
	if err != nil {
		return err
	}
	for name, v := range man.Versions {
		ver := &version{m: v, args: map[string]*hwnpu.Buffer{}}
		switch name {
		case "decode":
			m.decode = ver // its run is made once every program's PDIs are in the heap
		case "prompt":
			m.prompt = ver
		case "placed":
			m.placed = ver
		case "verify":
			m.verify = ver
		case "vision":
			m.vision = ver
		case "audio":
			m.audio = ver
		default:
			return fmt.Errorf("iron: unknown version %s", name)
		}
		if v.ELF != "" {
			if ver.prog, err = kr.Load(ctx, hwnpu.ProgramSource{Path: filepath.Join(dir, v.ELF)}); err != nil {
				return fmt.Errorf("iron: %s: %w", name, err)
			}
		}
		for _, a := range v.Arguments {
			if _, ok := m.banks[a.Kind]; ok {
				continue
			}
			var opts []hwnpu.BufferOption
			if a.Kind == "feedback" && len(v.Fed) > 0 {
				// the fed commands' ctrl scratchpads: device heap memory, as the scratchpad
				// a run makes for itself (IRON's feedback is the next run's scratchpad)
				opts = append(opts, hwnpu.WithBufferKind(hwnpu.BufferInstructions))
			}
			buf, err := dev.AllocBuffer(max(a.Bytes, 4096), opts...)
			if err != nil {
				return fmt.Errorf("iron: %s: argument %s: %w", name, a.Kind, err)
			}
			ver.args[a.Kind] = buf
		}
		if len(v.Steps) > 0 {
			if err := m.loadSteps(ctx, kr, dir, ver); err != nil {
				return fmt.Errorf("iron: %s: %w", name, err)
			}
		}
		// where it runs, as chat's other loads say it (console.Log colors it): the host steps
		// (Qwen3.6's prompt experts, routed on the host) are work off the NPU
		where := "all on the NPU"
		if len(v.Host) > 0 {
			where = fmt.Sprintf("%d steps on the host, NOT all on the NPU", len(v.Host))
		}
		if len(v.Steps) > 0 {
			logf("%s: %d steps (an xclbin per step), %s", name, len(v.Steps), where)
		} else {
			logf("%s: %d commands, %s", name, len(v.Kernels), where)
		}
	}
	if m.decode == nil { // without a prompt chunk, Prompt runs the decode step a token at a time
		return errors.New("iron: the export needs a decode step")
	}
	if _, err := m.caches(); err != nil {
		return err
	}
	if m.decode.m.ELF != "" {
		if err := m.decodeRuns(); err != nil {
			return err
		}
	}
	m.released = false
	return nil
}

// Release frees everything the model holds on the device (the banks are pinned host memory
// that the process's resident size does not show: 1.8 to 9 GB for Gemma 4 E2B), keeping what
// the host needs to answer about the model and to Reload it: the manifest, the host tables.
// The caches and the states go with it, so a sequence starts again after Reload; one to come
// back to is copied out first (ExportRows, StateAt). The model must not run until Reload.
func (m *Model) Release() error {
	var errs []error
	for _, v := range m.versions() {
		if v == nil {
			continue
		}
		errs = append(errs, v.closeRuns()...)
		if v.barrier != nil {
			errs = append(errs, v.barrier.Close())
			v.barrier = nil
		}
		if v.stepRun != nil {
			errs = append(errs, v.stepRun.Close())
			v.stepRun = nil
		}
		if v.stepProg != nil {
			errs = append(errs, v.stepProg.Close())
			v.stepProg = nil
		}
		errs = append(errs, v.closeSteps()...)
		for k, b := range v.args {
			errs = append(errs, b.Close())
			delete(v.args, k)
		}
		if v.prog != nil {
			errs = append(errs, v.prog.Close())
			v.prog = nil
		}
	}
	for _, p := range m.xclbins {
		errs = append(errs, p.Close())
	}
	for _, b := range m.banks {
		errs = append(errs, b.Close())
	}
	m.xclbins, m.banks, m.kv, m.dev = nil, nil, nil, nil
	m.released = true
	return errors.Join(errs...)
}

// Released reports whether the model is released (Release) and not loaded again.
func (m *Model) Released() bool { return m.released }

// Reload loads the model released by Release onto dev again: the weights from the export, the
// caches and states at zero.
func (m *Model) Reload(ctx context.Context, dev *hwnpu.Device) error {
	if !m.released {
		return errors.New("iron: Reload of a model that is not released")
	}
	if err := m.load(ctx, dev); err != nil {
		m.Release()
		return err
	}
	return nil
}

// loadBanks allocates the arena's banks on m.dev and reads the weights into them: of an export
// with towers, only the start of each bank that the text's versions use (textBanks), until a
// tower runs (growBanks).
func (m *Model) loadBanks() error {
	text := m.Manifest.textBanks()
	for kind, b := range m.Manifest.Banks {
		n := b.Bytes
		if t, ok := text[kind]; ok {
			n = t
		}
		buf, err := m.dev.AllocBuffer(n)
		if err != nil {
			return fmt.Errorf("iron: bank %s (%d MiB): %w", kind, n>>20, err)
		}
		m.banks[kind] = buf
		if b.Image != "" {
			if err := readAt(filepath.Join(m.dir, b.Image), 0, buf.Bytes()); err != nil {
				return fmt.Errorf("iron: bank %s: %w", kind, err)
			}
		}
		if err := buf.SyncToDevice(0, buf.Size()); err != nil {
			return err
		}
		if n < b.Bytes {
			m.logf("bank %s: %d MiB, %d MiB more when a tower runs", kind, n>>20, (b.Bytes-n)>>20)
		} else {
			m.logf("bank %s: %d MiB", kind, n>>20)
		}
	}
	return nil
}

// growBanks gives the banks loadBanks left short their whole size, for a tower and the prompt
// chunks that read its soft tokens: a new buffer with the text's part as the device left it
// (its weights, and the caches and states of the sequence, which goes on) and the towers'
// weights read from the bank's image. The old bank and the decode step's run, bound to it, go;
// Step makes the run again. Until Release, the banks stay whole; Reload leaves the towers out
// again. For a moment both buffers are pinned (Gemma 4 E2B in 4 bits: 1.7 and 2.6 GiB).
func (m *Model) growBanks() error {
	if m.released {
		return errReleased
	}
	t0, grown := time.Now(), 0
	for kind, b := range m.Manifest.Banks {
		old := m.banks[kind]
		if old == nil || old.Size() >= b.Bytes {
			continue
		}
		n := old.Size()
		buf, err := m.dev.AllocBuffer(b.Bytes)
		if err != nil {
			return fmt.Errorf("iron: bank %s with the towers (%d MiB): %w", kind, b.Bytes>>20, err)
		}
		if err := old.SyncFromDevice(0, n); err != nil {
			buf.Close()
			return err
		}
		copy(buf.Bytes(), old.Bytes()[:n])
		if b.Image != "" {
			if err := readAt(filepath.Join(m.dir, b.Image), int64(n), buf.Bytes()[n:]); err != nil {
				buf.Close()
				return fmt.Errorf("iron: bank %s: %w", kind, err)
			}
		}
		if err := buf.SyncToDevice(0, buf.Size()); err != nil {
			buf.Close()
			return err
		}
		m.dropDecode()
		m.banks[kind] = buf
		if err := old.Close(); err != nil {
			return err
		}
		grown += b.Bytes - n
	}
	if grown > 0 {
		m.logf("towers loaded: %d MiB more in %.1f s", grown>>20, time.Since(t0).Seconds())
	}
	return nil
}

// textBanks is how much of each bank the text's versions (the decode step and the prompt
// chunk) use, for the banks they use only the start of: what loadBanks allocates of an export
// with towers until one runs. It is nil for an export without towers, one packaged as an
// xclbin per step (whose runs are bound to the banks), and when a text version places a buffer
// past what its argument says it uses.
func (man *Manifest) textBanks() map[string]int {
	if man.Versions["vision"] == nil && man.Versions["audio"] == nil {
		return nil
	}
	out := map[string]int{}
	for _, name := range []string{"decode", "prompt"} {
		v := man.Versions[name]
		if v == nil || len(v.Steps) > 0 {
			return nil
		}
		for _, a := range v.Arguments {
			if _, ok := man.Banks[a.Kind]; ok {
				out[a.Kind] = max(out[a.Kind], a.Bytes)
			}
		}
	}
	for _, name := range []string{"decode", "prompt"} {
		for _, p := range man.Versions[name].Buffers {
			if n, ok := out[p.Argument]; ok && p.Offset+p.Bytes > n {
				return nil
			}
		}
	}
	for kind, n := range out {
		if n <= 0 || n >= man.Banks[kind].Bytes {
			delete(out, kind)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// readAt fills dst from a file at off; past the file's end (an image without its trailing zeros)
// dst keeps the zeros of a new buffer.
func readAt(path string, off int64, dst []byte) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := f.ReadAt(dst, off); err != nil && !errors.Is(err, io.EOF) {
		return err
	}
	return nil
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

// Close releases the device memory and the host tables.
func (m *Model) Close() error {
	m.closeHostTables()
	if m.released {
		return nil
	}
	return m.Release()
}

func (m *Model) versions() []*version {
	return []*version{m.decode, m.prompt, m.placed, m.verify, m.vision, m.audio}
}

func (m *Model) version(name string) *version {
	switch name {
	case "decode":
		return m.decode
	case "prompt":
		return m.prompt
	case "placed":
		return m.placed
	case "verify":
		return m.verify
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
	b := buf.Bytes()[o.Offset : o.Offset+o.Bytes]
	if p := v.m.LogitParts; p > 0 { // two partial sums a row: [block][2][p]
		if len(m.parts) < o.Bytes/2 {
			m.parts = make([]float32, o.Bytes/2)
		}
		parts := m.parts[:o.Bytes/2]
		vecops.BF16ToF32(parts, b)
		out := make([]float32, min(o.Bytes/4, m.Manifest.Config.VocabSize))
		for i := 0; i < len(out); i += p {
			row := out[i:min(i+p, len(out))]
			copy(row, parts[2*i:])
			f32.Axpy(row, 1, parts[2*i+p:2*i+p+len(row)])
		}
		return out, nil
	}
	out := make([]float32, min(o.Bytes/2, m.Manifest.Config.VocabSize))
	vecops.BF16ToF32(out, b)
	return out, nil
}

// Step runs the decode step on token at position and returns the logits after it.
func (m *Model) Step(ctx context.Context, token int32, position int) ([]float32, error) {
	if m.released {
		return nil, errReleased
	}
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
		if err := m.decodeRuns(); err != nil {
			return nil, err
		}
	}
	if err := m.carry(v, values); err != nil {
		return nil, err
	}
	if len(v.runs) > 1 {
		// (a version on one resident array says it needs none: there the barrier's change of
		// hardware context puts the array's whole configuration back, 2.4 ms a step.
		// HWNPU_IRON_BARRIER=1 or 0 overrides the manifest, to measure.)
		barrier := !v.m.NoStepBarrier
		if e := os.Getenv("HWNPU_IRON_BARRIER"); e != "" {
			barrier = e != "0"
		}
		if barrier {
			if err := m.stepBarrier(ctx); err != nil {
				return nil, err
			}
		}
		if err := m.runCommands(ctx, v, values); err != nil {
			return nil, err
		}
		m.stepAt(position)
		return m.logits(v)
	}
	if err := writeWords(v.run.Scratchpad(), v.m.Kernels[0].Parameters, values); err != nil {
		return nil, err
	}
	if err := v.run.Run(ctx); err != nil {
		return nil, err
	}
	return m.logits(v)
}

// hostStep is v's host step of kind for layer, or nil.
func (v *version) hostStep(kind string, layer int) *HostStep {
	for i := range v.m.Host {
		if h := &v.m.Host[i]; h.Kind == kind && h.Layer == layer {
			return h
		}
	}
	return nil
}

// decodeRuns makes the decode step's runs, one a command, a fed command's scratchpad its
// buffer of the feedback argument.
func (m *Model) decodeRuns() error {
	v := m.decode
	fed := map[int]Fed{}
	for _, f := range v.m.Fed {
		fed[f.Command] = f
	}
	runs := make([]*hwnpu.Run, 0, len(v.m.Kernels))
	fail := func(err error) error {
		for _, r := range runs {
			r.Close()
		}
		return err
	}
	for k := range v.m.Kernels {
		run, err := m.newRun(v, k)
		if err != nil {
			return fail(fmt.Errorf("iron: decode command %d: %w", k, err))
		}
		runs = append(runs, run)
		f, ok := fed[k]
		if !ok {
			continue
		}
		p, ok := v.m.Buffers[f.Buffer]
		if !ok || p.Argument != "feedback" || 4*f.Words > p.Bytes {
			return fail(fmt.Errorf("iron: decode command %d is fed from %s, not a buffer of the feedback argument", k, f.Buffer))
		}
		// the whole buffer: the device writes its first Words words, the host the command's others
		if err := run.UseScratchpad(m.buffer(v, "feedback"), p.Offset, max(4*f.Words, p.Bytes)); err != nil {
			return fail(fmt.Errorf("iron: decode command %d: %w", k, err))
		}
	}
	v.run, v.runs = runs[0], runs
	return nil
}

// runCommands runs every command of v back to back: the host's words in each command the
// device does not feed, then all submitted before the first wait (a hardware context runs its
// commands in order, so a fed command reads what the one before it wrote).
func (m *Model) runCommands(ctx context.Context, v *version, values map[string]int64) error {
	// a fed command's Words words from First are the device's (the command before writes them:
	// the host must leave them alone); its others, if it has any, the host's
	fed := map[int]Fed{}
	for _, f := range v.m.Fed {
		fed[f.Command] = f
	}
	var fences []hwnpu.Fence
	var err error
	// The host's words of the fed commands, into their scratchpads (buffers of the feedback
	// argument) before any command runs: a host write of a word shares a cache line with the
	// device's words, which a command running meanwhile would write under it.
	for k, f := range fed {
		var params []Param
		for _, p := range v.m.Kernels[k].Parameters {
			if p.Index < f.First || p.Index >= f.First+f.Words {
				params = append(params, p)
			}
		}
		if len(params) == 0 {
			continue
		}
		p := v.m.Buffers[f.Buffer]
		buf := m.buffer(v, "feedback")
		if err := writeWords(buf.Bytes()[p.Offset:p.Offset+p.Bytes], params, values); err != nil {
			return err
		}
		if err := buf.SyncToDevice(p.Offset, p.Bytes); err != nil {
			return err
		}
	}
	// HWNPU_IRON_TIMING: each command waited for and timed alone
	timing := os.Getenv("HWNPU_IRON_TIMING") != ""
	for k, run := range v.runs {
		if _, ok := fed[k]; !ok {
			if err = writeWords(run.Scratchpad(), v.m.Kernels[k].Parameters, values); err != nil {
				break
			}
		}
		var f hwnpu.Fence
		if f, err = run.Submit(); err != nil {
			err = fmt.Errorf("iron: decode command %d: %w", k, err)
			break
		}
		if timing {
			t0 := time.Now()
			if err = f.Wait(ctx); err != nil {
				break
			}
			fmt.Fprintf(os.Stderr, "decode command %d: %.3f ms\n", k, time.Since(t0).Seconds()*1000)
			continue
		}
		fences = append(fences, f)
	}
	for k, f := range fences {
		if werr := f.Wait(ctx); werr != nil && err == nil {
			err = fmt.Errorf("iron: decode command %d: %w", k, werr)
		}
	}
	return err
}

// closeRuns closes v's kept runs.
func (v *version) closeRuns() []error {
	var errs []error
	if len(v.runs) == 0 && v.run != nil {
		errs = append(errs, v.run.Close())
	}
	for _, r := range v.runs {
		errs = append(errs, r.Close())
	}
	v.run, v.runs = nil, nil
	return errs
}

// Prompt runs tokens from position start in prompt chunks and returns the logits after the
// last one. The cache holds the tokens before start.
func (m *Model) Prompt(ctx context.Context, tokens []int32, start int) ([]float32, error) {
	C := m.Manifest.Config.PrefillChunk
	if len(tokens) == 0 || start+len(tokens) > m.Manifest.Config.MaxSeqLen {
		return nil, fmt.Errorf("iron: %d tokens from %d do not fit the cache of %d", len(tokens), start, m.Manifest.Config.MaxSeqLen)
	}
	if m.prompt == nil {
		// no prompt chunk (Qwen3.6's experts are routed a token at a time): decode steps
		var logits []float32
		for i, t := range tokens {
			var err error
			if logits, err = m.Step(ctx, t, start+i); err != nil {
				return nil, err
			}
		}
		return logits, nil
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
	if err := m.growBanks(); err != nil {
		return nil, err
	}
	if len(ids) == 0 || len(ids) > m.Manifest.Config.PrefillChunk || start+len(ids) > m.Manifest.Config.MaxSeqLen {
		return nil, fmt.Errorf("iron: a placed chunk of %d tokens from %d", len(ids), start)
	}
	return m.chunkOn(ctx, m.placed, ids, start, before)
}

func (m *Model) chunkOn(ctx context.Context, v *version, ids []int32, start int, before []int32) ([]float32, error) {
	if m.released {
		return nil, errReleased
	}
	rows := len(ids)
	timing := os.Getenv("HWNPU_IRON_TIMING") != ""
	if err := m.splitKV(); err != nil { // the decode steps' rows into the keys and values the chunk reads
		return nil, err
	}
	tHost := time.Now()
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
	if err := m.convPairsIn(v, start); err != nil {
		return nil, err
	}
	tBarrier := time.Now()
	if err := m.chunkBarrier(ctx); err != nil {
		return nil, err
	}
	if timing {
		fmt.Fprintf(os.Stderr, "prompt chunk (%d rows from %d): inputs and host tables %.1f ms, barrier %.1f ms\n", rows, start,
			tBarrier.Sub(tHost).Seconds()*1000, time.Since(tBarrier).Seconds()*1000)
	}
	routes := map[int]*moeRoute{} // by layer: its moe_route's, until its moe_combine
	for k := range v.m.Kernels {
		runOnce := func() error {
			t0 := time.Now()
			run, err := m.newRun(v, k)
			if errors.Is(err, syscall.ENOSPC) && (m.decode.run != nil || m.verify != nil && m.verify.run != nil) {
				// the device heap holds no more control code: release the decode step's (as IRON
				// releases a run) and the check of drafts', which Step and Verify make again
				m.dropDecode()
				run, err = m.newRun(v, k)
			}
			t1 := time.Now()
			if err != nil {
				return fmt.Errorf("iron: prompt command %d: %w", k, err)
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
				return fmt.Errorf("iron: prompt command %d: %w", k, err)
			}
			return nil
		}
		var route *HostStep
		for i := range v.m.Host {
			h := &v.m.Host[i]
			if h.Command != k {
				continue
			}
			th := time.Now()
			switch h.Kind {
			case "moe_route", "moe_route_mx4":
				plan := m.moeHostRoute
				if h.Kind == "moe_route_mx4" {
					plan = m.moeHostRouteMX4
				}
				rt, err := plan(v, h, rows)
				if err != nil {
					return nil, err
				}
				routes[h.Layer], route = rt, h
			case "moe_combine":
				if err := m.moeHostCombine(v, h, routes[h.Layer], rows); err != nil {
					return nil, err
				}
				delete(routes, h.Layer)
			default:
				return nil, fmt.Errorf("iron: a host step %s before prompt command %d", h.Kind, k)
			}
			if timing {
				fmt.Fprintf(os.Stderr, "host %s of layer %d: %.1f ms\n", h.Kind, h.Layer, time.Since(th).Seconds()*1000)
			}
		}
		if route == nil {
			if err := runOnce(); err != nil {
				return nil, err
			}
			continue
		}
		combine := v.hostStep("moe_combine", route.Layer)
		if combine == nil {
			return nil, fmt.Errorf("iron: layer %d's moe_route without a moe_combine", route.Layer)
		}
		tm := time.Now()
		if err := m.runMoECommand(ctx, v, routes[route.Layer], route, combine, runOnce, values); err != nil {
			return nil, err
		}
		if timing {
			fmt.Fprintf(os.Stderr, "moe command %d: %.1f ms in all\n", k, time.Since(tm).Seconds()*1000)
		}
	}
	if err := m.convPairsOut(v, start, rows); err != nil {
		return nil, err
	}
	if err := m.joinKV(start, start+rows); err != nil { // the chunk's rows into the decode step's arrays
		return nil, err
	}
	tLogits := time.Now()
	logits, err := m.logits(v)
	if timing {
		fmt.Fprintf(os.Stderr, "prompt chunk (%d rows from %d): logits %.1f ms\n", rows, start, time.Since(tLogits).Seconds()*1000)
	}
	return logits, err
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
	if p.Offset+p.Bytes > buf.Size() {
		return nil, p, fmt.Errorf("iron: %s's buffer %s is in the towers' part of the bank, loaded when a tower runs", version, name)
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

// chunkBarrier runs an empty command in the decode step's hardware context before a prompt
// chunk: one of the ELF's empty instances (mlir-aie's aiecc_empty, no control code), 0.3 to 0.5
// ms. Without it, a prompt chunk's first command can hang until amdxdna's timeout
// (ERT_CMD_STATE_TIMEOUT) right after the chunk before, in exports whose commands are cut in
// some places (design.md 16.80 and 16.84); the first chunk always runs, and so does a chunk after
// any other context's command (a decode step, another process's). The cause is not known; a
// pause between the chunks does not help. Nothing without an empty instance (an xclbin per
// step) runs one.
func (m *Model) chunkBarrier(ctx context.Context) error {
	v := m.decode
	if v == nil || v.prog == nil {
		return nil
	}
	if v.barrier == nil {
		run, err := m.emptyRun(v, v.prog)
		if err != nil || run == nil {
			return err
		}
		v.barrier = run
	}
	if err := v.barrier.Run(ctx); err != nil {
		return fmt.Errorf("iron: the chunk barrier: %w", err)
	}
	return nil
}

// emptyRun makes a run of prog's empty instance (mlir-aie's aiecc_empty, no control code) with
// v's arguments bound, or nil when prog has none.
func (m *Model) emptyRun(v *version, prog *hwnpu.Program) (*hwnpu.Run, error) {
	name := ""
	for _, k := range prog.Kernels() {
		if strings.HasSuffix(k, ":aiecc_empty") {
			name = k
			break
		}
	}
	if name == "" {
		return nil, nil
	}
	kern, err := prog.Kernel(name)
	if err != nil {
		return nil, err
	}
	run, err := kern.NewRun()
	if err != nil {
		return nil, err
	}
	// Never read (it moves nothing), but every argument is bound, as for any run.
	for _, a := range v.m.Arguments {
		if a.Index < len(kern.Args()) {
			run.SetBuffer(a.Index, m.buffer(v, a.Kind))
		}
	}
	if err := run.Err(); err != nil {
		run.Close()
		return nil, fmt.Errorf("iron: %s: %w", name, err)
	}
	return run, nil
}

// stepBarrier runs an empty command in a hardware context of its own (the decode ELF loaded a
// second time) before a decode step of several commands. Without it, Qwen3.6 with the short
// convolution in the GDN operator hung the next step's first command (ERT_CMD_STATE_TIMEOUT)
// right after the last layer's experts command, every time (design.md 16.90, 16.92); an empty
// command in the decode step's own context does not help, as a pause did not for chunkBarrier.
// About 0.3 ms a step.
func (m *Model) stepBarrier(ctx context.Context) error {
	v := m.decode
	if v.stepRun == nil {
		if v.stepProg == nil {
			kr, err := m.dev.Kernel()
			if err != nil {
				return err
			}
			if v.stepProg, err = kr.Load(ctx, hwnpu.ProgramSource{Path: filepath.Join(m.dir, v.m.ELF)}); err != nil {
				return fmt.Errorf("iron: the step barrier: %w", err)
			}
		}
		run, err := m.emptyRun(v, v.stepProg)
		if err != nil || run == nil {
			return err
		}
		v.stepRun = run
	}
	if err := v.stepRun.Run(ctx); err != nil {
		return fmt.Errorf("iron: the step barrier: %w", err)
	}
	return nil
}

// dropDecode releases the decode step's run, whose control code (21 MiB for Gemma 4 E2B) a
// tower's (up to 40 MiB) may not fit beside in the device heap; Step makes it again. The chunk
// barrier goes too: it is bound to the banks, which growBanks replaces (its run on the closed
// bank failed with ENOENT, the first prompt chunk after a tower); chunkBarrier makes it again.
// The check of drafts' kept runs go too (Verify makes them again).
func (m *Model) dropDecode() {
	if m.verify != nil {
		m.verify.closeRuns()
	}
	if m.decode == nil {
		return
	}
	m.decode.closeRuns()
	if m.decode.barrier != nil {
		m.decode.barrier.Close()
		m.decode.barrier = nil
	}
	if m.decode.stepRun != nil {
		m.decode.stepRun.Close()
		m.decode.stepRun = nil
	}
}

// runTower runs every command of a tower version on its input, with the grid's words at row.
func (m *Model) runTower(ctx context.Context, v *version, input string, data []byte, own map[string]int64, row int) error {
	if err := m.growBanks(); err != nil {
		return err
	}
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
