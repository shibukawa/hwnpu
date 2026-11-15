package iron

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/console"
)

// GraphManifest is manifest.json of a graph exported by tools/amd/iron_graph.py: one IRON graph
// called with inputs (an image model's text encoder, a denoiser step, a VAE decoder), not a
// language model.
type GraphManifest struct {
	Version int             `json:"version"`
	Kind    string          `json:"kind"` // "graph"
	Name    string          `json:"name"`
	Banks   map[string]Bank `json:"banks"`
	Graph   *Version        `json:"graph"`
	// Meta is the application's own description (shapes, what the inputs hold).
	Meta json.RawMessage `json:"meta"`
	// Variants are other images of the banks for the same program: by variant, a bank's
	// image by kind (a DiT's groups of blocks, one ELF with each group's weights). Variant 0
	// is the banks' own images.
	Variants []map[string]string `json:"variants"`
}

// ReadGraphManifest reads dir/manifest.json of a graph export.
func ReadGraphManifest(dir string) (*GraphManifest, error) {
	b, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return nil, err
	}
	var m GraphManifest
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, fmt.Errorf("iron: %s: manifest: %w", dir, err)
	}
	if m.Version != ManifestVersion || m.Kind != "graph" {
		return nil, fmt.Errorf("iron: %s: not a graph export (version %d, kind %q)", dir, m.Version, m.Kind)
	}
	v := m.Graph
	if v == nil || v.ELF == "" || len(v.Kernels) == 0 {
		return nil, fmt.Errorf("iron: %s: the graph is no full ELF", dir)
	}
	if len(v.Encoders) > 0 || v.Carried != nil {
		return nil, fmt.Errorf("iron: %s: a graph takes no gathers and carries nothing", dir)
	}
	return &m, nil
}

// PinnedBytes reads the graph export in dir and returns what OpenGraph pins for it with one set
// of banks (its banks and its other arguments) and, of that, its banks: what
// GraphOptions.Prefetch pins in addition.
func PinnedBytes(dir string) (pinned, banks int64, err error) {
	m, err := ReadGraphManifest(dir)
	if err != nil {
		return 0, 0, err
	}
	for _, b := range m.Banks {
		banks += int64(max(b.Bytes, 4096))
	}
	pinned = banks
	for _, a := range m.Graph.Arguments {
		if _, ok := m.Banks[a.Kind]; !ok {
			pinned += int64(max(a.Bytes, 4096))
		}
	}
	return pinned, banks, nil
}

// Graph is a loaded graph export: its banks, its program and its own buffers.
type Graph struct {
	Manifest *GraphManifest
	dev      *hwnpu.Device
	dir      string
	banks    map[string]*hwnpu.Buffer   // the variant in use
	variants []map[string]*hwnpu.Buffer // every variant's banks (one, with no variants or swapped)
	// swapped (GraphOptions.SwapVariants): one set of banks, the variant in use (cur) read into
	// them by Use
	swapped bool
	cur     int
	// prefetch (GraphOptions.Prefetch): a second set of banks, spare, the next variant read
	// into it while the commands run (pending, until Use takes it)
	prefetch bool
	spare    map[string]*hwnpu.Buffer
	pending  chan prefetched
	v        *version
	// keep (GraphOptions.KeepRuns): each command's run, made by the first Run
	keep bool
	runs []*hwnpu.Run
	// barrier (GraphOptions.Barrier): the ELF loaded again and its empty command
	barrier     bool
	lent        map[string]bool // the kinds whose buffers GraphOptions.Buffers lent (not closed)
	barrierProg *hwnpu.Program
	barrierRun  *hwnpu.Run
	// Timing, when set, is told each command's run time.
	Timing func(command int, run time.Duration)
	// After, when set, is called after each command of Run, before the next: a host step
	// between two commands, which reads and writes the graph's named buffers (Named). An
	// error ends the Run.
	After func(command int) error
}

// Named returns the host view of the buffer the export placed under name (a state the graph
// was given a name for: manifest's buffers), and syncs of it from and to the device.
func (g *Graph) Named(name string) (b []byte, fromDevice, toDevice func() error, err error) {
	p, ok := g.v.m.Buffers[name]
	if !ok {
		return nil, nil, nil, fmt.Errorf("iron: %s: no buffer %s", g.Manifest.Name, name)
	}
	buf := g.buffer(p.Argument)
	if buf == nil || p.Offset+p.Bytes > buf.Size() {
		return nil, nil, nil, fmt.Errorf("iron: %s: buffer %s is outside %s", g.Manifest.Name, name, p.Argument)
	}
	return buf.Bytes()[p.Offset : p.Offset+p.Bytes],
		func() error { return buf.SyncFromDevice(p.Offset, p.Bytes) },
		func() error { return buf.SyncToDevice(p.Offset, p.Bytes) }, nil
}

// Prefetch is the image pipelines' GraphOptions.Prefetch: on unless HWNPU_IRON_PREFETCH=0.
var Prefetch = os.Getenv("HWNPU_IRON_PREFETCH") != "0"

// GraphOptions are OpenGraphWith's choices.
type GraphOptions struct {
	// SwapVariants keeps one set of banks for all the variants and reads a variant's images
	// into them when Use selects it, instead of keeping every variant's banks on the device.
	// The banks' images are the weights the export placed first (iron_graph.py: the
	// parameters, then the work area); the work area is reused, as no call leaves anything
	// in it for the next. Anima's DiT (7 groups of 4 blocks at 1024 x 1024, q8): pinned
	// 7 x 464 MiB -> 464 MiB, at a read of 304 MiB a group (page cache to the bank).
	SwapVariants bool
	// Prefetch (with SwapVariants) keeps a second set of banks and reads the variant after the
	// one Use selected ((i + 1) % variants) into it while the commands run, so that Use of
	// that variant only swaps the sets: a DiT's groups are used in turn every step. Pinned
	// one set of banks more; the read (page cache to the bank, 10-18 GB/s) no longer waits.
	Prefetch bool
	// KeepRuns makes each command's run once and keeps it, its control code in the device
	// heap (64 MiB on Linux 7.0, shared by every program), instead of making it for each call:
	// for a small graph called often (a face detector on each frame of a camera). Making a run
	// allocates, copies and flushes the control code: YuNet at 320 x 256 took 1.6 ms of the
	// host's CPU a call, kept 1.1 ms (the rest is the submission and the wait in the kernel).
	KeepRuns bool
	// Barrier runs an empty command in a hardware context of its own (the ELF loaded a second
	// time) before each Run, as the language models' stepBarrier. Without it, a graph whose
	// commands write a state at a per-call position (Qwen-Image-2.1's dit_text: each chunk's
	// keys and values into the causal cache from start) hung the next Run's first command
	// (ERT_CMD_STATE_TIMEOUT); the first Run is always right.
	Barrier bool
	// Buffers lends device buffers by bank or argument kind (each at least as large as the
	// kind's): the graph uses them instead of allocating its own and does not close them. A
	// bank's is filled from its image and zeroed past it (the states a graph expects zero);
	// an argument's is taken as it is. Allocating a device buffer pins and zeroes its pages in
	// the kernel, 0.33 s a GiB, and closing it 0.11 s a GiB: a VAE's stages, each a graph with
	// work areas of 1 to 3 GiB, share one (anima.VAE). Not with variants.
	Buffers map[string]*hwnpu.Buffer
}

// ClearParallel zeroes b on several threads (a lent bank's work area: GiBs).
func ClearParallel(b []byte) {
	const part = 64 << 20
	if len(b) <= part {
		clear(b)
		return
	}
	n := min(8, (len(b)+part-1)/part)
	step := (len(b)/n + 4095) &^ 4095
	var wg sync.WaitGroup
	for o := 0; o < len(b); o += step {
		wg.Add(1)
		go func(c []byte) {
			defer wg.Done()
			clear(c)
		}(b[o:min(o+step, len(b))])
	}
	wg.Wait()
}

// OpenGraph loads the graph exported in dir onto dev.
func OpenGraph(ctx context.Context, dev *hwnpu.Device, dir string) (*Graph, error) {
	return OpenGraphWith(ctx, dev, dir, GraphOptions{})
}

// OpenGraphWith is OpenGraph with options.
func OpenGraphWith(ctx context.Context, dev *hwnpu.Device, dir string, opt GraphOptions) (*Graph, error) {
	defer console.Loading(dir)()
	man, err := ReadGraphManifest(dir)
	if err != nil {
		return nil, err
	}
	g := &Graph{Manifest: man, dev: dev, dir: dir, keep: opt.KeepRuns, barrier: opt.Barrier}
	fail := func(err error) (*Graph, error) {
		g.Close()
		return nil, fmt.Errorf("iron: %s: %w", man.Name, err)
	}
	images := man.Variants
	if len(images) == 0 {
		images = []map[string]string{nil}
	}
	if opt.SwapVariants && len(images) > 1 {
		g.swapped = true
		g.prefetch = opt.Prefetch
		images = images[:1]
	}
	lent := func(kind string, n int) *hwnpu.Buffer {
		if b := opt.Buffers[kind]; b != nil && b.Size() >= n && len(images) == 1 && !g.swapped {
			if g.lent == nil {
				g.lent = map[string]bool{}
			}
			g.lent[kind] = true
			return b
		}
		return nil
	}
	for i, img := range images {
		banks := map[string]*hwnpu.Buffer{}
		g.variants = append(g.variants, banks)
		for kind, b := range man.Banks {
			need := max(b.Bytes, 4096)
			buf := lent(kind, need)
			own := buf == nil
			if own {
				var err error
				if buf, err = dev.AllocBuffer(need); err != nil {
					return fail(fmt.Errorf("bank %s (%d MiB): %w", kind, b.Bytes>>20, err))
				}
			}
			banks[kind] = buf
			name := b.Image
			if s, ok := img[kind]; ok {
				name = s
			}
			read := 0
			if name != "" {
				var err error
				if read, err = readPrefix(filepath.Join(dir, name), buf.Bytes()[:need]); err != nil {
					return fail(fmt.Errorf("variant %d: bank %s: %w", i, kind, err))
				}
			}
			if !own { // what the graph before left there
				ClearParallel(buf.Bytes()[read:need])
			}
			if err := buf.SyncToDevice(0, need); err != nil {
				return fail(err)
			}
		}
	}
	g.banks = g.variants[0]
	if g.prefetch {
		g.spare = map[string]*hwnpu.Buffer{}
		for kind, b := range man.Banks {
			buf, err := dev.AllocBuffer(max(b.Bytes, 4096))
			if err != nil {
				return fail(fmt.Errorf("bank %s (%d MiB), the prefetch's: %w", kind, b.Bytes>>20, err))
			}
			g.spare[kind] = buf
		}
	}
	kr, err := dev.Kernel()
	if err != nil {
		return fail(err)
	}
	v := &version{m: man.Graph, args: map[string]*hwnpu.Buffer{}}
	g.v = v
	if v.prog, err = kr.Load(ctx, hwnpu.ProgramSource{Path: filepath.Join(dir, v.m.ELF)}); err != nil {
		return fail(err)
	}
	for _, a := range v.m.Arguments {
		if _, ok := g.banks[a.Kind]; ok {
			continue
		}
		buf := lent(a.Kind, max(a.Bytes, 4096))
		if buf == nil {
			if buf, err = dev.AllocBuffer(max(a.Bytes, 4096)); err != nil {
				return fail(fmt.Errorf("argument %s: %w", a.Kind, err))
			}
		}
		v.args[a.Kind] = buf
	}
	g.startPrefetch(1)
	return g, nil
}

// prefetched is what a prefetch read: the variant, its bytes, and its error.
type prefetched struct {
	variant int
	read    int64
	err     error
}

// startPrefetch reads variant i into the spare banks in the background (GraphOptions.Prefetch).
func (g *Graph) startPrefetch(i int) {
	if !g.prefetch || g.pending != nil {
		return
	}
	i %= len(g.Manifest.Variants)
	ch := make(chan prefetched, 1)
	g.pending = ch
	spare := g.spare
	go func() {
		n, err := g.readVariant(i, spare)
		ch <- prefetched{i, n, err}
	}()
}

// takePrefetch waits for the prefetch under way, if any, and returns it.
func (g *Graph) takePrefetch() (prefetched, bool) {
	if g.pending == nil {
		return prefetched{}, false
	}
	r := <-g.pending
	g.pending = nil
	return r, true
}

// readVariant reads variant i's images into banks and syncs them to the device.
func (g *Graph) readVariant(i int, banks map[string]*hwnpu.Buffer) (int64, error) {
	var read int64
	for kind, b := range g.Manifest.Banks {
		name := b.Image
		if s, ok := g.Manifest.Variants[i][kind]; ok {
			name = s
		}
		if name == "" {
			continue
		}
		buf := banks[kind]
		n, err := readPrefix(filepath.Join(g.dir, name), buf.Bytes())
		if err != nil {
			return read, fmt.Errorf("iron: %s: variant %d: bank %s: %w", g.Manifest.Name, i, kind, err)
		}
		if err := buf.SyncToDevice(0, n); err != nil {
			return read, err
		}
		read += int64(n)
	}
	return read, nil
}

// Close releases the device memory.
func (g *Graph) Close() error {
	var errs []error
	g.takePrefetch() // its buffers are closed below
	for _, b := range g.spare {
		errs = append(errs, b.Close())
	}
	g.spare = nil
	for _, r := range g.runs {
		if r != nil {
			errs = append(errs, r.Close())
		}
	}
	g.runs = nil
	if g.barrierRun != nil {
		errs = append(errs, g.barrierRun.Close())
		g.barrierRun = nil
	}
	if g.barrierProg != nil {
		errs = append(errs, g.barrierProg.Close())
		g.barrierProg = nil
	}
	if v := g.v; v != nil {
		for kind, b := range v.args {
			if !g.lent[kind] {
				errs = append(errs, b.Close())
			}
		}
		if v.prog != nil {
			errs = append(errs, v.prog.Close())
		}
	}
	for _, banks := range g.variants {
		for kind, b := range banks {
			if !g.lent[kind] {
				errs = append(errs, b.Close())
			}
		}
	}
	return errors.Join(errs...)
}

// Variants is the number of bank variants (1 without any).
func (g *Graph) Variants() int {
	if g.swapped {
		return len(g.Manifest.Variants)
	}
	return len(g.variants)
}

// Use selects the variant the next runs bind. With SwapVariants it reads the variant's images
// into the banks (when it is not the one there).
func (g *Graph) Use(i int) error {
	if i < 0 || i >= g.Variants() {
		return fmt.Errorf("iron: %s: no variant %d of %d", g.Manifest.Name, i, g.Variants())
	}
	if !g.swapped {
		g.banks = g.variants[i]
		return nil
	}
	if i == g.cur {
		return nil
	}
	var read int64
	done := console.Reading() // preparing, in the times and HWNPU_ENERGY's energy
	defer func() { done(read) }()
	if r, ok := g.takePrefetch(); ok && r.variant == i && r.err == nil {
		g.banks, g.spare = g.spare, g.banks
		g.variants[0] = g.banks // the set in use, as Close and Variants read it
		g.cur, read = i, r.read
		g.startPrefetch(i + 1)
		return nil
	}
	// No prefetch of i (or it failed): read it here. The spare banks may hold part of one.
	n, err := g.readVariant(i, g.banks)
	read = n
	if err != nil {
		g.cur = -1 // the banks hold part of one variant and part of another
		return err
	}
	g.cur = i
	g.startPrefetch(i + 1)
	return nil
}

// readPrefix reads the file at path into the start of dst and returns how many bytes it read.
// A large file is read by several threads: a bank's image comes from the page cache, a copy a
// thread makes at 5 to 7 GB/s (a DiT's groups are read again every step: FLUX.2 klein's 3.9 GB
// took 0.55 s a step on one thread).
func readPrefix(path string, dst []byte) (int, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	const part = 32 << 20
	n := len(dst)
	if st, err := f.Stat(); err == nil && st.Size() < int64(n) {
		n = int(st.Size())
	}
	if n <= part {
		n, err := io.ReadFull(f, dst)
		if errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF) {
			err = nil
		}
		return n, err
	}
	threads := min(8, (n+part-1)/part)
	step := (n/threads + 4095) &^ 4095
	errs := make([]error, threads)
	var wg sync.WaitGroup
	for t := 0; t < threads; t++ {
		lo, hi := t*step, min((t+1)*step, n)
		if lo >= hi {
			break
		}
		wg.Add(1)
		go func(t, lo, hi int) {
			defer wg.Done()
			for o := lo; o < hi; {
				k, err := f.ReadAt(dst[o:hi], int64(o))
				o += k
				if err != nil {
					if o < hi {
						errs[t] = fmt.Errorf("%s: read at %d: %w", path, o, err)
					}
					return
				}
			}
		}(t, lo, hi)
	}
	wg.Wait()
	return n, errors.Join(errs...)
}

func (g *Graph) buffer(kind string) *hwnpu.Buffer {
	if b, ok := g.banks[kind]; ok {
		return b
	}
	return g.v.args[kind]
}

func findHandle(hs []Handle, name string) (*Handle, bool) {
	for i := range hs {
		if hs[i].Name == name {
			return &hs[i], true
		}
	}
	return nil, false
}

// Inputs and Outputs are the graph's handles.
func (g *Graph) Inputs() []Handle  { return g.v.m.Inputs }
func (g *Graph) Outputs() []Handle { return g.v.m.Outputs }

// Input returns the host view of input name, which a caller fills before Run.
func (g *Graph) Input(name string) ([]byte, error) {
	h, ok := findHandle(g.v.m.Inputs, name)
	if !ok {
		return nil, fmt.Errorf("iron: %s: no input %s", g.Manifest.Name, name)
	}
	return g.buffer(h.Argument).Bytes()[h.Offset : h.Offset+h.Bytes], nil
}

// SetInput copies data into input name, zeros after it.
func (g *Graph) SetInput(name string, data []byte) error {
	dst, err := g.Input(name)
	if err != nil {
		return err
	}
	if len(data) > len(dst) {
		return fmt.Errorf("iron: %s: %d bytes for input %s of %d", g.Manifest.Name, len(data), name, len(dst))
	}
	copy(dst, data)
	clear(dst[len(data):])
	return nil
}

// Output returns output name as the device left it after Run.
func (g *Graph) Output(name string) ([]byte, error) {
	h, ok := findHandle(g.v.m.Outputs, name)
	if !ok {
		return nil, fmt.Errorf("iron: %s: no output %s", g.Manifest.Name, name)
	}
	buf := g.buffer(h.Argument)
	if err := buf.SyncFromDevice(h.Offset, h.Bytes); err != nil {
		return nil, err
	}
	return buf.Bytes()[h.Offset : h.Offset+h.Bytes], nil
}

// Run syncs the inputs to the device and runs every command with the per-call values.
func (g *Graph) Run(ctx context.Context, values map[string]int64) error {
	v := g.v
	if g.barrier {
		if err := g.runBarrier(ctx); err != nil {
			return err
		}
	}
	for _, h := range v.m.Inputs {
		if err := g.buffer(h.Argument).SyncToDevice(h.Offset, h.Bytes); err != nil {
			return err
		}
	}
	all := make(map[string]int64, len(v.m.Values)+len(values))
	for _, n := range v.m.Values {
		all[n] = 0
	}
	for k, x := range values {
		all[k] = x
	}
	for _, a := range v.m.Addresses {
		base, ok := g.buffer(a.Argument).DeviceAddress()
		if !ok {
			return errors.New("iron: the device does not expose buffer addresses")
		}
		addr := base + uint64(a.Offset)
		switch a.Member {
		case "base_lo":
			all[a.Name] = int64(addr & (1<<a.LowBits - 1))
		case "base_hi":
			all[a.Name] = int64(addr >> a.LowBits)
		default:
			return fmt.Errorf("iron: address value %s of %s", a.Name, a.Member)
		}
	}
	for k := range v.m.Kernels {
		kern, err := v.prog.Kernel(v.m.Kernels[k].Name)
		if err != nil {
			return err
		}
		var run *hwnpu.Run
		if g.keep && k < len(g.runs) && g.runs[k] != nil {
			run = g.runs[k]
		} else if run, err = kern.NewRun(); err != nil {
			return err
		}
		if g.keep {
			if len(g.runs) < len(v.m.Kernels) {
				g.runs = make([]*hwnpu.Run, len(v.m.Kernels))
			}
			g.runs[k] = run
		}
		for _, a := range v.m.Arguments {
			run.SetBuffer(a.Index, g.buffer(a.Kind))
		}
		err = run.Err()
		if err == nil {
			err = writeWords(run.Scratchpad(), v.m.Kernels[k].Parameters, all)
		}
		t0 := time.Now()
		if err == nil {
			err = run.Run(ctx)
		}
		if g.Timing != nil {
			g.Timing(k, time.Since(t0))
		}
		if !g.keep {
			run.Close()
		}
		if err != nil {
			return fmt.Errorf("iron: %s: command %d: %w", g.Manifest.Name, k, err)
		}
		if g.After != nil {
			if err := g.After(k); err != nil {
				return fmt.Errorf("iron: %s: after command %d: %w", g.Manifest.Name, k, err)
			}
		}
	}
	return nil
}

// runBarrier runs the empty command of the graph's ELF loaded a second time (GraphOptions.Barrier).
func (g *Graph) runBarrier(ctx context.Context) error {
	if g.barrierRun == nil {
		if g.barrierProg == nil {
			kr, err := g.dev.Kernel()
			if err != nil {
				return err
			}
			if g.barrierProg, err = kr.Load(ctx, hwnpu.ProgramSource{Path: filepath.Join(g.dir, g.v.m.ELF)}); err != nil {
				return fmt.Errorf("iron: %s: the barrier: %w", g.Manifest.Name, err)
			}
		}
		name := ""
		for _, k := range g.barrierProg.Kernels() {
			if strings.HasSuffix(k, ":aiecc_empty") {
				name = k
				break
			}
		}
		if name == "" {
			return fmt.Errorf("iron: %s: the barrier: the ELF has no empty instance", g.Manifest.Name)
		}
		kern, err := g.barrierProg.Kernel(name)
		if err != nil {
			return err
		}
		run, err := kern.NewRun()
		if err != nil {
			return err
		}
		for _, a := range g.v.m.Arguments {
			if a.Index < len(kern.Args()) {
				run.SetBuffer(a.Index, g.buffer(a.Kind))
			}
		}
		if err := run.Err(); err != nil {
			run.Close()
			return fmt.Errorf("iron: %s: the barrier: %w", g.Manifest.Name, err)
		}
		g.barrierRun = run
	}
	if err := g.barrierRun.Run(ctx); err != nil {
		return fmt.Errorf("iron: %s: the barrier: %w", g.Manifest.Name, err)
	}
	return nil
}
