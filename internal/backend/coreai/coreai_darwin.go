// Package coreai implements the Apple backend on Core AI (macOS 27 and later), through a small
// Swift shim (shim/shim.swift, embedded as libcaishim.dylib and loaded with purego). It takes the
// same programs as the Core ML backend, .mlpackage directories of MIL, and lowers them to Core AI
// programs (internal/coreai/ailower), kept in the compile cache. Core AI specializes a program for
// the device on its first load and keeps the result in its own cache
// (~/Library/Caches/coreai-cache), which holds far more than Core ML's Neural Engine cache.
//
// Placement is read from that cache: the MPSGraph delegate's manifest marks a program run
// entirely on the Neural Engine (mps.fullyPlacedOnANE).
package coreai

import (
	"bytes"
	"context"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"syscall"
	"unsafe"

	"github.com/ebitengine/purego"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/coreai/ailower"
	"github.com/shibukawa/hwnpu/internal/mem"
	"github.com/shibukawa/hwnpu/internal/types"
)

//go:embed libcaishim.dylib
var shimLib []byte

var (
	loadOnce sync.Once
	loadErr  error

	caiArch  func() *byte
	caiFree  func(*byte)
	caiOpen  func(path *byte, kind int32, retried *int32, err **byte) uintptr
	caiClose func(h uintptr)
	caiNames func(h uintptr, which int32) *byte
	caiRun   func(h uintptr, ins, outs *uintptr, err **byte) int32
)

// kindNeuralEngine is the shim's specialization: prefer the Neural Engine. kindCachedOnly loads
// from Core AI's cache alone, failing when the program is not there.
const (
	kindNeuralEngine = 2
	kindCachedOnly   = 0x10
	// lowDisk is the free disk below which a program off the Neural Engine is put down to it
	lowDisk = 10 << 30
)

func load() error {
	loadOnce.Do(func() {
		if _, err := os.Stat("/System/Library/Frameworks/CoreAI.framework"); err != nil {
			loadErr = fmt.Errorf("coreai: Core AI needs macOS 27 or later: %w", types.ErrUnsupported)
			return
		}
		sum := sha256.Sum256(shimLib)
		dir := filepath.Join(os.TempDir(), "hwnpu-coreai")
		if err := os.MkdirAll(dir, 0o755); err != nil {
			loadErr = err
			return
		}
		p := filepath.Join(dir, hex.EncodeToString(sum[:8])+".dylib")
		if _, err := os.Stat(p); err != nil {
			tmp := fmt.Sprintf("%s.%d.tmp", p, os.Getpid())
			if err := os.WriteFile(tmp, shimLib, 0o755); err != nil {
				loadErr = err
				return
			}
			if err := os.Rename(tmp, p); err != nil {
				loadErr = err
				return
			}
		}
		h, err := purego.Dlopen(p, purego.RTLD_NOW|purego.RTLD_LOCAL)
		if err != nil {
			loadErr = fmt.Errorf("coreai: load the shim: %w", err)
			return
		}
		purego.RegisterLibFunc(&caiArch, h, "cai_arch")
		purego.RegisterLibFunc(&caiFree, h, "cai_free")
		purego.RegisterLibFunc(&caiOpen, h, "cai_open")
		purego.RegisterLibFunc(&caiClose, h, "cai_close")
		purego.RegisterLibFunc(&caiNames, h, "cai_names")
		purego.RegisterLibFunc(&caiRun, h, "cai_run")
	})
	return loadErr
}

// goString copies a string the shim allocated (strdup) and frees it.
func goString(p *byte) string {
	if p == nil {
		return ""
	}
	n := 0
	for *(*byte)(unsafe.Add(unsafe.Pointer(p), n)) != 0 {
		n++
	}
	s := string(unsafe.Slice(p, n))
	caiFree(p)
	return s
}

func cString(s string) *byte { b := append([]byte(s), 0); return &b[0] }

// Arch is the device's architecture name for Core AI (h15g for an M3), "" when Core AI is not
// available.
func Arch() string {
	if load() != nil {
		return ""
	}
	return goString(caiArch())
}

// Provider is the Core AI backend provider.
type Provider struct{}

func (Provider) Name() string { return "coreai" }

func (Provider) Enumerate(ctx context.Context) ([]types.DeviceInfo, error) {
	if err := load(); err != nil {
		if errors.Is(err, types.ErrUnsupported) {
			return nil, nil
		}
		return nil, err
	}
	arch := goString(caiArch())
	if arch == "" {
		return nil, nil
	}
	return []types.DeviceInfo{{
		Vendor:  types.VendorApple,
		Backend: "coreai",
		Name:    "Apple Neural Engine (Core AI, " + arch + ")",
		Caps:    types.CapGraph | types.CapCompile | types.CapHostCoherent | types.CapStrictVerify,
		Driver:  "Core AI",
	}}, nil
}

func (Provider) Open(ctx context.Context, info types.DeviceInfo, opts backend.Options) (backend.Device, error) {
	if err := load(); err != nil {
		return nil, err
	}
	return &device{opts: opts}, nil
}

type device struct{ opts backend.Options }

func (d *device) AllocBuffer(size int, kind types.BufferKind) (backend.Buffer, error) {
	b, err := mem.Alloc(size)
	if err != nil {
		return nil, err
	}
	return &buffer{b: b}, nil
}

func (d *device) Close() error { return nil }

type buffer struct{ b []byte }

func (b *buffer) Bytes() []byte                   { return b.b }
func (b *buffer) SyncToDevice(off, n int) error   { return nil }
func (b *buffer) SyncFromDevice(off, n int) error { return nil }
func (b *buffer) Close() error {
	err := mem.Free(b.b)
	b.b = nil
	return err
}

// asset is the Core AI program of the package at path: lowered once into the compile cache (or a
// temporary directory without one), with its bytecode's hash. ours marks an asset of the compile
// cache, whose bytecode Compile empties once Core AI has cached the program (emptyBytecode).
type asset struct {
	dir    string
	cached bool
	ours   bool
	owned  string // a temporary directory to remove on Close
}

func lowerInto(path, dst string) error {
	p, err := ailower.LowerPackage(path)
	if err != nil {
		return err
	}
	return p.WriteAsset(dst)
}

func (d *device) asset(path string) (asset, error) {
	if strings.EqualFold(filepath.Ext(path), ".aimodel") {
		return asset{dir: path, cached: true}, nil
	}
	if d.opts.CacheDir == "" {
		tmp, err := os.MkdirTemp("", "hwnpu-coreai-*.aimodel")
		if err != nil {
			return asset{}, err
		}
		if err := lowerInto(path, tmp); err != nil {
			os.RemoveAll(tmp)
			return asset{}, err
		}
		return asset{dir: tmp, owned: tmp}, nil
	}
	digest, err := cache.TreeDigest(path)
	if err != nil {
		return asset{}, fmt.Errorf("coreai: %w", err)
	}
	key := cache.Key([]byte("hwnpu-coreai-v1"), digest, []byte(ailower.Producer))
	if p, ok := cache.Lookup(d.opts.CacheDir, key, ".aimodel"); ok {
		return asset{dir: p, cached: true, ours: true}, nil
	}
	p, err := cache.Store(d.opts.CacheDir, key, ".aimodel", func(dst string) error { return lowerInto(path, dst) })
	if err != nil {
		return asset{}, err
	}
	cache.SetSource(d.opts.CacheDir, key, path)
	return asset{dir: p, ours: true}, nil
}

// bytecodeFile is the asset's program, which holds the weights.
const bytecodeFile = "main.mlirb"

// emptied reports whether the asset's bytecode was emptied (emptyBytecode).
func emptied(dir string) bool {
	info, err := os.Stat(filepath.Join(dir, bytecodeFile))
	return err == nil && info.Size() == 0
}

// emptyBytecode empties the asset's bytecode once Core AI has cached the program: Core AI keys
// its cache by main.hash and reads the cached program alone, so the weights are not kept three
// times (the package, the asset, Core AI's cache: 17 GB each for Qwen-Image). The file is
// replaced, not truncated, so that a program loaded from it keeps its bytes.
func emptyBytecode(dir string) error {
	tmp := filepath.Join(dir, bytecodeFile+".tmp")
	if err := os.WriteFile(tmp, nil, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, filepath.Join(dir, bytecodeFile))
}

// relower writes the asset of an emptied bytecode again, from the package at path.
func relower(path, dir string) error {
	tmp := dir + ".relower.tmp"
	os.RemoveAll(tmp)
	defer os.RemoveAll(tmp)
	if err := lowerInto(path, tmp); err != nil {
		return err
	}
	ents, err := os.ReadDir(tmp)
	if err != nil {
		return err
	}
	for _, e := range ents {
		if err := os.Rename(filepath.Join(tmp, e.Name()), filepath.Join(dir, e.Name())); err != nil {
			return err
		}
	}
	return nil
}

type graph struct {
	h         uintptr
	args      []types.ArgInfo
	order     [2][]int // the shim's inputs and outputs, as indices into args
	placement types.PlacementReport
	cached    bool
	owned     string
	mu        sync.Mutex
}

func (d *device) Compile(ctx context.Context, src types.GraphSource) (backend.Graph, error) {
	if src.Format != types.GraphFormatNative || src.Path == "" {
		return nil, fmt.Errorf("coreai: only GraphFormatNative with Path (.mlpackage or .aimodel) is supported: %w", types.ErrUnsupported)
	}
	if src.Function != "" || src.Units != "" {
		return nil, fmt.Errorf("coreai: functions and compute units are not supported: %w", types.ErrUnsupported)
	}
	a, err := d.asset(src.Path)
	if err != nil {
		return nil, err
	}
	dir := a.dir
	g := &graph{cached: a.cached, owned: a.owned}
	var retried int32
	var e *byte
	stub := a.ours && emptied(dir)
	if stub {
		// Core AI's cache alone: without its entry (another OS, another executable, a purge)
		// the bytecode is written again and compiled
		g.h = caiOpen(cString(dir), kindNeuralEngine|kindCachedOnly, &retried, &e)
		if g.h == 0 {
			msg := goString(e)
			if os.Getenv("HWNPU_COREAI_DEBUG") != "" {
				fmt.Fprintf(os.Stderr, "coreai: %s: not from Core AI's cache (%d): %s\n", filepath.Base(src.Path), retried, msg)
			}
			if err := relower(src.Path, dir); err != nil {
				return nil, fmt.Errorf("coreai: lower %s again: %w", src.Path, err)
			}
			stub, g.cached = false, false
		}
	}
	if g.h == 0 {
		g.h = caiOpen(cString(dir), kindNeuralEngine, &retried, &e)
		if g.h != 0 && retried == 1 {
			// Core AI had an entry that failed to load: dropped and compiled again
			msg := goString(e)
			e = nil
			if os.Getenv("HWNPU_COREAI_DEBUG") != "" {
				fmt.Fprintf(os.Stderr, "coreai: %s: the cached entry failed (%s), compiled again\n", filepath.Base(src.Path), msg)
			}
		}
	}
	if g.h == 0 {
		msg := goString(e)
		g.Close()
		return nil, fmt.Errorf("coreai: load %s: %s", src.Path, msg)
	}
	if a.ours && !stub && os.Getenv("HWNPU_COREAI_KEEP_BYTECODE") == "" {
		if err := emptyBytecode(dir); err != nil {
			g.Close()
			return nil, fmt.Errorf("coreai: %w", err)
		}
	}
	if err := g.describe(dir); err != nil {
		g.Close()
		return nil, err
	}
	var entries []string
	g.placement, entries = placement(dir)
	if g.placement.Verified && !g.placement.OnNPU() {
		// Core AI caches a program that the Neural Engine's compiler failed on (out of disk or
		// memory: "Failed to allocate memory for file backing") as it fell back, on the GPU, and
		// the persistent cache would keep it there: dropped, the next load compiles it again
		for _, e := range entries {
			os.RemoveAll(filepath.Dir(filepath.Dir(e)))
		}
		// the Neural Engine's compiled program lives apart from Core AI's entry (aned's cache, a
		// few GB per large program, purged first when the disk fills): short of disk, every load
		// compiles again and falls to the GPU
		var st syscall.Statfs_t
		if syscall.Statfs(dir, &st) == nil {
			if free := st.Bavail * uint64(st.Bsize); free < lowDisk {
				fmt.Fprintf(os.Stderr, "coreai: %s fell back from the Neural Engine with %.1f GB of disk free: the Neural Engine's compiler needs a few GB per program\n", filepath.Base(src.Path), float64(free)/1e9)
			}
		}
	}
	return g, nil
}

// describe makes the arguments, inputs then outputs each sorted by name (as the Core ML
// backend), from the program's features.
func (g *graph) describe(dir string) error {
	ins, outs, err := ailower.ReadIO(dir)
	if err != nil {
		return fmt.Errorf("coreai: %w", err)
	}
	f := struct{ Inputs, Outputs []ailower.Feature }{ins, outs}
	byName := map[string]ailower.Feature{}
	for _, x := range append(append([]ailower.Feature{}, f.Inputs...), f.Outputs...) {
		byName[x.Name] = x
	}
	index := map[string]int{}
	for k, list := range [2][]ailower.Feature{f.Inputs, f.Outputs} {
		names := make([]string, len(list))
		for i, x := range list {
			names[i] = x.Name
		}
		sort.Strings(names)
		kind := types.ArgInput
		if k == 1 {
			kind = types.ArgOutput
		}
		for _, n := range names {
			x := byName[n]
			dt := map[string]types.DType{"f16": types.F16, "f32": types.F32, "i32": types.I32}[x.Elem]
			if dt == types.DTypeUnknown {
				return fmt.Errorf("coreai: %s %q of type %s: %w", kind, n, x.Elem, types.ErrUnsupported)
			}
			bytes := dt.Size()
			for _, v := range x.Shape {
				bytes *= v
			}
			index[n] = len(g.args)
			g.args = append(g.args, types.ArgInfo{Index: len(g.args), Kind: kind, Name: n, Dims: append([]int(nil), x.Shape...), DType: dt, ModelDType: dt, Bytes: bytes})
		}
	}
	for which := 0; which < 2; which++ {
		for _, n := range strings.Split(goString(caiNames(g.h, int32(which))), "\n") {
			i, ok := index[n]
			if !ok {
				return fmt.Errorf("coreai: the program has %q, not in its description", n)
			}
			g.order[which] = append(g.order[which], i)
		}
	}
	return nil
}

func (g *graph) Args() []types.ArgInfo            { return g.args }
func (g *graph) Cached() bool                     { return g.cached }
func (g *graph) Placement() types.PlacementReport { return g.placement }
func (g *graph) NewRequest() (backend.Request, error) {
	return &request{g: g, binds: make([]*buffer, len(g.args))}, nil
}

func (g *graph) Close() error {
	if g.h != 0 {
		caiClose(g.h)
		g.h = 0
	}
	if g.owned != "" {
		os.RemoveAll(g.owned)
	}
	return nil
}

// placement reads Core AI's cache entry of the program: its MPSGraph delegate's manifest says
// whether everything runs on the Neural Engine. It returns the entries too (model.aimodelx).
func placement(dir string) (types.PlacementReport, []string) {
	rep := types.PlacementReport{Method: "coreai-cache-manifest"}
	hash, err := os.ReadFile(filepath.Join(dir, "main.hash"))
	if err != nil {
		rep.Reason = err.Error()
		return rep, nil
	}
	home, _ := os.UserHomeDir()
	// the default cache (Library/Caches) and the persistent one (Library/Application Support,
	// where the shim specializes)
	var entries []string
	for _, root := range []string{filepath.Join(home, "Library", "Application Support"), filepath.Join(home, "Library", "Caches")} {
		e, _ := filepath.Glob(filepath.Join(root, "coreai-cache", "*", "*", hex.EncodeToString(hash), "*", "model.aimodelx"))
		entries = append(entries, e...)
	}
	if len(entries) == 0 {
		rep.Reason = "no Core AI cache entry for the program"
		return rep, nil
	}
	rep.Verified = true
	for _, e := range entries {
		m, err := os.ReadFile(filepath.Join(e, "main-this-delegates", "MPSGraph", "mpsExecutable.mpsgraphpackage", "manifest.plist"))
		if err == nil && bytes.Contains(m, []byte("mps.fullyPlacedOnANE")) {
			return rep, entries
		}
	}
	rep.Ops = []types.OpPlacement{{Name: "program", Type: "program", Device: "gpu"}}
	return rep, entries
}

type request struct {
	g     *graph
	binds []*buffer
	mu    sync.Mutex
}

func (r *request) Bind(index int, b backend.Buffer) error {
	cb, ok := b.(*buffer)
	if !ok {
		return errors.New("coreai: buffer was not allocated by this device")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.binds[index] = cb
	return nil
}

func (r *request) Submit() (backend.Fence, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for i, b := range r.binds {
		if b == nil {
			return nil, fmt.Errorf("coreai: argument %d (%s) is not bound", i, r.g.args[i].Name)
		}
	}
	ptrs := func(order []int) []uintptr {
		out := make([]uintptr, len(order)+1)
		for i, a := range order {
			out[i] = uintptr(unsafe.Pointer(&r.binds[a].b[0]))
		}
		return out
	}
	ins, outs := ptrs(r.g.order[0]), ptrs(r.g.order[1])
	r.g.mu.Lock()
	var e *byte
	rc := caiRun(r.g.h, &ins[0], &outs[0], &e)
	r.g.mu.Unlock()
	f := &fence{}
	if rc != 0 {
		f.err = fmt.Errorf("coreai: run: %s", goString(e))
	}
	return f, nil
}

func (r *request) Close() error { return nil }

type fence struct{ err error }

func (f *fence) Wait(ctx context.Context) error { return f.err }
func (f *fence) Done() (bool, error)            { return true, f.err }
