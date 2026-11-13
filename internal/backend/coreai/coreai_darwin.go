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
	"regexp"
	"sort"
	"strings"
	"sync"
	"syscall"
	"unsafe"

	"github.com/ebitengine/purego"

	"github.com/shibukawa/hwnpu/internal/backend"
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
	// caiForget deletes Core AI's cache entries of an asset; nil with a shim from before it
	caiForget func(path *byte)
)

// kindNeuralEngine is the shim's specialization: prefer the Neural Engine.
const kindNeuralEngine = 2

// lowDisk is the free disk below which a program off the Neural Engine is put down to it.
const lowDisk = 10 << 30

// kindNoRetry is a flag of the shim's kind: on a failure, do not delete Core AI's cache entries
// and load again (an asset whose bytecode is cut, which would make Core AI parse nothing and
// abort).
const kindNoRetry = 0x100

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
		if _, err := purego.Dlsym(h, "cai_forget"); err == nil {
			purego.RegisterLibFunc(&caiForget, h, "cai_forget")
		}
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

// asset is the Core AI program of the package at path and its function: lowered once into the
// compile cache, with the packages beside it of the same weights (resolve), or alone into a
// temporary directory without a cache. stored is the cache's entry (the asset, "" in a temporary
// directory): what Compile may cut.
func (d *device) asset(path string) (dir, fn string, cached bool, owned, stored string, err error) {
	if d.opts.CacheDir != "" {
		dir, fn, cached, err = resolve(d.opts.CacheDir, path)
		return dir, fn, cached, "", dir, err
	}
	tmp, err := os.MkdirTemp("", "hwnpu-coreai-*.aimodel")
	if err != nil {
		return "", "", false, "", "", err
	}
	if err := lowerOne(path, tmp); err != nil {
		os.RemoveAll(tmp)
		return "", "", false, "", "", err
	}
	return tmp, "main", false, tmp, "", nil
}

// DropBytecodeEnv set to 1 cuts the bytecode of the compile cache's assets once Core AI's cache
// holds their programs: the last resort when the disk runs short (design.md 16.174 keeps them by
// default). The bytecode holds the weights, and Core AI's entry holds them again; Core AI loads a
// program from its entry by main.hash alone (Chroma1-Flash at 512 x 512: the compile cache 28 ->
// 15 GB, a DiT group loaded in 0.5 s rather than 6.5 s). When the entry is gone, the program is
// lowered again from its package, and Core AI compiles it as the first time.
const DropBytecodeEnv = "HWNPU_COREAI_DROP_BYTECODE"

// Parsing a cut bytecode aborts the process, so a cut asset (whether cut now or before) is opened
// only while its entry is there (hasEntry), with kindNoRetry; otherwise it is lowered again.

// isCut reports whether the asset's bytecode was cut.
func isCut(dir string) bool {
	fi, err := os.Stat(filepath.Join(dir, "main.mlirb"))
	return err == nil && fi.Size() == 0
}

// hasEntry reports whether Core AI's cache holds the program of the asset at dir for this
// process: its entries are under the OS's build and the executable's name
// (~/Library/Caches/coreai-cache/26B123/hwnpu-image/HASH/DEVICE/model.aimodelx). An entry
// without main-this.odix makes Core AI compile the bytecode again (and abort on a cut one); one
// without its delegates fails to load, and is lowered again. Anything it cannot read says no.
func hasEntry(dir string) bool {
	hash, err := os.ReadFile(filepath.Join(dir, "main.hash"))
	if err != nil {
		return false
	}
	build, err := syscall.Sysctl("kern.osversion")
	if err != nil || build == "" {
		return false
	}
	exe, err := os.Executable()
	if err != nil {
		return false
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return false
	}
	entries, _ := filepath.Glob(filepath.Join(home, "Library", "Caches", "coreai-cache", build, filepath.Base(exe), hex.EncodeToString(hash), "*", "model.aimodelx", "main-this.odix"))
	for _, e := range entries {
		if fi, err := os.Stat(e); err == nil && fi.Mode().IsRegular() && fi.Size() > 0 {
			return true
		}
	}
	return false
}

// cut cuts the asset's bytecode once Core AI's cache holds its program. The file is replaced, not
// truncated: Core AI may still map the old one.
func cut(dir string) error {
	if isCut(dir) || !hasEntry(dir) {
		return nil
	}
	p := filepath.Join(dir, "main.mlirb")
	tmp := p + ".cut"
	if err := os.WriteFile(tmp, nil, 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, p); err != nil {
		os.Remove(tmp)
		return err
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
	path, fn := splitFunction(src.Path)
	dir, cached, owned, stored := path, true, "", ""
	if !strings.EqualFold(filepath.Ext(path), ".aimodel") {
		var err error
		if dir, fn, cached, owned, stored, err = d.asset(path); err != nil {
			return nil, err
		}
	}
	if stored == "" && isCut(dir) && !hasEntry(dir) {
		return nil, fmt.Errorf("coreai: %s: the bytecode was cut and Core AI's cache entry is gone: lower the package again", src.Path)
	}
	g := &graph{cached: cached, owned: owned}
	open := func() string {
		kind := int32(kindNeuralEngine)
		if isCut(dir) {
			kind |= kindNoRetry
		}
		name := dir
		if fn != "main" {
			name += "#" + fn
		}
		var retried int32
		var e *byte
		g.h = caiOpen(cString(name), kind, &retried, &e)
		if g.h == 0 {
			return goString(e)
		}
		if retried == 1 {
			// Core AI had an entry that failed to load (aned purged its compiled program when the
			// disk ran short): dropped and compiled again; e is the first attempt's error
			msg := goString(e)
			if os.Getenv("HWNPU_COREAI_DEBUG") != "" {
				fmt.Fprintf(os.Stderr, "coreai: %s: the cached entry failed (%s), compiled again\n", filepath.Base(src.Path), msg)
			}
		}
		return ""
	}
	msg := open()
	if g.h == 0 && stored != "" && isCut(stored) {
		// the entry would not load: lowered again, and loaded as the first time
		if os.RemoveAll(stored) == nil {
			var err error
			if dir, fn, g.cached, _, stored, err = d.asset(path); err != nil {
				return nil, err
			}
			msg = open()
		}
	}
	if g.h == 0 {
		g.Close()
		return nil, fmt.Errorf("coreai: load %s: %s", src.Path, msg)
	}
	if stored != "" && os.Getenv(DropBytecodeEnv) == "1" {
		cut(stored) // best effort: an asset left whole only takes more disk
	}
	if err := g.describe(dir, fn); err != nil {
		g.Close()
		return nil, err
	}
	var offNPU []string
	g.placement, offNPU = placement(dir, fn)
	if len(offNPU) > 0 {
		// Core AI caches a program that the Neural Engine's compiler failed on (out of disk or
		// memory: "Failed to allocate memory for file backing") as it fell back, on the GPU, and
		// would keep it there: dropped, the next load compiles it again
		for _, e := range offNPU {
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

// splitFunction splits PROGRAM.aimodel#NAME, the function NAME of a program with several
// (ailower.LowerFunctions), into the program's path and NAME; main for any other path.
func splitFunction(path string) (string, string) {
	if i := strings.LastIndexByte(path, '#'); i > 0 && strings.EqualFold(filepath.Ext(path[:i]), ".aimodel") && i+1 < len(path) {
		return path[:i], path[i+1:]
	}
	return path, "main"
}

// describe makes the arguments, inputs then outputs each sorted by name (as the Core ML
// backend), from the program's features.
func (g *graph) describe(dir, fn string) error {
	ins, outs, err := ailower.ReadFunctionIO(dir, fn)
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
// whether everything of the function fn runs on the Neural Engine. When nothing says so, it
// returns the entries (model.aimodelx) whose program is off the Neural Engine too.
func placement(dir, fn string) (types.PlacementReport, []string) {
	rep := types.PlacementReport{Method: "coreai-cache-manifest"}
	hash, err := os.ReadFile(filepath.Join(dir, "main.hash"))
	if err != nil {
		rep.Reason = err.Error()
		return rep, nil
	}
	home, _ := os.UserHomeDir()
	pattern := filepath.Join(home, "Library", "Caches", "coreai-cache", "*", "*", hex.EncodeToString(hash), "*", "model.aimodelx")
	entries, _ := filepath.Glob(pattern)
	if len(entries) == 0 {
		rep.Reason = "no Core AI cache entry for the program"
		return rep, nil
	}
	rep.Verified = true
	for _, e := range entries {
		// the delegates are named after main.mlirb, whatever functions it has
		m, err := os.ReadFile(filepath.Join(e, "main-this-delegates", "MPSGraph", "mpsExecutable.mpsgraphpackage", "manifest.plist"))
		if err == nil && fullyPlaced(m, fn) {
			return rep, nil
		}
	}
	rep.Ops = []types.OpPlacement{{Name: "program", Type: "program", Device: "gpu"}}
	return rep, entries
}

// entryKey is a key of the manifest's Entry Function Attributes: a function's name, a 22
// character hash and a number (f8_Sfb_LWKqZ7kLACNxv16tNg_1).
var entryKey = regexp.MustCompile(`<key>[A-Za-z0-9_]+_[A-Za-z0-9_-]{22}_[0-9]+</key>`)

// fullyPlaced reports whether the manifest marks the function fn fully placed on the Neural
// Engine: mps.fullyPlacedOnANE between its entry's key and the next entry's. A manifest without
// such an entry is fully placed when it has the mark anywhere (as with one function).
func fullyPlaced(manifest []byte, fn string) bool {
	const mark = "mps.fullyPlacedOnANE"
	start := regexp.MustCompile(`<key>` + regexp.QuoteMeta(fn) + `_[A-Za-z0-9_-]{22}_[0-9]+</key>`).FindIndex(manifest)
	if start == nil {
		return bytes.Contains(manifest, []byte(mark))
	}
	rest := manifest[start[1]:]
	if next := entryKey.FindIndex(rest); next != nil {
		rest = rest[:next[0]]
	}
	return bytes.Contains(rest, []byte(mark))
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
