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
// temporary directory without a cache.
func (d *device) asset(path string) (dir, fn string, cached bool, owned string, err error) {
	if d.opts.CacheDir != "" {
		dir, fn, cached, err = resolve(d.opts.CacheDir, path)
		return dir, fn, cached, "", err
	}
	tmp, err := os.MkdirTemp("", "hwnpu-coreai-*.aimodel")
	if err != nil {
		return "", "", false, "", err
	}
	if err := lowerOne(path, tmp); err != nil {
		os.RemoveAll(tmp)
		return "", "", false, "", err
	}
	return tmp, "main", false, tmp, nil
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
	dir, fn := splitFunction(src.Path)
	cached, owned := true, ""
	if !strings.EqualFold(filepath.Ext(dir), ".aimodel") {
		var err error
		if dir, fn, cached, owned, err = d.asset(dir); err != nil {
			return nil, err
		}
	}
	g := &graph{cached: cached, owned: owned}
	var retried int32
	var e *byte
	open := dir
	if fn != "main" {
		open += "#" + fn
	}
	g.h = caiOpen(cString(open), kindNeuralEngine, &retried, &e)
	if g.h == 0 {
		msg := goString(e)
		g.Close()
		return nil, fmt.Errorf("coreai: load %s: %s", src.Path, msg)
	}
	if err := g.describe(dir, fn); err != nil {
		g.Close()
		return nil, err
	}
	g.placement = placement(dir, fn)
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
// whether everything of the function fn runs on the Neural Engine.
func placement(dir, fn string) types.PlacementReport {
	rep := types.PlacementReport{Method: "coreai-cache-manifest"}
	hash, err := os.ReadFile(filepath.Join(dir, "main.hash"))
	if err != nil {
		rep.Reason = err.Error()
		return rep
	}
	home, _ := os.UserHomeDir()
	pattern := filepath.Join(home, "Library", "Caches", "coreai-cache", "*", "*", hex.EncodeToString(hash), "*", "model.aimodelx")
	entries, _ := filepath.Glob(pattern)
	if len(entries) == 0 {
		rep.Reason = "no Core AI cache entry for the program"
		return rep
	}
	rep.Verified = true
	for _, e := range entries {
		// the delegates are named after main.mlirb, whatever functions it has
		m, err := os.ReadFile(filepath.Join(e, "main-this-delegates", "MPSGraph", "mpsExecutable.mpsgraphpackage", "manifest.plist"))
		if err == nil && fullyPlaced(m, fn) {
			return rep
		}
	}
	rep.Ops = []types.OpPlacement{{Name: "program", Type: "program", Device: "gpu"}}
	return rep
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
