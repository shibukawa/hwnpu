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
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"unsafe"

	"github.com/ebitengine/purego"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/console"
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
	// caiInputArray makes an array of an input, whose storage the caller fills in place, and
	// caiArrayFree frees it; nil with a shim from before them
	caiInputArray func(h uintptr, index int32) unsafe.Pointer
	caiArrayFree  func(p unsafe.Pointer)
)

// kindNeuralEngine is the shim's specialization: prefer the Neural Engine. kindCPU: the CPU alone.
const (
	kindCPU          = 1
	kindNeuralEngine = 2
)

// UnitsEnv names the environment variable that moves the programs off the Neural Engine, for
// telling a lowering's errors from the Neural Engine's (and checking a lowering while another
// process has the Neural Engine): "cpu" runs them on Core AI's CPU alone. As Core ML's
// HWNPU_COREML_UNITS. A reduce_mean whose sum passes fp16's range (a mean over 57600 values
// above 1.1) overflows there, where the Neural Engine and the GPU are right (docs/design/vision-models.md).
const UnitsEnv = "HWNPU_COREAI_UNITS"

// CopyEnv names the environment variable that, when set, gives the programs' inputs buffers of
// the host's, copied into Core AI's arrays at every run, instead of those arrays themselves
// (graph.AllocInput): for telling what the copies cost, and whether an error is of the arrays.
const CopyEnv = "HWNPU_COREAI_COPY"

// OnCPU reports whether UnitsEnv asks for the CPU.
func OnCPU() bool { return os.Getenv(UnitsEnv) == "cpu" }

// lowDisk is the free disk below which a program off the Neural Engine is put down to it.
const lowDisk = 10 << 30

// kindNoRetry is a flag of the shim's kind: on a failure, do not delete Core AI's cache entries
// and load again (an asset whose bytecode is cut, which would make Core AI parse nothing and
// abort).
const kindNoRetry = 0x100

// prebuiltOnce says once a process that its programs come from prebuilt ones.
var prebuiltOnce sync.Once

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
		if _, err := purego.Dlsym(h, "cai_input_array"); err == nil {
			purego.RegisterLibFunc(&caiInputArray, h, "cai_input_array")
			purego.RegisterLibFunc(&caiArrayFree, h, "cai_array_free")
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

// buffer is the host's memory, or the storage of an array of Core AI's (array: graph.AllocInput).
type buffer struct {
	b     []byte
	array unsafe.Pointer
}

func (b *buffer) Bytes() []byte                   { return b.b }
func (b *buffer) SyncToDevice(off, n int) error   { return nil }
func (b *buffer) SyncFromDevice(off, n int) error { return nil }
func (b *buffer) Close() error {
	if b.array != nil {
		caiArrayFree(b.array)
		b.b, b.array = nil, nil
		return nil
	}
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
// holds their programs: the last resort when the disk runs short (docs/design/apple-coreai.md keeps them by
// default). The bytecode holds the weights, and Core AI's entry holds them again; Core AI loads a
// program from its entry by main.hash alone (Chroma1-Flash at 512 x 512: the compile cache 28 ->
// 15 GB, a DiT group loaded in 0.5 s rather than 6.5 s). When the entry is gone, the program is
// lowered again from its package, and Core AI compiles it as the first time.
const DropBytecodeEnv = "HWNPU_COREAI_DROP_BYTECODE"

// PrebuiltEnv set to off makes Core AI compile a program on the device even when a prebuilt one
// (coreai-build's .aimodelc, Prebuilt) lies beside it.
const PrebuiltEnv = "HWNPU_COREAI_PREBUILT"

// Prebuilt is the program coreai-build compiled for this device's architecture for the asset at
// dir, or "" without one of the same source (main.hash): X.ARCH.aimodelc beside X.aimodel
// (hwnpu-dev models prebuild), else the packed one beside the export (PackedPath, from the shared
// layers; Assemble makes it whole). Core AI loads it without the Neural Engine compiler's pass
// over the program: the device finishes the compiled regions (ANELargeModelCompilerService), 3
// to 5 times faster for a large program, and fills the same cache entry (the hash is the
// source's), so the asset itself loads from the entry afterwards (docs/design/apple-coreai.md). The program
// it holds runs a few percent slower than one compiled on the device.
func Prebuilt(dir string) string {
	if os.Getenv(PrebuiltEnv) == "off" || LeftToDevice(dir) {
		return ""
	}
	arch := Arch()
	if arch == "" {
		return ""
	}
	want, err := os.ReadFile(filepath.Join(dir, "main.hash"))
	if err != nil {
		return ""
	}
	for _, p := range []string{PrebuiltPath(dir, arch), PackedPath(dir, arch)} {
		if p == "" {
			continue
		}
		have, err := os.ReadFile(filepath.Join(p, "main.hash"))
		if err == nil && bytes.Equal(want, have) && onANE(p) {
			return p
		}
	}
	return ""
}

// fetchedPrebuilt are the exports FetchPrebuilt was called for, by export and architecture.
var fetchedPrebuilt sync.Map

// fetchPrebuilt asks the shared layers (FetchPrebuilt) once a process for the prebuilt programs
// of the export that holds the asset at dir, which Core AI would compile.
func fetchPrebuilt(ctx context.Context, dir string) {
	export, ok := ExportOf(dir)
	if !ok || FetchPrebuilt == nil || os.Getenv(PrebuiltEnv) == "off" {
		return
	}
	arch := Arch()
	if arch == "" {
		return
	}
	once, _ := fetchedPrebuilt.LoadOrStore(export+"\x00"+arch, new(sync.Once))
	once.(*sync.Once).Do(func() { FetchPrebuilt(ctx, export, arch) })
}

// openable is the prebuilt program pb for Core AI to open: pb, or, packed (its weights a map onto
// the asset's bytecode at dir), assembled in a temporary directory that done removes.
func openable(pb, dir string) (string, func(), error) {
	if !packed(pb) {
		return pb, func() {}, nil
	}
	cache.Touch(pb) // the export's prebuilt programs, a unit of the cache's LRU of their own
	tmp, err := os.MkdirTemp(filepath.Dir(dir), ".assemble-*")
	if err != nil {
		return "", nil, err
	}
	out := filepath.Join(tmp, filepath.Base(pb))
	if err := Assemble(pb, dir, out); err != nil {
		os.RemoveAll(tmp)
		return "", nil, err
	}
	return out, func() { os.RemoveAll(tmp) }, nil
}

// packed reports whether the prebuilt program at pb has a file as a map (Pack).
func packed(pb string) bool {
	found := false
	filepath.WalkDir(pb, func(p string, d fs.DirEntry, err error) error {
		if err == nil && !d.IsDir() && strings.HasSuffix(p, MapSuffix) {
			found = true
			return filepath.SkipAll
		}
		return nil
	})
	return found
}

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
	d := entryDir(dir)
	if d == "" {
		return false
	}
	entries, _ := filepath.Glob(filepath.Join(d, "*", "model.aimodelx", "main-this.odix"))
	for _, e := range entries {
		if fi, err := os.Stat(e); err == nil && fi.Mode().IsRegular() && fi.Size() > 0 {
			return true
		}
	}
	return false
}

// entryDir is the folder of Core AI's cache that holds the entries of the asset at dir for this
// process (hasEntry), "" when it cannot be told.
func entryDir(dir string) string {
	hash, err := os.ReadFile(filepath.Join(dir, "main.hash"))
	if err != nil {
		return ""
	}
	base := coreAICacheDir()
	exe, err := os.Executable()
	if base == "" || err != nil {
		return ""
	}
	return filepath.Join(base, filepath.Base(exe), hex.EncodeToString(hash))
}

// coreAICacheDir is Core AI's cache of this OS build, its entries under the executables' names.
func coreAICacheDir() string {
	build, err := syscall.Sysctl("kern.osversion")
	if err != nil || build == "" {
		return ""
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, "Library", "Caches", "coreai-cache", build)
}

func init() { cache.SetRestorable(restorableEntries) }

// restorableEntries are the entries of Core AI's cache (of hwnpu's executables, this OS build)
// of the programs in the cache's exports that have a prebuilt program for this device: they hold
// the program's weights again, and the prebuilt program makes them again in seconds rather than
// the Neural Engine compiler's minutes, so a disk short of room loses them first (cache.Room,
// docs/cache-sharing-design.md 7.7).
func restorableEntries() []string {
	base := coreAICacheDir()
	if base == "" || os.Getenv(PrebuiltEnv) == "off" || Arch() == "" {
		return nil
	}
	hashes := map[string]bool{}
	filepath.WalkDir(filepath.Join(cache.Root(), "models"), func(p string, d fs.DirEntry, err error) error {
		if err != nil || !d.IsDir() {
			return nil
		}
		switch {
		case strings.HasSuffix(p, PackedSuffix), strings.HasSuffix(p, ".aimodelc"):
			return filepath.SkipDir
		case strings.EqualFold(filepath.Ext(p), ".aimodel"):
			if Prebuilt(p) != "" {
				if h, err := os.ReadFile(filepath.Join(p, "main.hash")); err == nil {
					hashes[hex.EncodeToString(h)] = true
				}
			}
			return filepath.SkipDir
		}
		return nil
	})
	exes, _ := os.ReadDir(base)
	var out []string
	for _, e := range exes {
		if !e.IsDir() || !strings.HasPrefix(e.Name(), "hwnpu") {
			continue
		}
		for h := range hashes {
			if p := filepath.Join(base, e.Name(), h); isDir(p) {
				out = append(out, p)
			}
		}
	}
	return out
}

func isDir(p string) bool { fi, err := os.Stat(p); return err == nil && fi.IsDir() }

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
	onANE     bool // counted in aneLive
	mu        sync.Mutex
}

// aneLive is the programs this process has loaded on the Neural Engine and not closed. An M2
// takes 112 at a time: the next fails to load there ("no ANE resources") and Core AI runs it on
// the GPU instead (aicheck limits, docs/design/apple-neural-engine.md).
var aneLive atomic.Int64

// aneProgramLimit is the programs a process had on the Neural Engine at once on an M2.
const aneProgramLimit = 112

func (d *device) Compile(ctx context.Context, src types.GraphSource) (backend.Graph, error) {
	if src.Format != types.GraphFormatNative || src.Path == "" {
		return nil, fmt.Errorf("coreai: only GraphFormatNative with Path (.mlpackage or .aimodel) is supported: %w", types.ErrUnsupported)
	}
	if src.Function != "" || src.Units != "" {
		return nil, fmt.Errorf("coreai: functions and compute units are not supported: %w", types.ErrUnsupported)
	}
	path, fn := splitFunction(src.Path)
	cache.Touch(path) // the model it is of is used now (the cache's LRU)
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
		if OnCPU() {
			kind = kindCPU
		}
		if isCut(dir) {
			kind |= kindNoRetry
		}
		name, prebuilt := dir, false
		if !isCut(dir) && !hasEntry(dir) {
			// no entry yet: a prebuilt program fills it, here or from the shared layers
			pb := Prebuilt(dir)
			if pb == "" {
				fetchPrebuilt(ctx, dir)
				pb = Prebuilt(dir)
			}
			if pb != "" {
				p, done, err := openable(pb, dir)
				if err != nil {
					fmt.Fprintf(os.Stderr, "coreai: %s: %v; compiled on the device\n", filepath.Base(pb), err)
				} else {
					defer done() // Core AI's entry holds the program once it is open
					name, prebuilt = p, true
				}
			}
		}
		if fn != "main" {
			name += "#" + fn
		}
		var retried int32
		var e *byte
		g.h = caiOpen(cString(name), kind, &retried, &e)
		if g.h == 0 && prebuilt {
			// the prebuilt program would not load (another toolchain, a corrupt file): compiled
			// on the device as without it
			msg := goString(e)
			fmt.Fprintf(os.Stderr, "coreai: %s: the prebuilt program failed (%s), compiled on the device\n", filepath.Base(name), msg)
			name = dir
			if fn != "main" {
				name += "#" + fn
			}
			g.h = caiOpen(cString(name), kind, &retried, &e)
		} else if g.h != 0 && prebuilt {
			prebuiltOnce.Do(func() {
				console.Notef("  prebuilt programs (%s) finished on the device in place of the Neural Engine compiler (%s=off compiles them)\n", Arch(), PrebuiltEnv)
			})
		}
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
	if d := entryDir(dir); d != "" {
		cache.Use(d) // its entry is not removed to make room while this process runs
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
	if OnCPU() { // the cache's entry may be another run's, for the Neural Engine
		g.placement = types.PlacementReport{Method: UnitsEnv, Verified: true, Ops: []types.OpPlacement{{Name: "program", Type: "program", Device: "cpu"}}}
		offNPU = nil
	}
	if g.placement.OnNPU() && !OnCPU() {
		g.onANE = true
		aneLive.Add(1)
	}
	if n := aneLive.Load(); len(offNPU) > 0 && n >= aneProgramLimit-8 {
		fmt.Fprintf(os.Stderr, "coreai: %s fell back from the Neural Engine with %d programs of this process on it: it takes about %d at a time (an M2)\n", filepath.Base(src.Path), n, aneProgramLimit)
	}
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
		if fn != "main" && stored != "" {
			// a function of a group off the Neural Engine, maybe for another member's (ungroup):
			// lowered alone, and loaded as the first time
			if ok, err := ungroup(d.opts.CacheDir, path); err == nil && ok {
				g.Close()
				return d.Compile(ctx, src)
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
			dt := map[string]types.DType{"f16": types.F16, "f32": types.F32, "i32": types.I32, "i8": types.I8, "i16": types.I16, "i4": types.I4, "u4": types.U4}[x.Elem]
			if dt == types.DTypeUnknown {
				return fmt.Errorf("coreai: %s %q of type %s: %w", kind, n, x.Elem, types.ErrUnsupported)
			}
			bytes := dt.Size()
			if dt == types.I4 || dt == types.U4 {
				bytes = 1
			}
			for _, v := range x.Shape {
				bytes *= v
			}
			if dt == types.I4 || dt == types.U4 { // two a byte
				bytes = (bytes + 1) / 2
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

// AllocInput makes the buffer of the input index as an array of Core AI's, which a run takes as
// it is: a buffer of the host's is copied into one at every run, and a program's caches are tens
// of megabytes (an M1 copies about 30 GB/s: PLaMo 3 Translate 2B's 84 MiB of rings took 5.8 of
// its step's 73 ms, docs/design/apple-coreai.md). It may be bound to any input of that shape, of this
// program or another. nil when the shim has no such array for the input (backend.InputAllocator).
func (g *graph) AllocInput(index int) (backend.Buffer, error) {
	if caiInputArray == nil || g.h == 0 || os.Getenv(CopyEnv) != "" {
		return nil, nil
	}
	if index < 0 || index >= len(g.args) || g.args[index].Kind != types.ArgInput {
		return nil, fmt.Errorf("coreai: argument %d is not an input", index)
	}
	for i, a := range g.order[0] {
		if a != index {
			continue
		}
		g.mu.Lock()
		p := caiInputArray(g.h, int32(i))
		g.mu.Unlock()
		if p == nil {
			return nil, nil
		}
		return &buffer{b: unsafe.Slice((*byte)(p), g.args[index].Bytes), array: p}, nil
	}
	return nil, nil
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
	if g.onANE {
		g.onANE = false
		aneLive.Add(-1)
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
