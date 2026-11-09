// Package onnxrt runs ONNX models on the CPU with ONNX Runtime's shared library, loaded at run
// time through purego (no cgo): the CPU beside the NPU, for a model the definitions place there
// (models.yaml's device: a small model whose operators cost more on the NPU than its
// arithmetic, docs/amd-npu-models.md 17).
//
// The library is HWNPU_ONNXRUNTIME (a path) or the system's libonnxruntime (onnxruntime.dll on
// Windows, libonnxruntime.dylib on macOS); ONNX Runtime's releases ship it
// (github.com/microsoft/onnxruntime/releases, lib/). Only float32 tensors are read and
// written. A session runs on the calling thread unless asked for more, and never spins.
package onnxrt

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/shibukawa/hwnpu/internal/dl"
)

// apiVersion is the OrtApi this package asks for: every function it calls is in it, and any
// later library serves it (the table only grows).
const apiVersion = 16

// The functions' places in OrtApi (onnxruntime_c_api.h, the order of struct OrtApi's members).
const (
	fnGetErrorMessage                  = 2
	fnCreateEnv                        = 3
	fnCreateSessionFromArray           = 8
	fnRun                              = 9
	fnCreateSessionOptions             = 10
	fnSetSessionGraphOptimizationLevel = 23
	fnSetIntraOpNumThreads             = 24
	fnSetInterOpNumThreads             = 25
	fnSessionGetInputCount             = 30
	fnSessionGetOutputCount            = 31
	fnSessionGetInputName              = 36
	fnSessionGetOutputName             = 37
	fnCreateTensorWithDataAsOrtValue   = 49
	fnGetTensorMutableData             = 51
	fnGetDimensionsCount               = 61
	fnGetDimensions                    = 62
	fnGetTensorTypeAndShape            = 65
	fnCreateCpuMemoryInfo              = 69
	fnAllocatorFree                    = 76
	fnGetAllocatorWithDefaultOptions   = 78
	fnReleaseEnv                       = 92
	fnReleaseStatus                    = 93
	fnReleaseMemoryInfo                = 94
	fnReleaseSession                   = 95
	fnReleaseValue                     = 96
	fnReleaseTensorTypeAndShapeInfo    = 99
	fnReleaseSessionOptions            = 100
	fnAddSessionConfigEntry            = 130
)

const (
	loggingWarning   = 2  // ORT_LOGGING_LEVEL_WARNING
	optimizeAll      = 99 // ORT_ENABLE_ALL
	arenaAllocator   = 1  // OrtArenaAllocator
	memTypeDefault   = 0  // OrtMemTypeDefault
	elementTypeFloat = 1  // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
)

// Runtime is the loaded library and its environment.
type Runtime struct {
	Path    string
	Version string
	api     uintptr // const OrtApi*
	env     uintptr
	mem     uintptr // the CPU's OrtMemoryInfo
	alloc   uintptr // the default allocator
}

var (
	loadOnce sync.Once
	loaded   *Runtime
	loadErr  error
)

// candidates are the library's paths to try: HWNPU_ONNXRUNTIME, then the system's names.
func candidates() []string {
	c := []string{os.Getenv("HWNPU_ONNXRUNTIME")}
	switch runtime.GOOS {
	case "windows":
		c = append(c, "onnxruntime.dll")
	case "darwin":
		c = append(c, "libonnxruntime.dylib", "/opt/homebrew/lib/libonnxruntime.dylib", "/usr/local/lib/libonnxruntime.dylib")
	default:
		c = append(c, "libonnxruntime.so", "libonnxruntime.so.1")
	}
	return c
}

// Load loads ONNX Runtime once and makes its environment.
func Load() (*Runtime, error) {
	loadOnce.Do(func() { loaded, loadErr = load() })
	return loaded, loadErr
}

func load() (*Runtime, error) {
	lib, err := dl.Open(candidates()...)
	if err != nil {
		return nil, fmt.Errorf("onnxrt: ONNX Runtime's library (set HWNPU_ONNXRUNTIME to libonnxruntime's path; github.com/microsoft/onnxruntime/releases ships it in lib/): %w", err)
	}
	base, err := lib.Sym("OrtGetApiBase")
	if err != nil {
		return nil, fmt.Errorf("onnxrt: %s: %w", lib.Path, err)
	}
	ab, _, _ := purego.SyscallN(base)
	if ab == 0 {
		return nil, fmt.Errorf("onnxrt: %s: OrtGetApiBase returned nothing", lib.Path)
	}
	// struct OrtApiBase { const OrtApi* (*GetApi)(uint32_t); const char* (*GetVersionString)(void); }
	fns := (*[2]uintptr)(cmem(ab))
	r := &Runtime{Path: lib.Path}
	if v, _, _ := purego.SyscallN(fns[1]); v != 0 {
		r.Version = goString(v)
	}
	if r.api, _, _ = purego.SyscallN(fns[0], apiVersion); r.api == 0 {
		return nil, fmt.Errorf("onnxrt: %s (ONNX Runtime %s) does not serve API version %d", lib.Path, r.Version, apiVersion)
	}
	id := cstr("hwnpu")
	if err := r.call(fnCreateEnv, loggingWarning, ptr(&id[0]), ptr(&r.env)); err != nil {
		return nil, err
	}
	if err := r.call(fnCreateCpuMemoryInfo, arenaAllocator, memTypeDefault, ptr(&r.mem)); err != nil {
		return nil, err
	}
	if err := r.call(fnGetAllocatorWithDefaultOptions, ptr(&r.alloc)); err != nil {
		return nil, err
	}
	runtime.KeepAlive(id)
	return r, nil
}

// fn is the address of OrtApi's function i.
func (r *Runtime) fn(i int) uintptr {
	return *(*uintptr)(unsafe.Add(cmem(r.api), i*int(unsafe.Sizeof(uintptr(0)))))
}

// call calls function i, which returns an OrtStatus* (nil: done), as an error.
func (r *Runtime) call(i int, args ...uintptr) error {
	st, _, _ := purego.SyscallN(r.fn(i), args...)
	if st == 0 {
		return nil
	}
	msg, _, _ := purego.SyscallN(r.fn(fnGetErrorMessage), st)
	err := errors.New("onnxrt: " + goString(msg))
	purego.SyscallN(r.fn(fnReleaseStatus), st)
	return err
}

func (r *Runtime) release(i int, p uintptr) {
	if p != 0 {
		purego.SyscallN(r.fn(i), p)
	}
}

// Session is a loaded model.
type Session struct {
	r       *Runtime
	s       uintptr
	Inputs  []string
	Outputs []string
	cin     [][]byte // the names as C strings
	cout    [][]byte
}

// Options are a session's choices.
type Options struct {
	// Threads runs each operator on this many threads (0 or 1: the calling thread alone, the
	// least CPU and energy for a small model). The threads never spin waiting for work.
	Threads int
}

// Open loads the model (ONNX bytes) into a session.
func (r *Runtime) Open(model []byte, o Options) (*Session, error) {
	po := new(uintptr) // out parameters live on the heap: a stack may move under a uintptr
	if err := r.call(fnCreateSessionOptions, ptr(po)); err != nil {
		return nil, err
	}
	opts := *po
	defer r.release(fnReleaseSessionOptions, opts)
	th := max(1, o.Threads)
	if err := r.call(fnSetIntraOpNumThreads, opts, uintptr(th)); err != nil {
		return nil, err
	}
	if err := r.call(fnSetInterOpNumThreads, opts, 1); err != nil {
		return nil, err
	}
	if err := r.call(fnSetSessionGraphOptimizationLevel, opts, optimizeAll); err != nil {
		return nil, err
	}
	for _, kv := range [][2]string{{"session.intra_op.allow_spinning", "0"}, {"session.inter_op.allow_spinning", "0"}} {
		k, v := cstr(kv[0]), cstr(kv[1])
		if err := r.call(fnAddSessionConfigEntry, opts, ptr(&k[0]), ptr(&v[0])); err != nil {
			return nil, err
		}
		runtime.KeepAlive(k)
		runtime.KeepAlive(v)
	}
	s := &Session{r: r}
	if len(model) == 0 {
		return nil, errors.New("onnxrt: an empty model")
	}
	var pin runtime.Pinner
	pin.Pin(&model[0])
	defer pin.Unpin()
	if err := r.call(fnCreateSessionFromArray, r.env, ptr(&model[0]), uintptr(len(model)), opts, ptr(&s.s)); err != nil {
		return nil, err
	}
	runtime.KeepAlive(model)
	var err error
	if s.Inputs, err = s.names(fnSessionGetInputCount, fnSessionGetInputName); err == nil {
		s.Outputs, err = s.names(fnSessionGetOutputCount, fnSessionGetOutputName)
	}
	if err != nil {
		s.Close()
		return nil, err
	}
	for _, n := range s.Inputs {
		s.cin = append(s.cin, cstr(n))
	}
	for _, n := range s.Outputs {
		s.cout = append(s.cout, cstr(n))
	}
	return s, nil
}

func (s *Session) names(count, name int) ([]string, error) {
	n := new(uintptr)
	if err := s.r.call(count, s.s, ptr(n)); err != nil {
		return nil, err
	}
	out := make([]string, *n)
	p := new(uintptr)
	for i := range out {
		if err := s.r.call(name, s.s, uintptr(i), s.r.alloc, ptr(p)); err != nil {
			return nil, err
		}
		out[i] = goString(*p)
		if err := s.r.call(fnAllocatorFree, s.r.alloc, *p); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// Close releases the session.
func (s *Session) Close() error {
	s.r.release(fnReleaseSession, s.s)
	s.s = 0
	return nil
}

// Tensor is a float32 tensor: its shape and its values, row major.
type Tensor struct {
	Shape []int64
	Data  []float32
}

// Run runs the model on inputs (by name, every input) and returns every output by name.
func (s *Session) Run(inputs map[string]Tensor) (map[string]Tensor, error) {
	r := s.r
	var pin runtime.Pinner
	defer pin.Unpin()
	vals := make([]uintptr, len(s.Inputs))
	defer func() {
		for _, v := range vals {
			r.release(fnReleaseValue, v)
		}
	}()
	names := make([]uintptr, len(s.Inputs))
	for i, n := range s.Inputs {
		t, ok := inputs[n]
		if !ok {
			return nil, fmt.Errorf("onnxrt: no input %s", n)
		}
		count := int64(1)
		for _, d := range t.Shape {
			count *= d
		}
		if count != int64(len(t.Data)) || len(t.Data) == 0 || len(t.Shape) == 0 {
			return nil, fmt.Errorf("onnxrt: input %s: %d values for the shape %v", n, len(t.Data), t.Shape)
		}
		pin.Pin(&t.Data[0])
		pin.Pin(&t.Shape[0])
		if err := r.call(fnCreateTensorWithDataAsOrtValue, r.mem, ptr(&t.Data[0]), uintptr(4*len(t.Data)),
			ptr(&t.Shape[0]), uintptr(len(t.Shape)), elementTypeFloat, ptr(&vals[i])); err != nil {
			return nil, fmt.Errorf("onnxrt: input %s: %w", n, err)
		}
		pin.Pin(&s.cin[i][0])
		names[i] = ptr(&s.cin[i][0])
	}
	onames := make([]uintptr, len(s.Outputs))
	for i := range s.Outputs {
		pin.Pin(&s.cout[i][0])
		onames[i] = ptr(&s.cout[i][0])
	}
	outs := make([]uintptr, len(s.Outputs))
	pin.Pin(&names[0])
	pin.Pin(&vals[0])
	pin.Pin(&onames[0])
	pin.Pin(&outs[0])
	if err := r.call(fnRun, s.s, 0, ptr(&names[0]), ptr(&vals[0]), uintptr(len(vals)),
		ptr(&onames[0]), uintptr(len(onames)), ptr(&outs[0])); err != nil {
		return nil, err
	}
	defer func() {
		for _, v := range outs {
			r.release(fnReleaseValue, v)
		}
	}()
	res := make(map[string]Tensor, len(outs))
	for i, v := range outs {
		t, err := r.tensor(v)
		if err != nil {
			return nil, fmt.Errorf("onnxrt: output %s: %w", s.Outputs[i], err)
		}
		res[s.Outputs[i]] = t
	}
	return res, nil
}

// tensor copies a float32 OrtValue out.
func (r *Runtime) tensor(v uintptr) (Tensor, error) {
	pi := new(uintptr)
	if err := r.call(fnGetTensorTypeAndShape, v, ptr(pi)); err != nil {
		return Tensor{}, err
	}
	info := *pi
	defer r.release(fnReleaseTensorTypeAndShapeInfo, info)
	pn := new(uintptr)
	if err := r.call(fnGetDimensionsCount, info, ptr(pn)); err != nil {
		return Tensor{}, err
	}
	n := *pn
	t := Tensor{Shape: make([]int64, n)}
	count := 1
	if n > 0 {
		if err := r.call(fnGetDimensions, info, ptr(&t.Shape[0]), n); err != nil {
			return Tensor{}, err
		}
	}
	for _, d := range t.Shape {
		count *= int(d)
	}
	pd := new(uintptr)
	if err := r.call(fnGetTensorMutableData, v, ptr(pd)); err != nil {
		return Tensor{}, err
	}
	data := *pd
	t.Data = make([]float32, count)
	if count > 0 {
		copy(t.Data, unsafe.Slice((*float32)(cmem(data)), count))
	}
	return t, nil
}

// ReadModel reads an ONNX file.
func ReadModel(path string) ([]byte, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("onnxrt: %s: %w", filepath.Base(path), err)
	}
	return b, nil
}

func cstr(s string) []byte { return append([]byte(s), 0) }

// ptr is p's address for a C call. Every p here is on the heap (new, a field of a heap
// struct, a slice's array) or pinned: a goroutine's stack may move, and a uintptr does not
// keep what it points at.
func ptr[T any](p *T) uintptr { return uintptr(unsafe.Pointer(p)) }

func goString(p uintptr) string {
	if p == 0 {
		return ""
	}
	n := 0
	for *(*byte)(unsafe.Add(cmem(p), n)) != 0 {
		n++
	}
	return string(unsafe.Slice((*byte)(cmem(p)), n))
}

// cmem is the address p, which ONNX Runtime gave (memory Go does not manage), as a pointer.
func cmem(p uintptr) unsafe.Pointer { return *(*unsafe.Pointer)(unsafe.Pointer(&p)) }
