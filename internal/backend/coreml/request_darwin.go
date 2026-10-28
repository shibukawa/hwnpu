package coreml

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/ebitengine/purego/objc"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/types"
)

// outputCopies counts outputs that Core ML did not write into the bound buffer and that
// had to be copied. It is exposed for tests and diagnostics.
var outputCopies atomic.Int64

type request struct {
	g      *graph
	binds  []*buffer
	padded []*buffer // per argument: the padded buffer Core ML runs on, or nil

	mu       sync.Mutex
	inflight *fence
	closed   bool
}

func (r *request) Bind(index int, b backend.Buffer) error {
	cb, ok := b.(*buffer)
	if !ok {
		return errors.New("coreml: buffer was not allocated by this device")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.inflight != nil {
		return errors.New("coreml: cannot bind while a submission is running")
	}
	r.binds[index] = cb
	return nil
}

func (r *request) Submit() (backend.Fence, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil, errors.New("coreml: request is closed")
	}
	if r.inflight != nil {
		if ok, _ := r.inflight.Done(); !ok {
			return nil, errors.New("coreml: previous submission is still running")
		}
	}
	for i, b := range r.binds {
		if b == nil {
			return nil, fmt.Errorf("coreml: argument %d (%s) is not bound", i, r.g.args[i].Name)
		}
	}
	f := &fence{done: make(chan struct{})}
	r.inflight = f
	go func() {
		f.err = withPool(r.predict)
		close(f.done)
	}()
	return f, nil
}

func (r *request) Close() error {
	r.mu.Lock()
	f := r.inflight
	r.closed = true
	r.mu.Unlock()
	if f != nil {
		<-f.done
	}
	for i, p := range r.padded {
		if p != nil {
			p.Close()
			r.padded[i] = nil
		}
	}
	return nil
}

// device returns the buffer Core ML sees for argument i: the caller's buffer, or the padded one.
func (r *request) device(i int) *buffer {
	if r.padded[i] != nil {
		return r.padded[i]
	}
	return r.binds[i]
}

// predict runs inside an autorelease pool on a locked thread.
func (r *request) predict() error {
	var owned []objc.ID
	defer func() {
		for _, o := range owned {
			release(o)
		}
	}()

	inputs := send(class("NSMutableDictionary"), "dictionary")
	backings := send(class("NSMutableDictionary"), "dictionary")
	for i, a := range r.g.args {
		if a.PaddedDims != nil && a.Kind == types.ArgInput {
			copy(r.padded[i].b[:a.Bytes], r.binds[i].b[:a.Bytes]) // the caller's rows; the rest stays zero
		}
		arr, err := multiArray(r.device(i), a)
		if err != nil {
			return err
		}
		owned = append(owned, arr)
		if a.Kind == types.ArgInput {
			fv := send(class("MLFeatureValue"), "featureValueWithMultiArray:", arr)
			send(inputs, "setObject:forKey:", fv, nsString(a.Name))
		} else {
			send(backings, "setObject:forKey:", arr, nsString(a.Name))
		}
	}

	var e objc.ID
	provider := send(send(class("MLDictionaryFeatureProvider"), "alloc"), "initWithDictionary:error:", inputs, &e)
	if provider == 0 {
		return nsError("create input provider", e)
	}
	owned = append(owned, provider)

	opts := send(send(class("MLPredictionOptions"), "alloc"), "init")
	owned = append(owned, opts)
	send(opts, "setOutputBackings:", backings)

	e = 0
	result := send(r.g.model, "predictionFromFeatures:options:error:", provider, opts, &e)
	if result == 0 {
		return nsError("predict", e)
	}

	// Core ML writes into the output backings when it can. Otherwise copy the result.
	for i, a := range r.g.args {
		if a.Kind != types.ArgOutput {
			continue
		}
		arr := send(send(result, "featureValueForName:", nsString(a.Name)), "multiArrayValue")
		if arr == 0 {
			return fmt.Errorf("coreml: output %q is missing from the prediction", a.Name)
		}
		dst := r.device(i).b
		src := sendPtr(arr, "dataPointer")
		if src != unsafe.Pointer(&dst[0]) {
			if dtypeOf(sendInt(arr, "dataType")) != a.DType || !isContiguous(intsOfArray(send(arr, "shape")), intsOfArray(send(arr, "strides"))) {
				return fmt.Errorf("coreml: output %q came back with a different type or layout", a.Name)
			}
			copy(dst[:physicalBytes(a)], unsafe.Slice((*byte)(src), physicalBytes(a)))
			outputCopies.Add(1)
		}
		if a.PaddedDims != nil {
			copy(r.binds[i].b[:a.Bytes], dst[:a.Bytes]) // keep only the caller's rows
		}
	}
	return nil
}

// multiArray wraps a buffer without copying. The caller releases the result.
func multiArray(b *buffer, a types.ArgInfo) (objc.ID, error) {
	dims := a.Dims
	if a.PaddedDims != nil {
		dims = a.PaddedDims // Core ML runs the padded shape
	}
	var e objc.ID
	arr := send(send(class("MLMultiArray"), "alloc"),
		"initWithDataPointer:shape:dataType:strides:deallocator:error:",
		unsafe.Pointer(&b.b[0]), nsArrayOfInts(dims), mlDataType(a.DType), nsArrayOfInts(contiguousStrides(dims)), objc.Block(0), &e)
	if arr == 0 {
		return 0, nsError("wrap "+a.Name, e)
	}
	return arr, nil
}

func contiguousStrides(dims []int) []int {
	s := make([]int, len(dims))
	acc := 1
	for i := len(dims) - 1; i >= 0; i-- {
		s[i] = acc
		acc *= dims[i]
	}
	return s
}

func isContiguous(shape, strides []int) bool {
	want := contiguousStrides(shape)
	if len(want) != len(strides) {
		return false
	}
	for i := range want {
		if shape[i] > 1 && want[i] != strides[i] {
			return false
		}
	}
	return true
}

type fence struct {
	done chan struct{}
	err  error
}

func (f *fence) Wait(ctx context.Context) error {
	select {
	case <-f.done:
		return f.err
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (f *fence) Done() (bool, error) {
	select {
	case <-f.done:
		return true, f.err
	default:
		return false, nil
	}
}

// physicalBytes is the size of the buffer Core ML runs on.
func physicalBytes(a types.ArgInfo) int {
	if a.PaddedDims == nil {
		return a.Bytes
	}
	n := a.DType.Size()
	for _, v := range a.PaddedDims {
		n *= v
	}
	return n
}
