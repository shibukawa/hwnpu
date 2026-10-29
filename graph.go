package hwnpu

import (
	"context"
	"errors"
	"fmt"

	"github.com/shibukawa/hwnpu/internal/backend"
)

// GraphRunner compiles graphs for a device.
type GraphRunner interface {
	// Compile loads and compiles a graph. In PlacementStrict mode it returns a *PlacementError
	// (errors.Is ErrNotOnNPU) when the graph does not run entirely on the NPU.
	Compile(ctx context.Context, src GraphSource) (*Graph, error)
}

type graphRunner struct{ d *Device }

func (r graphRunner) Compile(ctx context.Context, src GraphSource) (*Graph, error) {
	if r.d.be == nil {
		return nil, ErrClosed
	}
	g, err := r.d.be.Compile(ctx, src)
	if err != nil {
		return nil, err
	}
	rep := g.Placement()
	if r.d.opts.placement == PlacementStrict && !rep.OnNPU() {
		_ = g.Close()
		return nil, &PlacementError{Report: rep}
	}
	return &Graph{be: g, args: g.Args()}, nil
}

// Graph is a compiled graph.
type Graph struct {
	be   backend.Graph
	args []ArgInfo
}

// Args lists the graph arguments. Inputs come first; the order defines Bind indices.
func (g *Graph) Args() []ArgInfo { return append([]ArgInfo(nil), g.args...) }

// Arg looks up an argument by name.
func (g *Graph) Arg(name string) (ArgInfo, bool) {
	for _, a := range g.args {
		if a.Name == name {
			return a, true
		}
	}
	return ArgInfo{}, false
}

// Cached reports whether the graph was loaded from the WithCacheDir cache instead of compiled.
func (g *Graph) Cached() bool { return g.be.Cached() }

// Placement reports where the graph's operations were placed.
func (g *Graph) Placement() PlacementReport { return g.be.Placement() }

// NewRequest creates an execution request. A request is used by one goroutine at a time.
func (g *Graph) NewRequest() (*Request, error) {
	if g.be == nil {
		return nil, ErrClosed
	}
	r, err := g.be.NewRequest()
	if err != nil {
		return nil, err
	}
	return &Request{be: r, g: g}, nil
}

// Close releases the graph. Close its requests first.
func (g *Graph) Close() error {
	if g.be == nil {
		return ErrClosed
	}
	err := g.be.Close()
	g.be = nil
	return err
}

// Request binds buffers to graph arguments and runs the graph.
type Request struct {
	be backend.Request
	g  *Graph
}

// Bind attaches a buffer to the argument at index. The buffer must hold at least
// ArgInfo.Bytes bytes, laid out densely in the argument's DType and Dims.
func (r *Request) Bind(index int, b *Buffer) error {
	if r.be == nil {
		return ErrClosed
	}
	if index < 0 || index >= len(r.g.args) {
		return fmt.Errorf("hwnpu: argument index %d out of range [0,%d)", index, len(r.g.args))
	}
	if b == nil || b.be == nil {
		return errors.New("hwnpu: nil or closed buffer")
	}
	a := r.g.args[index]
	if b.Size() < a.Bytes {
		return fmt.Errorf("hwnpu: buffer of %d bytes is too small for %s", b.Size(), a)
	}
	return r.be.Bind(index, b.be)
}

// BindName attaches a buffer to the named argument.
func (r *Request) BindName(name string, b *Buffer) error {
	a, ok := r.g.Arg(name)
	if !ok {
		return fmt.Errorf("hwnpu: no argument named %q", name)
	}
	return r.Bind(a.Index, b)
}

// Fence waits for one submission.
type Fence interface {
	// Wait blocks until the submission finishes or ctx is done. On ctx cancellation the
	// submission keeps running; the request cannot be submitted again until it finishes.
	Wait(ctx context.Context) error
	// Done reports without blocking whether the submission finished, and its error.
	Done() (bool, error)
}

// Submit starts an execution with the bound buffers.
func (r *Request) Submit() (Fence, error) {
	if r.be == nil {
		return nil, ErrClosed
	}
	f, err := r.be.Submit()
	if err != nil {
		return nil, err
	}
	return f, nil
}

// ResetState sets the state of a model with state inputs (Core ML's MLState, which keeps a
// value such as a KV cache from one run to the next) back to zeros. A new request starts zeroed;
// for a model without states it does nothing.
func (r *Request) ResetState() error {
	if r.be == nil {
		return ErrClosed
	}
	if s, ok := r.be.(interface{ ResetState() error }); ok {
		return s.ResetState()
	}
	return nil
}

// Run submits and waits.
func (r *Request) Run(ctx context.Context) error {
	f, err := r.Submit()
	if err != nil {
		return err
	}
	return f.Wait(ctx)
}

// Close waits for a running submission and releases the request.
func (r *Request) Close() error {
	if r.be == nil {
		return ErrClosed
	}
	err := r.be.Close()
	r.be = nil
	return err
}
