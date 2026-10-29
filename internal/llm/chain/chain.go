// Package chain runs a model written in parts (hwnpu-convert's out.parts.json) as one model: it loads every part,
// a package of its own, and binds the output buffer of each part as the input x of the next, so
// that nothing is copied between them. Inputs other than x (such as an encoder's mask) are one
// buffer bound to every part that takes them.
package chain

import (
	"context"
	"fmt"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
)

// Part is one loaded part.
type Part struct {
	llm.PartRecord
	Placement hwnpu.PlacementReport
	Cached    bool // compiled from the device's cache

	g   *hwnpu.Graph
	req *hwnpu.Request
}

// Chain is a loaded model in parts.
type Chain struct {
	parts             []*Part
	inputs, outputs   map[string]*hwnpu.Buffer
	inShape, outShape map[string][]int
	out               *hwnpu.Buffer
	bufs              []*hwnpu.Buffer
}

// Open loads every part of the model at path on dev: a manifest of parts (out.parts.json) or a
// single package.
func Open(ctx context.Context, dev *hwnpu.Device, path string) (*Chain, error) {
	list, err := llm.ReadManifest(path)
	if err != nil {
		return nil, err
	}
	gr, err := dev.Graph()
	if err != nil {
		return nil, err
	}
	c := &Chain{inputs: map[string]*hwnpu.Buffer{}, outputs: map[string]*hwnpu.Buffer{}, inShape: map[string][]int{}, outShape: map[string][]int{}}
	alloc := func(size int) (*hwnpu.Buffer, error) {
		b, err := dev.AllocBuffer(size)
		if err == nil {
			c.bufs = append(c.bufs, b)
		}
		return b, err
	}
	for i, lp := range list.Parts {
		p := &Part{PartRecord: lp}
		if p.g, err = gr.Compile(ctx, hwnpu.GraphSource{Path: lp.Path}); err != nil {
			c.Close()
			return nil, fmt.Errorf("chain: part %d (%s): %w", i, lp.Path, err)
		}
		c.parts = append(c.parts, p)
		p.Placement, p.Cached = p.g.Placement(), p.g.Cached()
		if p.req, err = p.g.NewRequest(); err != nil {
			c.Close()
			return nil, err
		}
		var out *hwnpu.Buffer
		for _, a := range p.g.Args() {
			var b *hwnpu.Buffer
			switch {
			case a.Kind == hwnpu.ArgOutput:
				if b, err = alloc(a.Bytes); err != nil {
					c.Close()
					return nil, err
				}
				if a.Name == "y" {
					out = b
				} else {
					c.outputs[a.Name], c.outShape[a.Name] = b, a.Dims
				}
			case a.Name == "x" && i > 0:
				b = c.out
				if b.Size() < a.Bytes {
					c.Close()
					return nil, fmt.Errorf("chain: part %d takes x of %d bytes, the part before gives %d", i, a.Bytes, b.Size())
				}
			default:
				if b = c.inputs[a.Name]; b == nil {
					if b, err = alloc(a.Bytes); err != nil {
						c.Close()
						return nil, err
					}
					c.inputs[a.Name], c.inShape[a.Name] = b, a.Dims
				}
			}
			if err := p.req.Bind(a.Index, b); err != nil {
				c.Close()
				return nil, err
			}
		}
		if out == nil {
			c.Close()
			return nil, fmt.Errorf("chain: part %d has no output y", i)
		}
		c.out = out
	}
	c.outputs["y"] = c.out
	for _, a := range c.parts[len(c.parts)-1].g.Args() {
		if a.Kind == hwnpu.ArgOutput && a.Name == "y" {
			c.outShape["y"] = a.Dims
		}
	}
	return c, nil
}

// Program returns the chain as a decode.Program: its inputs and outputs by name (y is the last
// part's), and Run, which syncs the buffers around the parts.
func (c *Chain) Program(ctx context.Context) decode.Program {
	p := decode.Program{In: map[string][]byte{}, Out: map[string][]byte{}, InShape: c.inShape, OutShape: c.outShape}
	for n, b := range c.inputs {
		p.In[n] = b.Bytes()
	}
	for n, b := range c.outputs {
		p.Out[n] = b.Bytes()
	}
	p.Run = func() error {
		for _, b := range c.inputs {
			b.SyncToDevice(0, b.Size())
		}
		if err := c.Run(ctx); err != nil {
			return err
		}
		for _, b := range c.outputs {
			b.SyncFromDevice(0, b.Size())
		}
		return nil
	}
	return p
}

// Input returns the buffer of a model input (x, or another input such as mask).
func (c *Chain) Input(name string) *hwnpu.Buffer { return c.inputs[name] }

// Output returns the buffer of the last part's output.
func (c *Chain) Output() *hwnpu.Buffer { return c.out }

// Parts returns the loaded parts in order.
func (c *Chain) Parts() []*Part { return c.parts }

// OnNPU reports whether every part was verified to run on the NPU.
func (c *Chain) OnNPU() bool {
	for _, p := range c.parts {
		if !p.Placement.OnNPU() {
			return false
		}
	}
	return len(c.parts) > 0
}

// Run runs the parts in order.
func (c *Chain) Run(ctx context.Context) error {
	for i, p := range c.parts {
		if err := p.req.Run(ctx); err != nil {
			return fmt.Errorf("chain: part %d: %w", i, err)
		}
	}
	return nil
}

// Close releases the parts and the buffers.
func (c *Chain) Close() error {
	for _, p := range c.parts {
		if p.req != nil {
			p.req.Close()
		}
		if p.g != nil {
			p.g.Close()
		}
	}
	for _, b := range c.bufs {
		b.Close()
	}
	c.parts, c.bufs = nil, nil
	return nil
}
