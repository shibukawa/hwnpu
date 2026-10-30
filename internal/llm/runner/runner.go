// Package runner runs a converted model that reads all its input in one call (no KV cache): a
// package or an out.parts.json, on the NPU (internal/llm/chain) or with the CPU interpreter
// (internal/milcpu, slow, for checking on a machine without one). Inputs and the output y are
// float32; the device's fp16 buffers are filled and read here.
package runner

import (
	"context"
	"errors"
	"fmt"
	"io"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/chain"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Program is a loaded model.
type Program interface {
	// Inputs are the model's inputs and their shapes: x (the first part's), then the others.
	Inputs() map[string][]int
	// Output is the shape of y.
	Output() []int
	// Metadata is the first part's package metadata (llm.MetaArch, llm.MetaParam...).
	Metadata() map[string]string
	// OnNPU reports whether every part was verified to run on the NPU (false on the CPU).
	OnNPU() bool
	// Run computes y from the inputs, each as many values as its shape holds.
	Run(ctx context.Context, in map[string][]float32) ([]float32, error)
	Close() error
}

// Open loads the model at path on dev, or with the CPU interpreter when dev is nil.
func Open(ctx context.Context, dev *hwnpu.Device, path string) (Program, error) {
	man, err := llm.ReadManifest(path)
	if err != nil {
		return nil, err
	}
	tiers := man.Tiered()
	if len(tiers) != 1 || tiers[0].Cache > 0 {
		return nil, fmt.Errorf("runner: %s has a KV cache", path)
	}
	m, err := mlpackage.ReadModel(tiers[0].Parts[0].Path)
	if err != nil {
		return nil, err
	}
	meta := m.GetDescription().GetMetadata().GetUserDefined()
	if dev == nil {
		return openCPU(tiers[0].Parts, meta)
	}
	c, err := chain.Open(ctx, dev, path)
	if err != nil {
		return nil, err
	}
	return &devProgram{c: c, p: c.Program(ctx), meta: meta}, nil
}

func count(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

type devProgram struct {
	c    *chain.Chain
	p    decode.Program
	meta map[string]string
}

func (p *devProgram) Inputs() map[string][]int    { return p.p.InShape }
func (p *devProgram) Output() []int               { return p.p.OutShape["y"] }
func (p *devProgram) Metadata() map[string]string { return p.meta }
func (p *devProgram) OnNPU() bool                 { return p.c.OnNPU() }
func (p *devProgram) Close() error                { return p.c.Close() }

func (p *devProgram) Run(ctx context.Context, in map[string][]float32) ([]float32, error) {
	for name, buf := range p.p.In {
		v, ok := in[name]
		if !ok || len(v) != count(p.p.InShape[name]) {
			return nil, fmt.Errorf("runner: input %s needs %v values, got %d", name, p.p.InShape[name], len(v))
		}
		hwnpu.EncodeF16(buf, v)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := p.p.Run(); err != nil {
		return nil, err
	}
	y := make([]float32, count(p.Output()))
	hwnpu.DecodeF16(y, p.p.Out["y"])
	return y, nil
}

type cpuProgram struct {
	parts   []*milcpu.Program
	closers []io.Closer
	inputs  map[string][]int
	out     []int
	meta    map[string]string
}

func openCPU(parts []llm.PartRecord, meta map[string]string) (*cpuProgram, error) {
	p := &cpuProgram{inputs: map[string][]int{}, meta: meta}
	for i, part := range parts {
		prog, f, err := milcpu.OpenPackage(part.Path)
		if err != nil {
			p.Close()
			return nil, err
		}
		p.parts, p.closers = append(p.parts, prog), append(p.closers, f)
		for name, shape := range prog.Inputs() {
			if name != "x" || i == 0 {
				p.inputs[name] = shape
			}
		}
		m, err := mlpackage.ReadModel(part.Path)
		if err != nil {
			p.Close()
			return nil, err
		}
		for _, o := range m.GetDescription().GetOutput() {
			if o.GetName() == "y" {
				p.out = nil
				for _, d := range o.GetType().GetMultiArrayType().GetShape() {
					p.out = append(p.out, int(d))
				}
			}
		}
	}
	if p.out == nil {
		p.Close()
		return nil, errors.New("runner: no output y")
	}
	return p, nil
}

func (p *cpuProgram) Inputs() map[string][]int    { return p.inputs }
func (p *cpuProgram) Output() []int               { return p.out }
func (p *cpuProgram) Metadata() map[string]string { return p.meta }
func (p *cpuProgram) OnNPU() bool                 { return false }

func (p *cpuProgram) Run(ctx context.Context, in map[string][]float32) ([]float32, error) {
	var y []float32
	for i, part := range p.parts {
		feed := map[string]milcpu.Tensor{}
		for name, shape := range part.Inputs() {
			v := in[name]
			if name == "x" && i > 0 {
				v = y
			}
			if len(v) != count(shape) {
				return nil, fmt.Errorf("runner: input %s needs %v values, got %d", name, shape, len(v))
			}
			feed[name] = milcpu.Tensor{Shape: shape, Data: v}
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		res, err := part.Run(feed, milcpu.Options{})
		if err != nil {
			return nil, err
		}
		y = res["y"].Data
	}
	return y, nil
}

func (p *cpuProgram) Close() error {
	for _, c := range p.closers {
		c.Close()
	}
	return nil
}
