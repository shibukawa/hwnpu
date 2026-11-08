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
	"os"

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
	meta, err := llm.ReadMeta(man)
	if err != nil {
		return nil, err
	}
	if dev == nil {
		return openCPU(tiers[0].Parts, meta)
	}
	c, err := chain.Open(ctx, dev, path)
	if err != nil {
		return nil, err
	}
	// the program keeps the context it is taken with for every run, so it must not be the
	// caller's: hwnpu-chat loads under a signal.NotifyContext whose stop cancels it once the
	// model is up, and the vision encoder then failed every image with "context canceled"
	// (the context of a run is checked in Run)
	return &devProgram{c: c, p: c.Program(context.Background()), meta: meta}, nil
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
		// HWNPU_CPU_F16 rounds every operation's result to fp16, about the Neural Engine's
		// storage precision: to find values a program overflows on the NPU
		res, err := part.Run(feed, milcpu.Options{RoundF16: os.Getenv("HWNPU_CPU_F16") != ""})
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

// OpenSession loads a causal model converted with a KV cache (hwnpu-convert -cache), one or more
// packages of the same cache length and different tokens per call, on dev (nil: the CPU
// interpreter), and starts a decode session over them; the packages share one cache. Close the
// returned closer after the session.
func OpenSession(ctx context.Context, dev *hwnpu.Device, paths []string, m llm.Cached) (*decode.Session, io.Closer, error) {
	var closers multiCloser
	var sess *decode.Session
	var first *chain.Chain
	var shared map[string][]byte
	for i, path := range paths {
		var p decode.Program
		if dev != nil {
			var c *chain.Chain
			var err error
			if i == 0 {
				c, err = chain.Open(ctx, dev, path)
				first = c
			} else {
				c, err = first.Share(ctx, dev, path)
			}
			if err != nil {
				closers.Close()
				return nil, nil, err
			}
			closers = append(closers, c)
			p = c.Program(context.Background()) // not the caller's context: see Open
		} else {
			cp, c, err := cachedCPU(path, shared)
			if err != nil {
				closers.Close()
				return nil, nil, err
			}
			closers = append(closers, c)
			p = cp
			if i == 0 {
				shared = map[string][]byte{}
				for name, b := range p.In {
					if llm.CacheInput(name) {
						shared[name] = b
					}
				}
			}
		}
		var err error
		if i == 0 {
			sess, err = decode.New(p, m)
		} else {
			err = sess.Add(p)
		}
		if err != nil {
			closers.Close()
			return nil, nil, fmt.Errorf("runner: %s: %w", path, err)
		}
	}
	return sess, closers, nil
}

type multiCloser []io.Closer

func (m multiCloser) Close() error {
	for i := len(m) - 1; i >= 0; i-- {
		m[i].Close()
	}
	return nil
}

// cachedCPU loads a model with a KV cache as a decode.Program on the CPU interpreter: one package,
// or the parts of a manifest run in order as internal/llm/chain runs them (x of each part is y of
// the one before, an input named like an earlier part's output takes it), and for a model split
// at its attention (the manifest's Paged) in stages with its attention program. Inputs named in
// shared use those buffers.
func cachedCPU(path string, shared map[string][]byte) (decode.Program, io.Closer, error) {
	man, err := llm.ReadManifest(path)
	if err != nil {
		return decode.Program{}, nil, err
	}
	tiers := man.Tiered()
	if len(tiers) != 1 {
		return decode.Program{}, nil, fmt.Errorf("runner: %s: the CPU interpreter runs one cache length", path)
	}
	var closers multiCloser
	p := decode.Program{In: map[string][]byte{}, Out: map[string][]byte{}, InShape: map[string][]int{}, OutShape: map[string][]int{}}
	type part struct {
		prog *milcpu.Program
		last bool
	}
	var parts []part
	var stages [][]int
	var cur []int
	recs := tiers[0].Parts
	for i, rec := range recs {
		prog, f, err := milcpu.OpenPackage(rec.Path)
		if err != nil {
			closers.Close()
			return decode.Program{}, nil, err
		}
		closers = append(closers, f)
		outs, err := outputShapes(rec.Path)
		if err != nil {
			closers.Close()
			return decode.Program{}, nil, err
		}
		attends := -1
		for name, sh := range outs {
			if name == "y" && i < len(recs)-1 {
				continue
			}
			p.OutShape[name], p.Out[name] = sh, make([]byte, 2*count(sh))
			var layer int
			if _, err := fmt.Sscanf(name, "attn_q_%d", &layer); err == nil {
				attends = layer
			}
		}
		for name, sh := range prog.Inputs() {
			if name == "x" && i > 0 || p.Out[name] != nil {
				continue
			}
			p.InShape[name] = sh
			if b, ok := shared[name]; ok && len(b) == 2*count(sh) {
				p.In[name] = b
			} else {
				p.In[name] = make([]byte, 2*count(sh))
			}
		}
		parts = append(parts, part{prog: prog, last: i == len(recs)-1})
		cur = append(cur, i)
		if attends >= 0 && man.Paged > 0 {
			stages, cur = append(stages, cur), nil
			p.Attend = append(p.Attend, attends)
		}
	}
	if len(cur) == 0 && man.KVOnly && len(p.Attend) > 0 {
		// as internal/llm/chain: the last part computes keys and values nothing attends for
		p.Attend = p.Attend[:len(p.Attend)-1]
	} else {
		stages = append(stages, cur)
	}
	var y milcpu.Tensor
	run := func(list []int) error {
		for _, i := range list {
			in := map[string]milcpu.Tensor{}
			for name, sh := range parts[i].prog.Inputs() {
				if name == "x" && i > 0 {
					in[name] = y
					continue
				}
				b := p.In[name]
				if b == nil {
					b = p.Out[name]
				}
				v := make([]float32, count(sh))
				hwnpu.DecodeF16(v, b)
				in[name] = milcpu.Tensor{Shape: sh, Data: v}
			}
			res, err := parts[i].prog.Run(in, milcpu.Options{})
			if err != nil {
				return err
			}
			for name, o := range res {
				if name == "y" {
					y = o
					if !parts[i].last {
						continue
					}
				}
				hwnpu.EncodeF16(p.Out[name], o.Data)
			}
		}
		return nil
	}
	all := make([]int, len(parts))
	for i := range all {
		all[i] = i
	}
	p.Run = func() error { return run(all) }
	if man.Paged > 0 {
		if man.Attention == nil {
			closers.Close()
			return decode.Program{}, nil, fmt.Errorf("runner: %s: no attention program", path)
		}
		a, c, err := attentionCPU(man.Attention.Path, man.Paged)
		if err != nil {
			closers.Close()
			return decode.Program{}, nil, err
		}
		closers = append(closers, c)
		if man.Mass != nil {
			ms, c, err := massCPU(man.Mass.Path)
			if err != nil {
				closers.Close()
				return decode.Program{}, nil, err
			}
			closers = append(closers, c)
			a.Mass = ms
		}
		p.Run, p.Attn, p.Context, p.NoLogits = nil, a, man.Cache, man.KVOnly
		p.RunPart = func(i int) error { return run(stages[i]) }
	}
	return p, closers, nil
}

// cpuBuf is a decode.Buf in the Go heap.
type cpuBuf []byte

func (b cpuBuf) Bytes() []byte { return b }
func (b cpuBuf) Close() error  { return nil }

// attentionCPU loads the attention program of a model split at its attention on the CPU
// interpreter.
func attentionCPU(path string, chunk int) (*decode.Attention, io.Closer, error) {
	prog, f, err := milcpu.OpenPackage(path)
	if err != nil {
		return nil, nil, err
	}
	outs, err := outputShapes(path)
	if err != nil {
		f.Close()
		return nil, nil, err
	}
	in := prog.Inputs()
	a := &decode.Attention{Chunk: chunk, Q: make([]byte, 2*count(in["q"])), O: make([]byte, 2*count(outs["o"])), M: make([]byte, 2*count(outs["m"])), L: make([]byte, 2*count(outs["l"]))}
	a.NewBuf = func(n int) (decode.Buf, error) { return cpuBuf(make([]byte, n)), nil }
	a.Run = func(k, v, mask decode.Buf) error {
		vals := map[string]milcpu.Tensor{}
		for name, b := range map[string][]byte{"q": a.Q, "k": k.Bytes(), "v": v.Bytes(), "mask": mask.Bytes()} {
			x := make([]float32, count(in[name]))
			if 2*len(x) != len(b) {
				return fmt.Errorf("runner: attention input %s has %d bytes for %v", name, len(b), in[name])
			}
			hwnpu.DecodeF16(x, b)
			vals[name] = milcpu.Tensor{Shape: in[name], Data: x}
		}
		res, err := prog.Run(vals, milcpu.Options{})
		if err != nil {
			return err
		}
		hwnpu.EncodeF16(a.O, res["o"].Data)
		hwnpu.EncodeF16(a.M, res["m"].Data)
		hwnpu.EncodeF16(a.L, res["l"].Data)
		return nil
	}
	return a, f, nil
}

// massCPU loads the mass program of the two-level attention on the CPU interpreter.
func massCPU(path string) (*decode.Mass, io.Closer, error) {
	prog, f, err := milcpu.OpenPackage(path)
	if err != nil {
		return nil, nil, err
	}
	outs, err := outputShapes(path)
	if err != nil {
		f.Close()
		return nil, nil, err
	}
	in := prog.Inputs()
	ms := &decode.Mass{Q: make([]byte, 2*count(in["q"])), R: make([]byte, 2*count(in["r"])), W: make([]byte, 2*count(outs["w"]))}
	ms.Run = func(k, mask decode.Buf) error {
		vals := map[string]milcpu.Tensor{}
		for name, b := range map[string][]byte{"q": ms.Q, "r": ms.R, "k": k.Bytes(), "mask": mask.Bytes()} {
			x := make([]float32, count(in[name]))
			if 2*len(x) != len(b) {
				return fmt.Errorf("runner: mass input %s has %d bytes for %v", name, len(b), in[name])
			}
			hwnpu.DecodeF16(x, b)
			vals[name] = milcpu.Tensor{Shape: in[name], Data: x}
		}
		res, err := prog.Run(vals, milcpu.Options{})
		if err != nil {
			return err
		}
		hwnpu.EncodeF16(ms.W, res["w"].Data)
		return nil
	}
	return ms, f, nil
}

// outputShapes reads the outputs of a package and their shapes.
func outputShapes(path string) (map[string][]int, error) {
	m, err := mlpackage.ReadModel(path)
	if err != nil {
		return nil, err
	}
	out := map[string][]int{}
	for _, o := range m.GetDescription().GetOutput() {
		var sh []int
		for _, d := range o.GetType().GetMultiArrayType().GetShape() {
			sh = append(sh, int(d))
		}
		out[o.GetName()] = sh
	}
	return out, nil
}
