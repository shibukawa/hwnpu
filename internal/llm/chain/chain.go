// Package chain runs a model written in parts (hwnpu-convert's out.parts.json) as one model: it
// loads every part, a package of its own, and binds the output buffer of each part as the input x
// of the next, so that nothing is copied between them. An input named like an output of an
// earlier part is bound to that output (Gemma 4's per-layer inputs and shared keys and values);
// other inputs (such as an encoder's mask) are one buffer bound to every part that takes them.
//
// A model converted with several KV cache lengths (tiers) is loaded one tier at a time: the
// shortest first, since a step costs in proportion to the cache's length, and the next when the
// sequence outgrows it (Grow), its state carried over by the caller (decode.Session.Migrate). The
// next tier can be loaded in the background ahead of time (Preload).
package chain

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

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

// tier is one loaded set of parts with its buffers.
type tier struct {
	parts             []*Part
	inputs, outputs   map[string]*hwnpu.Buffer
	inShape, outShape map[string][]int
	passed            map[string]bool // outputs bound as inputs of a later part
	out               *hwnpu.Buffer
	bufs              []*hwnpu.Buffer
	loadTime          time.Duration
}

// Chain is a loaded model in parts.
type Chain struct {
	dev   *hwnpu.Device
	gr    hwnpu.GraphRunner
	tiers []llm.Tier
	index int   // the tier loaded
	cur   *tier // never nil after Open

	sharing bool // made by Share: its cache buffers belong to another chain, so it cannot grow

	mu    sync.Mutex
	pre   *preload // the next tier being loaded in the background, or nil
	ahead []*tier  // tiers after the loaded one that LoadAll has loaded, in order
}

type preload struct {
	done chan struct{}
	t    *tier
	err  error
}

// Open loads the model at path on dev: a manifest of parts (out.parts.json), whose first tier is
// loaded when it has several, or a single package.
func Open(ctx context.Context, dev *hwnpu.Device, path string) (*Chain, error) {
	man, err := llm.ReadManifest(path)
	if err != nil {
		return nil, err
	}
	return open(ctx, dev, man, nil)
}

// Share loads the model at path like Open, but binds its KV cache inputs (kv_*, win_*) to the
// buffers of the same inputs of c's loaded tier, so that both chains continue one sequence
// (decode.Session.Add): the same model converted with another number of tokens per call. The
// buffers stay c's; close the new chain before c, and do not Grow either chain while they share.
func (c *Chain) Share(ctx context.Context, dev *hwnpu.Device, path string) (*Chain, error) {
	man, err := llm.ReadManifest(path)
	if err != nil {
		return nil, err
	}
	shared := map[string]*hwnpu.Buffer{}
	for name, b := range c.cur.inputs {
		if strings.HasPrefix(name, "kv_") || strings.HasPrefix(name, "win_") {
			shared[name] = b
		}
	}
	return open(ctx, dev, man, shared)
}

func open(ctx context.Context, dev *hwnpu.Device, man *llm.Manifest, shared map[string]*hwnpu.Buffer) (*Chain, error) {
	gr, err := dev.Graph()
	if err != nil {
		return nil, err
	}
	c := &Chain{dev: dev, gr: gr, tiers: man.Tiered(), sharing: shared != nil}
	if c.cur, err = c.load(ctx, c.tiers[0].Parts, shared); err != nil {
		return nil, err
	}
	return c, nil
}

// load compiles and binds one tier's parts. Inputs named in shared are bound to those buffers
// instead of new ones, and are not owned.
func (c *Chain) load(ctx context.Context, list []llm.PartRecord, shared map[string]*hwnpu.Buffer) (*tier, error) {
	start := time.Now()
	t := &tier{inputs: map[string]*hwnpu.Buffer{}, outputs: map[string]*hwnpu.Buffer{}, inShape: map[string][]int{}, outShape: map[string][]int{}, passed: map[string]bool{}}
	alloc := func(size int) (*hwnpu.Buffer, error) {
		b, err := c.dev.AllocBuffer(size)
		if err == nil {
			t.bufs = append(t.bufs, b)
		}
		return b, err
	}
	var err error
	for i, lp := range list {
		p := &Part{PartRecord: lp}
		if p.g, err = c.gr.Compile(ctx, hwnpu.GraphSource{Path: lp.Path}); err != nil {
			t.close()
			return nil, fmt.Errorf("chain: part %d (%s): %w", i, lp.Path, err)
		}
		t.parts = append(t.parts, p)
		p.Placement, p.Cached = p.g.Placement(), p.g.Cached()
		if p.req, err = p.g.NewRequest(); err != nil {
			t.close()
			return nil, err
		}
		var out *hwnpu.Buffer
		for _, a := range p.g.Args() {
			var b *hwnpu.Buffer
			switch {
			case a.Kind == hwnpu.ArgOutput:
				if b, err = alloc(a.Bytes); err != nil {
					t.close()
					return nil, err
				}
				if a.Name == "y" {
					out = b
				} else {
					t.outputs[a.Name], t.outShape[a.Name] = b, a.Dims
				}
			case a.Name == "x" && i > 0:
				b = t.out
				if b.Size() < a.Bytes {
					t.close()
					return nil, fmt.Errorf("chain: part %d takes x of %d bytes, the part before gives %d", i, a.Bytes, b.Size())
				}
			case t.outputs[a.Name] != nil:
				b = t.outputs[a.Name]
				if b.Size() != a.Bytes {
					t.close()
					return nil, fmt.Errorf("chain: part %d takes %s of %d bytes, an earlier part outputs %d", i, a.Name, a.Bytes, b.Size())
				}
				t.passed[a.Name] = true
			default:
				if b = t.inputs[a.Name]; b == nil {
					if b = shared[a.Name]; b != nil {
						if b.Size() != a.Bytes {
							t.close()
							return nil, fmt.Errorf("chain: part %d takes %s of %d bytes, the shared buffer has %d", i, a.Name, a.Bytes, b.Size())
						}
					} else if b, err = alloc(a.Bytes); err != nil {
						t.close()
						return nil, err
					}
					t.inputs[a.Name], t.inShape[a.Name] = b, a.Dims
				}
			}
			if err := p.req.Bind(a.Index, b); err != nil {
				t.close()
				return nil, err
			}
		}
		if out == nil {
			t.close()
			return nil, fmt.Errorf("chain: part %d has no output y", i)
		}
		t.out = out
	}
	t.outputs["y"] = t.out
	for _, a := range t.parts[len(t.parts)-1].g.Args() {
		if a.Kind == hwnpu.ArgOutput && a.Name == "y" {
			t.outShape["y"] = a.Dims
		}
	}
	t.loadTime = time.Since(start)
	return t, nil
}

func (t *tier) close() {
	for _, p := range t.parts {
		if p.req != nil {
			p.req.Close()
		}
		if p.g != nil {
			p.g.Close()
		}
	}
	for _, b := range t.bufs {
		b.Close()
	}
	t.parts, t.bufs = nil, nil
}

func (t *tier) run(ctx context.Context) error {
	for i, p := range t.parts {
		if err := p.req.Run(ctx); err != nil {
			return fmt.Errorf("chain: part %d: %w", i, err)
		}
	}
	return nil
}

// program returns the tier as a decode.Program; Run syncs the buffers around the parts.
func (t *tier) program(ctx context.Context) decode.Program {
	p := decode.Program{In: map[string][]byte{}, Out: map[string][]byte{}, InShape: t.inShape, OutShape: t.outShape}
	for n, b := range t.inputs {
		p.In[n] = b.Bytes()
	}
	for n, b := range t.outputs {
		p.Out[n] = b.Bytes()
	}
	p.Run = func() error {
		for _, b := range t.inputs {
			b.SyncToDevice(0, b.Size())
		}
		if err := t.run(ctx); err != nil {
			return err
		}
		for n, b := range t.outputs {
			if !t.passed[n] {
				b.SyncFromDevice(0, b.Size())
			}
		}
		return nil
	}
	return p
}

// Program returns the loaded tier as a decode.Program: its inputs and outputs by name (y is the
// last part's). After Grow the program is a different one; take it again.
func (c *Chain) Program(ctx context.Context) decode.Program { return c.cur.program(ctx) }

// Input returns the buffer of a model input (x, or another input such as mask).
func (c *Chain) Input(name string) *hwnpu.Buffer { return c.cur.inputs[name] }

// Output returns the buffer of the last part's output.
func (c *Chain) Output() *hwnpu.Buffer { return c.cur.out }

// Parts returns the loaded parts in order.
func (c *Chain) Parts() []*Part { return c.cur.parts }

// LoadTime is how long the loaded tier took to compile and bind.
func (c *Chain) LoadTime() time.Duration { return c.cur.loadTime }

// OnNPU reports whether every loaded part was verified to run on the NPU.
func (c *Chain) OnNPU() bool {
	for _, p := range c.cur.parts {
		if !p.Placement.OnNPU() {
			return false
		}
	}
	return len(c.cur.parts) > 0
}

// Run runs the parts in order.
func (c *Chain) Run(ctx context.Context) error { return c.cur.run(ctx) }

// Tiers returns the KV cache length of every tier (0 for a model without one); the loaded tier is
// Tier().
func (c *Chain) Tiers() []int {
	var out []int
	for _, t := range c.tiers {
		out = append(out, t.Cache)
	}
	return out
}

// Tier is the index of the loaded tier.
func (c *Chain) Tier() int { return c.index }

// HasNext reports whether a longer tier follows the loaded one.
func (c *Chain) HasNext() bool { return c.index+1 < len(c.tiers) }

// LoadAll loads every tier after the loaded one now, so that Grow switches at once and nothing
// loads while the model runs. Every tier is then in memory. For a process that serves many
// sequences, this is the way to use tiers: loading a tier takes 6 to 30 seconds on the M3 and
// slows the predictions running meanwhile to less than half their speed.
func (c *Chain) LoadAll(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.pre != nil {
		return errors.New("chain: a tier is loading in the background")
	}
	if c.sharing {
		return errors.New("chain: a chain sharing another's cache cannot load more tiers")
	}
	for i := c.index + 1 + len(c.ahead); i < len(c.tiers); i++ {
		t, err := c.load(ctx, c.tiers[i].Parts, nil)
		if err != nil {
			return err
		}
		c.ahead = append(c.ahead, t)
	}
	return nil
}

// Preload starts loading the next tier in the background, if there is one and it is not loading
// or loaded already, so that Grow finds it ready. The load competes with the predictions running
// meanwhile (see LoadAll).
func (c *Chain) Preload(ctx context.Context) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.pre != nil || len(c.ahead) > 0 || !c.HasNext() || c.sharing {
		return
	}
	pre := &preload{done: make(chan struct{})}
	c.pre = pre
	parts := c.tiers[c.index+1].Parts
	go func() {
		pre.t, pre.err = c.load(ctx, parts, nil)
		close(pre.done)
	}()
}

// Grow switches to the next tier: it waits for the preload or loads the tier now, calls migrate
// with the new tier's program so that the caller carries the state over (decode.Session.Migrate),
// then releases the old tier. It returns the time spent waiting or loading and whether the tier
// had been preloaded.
func (c *Chain) Grow(ctx context.Context, migrate func(decode.Program) error) (waited time.Duration, preloaded bool, err error) {
	if !c.HasNext() {
		return 0, false, errors.New("chain: no longer tier to grow into")
	}
	if c.sharing {
		return 0, false, errors.New("chain: a chain sharing another's cache cannot grow")
	}
	start := time.Now()
	var next *tier
	c.mu.Lock()
	if len(c.ahead) > 0 {
		next, c.ahead = c.ahead[0], c.ahead[1:]
		preloaded = true
	}
	pre := c.pre
	c.mu.Unlock()
	if next == nil {
		preloaded = pre != nil
		if !preloaded {
			c.Preload(ctx)
			c.mu.Lock()
			pre = c.pre
			c.mu.Unlock()
		}
		select {
		case <-pre.done:
		case <-ctx.Done():
			return time.Since(start), preloaded, ctx.Err()
		}
		c.mu.Lock()
		c.pre = nil
		c.mu.Unlock()
		if pre.err != nil {
			return time.Since(start), preloaded, pre.err
		}
		next = pre.t
	}
	if err := migrate(next.program(ctx)); err != nil {
		next.close()
		return time.Since(start), preloaded, err
	}
	old := c.cur
	c.cur = next
	c.index++
	old.close()
	return time.Since(start), preloaded, nil
}

// Close releases the parts and the buffers, waiting for a preload to finish first.
func (c *Chain) Close() error {
	c.mu.Lock()
	pre := c.pre
	c.pre = nil
	c.mu.Unlock()
	if pre != nil {
		<-pre.done
		if pre.t != nil {
			pre.t.close()
		}
	}
	for _, t := range c.ahead {
		t.close()
	}
	c.ahead = nil
	if c.cur != nil {
		c.cur.close()
	}
	return nil
}
