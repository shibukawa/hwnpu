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
//
// A model split at its attention (llm.Paged, the manifest's Paged and Attention) runs in stages,
// the parts between two attention layers, with the program without weights that attends over a
// chunk of the cache loaded beside them (decode.Program.RunPart and Attn): the session holds that
// cache in buffers of the device (decode.Attention.NewBuf).
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
	// the part's buffers that a stage of a model split at its attention syncs around it
	ins, outs []*hwnpu.Buffer
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

	// of a model split at its attention
	stages  [][]*Part // the parts between two attention layers
	attend  []int     // the layer attended after each stage but the last
	attn    *attention
	paged   int  // the chunk length
	context int  // the longest sequence (the manifest's Cache)
	kvOnly  bool // the parts write the caches and output no logits (llm.Manifest.KVOnly)
}

// attention is the loaded program without weights that attends over one chunk of the cache.
type attention struct {
	g          *hwnpu.Graph
	req        *hwnpu.Request
	k, v, mask int // the arguments bound per call
	q, o, m, l *hwnpu.Buffer
	bufs       []*hwnpu.Buffer
	dev        *hwnpu.Device
	placement  hwnpu.PlacementReport
	mass       *mass // the two-level attention's, nil when the manifest has none
}

// mass is the loaded program without weights that weighs a chunk of bucket summaries for the
// two-level attention (decode.Sparse).
type mass struct {
	g         *hwnpu.Graph
	req       *hwnpu.Request
	k, mask   int
	q, r, w   *hwnpu.Buffer
	bufs      []*hwnpu.Buffer
	placement hwnpu.PlacementReport
}

// Chain is a loaded model in parts.
type Chain struct {
	dev   *hwnpu.Device
	units string // GraphSource.Units of every part
	gr    hwnpu.GraphRunner
	tiers []llm.Tier
	index int   // the tier loaded
	cur   *tier // never nil after Open

	sharing bool              // made by Share: its cache buffers belong to another chain, so it cannot grow
	meta    map[string]string // the parts' metadata (llm.ReadMeta), nil when unread

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
	return open(ctx, dev, man, nil, "")
}

// OpenOn is Open with every part run on the units given (hwnpu.GraphSource.Units: "gpu", "cpu";
// "" the Neural Engine).
func OpenOn(ctx context.Context, dev *hwnpu.Device, path, units string) (*Chain, error) {
	man, err := llm.ReadManifest(path)
	if err != nil {
		return nil, err
	}
	return open(ctx, dev, man, nil, units)
}

// Share loads the model at path like Open, but binds its KV cache inputs (kv_*, win_*, st_*: llm.CacheInput) to the
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
		if llm.CacheInput(name) {
			shared[name] = b
		}
	}
	return open(ctx, dev, man, shared, "")
}

func open(ctx context.Context, dev *hwnpu.Device, man *llm.Manifest, shared map[string]*hwnpu.Buffer, units string) (*Chain, error) {
	gr, err := dev.Graph()
	if err != nil {
		return nil, err
	}
	c := &Chain{dev: dev, gr: gr, tiers: man.Tiered(), sharing: shared != nil, units: units}
	c.meta, _ = llm.ReadMeta(man) // conversions from before the metadata have none
	if c.cur, err = c.load(ctx, c.tiers[0].Parts, shared); err != nil {
		return nil, err
	}
	if man.Paged > 0 {
		if err := c.cur.loadPaged(ctx, c, man); err != nil {
			c.cur.close()
			return nil, err
		}
	}
	return c, nil
}

// loadPaged loads the attention program of a model split at its attention and groups the parts
// into the stages between the attention layers: a stage ends with a part that outputs attn_q_N.
func (t *tier) loadPaged(ctx context.Context, c *Chain, man *llm.Manifest) error {
	if man.Attention == nil || man.Cache <= 0 || len(man.Tiers) > 0 {
		return fmt.Errorf("chain: a model split at its attention needs its attention program and one cache length")
	}
	t.paged, t.context, t.kvOnly = man.Paged, man.Cache, man.KVOnly
	var cur []*Part
	for i, p := range t.parts {
		cur = append(cur, p)
		layer := -1
		for _, a := range p.g.Args() {
			if a.Kind == hwnpu.ArgOutput && strings.HasPrefix(a.Name, "attn_q_") {
				if _, err := fmt.Sscanf(a.Name, "attn_q_%d", &layer); err != nil {
					return fmt.Errorf("chain: part %d: output %s", i, a.Name)
				}
			}
		}
		if layer >= 0 {
			t.stages, t.attend, cur = append(t.stages, cur), append(t.attend, layer), nil
		}
	}
	if len(cur) == 0 && t.kvOnly {
		// the last part computes the keys and values of a layer whose output no cache needs:
		// the session writes them, and nothing attends for it
		t.attend = t.attend[:len(t.attend)-1]
	} else {
		if len(cur) == 0 || len(t.attend) == 0 {
			return fmt.Errorf("chain: the parts of a model split at its attention must alternate with it, ending with a part")
		}
		t.stages = append(t.stages, cur)
	}
	a := &attention{dev: c.dev}
	t.attn = a
	var err error
	if a.g, err = c.gr.Compile(ctx, hwnpu.GraphSource{Path: man.Attention.Path, Units: c.units}); err != nil {
		return fmt.Errorf("chain: the attention program (%s): %w", man.Attention.Path, err)
	}
	a.placement = a.g.Placement()
	if a.req, err = a.g.NewRequest(); err != nil {
		return err
	}
	for _, arg := range a.g.Args() {
		switch arg.Name {
		case "k":
			a.k = arg.Index
			continue
		case "v":
			a.v = arg.Index
			continue
		case "mask":
			a.mask = arg.Index
			continue
		}
		b, err := c.dev.AllocBuffer(arg.Bytes)
		if err != nil {
			return err
		}
		a.bufs = append(a.bufs, b)
		if err := a.req.Bind(arg.Index, b); err != nil {
			return err
		}
		switch arg.Name {
		case "q":
			a.q = b
		case "o":
			a.o = b
		case "m":
			a.m = b
		case "l":
			a.l = b
		}
	}
	if a.q == nil || a.o == nil || a.m == nil || a.l == nil {
		return fmt.Errorf("chain: the attention program does not have q, o, m and l")
	}
	if man.Mass != nil {
		if a.mass, err = loadMass(ctx, c, man.Mass.Path); err != nil {
			return err
		}
	}
	return nil
}

// loadMass loads the mass program of the two-level attention.
func loadMass(ctx context.Context, c *Chain, path string) (*mass, error) {
	ms := &mass{}
	var err error
	if ms.g, err = c.gr.Compile(ctx, hwnpu.GraphSource{Path: path, Units: c.units}); err != nil {
		return nil, fmt.Errorf("chain: the mass program (%s): %w", path, err)
	}
	ms.placement = ms.g.Placement()
	if ms.req, err = ms.g.NewRequest(); err != nil {
		return nil, err
	}
	for _, arg := range ms.g.Args() {
		switch arg.Name {
		case "k":
			ms.k = arg.Index
			continue
		case "mask":
			ms.mask = arg.Index
			continue
		}
		b, err := c.dev.AllocBuffer(arg.Bytes)
		if err != nil {
			return nil, err
		}
		ms.bufs = append(ms.bufs, b)
		if err := ms.req.Bind(arg.Index, b); err != nil {
			return nil, err
		}
		switch arg.Name {
		case "q":
			ms.q = b
		case "r":
			ms.r = b
		case "w":
			ms.w = b
		}
	}
	if ms.q == nil || ms.r == nil || ms.w == nil {
		return nil, fmt.Errorf("chain: the mass program does not have q, r and w")
	}
	return ms, nil
}

// program is the attention as decode takes it, run under ctx (that of the tier's program, not
// the load's: hwnpu-chat cancels the context it loads under once the model is up).
func (a *attention) program(ctx context.Context, chunk int) *decode.Attention {
	return &decode.Attention{
		Chunk: chunk,
		Q:     a.q.Bytes(), O: a.o.Bytes(), M: a.m.Bytes(), L: a.l.Bytes(),
		NewBuf: func(n int) (decode.Buf, error) { return a.dev.AllocBuffer(n) },
		Run: func(k, v, mask decode.Buf) error {
			for _, x := range []struct {
				i int
				b decode.Buf
			}{{a.k, k}, {a.v, v}, {a.mask, mask}} {
				hb, ok := x.b.(*hwnpu.Buffer)
				if !ok {
					return fmt.Errorf("chain: a chunk not allocated by the attention program")
				}
				hb.SyncToDevice(0, hb.Size())
				if err := a.req.Bind(x.i, hb); err != nil {
					return err
				}
			}
			a.q.SyncToDevice(0, a.q.Size())
			if err := a.req.Run(ctx); err != nil {
				return err
			}
			for _, b := range []*hwnpu.Buffer{a.o, a.m, a.l} {
				b.SyncFromDevice(0, b.Size())
			}
			return nil
		},
		Mass: a.mass.program(ctx),
	}
}

// program is the mass program as decode takes it, nil for none.
func (ms *mass) program(ctx context.Context) *decode.Mass {
	if ms == nil {
		return nil
	}
	return &decode.Mass{
		Q: ms.q.Bytes(), R: ms.r.Bytes(), W: ms.w.Bytes(),
		Run: func(k, mask decode.Buf) error {
			for _, x := range []struct {
				i int
				b decode.Buf
			}{{ms.k, k}, {ms.mask, mask}} {
				hb, ok := x.b.(*hwnpu.Buffer)
				if !ok {
					return fmt.Errorf("chain: a chunk not allocated by the attention program")
				}
				hb.SyncToDevice(0, hb.Size())
				if err := ms.req.Bind(x.i, hb); err != nil {
					return err
				}
			}
			ms.q.SyncToDevice(0, ms.q.Size())
			ms.r.SyncToDevice(0, ms.r.Size())
			if err := ms.req.Run(ctx); err != nil {
				return err
			}
			ms.w.SyncFromDevice(0, ms.w.Size())
			return nil
		},
	}
}

func (ms *mass) close() {
	if ms == nil {
		return
	}
	if ms.req != nil {
		ms.req.Close()
	}
	if ms.g != nil {
		ms.g.Close()
	}
	for _, b := range ms.bufs {
		b.Close()
	}
}

func (a *attention) close() {
	a.mass.close()
	if a.req != nil {
		a.req.Close()
	}
	if a.g != nil {
		a.g.Close()
	}
	for _, b := range a.bufs {
		b.Close()
	}
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
		if p.g, err = c.gr.Compile(ctx, hwnpu.GraphSource{Path: lp.Path, Units: c.units}); err != nil {
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
			if a.Kind == hwnpu.ArgOutput {
				p.outs = append(p.outs, b)
			} else {
				p.ins = append(p.ins, b)
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
	if t.attn != nil {
		t.attn.close()
		t.attn = nil
	}
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
	if t.attn != nil {
		p.Attend, p.Context, p.Attn, p.NoLogits = t.attend, t.context, t.attn.program(ctx, t.paged), t.kvOnly
		p.RunPart = func(i int) error {
			for _, part := range t.stages[i] {
				for _, b := range part.ins {
					b.SyncToDevice(0, b.Size())
				}
				if err := part.req.Run(ctx); err != nil {
					return fmt.Errorf("chain: %s: %w", part.Path, err)
				}
				for _, b := range part.outs {
					b.SyncFromDevice(0, b.Size())
				}
			}
			return nil
		}
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

// Meta is the metadata of the parts (llm.ReadMeta), nil for a conversion from before it.
func (c *Chain) Meta() map[string]string { return c.meta }

// Program returns the loaded tier as a decode.Program: its inputs and outputs by name (y is the
// last part's) and the parts' metadata. After Grow the program is a different one; take it again.
func (c *Chain) Program(ctx context.Context) decode.Program {
	p := c.cur.program(ctx)
	p.Meta = c.meta
	return p
}

// Input returns the buffer of a model input (x, or another input such as mask).
func (c *Chain) Input(name string) *hwnpu.Buffer { return c.cur.inputs[name] }

// Output returns the buffer of the last part's output.
func (c *Chain) Output() *hwnpu.Buffer { return c.cur.out }

// Parts returns the loaded parts in order.
func (c *Chain) Parts() []*Part { return c.cur.parts }

// LoadTime is how long the loaded tier took to compile and bind.
func (c *Chain) LoadTime() time.Duration { return c.cur.loadTime }

// OnNPU reports whether every loaded part, and the attention program of a model split at its
// attention (with its mass program), was verified to run on the NPU.
func (c *Chain) OnNPU() bool {
	for _, p := range c.cur.parts {
		if !p.Placement.OnNPU() {
			return false
		}
	}
	if a := c.cur.attn; a != nil && (!a.placement.OnNPU() || a.mass != nil && !a.mass.placement.OnNPU()) {
		return false
	}
	return len(c.cur.parts) > 0
}

// HasMass reports whether the model split at its attention has the mass program of the two-level
// attention (decode.Sparse).
func (c *Chain) HasMass() bool { return c.cur.attn != nil && c.cur.attn.mass != nil }

// Paged is the chunk length of a model split at its attention, 0 for another.
func (c *Chain) Paged() int { return c.cur.paged }

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
