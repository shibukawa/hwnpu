package flow

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"sync"
)

// A network is nodes joined by edges: sources (Source), nodes of one input and one output
// (Func, Each, Tap, and the nodes of models the packages of models make), and sinks (Sink).
// A node's output may go to several nodes; a node's input comes from one.
//
//	mic := flow.Source[[]float32]("mic").From(stream)
//	log := flow.Tap("heard", func(c flow.Chunk[flow.Text]) { fmt.Println(c.Data.Text) })
//	out := flow.Sink[[]float32]("speech").Do(play)
//	mic.Connect(vad)
//	vad.Connect(hear)
//	hear.Connect(log)
//	log.Connect(speak, flow.Keep(4))
//	speak.Connect(out)
//	err := flow.Run(ctx, mic)
//
// The network is the same streams joined the same way (a node is a function from a stream to
// a stream, an edge passes one's chunks on), written down as a structure, so that what runs it
// knows the models it holds and how they are joined.

// element is a part of a network.
type element interface{ node() *base }

// base is what every part of a network has.
type base struct {
	name string
	kind string
	// Meta is what the node carries for whoever builds on the network: the ai package keeps
	// the model a node runs here.
	Meta any

	fed  string // the node its input comes from, "" for none yet
	outs []outRef
}

// outRef is an edge out of a node, as Walk sees it.
type outRef struct {
	to     element
	policy Policy
}

func (b *base) node() *base { return b }

// Name is the node's name, for errors and for showing the network.
func (b *base) Name() string { return b.name }

// Input is what takes chunks of T: a node whose input is T, or a sink of T. Only this
// package's types are Inputs.
type Input[T any] interface {
	element
	start(ctx context.Context, in Stream[T], r *running)
}

// Root is where a network starts: a Source.
type Root interface {
	element
	begin(ctx context.Context, r *running) error
}

// Policy is how an edge passes chunks on.
type Policy struct {
	kind policyKind
	n    int
}

type policyKind int

const (
	direct policyKind = iota
	keep
	newest
)

// Keep makes an edge keep up to n chunks for the node after it, read in a goroutine of its
// own (Async): the node before goes on while the node after is busy, and no chunk is dropped.
// It is for chunks that must all arrive: sentences on their way to a speaker.
func Keep(n int) Policy { return Policy{keep, n} }

// Newest makes an edge pass on the newest chunk only (Latest): what came while the node after
// was busy is dropped. It is for chunks that go stale: a camera's frames into a detector.
func Newest() Policy { return Policy{kind: newest} }

func (p Policy) String() string {
	switch p.kind {
	case keep:
		return "keep " + strconv.Itoa(p.n)
	case newest:
		return "newest"
	}
	return ""
}

// through is s as the policy passes it on.
func through[T any](ctx context.Context, p Policy, s Stream[T]) Stream[T] {
	switch p.kind {
	case keep:
		return Async(ctx, s, p.n)
	case newest:
		return Latest(ctx, s)
	}
	return s
}

// edge is an edge out of a node of output T.
type edge[T any] struct {
	to     Input[T]
	policy Policy
}

// outputs are the edges out of a node of output T.
type outputs[T any] struct{ edges []edge[T] }

// connect adds the edge from → to.
func connect[T any](from *base, o *outputs[T], to Input[T], ps []Policy) {
	var p Policy
	if len(ps) > 0 {
		p = ps[len(ps)-1]
	}
	tb := to.node()
	if tb.fed != "" {
		panic(fmt.Sprintf("flow: %s already has its input from %s: a node has one input (%s cannot be connected to it too)", tb.name, tb.fed, from.name))
	}
	tb.fed = from.name
	o.edges = append(o.edges, edge[T]{to, p})
	from.outs = append(from.outs, outRef{to, p})
}

// fan gives the stream s of the node called name to the nodes after it: each edge its own
// reader, all chunks to each. Without an edge the stream is read to its end and dropped.
func (o *outputs[T]) fan(ctx context.Context, name string, s Stream[T], r *running) {
	switch len(o.edges) {
	case 0:
		drain(r, s, nil)
	case 1:
		e := o.edges[0]
		e.to.start(ctx, through(ctx, e.policy, s), r)
	default:
		for i, b := range broadcast(r, s, len(o.edges)) {
			e := o.edges[i]
			e.to.start(ctx, through(ctx, e.policy, b), r)
		}
	}
}

// running is a network being run.
type running struct {
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	mu  sync.Mutex
	err error
}

// fail ends the run with err, unless it has failed already.
func (r *running) fail(err error) {
	r.mu.Lock()
	if r.err == nil {
		r.err = err
	}
	r.mu.Unlock()
	r.cancel()
}

// drain reads s to its end in a goroutine, giving each chunk to fn (nil for none).
func drain[T any](r *running, s Stream[T], fn func(Chunk[T]) error) {
	r.wg.Go(func() {
		for c, err := range s {
			if err != nil {
				r.fail(err)
				return
			}
			if fn != nil {
				if err := fn(c); err != nil {
					r.fail(err)
					return
				}
			}
		}
	})
}

// broadcast reads s in a goroutine and gives every chunk to each of k streams, which are read
// independently: what a slower reader has not read yet waits for it in memory, and a reader
// that waits for the next chunk does not hold the others back. A reader that stops early stops
// receiving; the reading stops when all have stopped.
func broadcast[T any](r *running, s Stream[T], k int) []Stream[T] {
	type queue struct {
		items   []item[T]
		ended   bool // the source has ended
		stopped bool // the reader has stopped
		wake    chan struct{}
	}
	var mu sync.Mutex
	qs := make([]*queue, k)
	for i := range qs {
		qs[i] = &queue{wake: make(chan struct{}, 1)}
	}
	signal := func(q *queue) {
		select {
		case q.wake <- struct{}{}:
		default:
		}
	}
	r.wg.Go(func() {
		defer func() {
			mu.Lock()
			for _, q := range qs {
				q.ended = true
				signal(q)
			}
			mu.Unlock()
		}()
		for c, err := range s {
			live := 0
			mu.Lock()
			for _, q := range qs {
				if !q.stopped {
					q.items = append(q.items, item[T]{c, err})
					signal(q)
					live++
				}
			}
			mu.Unlock()
			if err != nil || live == 0 || r.ctx.Err() != nil {
				return
			}
		}
	})
	out := make([]Stream[T], k)
	for i, q := range qs {
		out[i] = func(yield func(Chunk[T], error) bool) {
			defer func() {
				mu.Lock()
				q.stopped, q.items = true, nil
				mu.Unlock()
			}()
			for {
				mu.Lock()
				var (
					it item[T]
					ok bool
				)
				if len(q.items) > 0 {
					it, ok = q.items[0], true
					q.items = q.items[1:]
				}
				ended := q.ended
				mu.Unlock()
				switch {
				case ok:
					if !yield(it.c, it.err) || it.err != nil {
						return
					}
				case ended:
					return
				default:
					select {
					case <-q.wake:
					case <-r.ctx.Done():
						yield(Chunk[T]{}, r.ctx.Err())
						return
					}
				}
			}
		}
	}
	return out
}

// NodeError is the error of a node of a network: Run's error says which node failed.
type NodeError struct {
	Node string
	Err  error
}

func (e *NodeError) Error() string { return e.Node + ": " + e.Err.Error() }
func (e *NodeError) Unwrap() error { return e.Err }

// named gives the errors that begin in the node called name its name. An error that comes
// through from a node before keeps that node's.
func named[T any](name string, s Stream[T]) Stream[T] {
	return func(yield func(Chunk[T], error) bool) {
		for c, err := range s {
			if err != nil {
				if ne := (*NodeError)(nil); !errors.As(err, &ne) && !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
					err = &NodeError{Node: name, Err: err}
				}
				yield(c, err)
				return
			}
			if !yield(c, nil) {
				return
			}
		}
	}
}

// Src is a source of a network: where its chunks come from (Source).
type Src[T any] struct {
	base
	outputs[T]
	s Stream[T]
}

// Source is a source of chunks of T called name, to be given its stream with From: a
// microphone, a recording, a camera. A network is declared with its sources and given their
// streams when it is run.
func Source[T any](name string) *Src[T] {
	return &Src[T]{base: base{name: name, kind: "source"}}
}

// From gives the source its stream, read when the network runs. A live source's stream is to
// keep what its reader is late for (flow.Async), or the edge after it drops it (Newest).
func (s *Src[T]) From(stream Stream[T]) *Src[T] {
	s.s = stream
	return s
}

// Connect passes the source's chunks on to a node or a sink, as the policy says (none: as the
// node after reads them). Several Connects give each all the chunks.
func (s *Src[T]) Connect(to Input[T], p ...Policy) { connect(&s.base, &s.outputs, to, p) }

func (s *Src[T]) begin(ctx context.Context, r *running) error {
	if s.s == nil {
		return fmt.Errorf("flow: the source %s has no stream (From gives it one)", s.name)
	}
	s.fan(ctx, s.name, named(s.name, s.s), r)
	return nil
}

// Node is a node of one input and one output: a function from a stream of A to a stream of B.
type Node[A, B any] struct {
	base
	outputs[B]
	fn func(ctx context.Context, in Stream[A]) Stream[B]
}

// Func makes a node of a function over streams, which has the whole say over what goes on:
// it may keep chunks until it has enough, give several for one, or none.
func Func[A, B any](name string, fn func(ctx context.Context, in Stream[A]) Stream[B]) *Node[A, B] {
	return &Node[A, B]{base: base{name: name, kind: "node"}, fn: fn}
}

// Stop is what the callback of Each returns to end its node without an error: the node gives
// nothing more, the nodes before it stop (a microphone is closed), and the nodes after it work
// off what they were given. It is how a node ends a network from inside: a spoken "that's all".
var Stop = errors.New("flow: stop")

// Each makes a node of a callback: fn is called with each chunk that comes, and what it gives
// to emit goes on to the nodes after, nothing, one chunk or several. A node that waits for the
// end of a sentence keeps the text in a variable and emits when the end comes; a node that
// hears a command changes another node's settings and emits nothing. end (optional) is called
// once the input has ended, to emit what was kept. fn returning Stop ends the node there; any
// other error of fn or end ends the network.
func Each[A, B any](name string, fn func(c Chunk[A], emit func(Chunk[B])) error, end ...func(emit func(Chunk[B])) error) *Node[A, B] {
	return Func(name, func(ctx context.Context, in Stream[A]) Stream[B] {
		return func(yield func(Chunk[B], error) bool) {
			stopped := false
			emit := func(c Chunk[B]) {
				if !stopped && !yield(c, nil) {
					stopped = true
				}
			}
			fail := func(err error) {
				if !stopped {
					yield(Chunk[B]{}, err)
				}
			}
			for c, err := range in {
				if err != nil {
					fail(err)
					return
				}
				if err := fn(c, emit); err != nil {
					if !errors.Is(err, Stop) {
						fail(err)
					}
					return
				}
				if stopped {
					return
				}
			}
			for _, e := range end {
				if stopped {
					return
				}
				if err := e(emit); err != nil {
					fail(err)
					return
				}
			}
		}
	})
}

// Tap makes a node that passes every chunk on as it is after showing it to fn: a log, a
// counter, a caption on a screen.
func Tap[T any](name string, fn func(Chunk[T])) *Node[T, T] {
	return Each(name, func(c Chunk[T], emit func(Chunk[T])) error {
		fn(c)
		emit(c)
		return nil
	})
}

// Connect passes the node's output on to a node or a sink, as the policy says (none: as the
// node after reads it). Several Connects give each all the chunks.
func (n *Node[A, B]) Connect(to Input[B], p ...Policy) { connect(&n.base, &n.outputs, to, p) }

func (n *Node[A, B]) start(ctx context.Context, in Stream[A], r *running) {
	n.fan(ctx, n.name, named(n.name, n.fn(ctx, in)), r)
}

// Snk is a sink of a network: where its chunks go (Sink).
type Snk[T any] struct {
	base
	fn func(Chunk[T]) error
}

// Sink is a sink of chunks of T called name, to be given what it does with them by Do: a
// player, a file, a screen.
func Sink[T any](name string) *Snk[T] {
	return &Snk[T]{base: base{name: name, kind: "sink"}}
}

// Do gives the sink what it does with each chunk, in the sink's goroutine. An error ends the
// network.
func (s *Snk[T]) Do(fn func(Chunk[T]) error) *Snk[T] {
	s.fn = fn
	return s
}

func (s *Snk[T]) start(ctx context.Context, in Stream[T], r *running) {
	name, fn := s.name, s.fn
	drain(r, in, func(c Chunk[T]) error {
		if fn == nil {
			return nil
		}
		if err := fn(c); err != nil {
			return &NodeError{Node: name, Err: err}
		}
		return nil
	})
}

// Run runs the network that starts at the sources: every node in goroutines as its edges
// need, until every source has ended and every chunk has gone through, an error (Run's error
// is a *NodeError that names the node it began in), or the cancel of ctx.
func Run(ctx context.Context, sources ...Root) error {
	if len(sources) == 0 {
		return errors.New("flow: a network needs a source")
	}
	parent := ctx
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	r := &running{ctx: ctx, cancel: cancel}
	for _, s := range sources {
		if err := s.begin(ctx, r); err != nil {
			r.fail(err)
			break
		}
	}
	r.wg.Wait()
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.err == nil {
		return parent.Err()
	}
	return r.err
}

// NodeInfo describes a node of a network (Walk).
type NodeInfo struct {
	Name string
	// Kind is source, node or sink.
	Kind string
	// Meta is what the node carries (the ai package: the model it runs).
	Meta any
	// To are the nodes its output goes to.
	To []EdgeInfo
}

// EdgeInfo describes an edge of a network.
type EdgeInfo struct {
	// To is the name of the node the edge goes to; Policy how it passes chunks on ("" for as
	// they are read, "keep N", "newest").
	To, Policy string
}

// Walk describes the nodes of the network that starts at the sources, each once, a node
// before the nodes its output goes to.
func Walk(sources ...Root) []NodeInfo {
	var out []NodeInfo
	seen := map[*base]bool{}
	var visit func(e element)
	visit = func(e element) {
		b := e.node()
		if seen[b] {
			return
		}
		seen[b] = true
		info := NodeInfo{Name: b.name, Kind: b.kind, Meta: b.Meta}
		for _, o := range b.outs {
			info.To = append(info.To, EdgeInfo{To: o.to.node().name, Policy: o.policy.String()})
		}
		out = append(out, info)
		for _, o := range b.outs {
			visit(o.to)
		}
	}
	for _, s := range sources {
		visit(s)
	}
	return out
}
