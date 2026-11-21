package flow

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"sync"
	"time"
)

// A network is nodes joined by edges: sources (Source), nodes of one input and one output
// (Func, Each, Tap, and the nodes of models the packages of models make), and sinks (Sink).
// A node's output may go to several nodes; a node's input comes from one.
//
//	mic := flow.Source[[]float32]("mic").From(stream)
//	log := flow.Tap("heard", func(c flow.Chunk[flow.Text]) { fmt.Println(c.Data.Text) })
//	out := flow.Sink[[]float32]("speech").Do(play)
//	mic.To(vad).To(hear).To(log).To(speak, flow.Keep(4)).To(out)
//	err := flow.Run(ctx, mic)
//
// The network is the same streams joined the same way (a node is a function from a stream to
// a stream, an edge passes one's chunks on), written down as a structure, so that what runs it
// knows the models it holds and how they are joined.

// Element is a part of a network: a source, a node or a sink. Only this package's types (and
// those that embed them) are Elements.
type Element interface {
	node() *base
	// inType and outType are the types of the chunks it takes and gives, nil for none.
	inType() reflect.Type
	outType() reflect.Type
	// linkTo joins its output to another element, whose input is of its output's type.
	linkTo(to Element, ps []Policy)
}

// Link is a place in a network to go on from: what To returns, so that joins are written in a
// row.
//
//	mic.To(vad).To(hear).To(trans).To(speak, flow.Keep(4)).To(out)
//
// A node's own To has the compiler check that the next element takes what the node gives. A
// Link's To checks it when it is called, as the network is built, and panics on a mismatch
// (Go's methods cannot carry the type along a row of calls): either way, before anything runs.
type Link struct{ at Element }

// Chain is the place to go on from e.
func Chain(e Element) *Link { return &Link{e} }

// To passes the output of the element before on to next, as the policy says, and returns the
// place to go on from next. Called twice on one Link, it gives both all the chunks.
func (l *Link) To(next Element, p ...Policy) *Link {
	check(l.at, next)
	l.at.linkTo(next, p)
	return &Link{next}
}

// check panics when to does not take what from gives.
func check(from, to Element) {
	out, in := from.outType(), to.inType()
	switch {
	case out == nil:
		panic(fmt.Sprintf("flow: %s is a sink: nothing comes after it (%s)", from.node().name, to.node().name))
	case in == nil:
		panic(fmt.Sprintf("flow: %s is a source: nothing comes before it (%s)", to.node().name, from.node().name))
	case out != in:
		panic(fmt.Sprintf("flow: %s gives %v and %s takes %v", from.node().name, out, to.node().name, in))
	}
}

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
	to     Element
	policy Policy
}

func (b *base) node() *base { return b }

// Name is the node's name, for errors and for showing the network.
func (b *base) Name() string { return b.name }

// Input is what takes chunks of T: a node whose input is T, or a sink of T. Only this
// package's types are Inputs.
type Input[T any] interface {
	Element
	start(ctx context.Context, in Stream[T], r *running)
}

// Root is where a network starts: a Source.
type Root interface {
	Element
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

// fanout gives the chunks of one output to the readers of its edges, each with a queue of its
// own: what a slower reader has not read yet waits for it in memory, and a reader that waits
// for its next chunk does not hold the others back. A reader that stops early stops receiving.
type fanout[T any] struct {
	r  *running
	mu sync.Mutex
	qs []*fanQueue[T]
}

type fanQueue[T any] struct {
	items   []item[T]
	ended   bool // the source has ended
	stopped bool // the reader has stopped
	wake    chan struct{}
}

func newFanout[T any](r *running, k int) *fanout[T] {
	f := &fanout[T]{r: r, qs: make([]*fanQueue[T], k)}
	for i := range f.qs {
		f.qs[i] = &fanQueue[T]{wake: make(chan struct{}, 1)}
	}
	return f
}

func (q *fanQueue[T]) signal() {
	select {
	case q.wake <- struct{}{}:
	default:
	}
}

// push gives it to reader to, or to every reader when to is negative, and reports how many
// readers are still reading.
func (f *fanout[T]) push(to int, it item[T]) (live int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for i, q := range f.qs {
		if q.stopped {
			continue
		}
		live++
		if to < 0 || to == i {
			q.items = append(q.items, it)
			q.signal()
		}
	}
	return live
}

// end tells the readers that nothing more comes.
func (f *fanout[T]) end() {
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, q := range f.qs {
		q.ended = true
		q.signal()
	}
}

// stream is reader i's stream.
func (f *fanout[T]) stream(i int) Stream[T] {
	q := f.qs[i]
	return func(yield func(Chunk[T], error) bool) {
		defer func() {
			f.mu.Lock()
			q.stopped, q.items = true, nil
			f.mu.Unlock()
		}()
		for {
			f.mu.Lock()
			var (
				it item[T]
				ok bool
			)
			if len(q.items) > 0 {
				it, ok = q.items[0], true
				q.items = q.items[1:]
			}
			ended := q.ended
			f.mu.Unlock()
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
				case <-f.r.ctx.Done():
					yield(Chunk[T]{}, f.r.ctx.Err())
					return
				}
			}
		}
	}
}

// broadcast reads s in a goroutine and gives every chunk to each of k streams (fanout). The
// reading stops when all the readers have stopped.
func broadcast[T any](r *running, s Stream[T], k int) []Stream[T] {
	f := newFanout[T](r, k)
	r.wg.Go(func() {
		defer f.end()
		for c, err := range s {
			if live := f.push(-1, item[T]{c, err}); err != nil || live == 0 || r.ctx.Err() != nil {
				return
			}
		}
	})
	out := make([]Stream[T], k)
	for i := range out {
		out[i] = f.stream(i)
	}
	return out
}

// routed is a chunk on its way to one of a node's edges.
type routed[T any] struct {
	to int
	c  Chunk[T]
}

// route gives the chunks of s to the nodes after the one called name, each chunk to the edge
// it names.
func (o *outputs[T]) route(ctx context.Context, name string, s Stream[routed[T]], r *running) {
	f := newFanout[T](r, len(o.edges))
	r.wg.Go(func() {
		defer f.end()
		for rc, err := range s {
			if err == nil && (rc.Data.to < 0 || rc.Data.to >= len(o.edges)) {
				err = &NodeError{Node: name, Err: fmt.Errorf("it gave a chunk to its output %d, and has %d (the outputs are numbered from 0, in the order of its Tos)", rc.Data.to, len(o.edges))}
			}
			if err != nil {
				f.push(-1, item[T]{err: err})
				return
			}
			if live := f.push(rc.Data.to, item[T]{c: rc.Data.c}); live == 0 || r.ctx.Err() != nil {
				return
			}
		}
	})
	for i, e := range o.edges {
		e.to.start(ctx, through(ctx, e.policy, f.stream(i)), r)
	}
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

// To passes the source's chunks on to a node or a sink, as the policy says (none: as the node
// after reads them), and returns the place to go on from there. Several Tos give each all the
// chunks.
func (s *Src[T]) To(to Input[T], p ...Policy) *Link {
	connect(&s.base, &s.outputs, to, p)
	return &Link{to}
}

func (s *Src[T]) inType() reflect.Type  { return nil }
func (s *Src[T]) outType() reflect.Type { return reflect.TypeFor[T]() }
func (s *Src[T]) linkTo(to Element, ps []Policy) {
	connect(&s.base, &s.outputs, to.(Input[T]), ps)
}

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
	// routing is fn for a node that says which of its outputs each chunk goes to (Route).
	routing func(ctx context.Context, in Stream[A]) Stream[routed[B]]
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

// EachOption is a setting of Each and Route.
type EachOption[B any] func(*eachOptions[B])

type eachOptions[B any] struct {
	end  func(emit func(Chunk[B])) error
	wait time.Duration
	idle func(emit func(Chunk[B])) error
}

// AtEnd has fn called once the node's input has ended, to emit what the callback kept.
func AtEnd[B any](fn func(emit func(Chunk[B])) error) EachOption[B] {
	return func(o *eachOptions[B]) { o.end = fn }
}

// WhenIdle has fn called when no chunk has come for wait since the last one, to emit what the
// callback kept so that it does not wait for ever: a node that holds text until a sentence ends
// lets it go when the speaker has stopped in the middle of one. fn is called once for each
// such pause.
func WhenIdle[B any](wait time.Duration, fn func(emit func(Chunk[B])) error) EachOption[B] {
	return func(o *eachOptions[B]) { o.wait, o.idle = wait, fn }
}

// Each makes a node of a callback: fn is called with each chunk that comes, and what it gives
// to emit goes on to the nodes after, nothing, one chunk or several. The callback decides when
// the nodes after it run: a node that waits for the end of a sentence keeps the text in a
// variable and emits when the end comes (AtEnd and WhenIdle let it emit what is left); a node
// that hears a command changes another node's settings and emits nothing. fn returning Stop
// ends the node there; any other error ends the network.
func Each[A, B any](name string, fn func(c Chunk[A], emit func(Chunk[B])) error, opts ...EachOption[B]) *Node[A, B] {
	return Func(name, func(ctx context.Context, in Stream[A]) Stream[B] {
		return each(ctx, in, opts, func(c Chunk[A], emit func(Chunk[B])) error { return fn(c, emit) })
	})
}

// Route makes a node of a callback that says where each chunk goes: emit(to, c) gives c to the
// node's output to, 0 for the node of its first To, 1 for that of the second. A chunk is given
// to one output, to several (emit called for each), or to none. It is how a program sorts what
// passes: commands one way and what is to be translated the other, each language to its own
// translator. The options' emit gives to the first output.
func Route[A, B any](name string, fn func(c Chunk[A], emit func(to int, c Chunk[B])) error, opts ...EachOption[B]) *Node[A, B] {
	n := &Node[A, B]{base: base{name: name, kind: "node"}}
	n.routing = func(ctx context.Context, in Stream[A]) Stream[routed[B]] {
		var ropts []EachOption[routed[B]]
		for _, opt := range opts {
			var o eachOptions[B]
			opt(&o)
			first := func(emit func(Chunk[routed[B]])) func(Chunk[B]) {
				return func(c Chunk[B]) { emit(Chunk[routed[B]]{Data: routed[B]{0, c}}) }
			}
			if end := o.end; end != nil {
				ropts = append(ropts, AtEnd(func(emit func(Chunk[routed[B]])) error { return end(first(emit)) }))
			}
			if idle := o.idle; idle != nil {
				ropts = append(ropts, WhenIdle(o.wait, func(emit func(Chunk[routed[B]])) error { return idle(first(emit)) }))
			}
		}
		return each(ctx, in, ropts, func(c Chunk[A], emit func(Chunk[routed[B]])) error {
			return fn(c, func(to int, c Chunk[B]) { emit(Chunk[routed[B]]{Data: routed[B]{to, c}}) })
		})
	}
	return n
}

// each is the stream of a callback node.
func each[A, B any](ctx context.Context, in Stream[A], opts []EachOption[B], fn func(c Chunk[A], emit func(Chunk[B])) error) Stream[B] {
	var o eachOptions[B]
	for _, opt := range opts {
		opt(&o)
	}
	return func(yield func(Chunk[B], error) bool) {
		stopped := false
		emit := func(c Chunk[B]) {
			if !stopped && !yield(c, nil) {
				stopped = true
			}
		}
		// call runs a callback and reports whether the node goes on
		call := func(f func() error) bool {
			if err := f(); err != nil {
				if !errors.Is(err, Stop) && !stopped {
					yield(Chunk[B]{}, err)
				}
				return false
			}
			return !stopped
		}
		ended := func() {
			if o.end != nil {
				call(func() error { return o.end(emit) })
			}
		}
		if o.idle == nil {
			for c, err := range in {
				if err != nil {
					if !stopped {
						yield(Chunk[B]{}, err)
					}
					return
				}
				if !call(func() error { return fn(c, emit) }) {
					return
				}
			}
			ended()
			return
		}
		// with WhenIdle the input is read in a goroutine of its own, so that a pause is seen
		rctx, cancel := context.WithCancel(ctx)
		ch := make(chan item[A])
		done := make(chan struct{})
		go func() {
			defer close(done)
			defer close(ch)
			for c, err := range in {
				select {
				case ch <- item[A]{c, err}:
				case <-rctx.Done():
					return
				}
				if err != nil {
					return
				}
			}
		}()
		defer func() {
			cancel()
			<-done
		}()
		timer := time.NewTimer(o.wait)
		timer.Stop()
		defer timer.Stop()
		waiting := false // a chunk came and no pause was reported since
		for {
			var pause <-chan time.Time
			if waiting {
				pause = timer.C
			}
			select {
			case it, ok := <-ch:
				if !ok {
					ended()
					return
				}
				if it.err != nil {
					if !stopped {
						yield(Chunk[B]{}, it.err)
					}
					return
				}
				if !call(func() error { return fn(it.c, emit) }) {
					return
				}
				timer.Reset(o.wait)
				waiting = true
			case <-pause:
				waiting = false
				if !call(func() error { return o.idle(emit) }) {
					return
				}
			case <-ctx.Done():
				if !stopped {
					yield(Chunk[B]{}, ctx.Err())
				}
				return
			}
		}
	}
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

// To passes the node's output on to a node or a sink, as the policy says (none: as the node
// after reads it), and returns the place to go on from there. Several Tos give each all the
// chunks.
func (n *Node[A, B]) To(to Input[B], p ...Policy) *Link {
	connect(&n.base, &n.outputs, to, p)
	return &Link{to}
}

func (n *Node[A, B]) inType() reflect.Type  { return reflect.TypeFor[A]() }
func (n *Node[A, B]) outType() reflect.Type { return reflect.TypeFor[B]() }
func (n *Node[A, B]) linkTo(to Element, ps []Policy) {
	connect(&n.base, &n.outputs, to.(Input[B]), ps)
}

func (n *Node[A, B]) start(ctx context.Context, in Stream[A], r *running) {
	if n.routing != nil {
		n.route(ctx, n.name, named(n.name, n.routing(ctx, in)), r)
		return
	}
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

func (s *Snk[T]) inType() reflect.Type  { return reflect.TypeFor[T]() }
func (s *Snk[T]) outType() reflect.Type { return nil }
func (s *Snk[T]) linkTo(Element, []Policy) {
	panic("flow: " + s.name + " is a sink: nothing comes after it")
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
	if err := complete(sources); err != nil {
		return err
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

// elements calls fn with every element of the network that starts at the sources, each once,
// an element before those its output goes to. An input of a node with several stands for its
// node.
func elements(sources []Root, fn func(e Element)) {
	seen := map[*base]bool{}
	var visit func(e Element)
	visit = func(e Element) {
		if p, ok := e.(ported); ok {
			e, _ = p.of()
		}
		b := e.node()
		if seen[b] {
			return
		}
		seen[b] = true
		fn(e)
		for _, o := range b.outs {
			visit(o.to)
		}
	}
	for _, s := range sources {
		visit(s)
	}
}

// complete finds a network that cannot run as it is: a node of two inputs with only one given.
func complete(sources []Root) (err error) {
	elements(sources, func(e Element) {
		if u, ok := e.(interface{ unfed() error }); ok && err == nil {
			err = u.unfed()
		}
	})
	return err
}

// Walk describes the nodes of the network that starts at the sources, each once, a node
// before the nodes its output goes to. An edge into a node of several inputs names the input:
// "render (2)".
func Walk(sources ...Root) []NodeInfo {
	var out []NodeInfo
	elements(sources, func(e Element) {
		b := e.node()
		info := NodeInfo{Name: b.name, Kind: b.kind, Meta: b.Meta}
		for _, o := range b.outs {
			info.To = append(info.To, EdgeInfo{To: o.to.node().name, Policy: o.policy.String()})
		}
		out = append(out, info)
	})
	return out
}
