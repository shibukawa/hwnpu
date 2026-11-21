package flow

import (
	"context"
	"fmt"
	"reflect"
	"sync"
)

// Call is nodes run once for each request: one value in, the nodes one after another over it,
// one value out. It is the other way an app is put together, beside the network that keeps
// running on what its sources give (Source, To): a server's answer to a request, a batch job's
// step, a picture and then its depth.
//
//	say := flow.NewCall[flow.Text, []float32]("say").Then(translate).Then(speak)
//	pcm, err := say.Do(ctx, flow.Text{Text: "こんにちは"})
//
// The nodes are those of a network (Func, Each, Tap, the nodes of models). A node belongs to
// one call or to one network; a model that is in both has a node in each. Calls may be made at
// the same time: each has its own streams through the nodes (a node of a model still runs one
// request at a time).
type Call[In, Out any] struct {
	src  *Src[In]
	last Element

	seal    sync.Once
	sealErr error
}

// NewCall is a call named name that takes an In and gives an Out, to be given its nodes with
// Then.
func NewCall[In, Out any](name string) *Call[In, Out] {
	return &Call[In, Out]{src: &Src[In]{base: base{name: name, kind: "call"}}}
}

// Then adds the node that runs next: on the request for the first, on the output of the node
// before for the others. It panics when next does not take what comes before it, as a Link's To
// does. The policy is the edge's, as in a network.
func (c *Call[In, Out]) Then(next Element, p ...Policy) *Call[In, Out] {
	from := Element(c.src)
	if c.last != nil {
		from = c.last
	}
	check(from, next)
	from.linkTo(next, p)
	c.last = next
	return c
}

// Name is the call's name.
func (c *Call[In, Out]) Name() string { return c.src.name }

// Nodes describes the call's nodes, as Walk does a network's: the call itself first.
func (c *Call[In, Out]) Nodes() []NodeInfo { return Walk(c.src) }

// resultKey is the context key of where a run of the call puts its results.
type resultKey struct{ call any }

// result is the sink at the end of a call: it gives each chunk to the run's collector.
type result[Out any] struct {
	base
	call any
}

func (r *result[Out]) inType() reflect.Type     { return reflect.TypeFor[Out]() }
func (r *result[Out]) outType() reflect.Type    { return nil }
func (r *result[Out]) linkTo(Element, []Policy) {}
func (r *result[Out]) start(ctx context.Context, in Stream[Out], run *running) {
	add, _ := ctx.Value(resultKey{r.call}).(func(Chunk[Out]))
	drain(run, in, func(c Chunk[Out]) error {
		if add != nil {
			add(c)
		}
		return nil
	})
}

// ready joins the last node to the call's result, at the first run.
func (c *Call[In, Out]) ready() error {
	c.seal.Do(func() {
		switch {
		case c.last == nil:
			c.sealErr = fmt.Errorf("flow: the call %s has no nodes (Then gives it them)", c.src.name)
		case c.last.outType() != reflect.TypeFor[Out]():
			c.sealErr = fmt.Errorf("flow: the call %s gives %v, and its last node %s gives %v", c.src.name, reflect.TypeFor[Out](), c.last.node().name, c.last.outType())
		default:
			c.last.linkTo(&result[Out]{base: base{name: c.src.name + " result", kind: "sink"}, call: c}, nil)
		}
	})
	return c.sealErr
}

// run runs the call's nodes over in once, giving each chunk that comes out to add.
func (c *Call[In, Out]) run(ctx context.Context, in In, add func(Chunk[Out])) error {
	if err := c.ready(); err != nil {
		return err
	}
	// this run's own source over the call's edges, and its own collector
	src := &Src[In]{base: c.src.base, outputs: c.src.outputs, s: Of(in, 0)}
	return Run(context.WithValue(ctx, resultKey{c}, add), src)
}

// All runs the call's nodes over in and returns everything that comes out, in order (the
// sentences of a text spoken one by one, the utterances of a recording).
func (c *Call[In, Out]) All(ctx context.Context, in In) ([]Out, error) {
	var mu sync.Mutex
	var out []Out
	err := c.run(ctx, in, func(ch Chunk[Out]) {
		mu.Lock()
		out = append(out, ch.Data)
		mu.Unlock()
	})
	return out, err
}

// Do runs the call's nodes over in and returns what comes out: the last chunk's data, the
// whole answer of nodes that give the answer so far before it. ErrEmpty is the error of a call
// whose nodes gave nothing.
func (c *Call[In, Out]) Do(ctx context.Context, in In) (Out, error) {
	all, err := c.All(ctx, in)
	var zero Out
	switch {
	case err != nil:
		return zero, err
	case len(all) == 0:
		return zero, fmt.Errorf("flow: the call %s: %w", c.src.name, ErrEmpty)
	}
	return all[len(all)-1], nil
}
