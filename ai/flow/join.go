package flow

import (
	"context"
	"fmt"
	"iter"
	"reflect"
	"sync"
)

// Join is a node of two inputs and one output: a program that waits for what two nodes give
// and passes on what it makes of them (Join2, Both, With). What gives A goes to In1 and what
// gives B to In2:
//
//	render := flow.With("render", draw)       // ticks, with the newest position of the head
//	clock.To(render.In1)
//	faces.To(render.In2, flow.Newest())
//	render.To(screen)
type Join[A, B, C any] struct {
	base
	outputs[C]
	// In1 and In2 are the node's inputs.
	In1 *Port[A]
	In2 *Port[B]

	fn func(ctx context.Context, a Stream[A], b Stream[B]) Stream[C]
}

// Port is an input of a node with several (Join.In1, Join.In2).
type Port[T any] struct {
	base
	owner    Element
	outT     reflect.Type
	link     func(to Element, ps []Policy)
	arrive   func(ctx context.Context, in Stream[T], r *running)
	position int
}

func (p *Port[T]) inType() reflect.Type           { return reflect.TypeFor[T]() }
func (p *Port[T]) outType() reflect.Type          { return p.outT }
func (p *Port[T]) linkTo(to Element, ps []Policy) { p.link(to, ps) }
func (p *Port[T]) start(ctx context.Context, in Stream[T], r *running) {
	p.arrive(ctx, in, r)
}

// ported is an input that belongs to another element, which Walk describes in its place.
type ported interface {
	of() (owner Element, input int)
}

func (p *Port[T]) of() (Element, int) { return p.owner, p.position }

// Join2 makes a node of two inputs of a function over their streams, which has the whole say
// over how they are read together: one after the other, each as it comes (Hold, Latest), in
// pairs. Both and With are the two common ways, made with it.
func Join2[A, B, C any](name string, fn func(ctx context.Context, a Stream[A], b Stream[B]) Stream[C]) *Join[A, B, C] {
	j := &Join[A, B, C]{base: base{name: name, kind: "node"}, fn: fn}
	// the node starts once both inputs have their streams
	var (
		mu     sync.Mutex
		a      Stream[A]
		b      Stream[B]
		gotten int
	)
	begin := func(ctx context.Context, r *running) {
		if gotten == 2 {
			j.fan(ctx, j.name, named(j.name, j.fn(ctx, a, b)), r)
		}
	}
	out := reflect.TypeFor[C]()
	j.In1 = &Port[A]{base: base{name: name + " (1)", kind: "input"}, owner: j, outT: out, link: j.linkTo, position: 1,
		arrive: func(ctx context.Context, in Stream[A], r *running) {
			mu.Lock()
			a, gotten = in, gotten+1
			mu.Unlock()
			begin(ctx, r)
		}}
	j.In2 = &Port[B]{base: base{name: name + " (2)", kind: "input"}, owner: j, outT: out, link: j.linkTo, position: 2,
		arrive: func(ctx context.Context, in Stream[B], r *running) {
			mu.Lock()
			b, gotten = in, gotten+1
			mu.Unlock()
			begin(ctx, r)
		}}
	return j
}

// To passes the node's output on to a node or a sink, as the policy says, and returns the
// place to go on from there.
func (j *Join[A, B, C]) To(to Input[C], p ...Policy) *Link {
	connect(&j.base, &j.outputs, to, p)
	return &Link{to}
}

func (j *Join[A, B, C]) inType() reflect.Type  { return nil } // through In1 and In2
func (j *Join[A, B, C]) outType() reflect.Type { return reflect.TypeFor[C]() }
func (j *Join[A, B, C]) linkTo(to Element, ps []Policy) {
	connect(&j.base, &j.outputs, to.(Input[C]), ps)
}

// unfed is the error of a join one of whose inputs nothing gives.
func (j *Join[A, B, C]) unfed() error {
	for i, fed := range []string{j.In1.fed, j.In2.fed} {
		if fed == "" {
			return fmt.Errorf("flow: nothing goes into input %d of %s: a node of two inputs runs when both have a node before them", i+1, j.name)
		}
	}
	return nil
}

type chunkPair[A, B any] struct {
	a Chunk[A]
	b Chunk[B]
}

// Both makes a node that waits for a chunk of each input and passes on what fn makes of the
// two: the first of one with the first of the other, the second with the second. It ends when
// either input ends. It is for two results of the same thing that are both needed: a picture's
// depth and its outlines, the left and the right transcript of a moment.
func Both[A, B, C any](name string, fn func(a Chunk[A], b Chunk[B], emit func(Chunk[C])) error) *Join[A, B, C] {
	return Join2(name, func(ctx context.Context, as Stream[A], bs Stream[B]) Stream[C] {
		return func(yield func(Chunk[C], error) bool) {
			next, stop := iter.Pull2(bs)
			defer stop()
			pairs := Stream[chunkPair[A, B]](func(yield func(Chunk[chunkPair[A, B]], error) bool) {
				for a, err := range as {
					if err != nil {
						yield(Chunk[chunkPair[A, B]]{}, err)
						return
					}
					b, err, ok := next()
					if !ok {
						return
					}
					if err != nil {
						yield(Chunk[chunkPair[A, B]]{}, err)
						return
					}
					if !yield(Chunk[chunkPair[A, B]]{Data: chunkPair[A, B]{a: a, b: b}}, nil) {
						return
					}
				}
			})
			each(ctx, pairs, nil, func(p Chunk[chunkPair[A, B]], emit func(Chunk[C])) error {
				return fn(p.Data.a, p.Data.b, emit)
			})(yield)
		}
	})
}

// With makes a node that runs on its first input and takes the newest chunk of its second
// along: fn is called with each chunk of the first, and with the newest of the second (ok is
// false before the second has given one). The second input is read as it comes and never
// waited for. It is for a stage that runs on its own beat with whatever another stage last
// found: a renderer's ticks with the head's position, sound with the newest setting.
func With[A, B, C any](name string, fn func(a Chunk[A], b Chunk[B], ok bool, emit func(Chunk[C])) error) *Join[A, B, C] {
	return Join2(name, func(ctx context.Context, as Stream[A], bs Stream[B]) Stream[C] {
		return func(yield func(Chunk[C], error) bool) {
			held := Hold(ctx, bs)
			defer held.Stop()
			each(ctx, as, nil, func(a Chunk[A], emit func(Chunk[C])) error {
				if err := held.Err(); err != nil {
					return err
				}
				b, ok := held.Latest()
				return fn(a, b, ok, emit)
			})(yield)
		}
	})
}
