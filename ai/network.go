package ai

import (
	"context"
	"errors"
	"io"
	"sync"

	"github.com/shibukawa/hwnpu/ai/flow"
)

// runtimeKey is the context key of the Runtime a network runs on.
type runtimeKey struct{}

// Node makes a node of a network or a call (flow) of a use: fn runs the use's model over the node's
// input. The model is the Runtime's (Use.Get), loaded before the network starts. The packages
// of models make their nodes with it (speech.TranscribeNode, text.TranslateNode); an app makes
// its own for a model it uses another way.
func Node[T io.Closer, A, B any](u Use[T], fn func(ctx context.Context, model T, in flow.Stream[A]) flow.Stream[B]) *flow.Node[A, B] {
	n := flow.Func(u.Name(), func(ctx context.Context, in flow.Stream[A]) flow.Stream[B] {
		return func(yield func(flow.Chunk[B], error) bool) {
			rt, ok := ctx.Value(runtimeKey{}).(*Runtime)
			if !ok {
				yield(flow.Chunk[B]{}, errors.New("ai: the node of a model runs in App.Run, Runtime.Run or a Call's Do, which give it its model"))
				return
			}
			m, err := u.Get(ctx, rt)
			if err != nil {
				yield(flow.Chunk[B]{}, err)
				return
			}
			fn(ctx, m, in)(yield)
		}
	})
	n.Meta = Requirement(u)
	return n
}

// From gives the app a source of its network (flow.Source) and returns the place to go on
// from it, so that the network is written in a row:
//
//	app := ai.NewApp("interpreter")
//	app.From(mic).To(vad).To(hear).To(trans).To(speak, flow.Keep(4)).To(out)
//	...
//	err := app.Run(ctx)
//
// The models of the network's nodes (speech.TranscribeNode(stt), ...) are the app's models, as
// if they were given to NewApp: Plan and the commands of the cli package see them. Nodes may be joined
// after the source too (hear.To(captions)), any time before the run.
func (a *App) From(source flow.Root) *flow.Link {
	a.roots = append(a.roots, source)
	return flow.Chain(source)
}

// Network describes the app's network (From) and then its calls (NewCall), a node before the
// nodes its output goes to.
func (a *App) Network() []flow.NodeInfo {
	out := flow.Walk(a.roots...)
	for _, c := range a.calls {
		out = append(out, c.Nodes()...)
	}
	return out
}

// requirements are the app's models: those registered (NewApp, Use) and those of the nodes of
// its network and its calls, each once.
func (a *App) requirements() []requirement {
	out := append([]requirement(nil), a.reqs...)
	for _, n := range a.Network() {
		u, ok := n.Meta.(Requirement)
		if !ok {
			continue
		}
		r := u.requirement()
		have := false
		for _, o := range out {
			have = have || o.key() == r.key()
		}
		if !have {
			out = append(out, r)
		}
	}
	return out
}

// Call is nodes run once for each request on an app's models (flow.Call): one value in, the
// nodes one after another over it, one value out. It is how an app that answers requests is
// put together (a server, a batch job), as From and To put together one that keeps running on
// what its sources give.
//
//	say := ai.NewCall[flow.Text, []float32](app, "say").Then(translate).Then(speak)
//	...
//	pcm, err := say.Do(ctx, rt, flow.Text{Text: "こんにちは"})
//
// The models of the call's nodes are the app's models. A model in a call and in the network,
// or in two calls, has a node in each (text.TranslateNode(tr, ...) twice): it is one model,
// loaded once.
type Call[In, Out any] struct{ c *flow.Call[In, Out] }

// NewCall is a call of app a named name that takes an In and gives an Out, to be given its
// nodes with Then.
func NewCall[In, Out any](a *App, name string) *Call[In, Out] {
	c := &Call[In, Out]{flow.NewCall[In, Out](name)}
	a.calls = append(a.calls, c.c)
	return c
}

// Then adds the node that runs next: on the request for the first, on the output of the node
// before for the others. It panics when next does not take what comes before it.
func (c *Call[In, Out]) Then(next flow.Element, p ...flow.Policy) *Call[In, Out] {
	c.c.Then(next, p...)
	return c
}

// Do runs the call's nodes over in on rt and returns what comes out (the last chunk's data).
// The models of its nodes are loaded at their first use, as Use.Get loads them. Calls may be
// made at the same time; a model runs one of them at a time.
func (c *Call[In, Out]) Do(ctx context.Context, rt *Runtime, in In) (Out, error) {
	return c.c.Do(context.WithValue(ctx, runtimeKey{}, rt), in)
}

// All is Do that returns everything that comes out, in order.
func (c *Call[In, Out]) All(ctx context.Context, rt *Runtime, in In) ([]Out, error) {
	return c.c.All(context.WithValue(ctx, runtimeKey{}, rt), in)
}

// Run opens the app's models and runs its network (From) to its end: Open, Runtime.Run and
// Close in one call.
func (a *App) Run(ctx context.Context, opts ...Option) error {
	rt, err := a.Open(ctx, opts...)
	if err != nil {
		return err
	}
	defer rt.Close()
	return rt.Run(ctx)
}

// Run runs the app's network (App.From) on the Runtime: it loads the models of the
// network's nodes first, one after another, so that no source is read before they are ready,
// then runs the network until its sources have ended and every chunk has gone through, a node
// fails (the error names it: *flow.NodeError), or ctx is cancelled.
func (rt *Runtime) Run(ctx context.Context) error {
	if len(rt.app.roots) == 0 {
		return errors.New("ai: the app has no network (App.From gives it its sources)")
	}
	for _, n := range flow.Walk(rt.app.roots...) {
		if u, ok := n.Meta.(Requirement); ok {
			if _, err := rt.get(ctx, u.requirement()); err != nil {
				return err
			}
		}
	}
	return flow.Run(context.WithValue(ctx, runtimeKey{}, rt), rt.app.roots...)
}

// ModelNode is the node of a model in a network, with the settings the model runs with there (O).
// The settings can be changed while the network runs (Update): the next chunk is read with the
// new ones. A use may have several nodes, each a place of its own in the network with its own
// settings (the left and the right microphone into one transcriber).
type ModelNode[A, B, O any] struct {
	*flow.Node[A, B]

	mu sync.Mutex
	o  O
}

// NewModelNode makes the node of a use with settings o: fn runs the model over the node's input,
// asking options for the settings of each chunk it reads. The packages of models make their
// nodes with it.
func NewModelNode[T io.Closer, A, B, O any](u Use[T], o O, fn func(ctx context.Context, model T, in flow.Stream[A], options func() O) flow.Stream[B]) *ModelNode[A, B, O] {
	s := &ModelNode[A, B, O]{o: o}
	s.Node = Node(u, func(ctx context.Context, model T, in flow.Stream[A]) flow.Stream[B] {
		return fn(ctx, model, in, s.Options)
	})
	return s
}

// Options are the node's settings now.
func (s *ModelNode[A, B, O]) Options() O {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.o
}

// Update changes the node's settings, from any goroutine: before the network runs (a command
// line's flags) or while it does (another node's callback that heard "in Spanish, please").
func (s *ModelNode[A, B, O]) Update(fn func(o *O)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	fn(&s.o)
}
