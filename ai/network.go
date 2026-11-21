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

// Node makes a node of a network (flow) of a use: fn runs the use's model over the node's
// input. The model is the Runtime's (Use.Get), loaded before the network starts. The packages
// of models make their nodes with it (speech.TranscribeNode, text.TranslateNode); an app makes
// its own for a model it uses another way.
func Node[T io.Closer, A, B any](u Use[T], fn func(ctx context.Context, model T, in flow.Stream[A]) flow.Stream[B]) *flow.Node[A, B] {
	n := flow.Func(u.Name(), func(ctx context.Context, in flow.Stream[A]) flow.Stream[B] {
		return func(yield func(flow.Chunk[B], error) bool) {
			rt, ok := ctx.Value(runtimeKey{}).(*Runtime)
			if !ok {
				yield(flow.Chunk[B]{}, errors.New("ai: the node of a model runs in App.Run or Runtime.Run, which give it its model"))
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

// Connect gives the app its network: the sources it starts at (flow.Source), with the nodes
// joined after them (Connect of each node). The models of the network's nodes are the app's
// models, as if they were given to NewApp: Plan and the commands of aicli see them. The joins
// may be made before or after this call.
//
//	app := ai.NewApp("interpreter")
//	app.Connect(mic)
//	mic.Connect(vad)
//	vad.Connect(hear)   // a node of a model: speech.TranscribeNode(stt)
//	hear.Connect(trans)
//	...
//	err := app.Run(ctx)
func (a *App) Connect(sources ...flow.Root) *App {
	a.roots = append(a.roots, sources...)
	return a
}

// Network describes the app's network (Connect), a node before the nodes its output goes to.
func (a *App) Network() []flow.NodeInfo { return flow.Walk(a.roots...) }

// requirements are the app's models: those registered (NewApp, Use) and those of its network's
// nodes, each once.
func (a *App) requirements() []requirement {
	out := append([]requirement(nil), a.reqs...)
	for _, n := range flow.Walk(a.roots...) {
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

// Run opens the app's models and runs its network (Connect) to its end: Open, Runtime.Run and
// Close in one call.
func (a *App) Run(ctx context.Context, opts ...Option) error {
	rt, err := a.Open(ctx, opts...)
	if err != nil {
		return err
	}
	defer rt.Close()
	return rt.Run(ctx)
}

// Run runs the app's network (App.Connect) on the Runtime: it loads the models of the
// network's nodes first, one after another, so that no source is read before they are ready,
// then runs the network until its sources have ended and every chunk has gone through, a node
// fails (the error names it: *flow.NodeError), or ctx is cancelled.
func (rt *Runtime) Run(ctx context.Context) error {
	if len(rt.app.roots) == 0 {
		return errors.New("ai: the app has no network (App.Connect gives it its sources)")
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
