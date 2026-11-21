package ai

import (
	"context"
	"fmt"
	"io"
	"slices"
	"strings"

	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/internal/airole"
)

// requirement is a model an app uses: the name it is picked by and the role it fills.
type requirement struct {
	name string
	role *airole.Role
	// as picks the model, and the way it runs, as a machine of this many GB would (Use.As); 0
	// for the machine's own memory.
	as int
	// tiers are other names for machines of more memory (Use.At), by memory, smallest first.
	tiers []tier
}

// tier is what a use is on machines of at least gb GB.
type tier struct {
	gb   int
	name string
}

// pick is the name the requirement goes by on a machine of machine GB: that of the tier of the
// most memory the machine has (gb: its memory), else its own name (gb: 0).
func (r requirement) pick(machine int) (name string, gb int) {
	name = r.name
	for _, t := range r.tiers {
		if t.gb <= machine {
			name, gb = t.name, t.gb
		}
	}
	return name, gb
}

// key names the requirement in a StoreFile and a Runtime: ROLE:NAME.
func (r requirement) key() string { return r.role.Name + ":" + r.name }

// Requirement is a model an app uses, for App to register: a Use of one of the packages of
// models (speech.Transcribe, text.Translate, vision.Depth, ...).
type Requirement interface {
	requirement() requirement
}

// Use is a model an app uses, loaded as a T. The packages of models make them
// (speech.Transcribe("transcribe")); an app keeps each in a variable, registers them in its App,
// and gets the model from a Runtime with Get:
//
//	var stt = speech.Transcribe("transcribe")
//	app := ai.NewApp("interpreter", stt)
//	...
//	t, err := stt.Get(ctx, rt) // a *speech.Transcriber
type Use[T io.Closer] struct{ req requirement }

// Define makes the Use of a role's model named name: a model, a family or a use of the
// definitions (models.yaml; hwnpu-chat list shows them). It is for hwnpu's own packages of
// models, which hold the roles; an app gets its uses from those.
func Define[T io.Closer](role *airole.Role, name string) Use[T] {
	return Use[T]{requirement{name: strings.TrimSpace(name), role: role}}
}

func (u Use[T]) requirement() requirement { return u.req }

// Name is the name the model is picked by.
func (u Use[T]) Name() string { return u.req.name }

// Role is the role the model fills (transcribe, translate, speak, ...).
func (u Use[T]) Role() string { return u.req.role.Name }

// At makes the use another model on machines of at least gb GB of memory: name is a model, a
// family (with a tag) or a use of the definitions, as the use's own name is. Of several At, the
// one of the most memory the machine has counts; a machine of less than all of them takes the
// use's own name. It is how an app sets its line-up of models by the machine's memory: a use
// alone picks the largest model the machine runs, and an app that runs several at once wants
// the main one large and the others small and fast.
//
//	tr := text.Translate("translate").       // 8 GB: the pick of the definitions (the 0.8B)
//		At(16, "plamo3translate:2b").        // 16 GB: beside a transcriber and a speaker
//		At(32, "plamo3translate:8b")         // 32 GB and up
func (u Use[T]) At(gb int, name string) Use[T] {
	name = strings.TrimSpace(name)
	if gb <= 0 || name == "" {
		panic("ai: Use.At takes a machine's memory in GB above 0 and a name of the definitions")
	}
	ts := slices.DeleteFunc(slices.Clone(u.req.tiers), func(t tier) bool { return t.gb == gb })
	ts = append(ts, tier{gb, name})
	slices.SortFunc(ts, func(a, b tier) int { return a.gb - b.gb })
	u.req.tiers = ts
	return u
}

// As makes the use pick its model, and the way the model runs (the programs of models.yaml's
// variants), as a machine of gb GB would, whatever memory this one has: a model that helps the
// app's main one kept small on every machine. 0 is the machine's own memory again. A name that
// is one model is that model whatever gb is; it then only runs the smaller way.
func (u Use[T]) As(gb int) Use[T] {
	u.req.as = max(gb, 0)
	return u
}

// Get is the use's model in rt, loaded at the first call (which takes from under a second to
// tens of seconds) and kept until Release or the Runtime's Close; later calls return the same
// one. A use the store has not prepared is ErrNotPrepared, unless the Runtime prepares it
// (WithAutoPrepare).
func (u Use[T]) Get(ctx context.Context, rt *Runtime) (T, error) {
	var zero T
	l, err := rt.get(ctx, u.req)
	if err != nil {
		return zero, err
	}
	t, ok := l.(T)
	if !ok {
		return zero, fmt.Errorf("ai: %s loads as a %T, not a %T", u.req.key(), l, zero)
	}
	return t, nil
}

// Release unloads the use's model from rt and frees its memory, for an app that runs its models
// one after another (a picture's generator, then its depth). A later Get loads it again.
func (u Use[T]) Release(rt *Runtime) error { return rt.release(u.req) }

// App is the set of models an application uses: what it registers to plan, download, compile
// and run them. The cli package gives an app's command line the commands that do so.
type App struct {
	// Name is the app's name, in its commands' help and its logs.
	Name string
	// Store is where the models come from and are kept; the zero value is that of hwnpu's
	// commands.
	Store Store

	reqs   []requirement
	roots  []flow.Root
	calls  []interface{ Nodes() []flow.NodeInfo }
	memory int
}

// NewApp is the app called name that uses the models uses.
func NewApp(name string, uses ...Requirement) *App {
	return (&App{Name: name}).Use(uses...)
}

// Use registers more models. One registered twice (the same name and role) counts once, as it
// was first registered.
func (a *App) Use(uses ...Requirement) *App {
	for _, u := range uses {
		r := u.requirement()
		if r.role == nil || r.name == "" {
			panic("ai: a Use not made by a package of models (speech.Transcribe, text.Translate, ...)")
		}
		if !slices.ContainsFunc(a.reqs, func(o requirement) bool { return o.key() == r.key() }) {
			a.reqs = append(a.reqs, r)
		}
	}
	return a
}

// Memory makes uses and families pick their models, and the models their programs, for a machine
// of gb GB instead of this one (a cache to ship for smaller machines): the uses' At count for
// that memory too. 0 is this machine's, with the memory chosen for a model (hwnpu memory,
// modelstore.Store.SetMemory).
func (a *App) Memory(gb int) *App {
	a.memory = gb
	return a
}

// registered is the app's model of r's name and role, as it was registered (with its At and As);
// ok is false when r is not one of the app's models.
func (a *App) registered(r requirement) (reg requirement, ok bool) {
	reqs := a.requirements()
	i := slices.IndexFunc(reqs, func(o requirement) bool { return o.key() == r.key() })
	if i < 0 {
		return r, false
	}
	return reqs[i], true
}
