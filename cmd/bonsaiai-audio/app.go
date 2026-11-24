package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/shibukawa/bonsaiai/ai"
	aicli "github.com/shibukawa/bonsaiai/ai/cli"
	"github.com/shibukawa/bonsaiai/internal/chat"
	"github.com/shibukawa/bonsaiai/modelstore"
)

// bonsaiai-audio runs its models through the ai package, as an app would: the model --model names
// is a use of the definitions (ai.Catalog), the flags that change how it is made or run a model
// of its own made from it (ai.Definition), prepared when it is not here (aicli.Ensure) and loaded
// by a Runtime that compiles it at its first load (ai.CompileOnLoad).

// store is where bonsaiai-audio's models come from: the commands' layers, with --models' files.
func store(models string) ai.Store {
	var files []string
	for _, f := range strings.Split(models, ",") {
		if f = strings.TrimSpace(f); f != "" {
			files = append(files, f)
		}
	}
	return ai.Store{Models: modelstore.Options{Files: files}}
}

// useFor is the use of the model name means for u's role (ai.Normalize), and the definitions
// it needs (a Hugging Face repository the definitions do not have). A name of none is an error
// after the models of the role's kind are listed.
func useFor[T io.Closer](st ai.Store, u ai.Use[T]) (ai.Use[T], []ai.Definition, error) {
	cat, err := st.Catalog()
	if err != nil {
		return u, nil, err
	}
	n, def, err := ai.Normalize(cat, u)
	if e := (*ai.NoModelError)(nil); errors.As(err, &e) {
		kind := u.Kinds()[0]
		fmt.Fprintf(os.Stderr, "no %s model %q; the %s models of the definitions:\n\n", kind, e.Name, kind)
		chat.List(os.Stderr, chat.Kind(kind))
		fmt.Fprintln(os.Stderr)
		return u, nil, fmt.Errorf("no %s model, family or use %q (bonsaiai-audio list %s shows them)", kind, e.Name, kind)
	}
	if err != nil {
		return u, nil, err
	}
	if def != nil {
		return n, []ai.Definition{*def}, nil
	}
	return n, nil, nil
}

// open prepares the app's models that are not here and opens its Runtime, whose first load of a
// model compiles it for the NPU. opts are the Runtime's others (ai.AllowOffNPU).
func open(ctx context.Context, app *ai.App, lw io.Writer, opts ...ai.Option) (*ai.Runtime, error) {
	if err := aicli.Ensure(ctx, app, aicli.PrepareOptions{Log: os.Stderr}); err != nil {
		return nil, err
	}
	return app.Open(ctx, append([]ai.Option{ai.CompileOnLoad(), ai.WithLog(lw)}, opts...)...)
}

// defined is the use u as the flags make it: u itself when d changes nothing, else a model of
// its own like u's (d, which the app defines).
func defined[T io.Closer](u ai.Use[T], defs []ai.Definition, d ai.Definition, as func(string) ai.Use[T]) (ai.Use[T], []ai.Definition) {
	if d.Programs == nil && d.Settings == nil && d.Device == "" {
		return u, defs
	}
	d.Like = u.Name()
	return as(d.ID()), append(defs, d)
}

// loaded is the model the Runtime loaded last, and how long that took.
func loaded(rt *ai.Runtime) (model string, took time.Duration) {
	ls := rt.Loaded()
	if len(ls) == 0 {
		return "", 0
	}
	return ls[len(ls)-1].Model, ls[len(ls)-1].Took
}

// loadedLine is the log's line of a model loaded: its name, how long it took, where it runs.
func loadedLine(rt *ai.Runtime, cpu, onNPU bool) string {
	ls := rt.Loaded()
	if len(ls) == 0 {
		return ""
	}
	l := ls[len(ls)-1]
	return fmt.Sprintf("loaded %s in %.1f s, %s", l.Model, l.Took.Seconds(), placement(cpu, onNPU))
}

// absolute is path made absolute (a definition's programs are).
func absolute(path string) string {
	if p, err := filepath.Abs(path); err == nil {
		return p
	}
	return path
}
