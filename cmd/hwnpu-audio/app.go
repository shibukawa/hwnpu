package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu/ai"
	aicli "github.com/shibukawa/hwnpu/ai/cli"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/modelstore"
)

// hwnpu-audio runs its models through the ai package, as an app would: the model --model names
// is a use of the definitions (ai.Catalog), the flags that change how it is made or run a model
// of its own made from it (ai.Definition), prepared when it is not here (aicli.Ensure) and loaded
// by a Runtime that compiles it at its first load (ai.CompileOnLoad).

// store is where hwnpu-audio's models come from: the commands' layers, with --models' files.
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
		return u, nil, fmt.Errorf("no %s model, family or use %q (hwnpu-audio list %s shows them)", kind, e.Name, kind)
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
// model compiles it for the NPU.
func open(ctx context.Context, app *ai.App, lw io.Writer) (*ai.Runtime, error) {
	if err := aicli.Ensure(ctx, app, aicli.PrepareOptions{Log: os.Stderr}); err != nil {
		return nil, err
	}
	return app.Open(ctx, ai.CompileOnLoad(), ai.WithLog(lw))
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
