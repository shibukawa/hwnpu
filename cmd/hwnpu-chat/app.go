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
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/modelstore"
)

// hwnpu-chat runs its models through the ai package, as an app would: the model --model names
// is a use of the definitions (ai.Catalog), the flags that change how it is made a model of its
// own made from it (ai.Definition), prepared when it is not here (aicli.Ensure) and loaded by a
// Runtime that compiles it at its first load (ai.CompileOnLoad). The flags that change only how
// it runs (--idle, --prompt-cache, --experts-key, --no-sparse) are the role's load settings
// (ai.LoadSettings), the sampling and the thinking each request's options.

// store is where hwnpu-chat's models come from: the commands' layers, with --models' files.
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
func useFor[T io.Closer](cat *ai.Catalog, u ai.Use[T]) (ai.Use[T], []ai.Definition, error) {
	n, def, err := ai.Normalize(cat, u)
	if e := (*ai.NoModelError)(nil); errors.As(err, &e) {
		kind := u.Kinds()[0]
		fmt.Fprintf(os.Stderr, "no %s model %q; the %s models of the definitions:\n\n", kind, e.Name, kind)
		chat.List(os.Stderr, chat.Kind(kind))
		fmt.Fprintln(os.Stderr)
		return u, nil, fmt.Errorf("no %s model, family or use %q (hwnpu-chat list shows them)", kind, e.Name)
	}
	if err != nil {
		return u, nil, err
	}
	if def != nil {
		return n, []ai.Definition{*def}, nil
	}
	return n, nil, nil
}

// setBackend makes the NPU --backend names the one the models are converted for and run on (the
// process's HWNPU_BACKEND, which the ai package reads), checked on a Mac.
func setBackend(ctx context.Context, name string) error {
	if name == "" {
		_, err := resolveBackend(ctx, "")
		return err
	}
	d, err := resolveBackend(ctx, name)
	if err != nil {
		return err
	}
	return os.Setenv(llm.BackendEnv, d)
}

// open prepares the app's models that are not here and opens its Runtime, whose first load of a
// model compiles it for the NPU.
func open(ctx context.Context, app *ai.App, opts ...ai.Option) (*ai.Runtime, error) {
	if err := aicli.Ensure(ctx, app, aicli.PrepareOptions{Log: os.Stderr}); err != nil {
		return nil, err
	}
	return app.Open(ctx, append([]ai.Option{ai.CompileOnLoad(), ai.WithLog(logw())}, opts...)...)
}

// prepared logs how long making the models ready took (console.Prepared), as hwnpu-chat did for
// the programs it converts: not for AMD's NPU, whose exports are loaded as they are.
func prepared(rt *ai.Runtime) {
	if rt.Target() != cache.AMDTarget {
		console.Prepared(logw())
	}
}

// absolute is path made absolute (a definition's programs are).
func absolute(path string) string {
	if p, err := filepath.Abs(expandHome(path)); err == nil {
		return p
	}
	return path
}

// absolutes are the paths of a comma separated list made absolute; nil for "".
func absolutes(list string) []string {
	var out []string
	for _, p := range split(list) {
		out = append(out, absolute(strings.TrimSpace(p)))
	}
	return out
}
