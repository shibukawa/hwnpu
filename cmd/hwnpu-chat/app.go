package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/internal/aiensure"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chatcmd"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/modelstore"
)

// hwnpu-chat runs its models through the ai package, as an app would: the model --model names
// is a use of the definitions (ai.Catalog), the flags that change how it is made a model of its
// own made from it (ai.Definition), prepared when it is not here (aiensure.Ensure, ai/cli's
// Ensure) and loaded by a Runtime that compiles it at its first load (ai.CompileOnLoad). The
// models run in hwnpud, and what the command asks of it besides them (its list, why a use picks
// a model, the memory kept) is internal/chatcmd's. The flags that change only how it runs
// (--idle, --prompt-cache, --experts-key, --no-sparse) are the role's load settings
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
// it needs (a Hugging Face repository the definitions do not have), in st's catalog cat. A name
// of none is an error after the models of the role's kind are listed.
func useFor[T io.Closer](st ai.Store, cat *ai.Catalog, u ai.Use[T]) (ai.Use[T], []ai.Definition, error) {
	return aiensure.ResolveUse(cat, u, aiensure.ResolveOptions{
		ListCommand: "hwnpu-chat list",
		List: func(w io.Writer, kind string) error {
			return chatcmd.List(context.Background(), st, w, chatcmd.ListRequest{Kinds: []string{kind}})
		},
	}, func(n ai.Use[T]) error {
		if !strings.EqualFold(n.Name(), u.Name()) {
			// what a use or a family picks, and why, as before
			return chatcmd.Picked(context.Background(), st, u.Name(), logw())
		}
		return nil
	})
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
	return os.Setenv(backendEnv, d)
}

// open prepares the app's models that are not here, saying what each takes to download and on
// the disk first (toFetch), and opens its Runtime, whose first load of a model compiles it for
// the NPU.
func open(ctx context.Context, app *ai.App, opts ...ai.Option) (*ai.Runtime, error) {
	return openAfter(ctx, app, said, nil, opts...)
}

// said is openAfter's before that says what the models that are not here take (toFetch) and
// goes on.
func said(p *ai.Plan) error {
	for _, line := range toFetch(p) {
		fmt.Fprintln(logw(), line)
	}
	return nil
}

// openAfter is open with what is done when a model is not here, before it is made ready
// (before: aiensure's Options.Before, whose error ends it), and once the models are ready,
// before the Runtime is opened (ready; nil for nothing): /model asks before a download, and
// lets the model at hand go before the next is loaded.
func openAfter(ctx context.Context, app *ai.App, before func(*ai.Plan) error, ready func(), opts ...ai.Option) (*ai.Runtime, error) {
	if err := aiensure.Ensure(ctx, app, aiensure.Options{Log: os.Stderr, Before: before}); err != nil {
		return nil, err
	}
	if ready != nil {
		ready()
	}
	return app.Open(ctx, append([]ai.Option{ai.CompileOnLoad(), ai.WithLog(logw())}, opts...)...)
}

// toFetch says, a line for each model of the plan that is not here, what making it ready takes:
// about how much is downloaded and how much it keeps on the disk (the definitions' bytes and
// disk; while it is prepared the two are there at once).
func toFetch(p *ai.Plan) []string {
	var lines []string
	seen := map[string]bool{}
	for _, it := range p.Items {
		if it.Ready || seen[it.Model] {
			continue
		}
		seen[it.Model] = true
		line := it.Model + " is not here yet"
		if size := fetchSize(it); size != "" {
			line += ": " + size
		}
		lines = append(lines, line)
	}
	return lines
}

// fetchSize is what making a model that is not here ready takes, from its plan; "" when the
// definitions do not say (a model with no form for this NPU has no size on its disk).
func fetchSize(it ai.Item) string {
	disk := cache.FormatSize(it.Disk)
	switch {
	case it.Download > 0 && it.Disk > 0:
		return fmt.Sprintf("about %s to download, %s kept on the disk", cache.FormatSize(it.Download), disk)
	case it.Download > 0:
		return fmt.Sprintf("about %s to download", cache.FormatSize(it.Download))
	case it.Disk > 0:
		return fmt.Sprintf("its checkpoint is here; %s kept on the disk once it is made ready", disk)
	}
	return ""
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
