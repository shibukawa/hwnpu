package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/shibukawa/bonsaiai/ai"
	aicli "github.com/shibukawa/bonsaiai/ai/cli"
	"github.com/shibukawa/bonsaiai/internal/chat"
	"github.com/shibukawa/bonsaiai/internal/llm"
)

// bonsaiai-image's analyses (depth, face-detect, detect, pose, segment, ocr, label, embed) run their
// models through the ai package's ai/vision, as an app would: the model --model names is a use
// of the definitions (ai.Normalize), the flags that change how it is made or run a model of its
// own made from it (ai.Definition: --device, --cpu, --threads, an export's directory or an ONNX
// file), prepared when it is not here (aicli.Ensure) and loaded by a Runtime that compiles it at
// its first load (ai.CompileOnLoad).

// useFor is the use of the model name means for u's role (ai.Normalize), and the definitions it
// needs (a Hugging Face repository the definitions do not have), saying what a family or a use
// picked when say. A name of none is an error after the models of the role's kind are listed.
func useFor[T io.Closer](st ai.Store, u ai.Use[T], say bool) (ai.Use[T], []ai.Definition, error) {
	cat, err := st.Catalog()
	if err != nil {
		return u, nil, err
	}
	n, def, err := ai.Normalize(cat, u)
	if e := (*ai.NoModelError)(nil); errors.As(err, &e) {
		kind := chat.Kind(u.Kinds()[0])
		// what there is, then the error (kong prints it last)
		fmt.Fprintf(os.Stderr, "no %s model %q; the %s models of the definitions:\n\n", kind, e.Name, kind)
		listModels(os.Stderr, kind)
		fmt.Fprintln(os.Stderr)
		return u, nil, fmt.Errorf("no %s model, family or use %q (bonsaiai-image list shows them)", kind, e.Name)
	}
	if err != nil {
		return u, nil, err
	}
	if def != nil {
		return n, []ai.Definition{*def}, nil
	}
	// what a family or a use picked, said as the commands say it
	if say {
		chat.LookupOn(u.Name(), 0, os.Stderr)
	}
	return n, nil, nil
}

// made is how the flags make a model of its own (ai.Definition): its Device, Settings and
// Programs, the rest from the model --model names.
type made struct {
	device   string
	settings map[string]string
	programs map[string][]string
}

// set gives the setting name value ("" gives none).
func (m *made) set(name, value string) {
	if value == "" {
		return
	}
	if m.settings == nil {
		m.settings = map[string]string{}
	}
	m.settings[name] = value
}

// threads gives the CPU's threads of --threads (1, the default, gives none).
func (m *made) threads(n int) {
	if n > 1 {
		m.set("threads", strconv.Itoa(n))
	}
}

// interpreter makes the model run its converted programs on the CPU interpreter (--cpu).
func (m *made) interpreter(on bool) {
	if on {
		m.device = "cpu"
		m.set("cpu", "interpreter")
	}
}

// program gives the programs name (a file or a directory of the user's, made absolute).
func (m *made) program(name, path string) {
	if abs, err := filepath.Abs(path); err == nil {
		path = abs
	}
	m.programs = map[string][]string{name: {path}}
}

// load prepares the model name means for the role of mk's uses, as the flags make it (how), when
// it is not here, and loads it: the model, the Runtime it is loaded in (closed by the caller)
// and the log's line of its load ("loaded NAME in 0.2 s").
func load[T io.Closer](ctx context.Context, name string, mk func(string) ai.Use[T], how made, lw io.Writer) (T, *ai.Runtime, string, error) {
	var none T
	st := ai.Store{}
	use, defs, err := useFor(st, mk(name), how.programs == nil)
	if err != nil {
		return none, nil, "", err
	}
	if how.device != "" || how.settings != nil || how.programs != nil {
		d := ai.Definition{Like: use.Name(), Device: how.device, Settings: how.settings, Programs: how.programs}
		use, defs = mk(d.ID()), append(defs, d)
	}
	st.Definitions = defs
	app := ai.NewApp("bonsaiai-image", use)
	app.Store = st
	if err := aicli.Ensure(ctx, app, aicli.PrepareOptions{AcceptLicenses: acceptLicenses, Log: os.Stderr}); err != nil {
		return none, nil, "", err
	}
	rt, err := app.Open(ctx, ai.CompileOnLoad(), ai.WithLog(lw))
	if err != nil {
		return none, nil, "", err
	}
	t, err := use.Get(ctx, rt)
	if err != nil {
		rt.Close()
		return none, nil, "", err
	}
	line := ""
	if ls := rt.Loaded(); len(ls) > 0 {
		l := ls[len(ls)-1]
		model, _, _ := strings.Cut(l.Model, "+") // a model of the flags' by the one it was made from
		line = fmt.Sprintf("loaded %s in %.1f s", model, l.Took.Seconds())
	}
	return t, rt, line, nil
}

// useBackend makes --backend the device backend the models are opened on (BONSAIAI_BACKEND, which
// ai reads).
func useBackend(name string) {
	if name != "" {
		os.Setenv(llm.BackendEnv, name)
	}
}
