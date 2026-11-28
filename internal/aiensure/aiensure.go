// Package aiensure makes an app's models ready before a command runs them, as ai/cli's Ensure
// does, without ai/cli's status of the NPU (internal/npustatus), which reads the NPU in this
// process: hwnpu-chat goes through hwnpud alone (docs/design/hwnpud.md, "Clients without the
// engine") and imports this. ai/cli's Ensure and its licenses are these.
package aiensure

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/internal/console"
)

// Options are ai/cli's PrepareOptions that Ensure reads.
type Options struct {
	// AcceptLicenses accepts the licenses that ask to be accepted (they are shown); without it
	// they are asked for when In is a terminal, and refused otherwise.
	AcceptLicenses bool
	// Quiet leaves out the lines of the downloads and conversions.
	Quiet bool
	// Log is where the progress is written (nil: the standard error), In where a license's
	// acceptance is read from (nil: the standard input).
	Log io.Writer
	In  *os.File
	// Before is called with the plan when a model is not here, before its licenses are asked
	// and it is downloaded and converted: a command says there what that takes (the items'
	// Download and Disk), or asks; its error ends Ensure. Nil for nothing.
	Before func(*ai.Plan) error
}

// Ensure makes the app's models that are not here ready for a command that runs them next: it
// downloads and converts what is missing and writes nothing when everything is here. It leaves
// the compiling to the command's first load (ai.CompileOnLoad).
func Ensure(ctx context.Context, app *ai.App, o Options) error {
	if o.Log == nil {
		o.Log = os.Stderr
	}
	if o.In == nil {
		o.In = os.Stdin
	}
	p, err := app.Plan(ctx)
	if err != nil {
		return err
	}
	if p.Ready() {
		return nil
	}
	if o.Before != nil {
		if err := o.Before(p); err != nil {
			return err
		}
	}
	p.NoCompile, p.Log, p.Brief = true, o.Log, o.Quiet
	if err := Licenses(p, o.AcceptLicenses, o.In, o.Log); err != nil {
		return err
	}
	return p.Run(ctx, nil)
}

// Licenses shows the licenses of the models to download that ask to be accepted, and accepts
// them: by accept, else by asking on a terminal (in).
func Licenses(p *ai.Plan, accept bool, in *os.File, out io.Writer) error {
	missing := map[string]bool{}
	for _, it := range p.Items {
		if !it.Ready {
			missing[it.Model] = true
		}
	}
	var ask []ai.License
	for _, l := range p.Licenses() {
		if missing[l.Model] && l.Accept && !l.Accepted {
			ask = append(ask, l)
		}
	}
	if len(ask) == 0 {
		return nil
	}
	for _, l := range ask {
		terms := ""
		if l.Terms != "" {
			terms = ": " + l.Terms
		}
		fmt.Fprintf(out, "%s is under the %s license%s\n  %s\n", l.Model, l.ID, terms, l.URL)
	}
	if !accept {
		if !console.Terminal(in) {
			return errors.New("these licenses ask to be accepted before the models are downloaded: accept them (--accept-license)")
		}
		fmt.Fprint(out, "Accept? [y/N] ")
		line, _ := bufio.NewReader(in).ReadString('\n')
		if a := strings.ToLower(strings.TrimSpace(line)); a != "y" && a != "yes" {
			return errors.New("the licenses were not accepted")
		}
	}
	p.AcceptLicenses = true
	return nil
}

// ResolveOptions configures how ResolveUse reports missing models or resolved family/use names.
type ResolveOptions struct {
	// ListCommand is the command shown in the final error when no model matches; if it contains
	// "%s", the role's primary kind is formatted into it.
	ListCommand string
	// ErrOut receives the list header and models on NoModelError (nil defaults to os.Stderr).
	ErrOut io.Writer
	// List writes the available models of kind to w when no model matches.
	List func(w io.Writer, kind string) error
}

// ResolveUse normalizes u against cat (ai.Normalize). On NoModelError it lists the available
// models of u's primary kind and returns a formatted error mentioning o.ListCommand.
func ResolveUse[T io.Closer](cat *ai.Catalog, u ai.Use[T], o ResolveOptions, onPicked func(resolved ai.Use[T]) error) (ai.Use[T], []ai.Definition, error) {
	n, def, err := ai.Normalize(cat, u)
	if e := (*ai.NoModelError)(nil); errors.As(err, &e) {
		kind := u.Kinds()[0]
		out := o.ErrOut
		if out == nil {
			out = os.Stderr
		}
		fmt.Fprintf(out, "no %s model %q; the %s models of the definitions:\n\n", kind, e.Name, kind)
		if o.List != nil {
			if lerr := o.List(out, kind); lerr != nil {
				return u, nil, lerr
			}
		}
		fmt.Fprintln(out)
		hint := o.ListCommand
		if strings.Contains(hint, "%s") {
			hint = fmt.Sprintf(hint, kind)
		}
		return u, nil, fmt.Errorf("no %s model, family or use %q (%s shows them)", kind, e.Name, hint)
	}
	if err != nil {
		return u, nil, err
	}
	if def != nil {
		return n, []ai.Definition{*def}, nil
	}
	if onPicked != nil {
		if err := onPicked(n); err != nil {
			return u, nil, err
		}
	}
	return n, nil, nil
}
