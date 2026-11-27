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
