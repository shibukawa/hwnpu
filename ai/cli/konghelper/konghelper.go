// Package konghelper has the commands of the cli package as kong's structs, to put in the
// command line of an app that parses it with kong (github.com/alecthomas/kong):
//
//	var app = ai.NewApp("interpreter")
//
//	var cmds struct {
//		Run    RunCmd            `cmd:"" default:"withargs" help:"Interpret a recording."`
//		Models konghelper.Models `cmd:"" help:"The models interpreter uses: list, download, prepare."`
//		Status konghelper.Status `cmd:"" help:"This machine's NPU and what uses it."`
//	}
//
//	func main() {
//		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
//		defer stop()
//		k := kong.Parse(&cmds, konghelper.Bind(ctx, app))
//		k.FatalIfErrorf(k.Run())
//	}
//
// Bind gives the commands the app and a context; the app's own commands may take them too
// (func (c *RunCmd) Run(ctx context.Context, app *ai.App) error), and open the app's Runtime
// with Open, which runs them only when "APP models prepare" has made every model ready and
// compiled for this machine's NPU: a command never waits for a download or a compile, which
// its user could not tell from the app being slow.
package konghelper

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/alecthomas/kong"

	"github.com/shibukawa/bonsaiai/ai"
	"github.com/shibukawa/bonsaiai/ai/cli"
)

// Bind is the kong option that gives the commands of this package, and the app's own, the
// context they run under and the app whose models they look after.
func Bind(ctx context.Context, app *ai.App) kong.Option {
	return kong.OptionFunc(func(k *kong.Kong) error {
		if err := kong.BindTo(ctx, (*context.Context)(nil)).Apply(k); err != nil {
			return err
		}
		return kong.Bind(app).Apply(k)
	})
}

// HelpWithoutArgs is the kong option that shows the app's help, every command, when it is run
// without arguments. An app whose main command is its default (`default:"withargs"`) would else
// show that command's help alone, without the others (models, status). kong.Parse reads the
// arguments after its options are applied, so this goes in its options.
func HelpWithoutArgs() kong.Option {
	return kong.OptionFunc(func(*kong.Kong) error {
		if len(os.Args) == 1 {
			os.Args = append(os.Args, "--help")
		}
		return nil
	})
}

// Models is the command of an app's models: models (the list), models download, models prepare.
type Models struct {
	List     ModelsList     `cmd:"" default:"1" help:"The models the app uses and its network: which are here, what the others take to download, what they take running beside what this machine has for models, their licenses (the command when none is named)."`
	Download ModelsDownload `cmd:"" help:"Download the models that are not here and convert their programs, without loading them: no NPU is needed, and the app then starts without the network."`
	Prepare  ModelsPrepare  `cmd:"" aliases:"compile" help:"Download and convert as download does, then load each model once on this machine's NPU, which compiles its programs for it, so that the app's first run does not wait for that."`
}

// ModelsList is models list (cli.List).
type ModelsList struct {
	JSON bool `help:"Write the plan as JSON."`
}

func (c ModelsList) Run(ctx context.Context, app *ai.App) error {
	return cli.List(ctx, app, cli.ListOptions{JSON: c.JSON})
}

// prepareFlags are what download and prepare share.
type prepareFlags struct {
	AcceptLicense bool `name:"accept-license" help:"Accept the licenses that ask to be accepted before their models are downloaded (they are shown; without this, a terminal is asked)."`
	Quiet         bool `short:"q" help:"Only the stages, not the lines of the downloads and conversions."`
}

func (f prepareFlags) options() cli.PrepareOptions {
	return cli.PrepareOptions{AcceptLicenses: f.AcceptLicense, Quiet: f.Quiet}
}

// ModelsDownload is models download (cli.Download).
type ModelsDownload struct{ prepareFlags }

func (c ModelsDownload) Run(ctx context.Context, app *ai.App) error {
	return cli.Download(ctx, app, c.options())
}

// ModelsPrepare is models prepare (cli.Prepare).
type ModelsPrepare struct{ prepareFlags }

func (c ModelsPrepare) Run(ctx context.Context, app *ai.App) error {
	return cli.Prepare(ctx, app, c.options())
}

// Ready fails unless every model of the app is prepared and compiled for this machine's NPU
// (by this program): the error names those that are not and says to run "APP models prepare".
// An app's command calls it (or Open) before it starts, so that it does not stop half way.
func Ready(ctx context.Context, app *ai.App) error {
	p, err := app.Plan(ctx)
	if err != nil {
		return err
	}
	if p.Compiled() {
		return nil
	}
	var not []string
	for _, it := range p.Items {
		switch {
		case !it.Ready:
			not = append(not, fmt.Sprintf("%s (%s) is not downloaded and converted", it.Use, it.Model))
		case !it.Compiled:
			not = append(not, fmt.Sprintf("%s (%s) is not compiled for this machine's NPU", it.Use, it.Model))
		}
	}
	return fmt.Errorf("%w: %s; \"%s models prepare\" makes them ready (\"%s models\" shows them)", ai.ErrNotPrepared, strings.Join(not, ", "), app.Name, app.Name)
}

// Open is App.Open after Ready.
func Open(ctx context.Context, app *ai.App, opts ...ai.Option) (*ai.Runtime, error) {
	if err := Ready(ctx, app); err != nil {
		return nil, err
	}
	return app.Open(ctx, opts...)
}

// Status is the command of the NPU's status (cli.Status), as bonsaiai status.
type Status struct {
	JSON  bool          `help:"Write all that was read as JSON."`
	Watch time.Duration `short:"w" help:"Show it again every this long (2s) until Ctrl-C: on a terminal, as top does." placeholder:"EVERY"`
	Wake  bool          `help:"Open the NPU and ask it too (its driver's and firmware's details): on Linux this wakes a sleeping NPU for a few seconds."`
}

func (c Status) Run(ctx context.Context, app *ai.App) error {
	return cli.Status(ctx, app, cli.StatusOptions{JSON: c.JSON, Watch: c.Watch, Wake: c.Wake})
}
