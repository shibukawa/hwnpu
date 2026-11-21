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
// with AutoPrepare, so that their first run makes the models ready.
package konghelper

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/alecthomas/kong"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/cli"
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

// AutoPrepare is the option of App.Open (and App.Run) for an app's commands: a model that is not
// ready is made ready at its first use, as "models prepare" makes them all ahead (downloaded and
// converted, then compiled as it loads), each step a line on the standard error. Only the models
// a command uses are prepared: one that runs without the picture model does not download it.
func AutoPrepare() ai.Option {
	return ai.WithAutoPrepare(func(p ai.Progress) {
		switch {
		case p.Message != "":
			fmt.Fprintf(os.Stderr, "    %s\n", p.Message)
		case p.Stage == ai.StagePrepare:
			fmt.Fprintf(os.Stderr, "%s (%s): not here yet; downloading and converting it (\"models prepare\" does this ahead)\n", p.Use, p.Model)
		case p.Stage == ai.StageDone:
			fmt.Fprintf(os.Stderr, "%s (%s): ready\n", p.Use, p.Model)
		}
	})
}

// Status is the command of the NPU's status (cli.Status), as hwnpu status.
type Status struct {
	JSON  bool          `help:"Write all that was read as JSON."`
	Watch time.Duration `short:"w" help:"Show it again every this long (2s) until Ctrl-C: on a terminal, as top does." placeholder:"EVERY"`
	Wake  bool          `help:"Open the NPU and ask it too (its driver's and firmware's details): on Linux this wakes a sleeping NPU for a few seconds."`
}

func (c Status) Run(ctx context.Context, app *ai.App) error {
	return cli.Status(ctx, app, cli.StatusOptions{JSON: c.JSON, Watch: c.Watch, Wake: c.Wake})
}
