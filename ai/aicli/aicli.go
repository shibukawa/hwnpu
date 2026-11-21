// Package aicli gives the command line of an app made with the ai package the commands that
// look after its models: what they are and what they take, downloading them ahead of the first
// run, compiling them for this machine's NPU, and the NPU's status. They are kong commands, to
// put in the app's own command line beside its commands:
//
//	var app = ai.NewApp("interpreter", stt, tr, tts)
//
//	var cli struct {
//		Run    RunCmd       `cmd:"" default:"withargs" help:"Interpret a recording."`
//		Models aicli.Models `cmd:"" help:"The models interpreter uses: list, download, prepare."`
//		Status aicli.Status `cmd:"" help:"This machine's NPU and what uses it."`
//	}
//
//	func main() {
//		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
//		defer stop()
//		k := kong.Parse(&cli, aicli.Bind(ctx, app))
//		k.FatalIfErrorf(k.Run())
//	}
//
// Bind gives the commands the app and a context; the app's own commands may take them too
// (func (c *RunCmd) Run(ctx context.Context, app *ai.App) error).
package aicli

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/alecthomas/kong"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/npustatus"
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
	List     ModelsList     `cmd:"" default:"1" help:"The models the app uses: which are here, what the others take to download, what they take running, their licenses (the command when none is named)."`
	Download ModelsDownload `cmd:"" help:"Download the models that are not here and convert their programs, without loading them: no NPU is needed, and the app then starts without the network."`
	Prepare  ModelsPrepare  `cmd:"" aliases:"compile" help:"Download and convert as download does, then load each model once on this machine's NPU, which compiles its programs for it, so that the app's first run does not wait for that."`
}

// ModelsList is models list.
type ModelsList struct {
	JSON bool `help:"Write the plan as JSON."`
}

// planJSON is ModelsList's JSON.
type planJSON struct {
	App           string       `json:"app"`
	Target        string       `json:"target"`
	MachineGB     int          `json:"machine_gb"`
	Ready         bool         `json:"ready"`
	Compiled      bool         `json:"compiled"`
	DownloadBytes int64        `json:"download_bytes"`
	MemoryGB      float64      `json:"memory_gb"`
	Items         []ai.Item    `json:"items"`
	Licenses      []ai.License `json:"licenses"`
}

func (c ModelsList) Run(ctx context.Context, app *ai.App) error {
	p, err := app.Plan(ctx)
	if err != nil {
		return err
	}
	if c.JSON {
		e := json.NewEncoder(os.Stdout)
		e.SetIndent("", "  ")
		return e.Encode(planJSON{App: app.Name, Target: p.Target(), MachineGB: p.MachineGB(), Ready: p.Ready(), Compiled: p.Compiled(),
			DownloadBytes: p.DownloadBytes(), MemoryGB: p.MemoryGB(), Items: p.Items, Licenses: p.Licenses()})
	}
	writePlan(os.Stdout, app, p)
	return nil
}

// state is how the list says where an item stands.
func state(it ai.Item, target string) string {
	switch {
	case !it.Ready:
		return "missing"
	case it.Bundle != "":
		return "bundled"
	case it.Compiled:
		return "compiled"
	case target == "":
		return "ready"
	}
	return "ready, not compiled"
}

func size(n int64) string {
	if n <= 0 {
		return "-"
	}
	return cache.FormatSize(n)
}

func gb(x float64) string {
	if x <= 0 {
		return "-"
	}
	return fmt.Sprintf("%.1f GB", x)
}

// writePlan writes the plan as a table and what it comes to.
func writePlan(w io.Writer, app *ai.App, p *ai.Plan) {
	target := p.Target()
	if target == "" {
		fmt.Fprintf(w, "%s's models (no NPU that runs them was found here: they can be downloaded, not loaded)\n\n", app.Name)
	} else {
		fmt.Fprintf(w, "%s's models, for this machine's NPU (%s)\n\n", app.Name, target)
	}
	licenses := map[string]ai.License{}
	for _, l := range p.Licenses() {
		licenses[l.Model] = l
	}
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "USE\tROLE\tMODEL\tSTATE\tDOWNLOAD\tMEMORY\tLICENSE")
	for _, it := range p.Items {
		l := licenses[it.Model]
		lic := l.ID
		if l.Class != "" && l.Class != "open" {
			lic += " (" + l.Class + ")"
		}
		if l.Accept && !l.Accepted {
			lic += ", to accept"
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", it.Use, it.Role, it.Model, state(it, target), size(it.Download), gb(it.MemoryGB), lic)
	}
	tw.Flush()
	fmt.Fprintln(w)
	running := fmt.Sprintf("about %s running together, of this machine's %d GB", gb(p.MemoryGB()), p.MachineGB())
	switch {
	case p.Compiled():
		fmt.Fprintf(w, "All here and compiled; %s.\n", running)
	case p.Ready():
		fmt.Fprintf(w, "All here; %s. \"models prepare\" compiles them for the NPU ahead of the first run.\n", running)
	default:
		down := "Their checkpoints are here or come from the sources"
		if n := p.DownloadBytes(); n > 0 {
			down = "About " + size(n) + " to download"
		}
		fmt.Fprintf(w, "%s; %s. \"models download\" downloads and converts; \"models prepare\" compiles for the NPU too.\n", down, running)
	}
	if p.MemoryGB() > float64(p.MachineGB()) {
		fmt.Fprintf(w, "Together they take more than the machine's memory: the app is to pick smaller models for it (ai.Use's At and As) or run them one after another (Release).\n")
	}
	for _, it := range p.Items {
		if strings.Contains(it.Why, "by the app") {
			fmt.Fprintf(w, "%s: %s\n", it.Use, it.Why)
		}
	}
}

// prepareFlags are what download and prepare share.
type prepareFlags struct {
	AcceptLicense bool `name:"accept-license" help:"Accept the licenses that ask to be accepted before their models are downloaded (they are shown; without this, a terminal is asked)."`
	Quiet         bool `short:"q" help:"Only the stages, not the lines of the downloads and conversions."`
}

// ModelsDownload is models download.
type ModelsDownload struct{ prepareFlags }

func (c ModelsDownload) Run(ctx context.Context, app *ai.App) error {
	return c.run(ctx, app, true)
}

// ModelsPrepare is models prepare.
type ModelsPrepare struct{ prepareFlags }

func (c ModelsPrepare) Run(ctx context.Context, app *ai.App) error {
	return c.run(ctx, app, false)
}

func (c prepareFlags) run(ctx context.Context, app *ai.App, noCompile bool) error {
	out := os.Stderr
	p, err := app.Plan(ctx)
	if err != nil {
		return err
	}
	p.NoCompile = noCompile
	if !noCompile && p.Target() == "" {
		fmt.Fprintln(out, "no NPU that runs the models was found here: downloading and converting only")
	}
	if err := c.licenses(p, os.Stdin, out); err != nil {
		return err
	}
	start := time.Now()
	stage := ""
	err = p.Run(ctx, func(pr ai.Progress) {
		head := fmt.Sprintf("[%d/%d] %s (%s)", pr.Item, pr.Items, pr.Use, pr.Model)
		switch {
		case pr.Message != "":
			if !c.Quiet {
				fmt.Fprintf(out, "    %s\n", pr.Message)
			}
		case pr.Stage == ai.StagePrepare:
			fmt.Fprintf(out, "%s: downloading and converting what is not here\n", head)
		case pr.Stage == ai.StageCompile:
			fmt.Fprintf(out, "%s: loading once, to compile for the NPU\n", head)
		case pr.Stage == ai.StageDone && stage != "":
			fmt.Fprintf(out, "%s: done\n", head)
		}
		stage = string(pr.Stage)
	})
	if err != nil {
		return err
	}
	if stage == "" {
		fmt.Fprintln(out, "nothing to do")
	} else {
		fmt.Fprintf(out, "done in %s\n", time.Since(start).Round(time.Second))
	}
	writePlan(os.Stdout, app, p)
	return nil
}

// licenses shows the licenses of the models to download that ask to be accepted, and accepts
// them: by --accept-license, else by asking on a terminal.
func (c prepareFlags) licenses(p *ai.Plan, in io.Reader, out io.Writer) error {
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
	if !c.AcceptLicense {
		if !console.Terminal(os.Stdin) {
			return errors.New("these licenses ask to be accepted before the models are downloaded: --accept-license accepts them")
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

// Status is the command of the NPU's status, as hwnpu status: the device, its driver and
// firmware, how busy it is, its power, the processes that use it and the memory each holds, and
// the cache of models.
type Status struct {
	JSON  bool          `help:"Write all that was read as JSON."`
	Watch time.Duration `short:"w" help:"Show it again every this long (2s) until Ctrl-C: on a terminal, as top does." placeholder:"EVERY"`
	Wake  bool          `help:"Open the NPU and ask it too (its driver's and firmware's details): on Linux this wakes a sleeping NPU for a few seconds."`
}

func (c Status) Run(ctx context.Context, app *ai.App) error {
	return npustatus.Cmd{JSON: c.JSON, Watch: c.Watch, Wake: c.Wake, Store: app.Store.Models}.Run(ctx)
}
