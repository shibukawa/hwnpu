// Package cli has the commands that look after an app's models, for the command line of an
// app made with the ai package: what its models are and what they take, downloading them ahead
// of the first run, compiling them for this machine's NPU, and the NPU's status.
//
// It depends on no command line library. An app calls the functions from the commands of
// whatever it parses its command line with:
//
//	cli.List(ctx, app, cli.ListOptions{})        // "myapp models"
//	cli.Download(ctx, app, cli.PrepareOptions{}) // "myapp models download"
//	cli.Prepare(ctx, app, cli.PrepareOptions{})  // "myapp models prepare"
//	cli.Status(ctx, app, cli.StatusOptions{})    // "myapp status"
//
// or hands its arguments over, for an app without one:
//
//	if handled, err := cli.Run(ctx, app, os.Args[1:]); handled {
//		// "models ..." or "status ..." was run
//	}
//
// The package konghelper has the same commands as kong's structs.
package cli

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/npustatus"
)

// ListOptions are how List writes.
type ListOptions struct {
	// JSON writes the plan as JSON instead of the table.
	JSON bool
	// Out is where it is written (nil: the standard output).
	Out io.Writer
}

// planJSON is List's JSON.
type planJSON struct {
	App           string       `json:"app"`
	Target        string       `json:"target"`
	MachineGB     int          `json:"machine_gb"`
	UsableGB      float64      `json:"usable_gb"`
	Fits          bool         `json:"fits"`
	Ready         bool         `json:"ready"`
	Compiled      bool         `json:"compiled"`
	DownloadBytes int64        `json:"download_bytes"`
	MemoryGB      float64      `json:"memory_gb"`
	InTurns       [][]string   `json:"in_turns,omitempty"`
	Items         []ai.Item    `json:"items"`
	Licenses      []ai.License `json:"licenses"`
}

// List writes the models the app uses and its network: which models are here, what the others
// take to download, what each takes running, what they take together beside what the machine
// has for models, and their licenses. It downloads nothing.
func List(ctx context.Context, app *ai.App, o ListOptions) error {
	out := o.Out
	if out == nil {
		out = os.Stdout
	}
	p, err := app.Plan(ctx)
	if err != nil {
		return err
	}
	if o.JSON {
		e := json.NewEncoder(out)
		e.SetIndent("", "  ")
		return e.Encode(planJSON{App: app.Name, Target: p.Target(), MachineGB: p.MachineGB(), UsableGB: p.UsableGB(), Fits: p.Fits(), Ready: p.Ready(), Compiled: p.Compiled(),
			DownloadBytes: p.DownloadBytes(), MemoryGB: p.MemoryGB(), InTurns: p.InTurns(), Items: p.Items, Licenses: p.Licenses()})
	}
	WritePlan(out, app, p)
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

// WritePlan writes a plan as List does: the table of the models, what they come to, and the
// app's network.
func WritePlan(w io.Writer, app *ai.App, p *ai.Plan) {
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
	var turns []string
	for _, g := range p.InTurns() {
		turns = append(turns, strings.Join(g, " and "))
	}
	together := "running together"
	if len(turns) > 0 {
		together = "at once (" + strings.Join(turns, "; ") + " in turns)"
	}
	running := fmt.Sprintf("about %s %s; hwnpu counts on about %.0f GB of a %d GB machine for models", gb(p.MemoryGB()), together, p.UsableGB(), p.MachineGB())
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
	if !p.Fits() {
		fmt.Fprintf(w, "Together they take more than hwnpu counts on for models here, and the NPU's memory cannot be swapped out: the other applications may be pushed to swap. The app is to pick smaller models for this memory (ai.Use's At and As) or run them one after another (App.InTurns).\n")
	}
	for _, it := range p.Items {
		if strings.Contains(it.Why, "by the app") {
			fmt.Fprintf(w, "%s: %s\n", it.Use, it.Why)
		}
	}
	if lines := network(app.Network()); len(lines) > 0 {
		fmt.Fprintf(w, "\nThe network:\n  %s\n", strings.Join(lines, "\n  "))
	}
}

// network writes a network as its chains, a line each: "mic → vad → transcribe (keep 4) → out".
// A node whose output goes to several nodes ends its line with them, and each goes on in a line
// of its own.
func network(nodes []flow.NodeInfo) []string {
	var lines []string
	line := ""
	for i, n := range nodes {
		if line == "" {
			line = n.Name
		}
		if len(n.To) == 1 && i+1 < len(nodes) && nodes[i+1].Name == n.To[0].To {
			line += " → " + edge(n.To[0])
			continue
		}
		if len(n.To) > 0 {
			var to []string
			for _, e := range n.To {
				to = append(to, edge(e))
			}
			line += " → " + strings.Join(to, ", ")
		}
		if strings.Contains(line, "→") {
			lines = append(lines, line)
		}
		line = ""
	}
	return lines
}

func edge(e flow.EdgeInfo) string {
	if e.Policy == "" {
		return e.To
	}
	return e.To + " (" + e.Policy + ")"
}

// PrepareOptions are how Download and Prepare run.
type PrepareOptions struct {
	// AcceptLicenses accepts the licenses that ask to be accepted before their models are
	// downloaded (they are shown). Without it they are asked for when In is a terminal, and
	// refused otherwise.
	AcceptLicenses bool
	// Quiet leaves out the lines of the downloads and conversions: only the stages are written.
	Quiet bool
	// Out is where the list is written at the end (nil: the standard output), Log where the
	// progress is (nil: the standard error), In where a license's acceptance is read from (nil:
	// the standard input).
	Out, Log io.Writer
	In       *os.File
}

// Download downloads the app's models that are not here and converts their programs, without
// loading them: no NPU is needed, and the app then starts without the network.
func Download(ctx context.Context, app *ai.App, o PrepareOptions) error {
	return prepare(ctx, app, o, true)
}

// Prepare downloads and converts as Download does, then loads each model once on this
// machine's NPU, which compiles its programs for it, so that the app's first run does not wait
// for that.
func Prepare(ctx context.Context, app *ai.App, o PrepareOptions) error {
	return prepare(ctx, app, o, false)
}

func prepare(ctx context.Context, app *ai.App, c PrepareOptions, noCompile bool) error {
	if c.Out == nil {
		c.Out = os.Stdout
	}
	if c.Log == nil {
		c.Log = os.Stderr
	}
	if c.In == nil {
		c.In = os.Stdin
	}
	out := c.Log
	p, err := app.Plan(ctx)
	if err != nil {
		return err
	}
	p.NoCompile = noCompile
	if !noCompile && p.Target() == "" {
		fmt.Fprintln(out, "no NPU that runs the models was found here: downloading and converting only")
	}
	if err := licenses(p, c, out); err != nil {
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
	WritePlan(c.Out, app, p)
	return nil
}

// licenses shows the licenses of the models to download that ask to be accepted, and accepts
// them: by AcceptLicenses, else by asking on a terminal.
func licenses(p *ai.Plan, c PrepareOptions, out io.Writer) error {
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
	if !c.AcceptLicenses {
		if !console.Terminal(c.In) {
			return errors.New("these licenses ask to be accepted before the models are downloaded: accept them (--accept-license)")
		}
		fmt.Fprint(out, "Accept? [y/N] ")
		line, _ := bufio.NewReader(c.In).ReadString('\n')
		if a := strings.ToLower(strings.TrimSpace(line)); a != "y" && a != "yes" {
			return errors.New("the licenses were not accepted")
		}
	}
	p.AcceptLicenses = true
	return nil
}

// StatusOptions are how Status writes.
type StatusOptions struct {
	// JSON writes all that was read as JSON.
	JSON bool
	// Watch shows it again every this long until ctx ends: on a terminal as top does.
	Watch time.Duration
	// Wake opens the NPU and asks it too (its driver's and firmware's details): on Linux this
	// wakes a sleeping NPU for a few seconds.
	Wake bool
}

// Status writes the NPU's status on the standard output, as hwnpu status does: the device, its
// driver and firmware, how busy it is, its power, the processes that use it and the memory
// each holds, and the app's cache of models.
func Status(ctx context.Context, app *ai.App, o StatusOptions) error {
	return npustatus.Cmd{JSON: o.JSON, Watch: o.Watch, Wake: o.Wake, Store: app.Store.Models}.Run(ctx)
}

// Usage is the help of the commands Run takes, for an app's own help.
const Usage = `  models [--json]                          the models the app uses and its network: which are here,
                                           what they take to download and running, their licenses
  models download [--accept-license] [-q]  download and convert what is not here (no NPU needed)
  models prepare [--accept-license] [-q]   download, convert, and compile for this machine's NPU
  status [--json] [-w EVERY] [--wake]      this machine's NPU: the device, how busy it is, what uses it`

// Run runs the command the arguments name, for an app that parses its command line itself:
// "models" (with "list", "download" or "prepare"; "compile" is prepare) and "status", with their
// flags (Usage). handled is false, and nothing is done, when the arguments begin with neither:
// they are the app's own.
func Run(ctx context.Context, app *ai.App, args []string) (handled bool, err error) {
	if len(args) == 0 {
		return false, nil
	}
	fs := flag.NewFlagSet(app.Name+" "+args[0], flag.ContinueOnError)
	switch args[0] {
	case "status":
		var o StatusOptions
		fs.BoolVar(&o.JSON, "json", false, "write all that was read as JSON")
		fs.DurationVar(&o.Watch, "w", 0, "show it again every this long (2s)")
		fs.BoolVar(&o.Wake, "wake", false, "open the NPU and ask it too")
		if err := fs.Parse(args[1:]); err != nil {
			return true, err
		}
		return true, Status(ctx, app, o)
	case "models":
		sub, rest := "list", args[1:]
		if len(rest) > 0 && !strings.HasPrefix(rest[0], "-") {
			sub, rest = rest[0], rest[1:]
		}
		var (
			lo ListOptions
			po PrepareOptions
		)
		switch sub {
		case "list":
			fs.BoolVar(&lo.JSON, "json", false, "write the plan as JSON")
		case "download", "prepare", "compile":
			fs.BoolVar(&po.AcceptLicenses, "accept-license", false, "accept the licenses that ask to be accepted")
			fs.BoolVar(&po.Quiet, "q", false, "only the stages")
		default:
			return true, fmt.Errorf("%s models: no command %q (list, download, prepare)", app.Name, sub)
		}
		if err := fs.Parse(rest); err != nil {
			return true, err
		}
		switch sub {
		case "list":
			return true, List(ctx, app, lo)
		case "download":
			return true, Download(ctx, app, po)
		}
		return true, Prepare(ctx, app, po)
	}
	return false, nil
}
