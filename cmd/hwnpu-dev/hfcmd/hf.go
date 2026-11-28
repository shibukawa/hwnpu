// Package hfcmd is hwnpu-dev models hf: putting the conversions the sources hold on Hugging Face, one
// repository a model (docs/prebuilt.md), and pinning in models.yaml the commit each target reads.
package hfcmd

import (
	"bufio"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"text/tabwriter"

	"github.com/goccy/go-yaml"

	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/devopt"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	shared "github.com/shibukawa/hwnpu/internal/store"
)

// Cmd is hwnpu-dev models hf; its subcommands take the definitions' flag of models
// (devopt.Defs).
type Cmd struct {
	Plan  PlanCmd  `cmd:"" help:"Show the layers, and what of each model's conversions they hold could go to Hugging Face now (N:MODEL/TARGET for push), or what to do first."`
	Push  PushCmd  `cmd:"" help:"Put conversions the layers hold in their models' repositories on Hugging Face (hub.push/MODEL), then pin the commit in models.yaml."`
	Check CheckCmd `cmd:"" help:"Check the models' repositories on Hugging Face at the commits models.yaml pins: what is missing there, or not what this hwnpu makes."`
	Sizes SizesCmd `cmd:"" help:"Show what each model downloads and keeps on the disk by the NPU's vendor, from the conversions the layers hold; with --write set models.yaml's disk."`
}

func load(d *devopt.Defs) error { return d.Load() }

func signalContext() (context.Context, context.CancelFunc) {
	return signal.NotifyContext(context.Background(), os.Interrupt)
}

// HubRepoName is the repository hwnpu-dev models hf push puts model m's conversions in: hub.push/MODEL.
func HubRepoName(m chat.Model) string { return chat.HubPush + "/" + m.Name }

// selector is N:MODEL/TARGET of the command line: the layer (-1: any), the model, the group
// and, narrower, what of it (coreai: the Core AI exports alone; coreai-ARCH: that prebuilt).
type selector struct {
	layer        int
	model        chat.Model
	group, which string
}

func parseSelector(s string) (selector, error) {
	sel := selector{layer: -1}
	if n, rest, ok := strings.Cut(s, ":"); ok {
		i, err := strconv.Atoi(n)
		if err != nil || i < 0 {
			return sel, fmt.Errorf("%s: the layer before : is a number of hwnpu-dev models hf plan's list", s)
		}
		sel.layer, s = i, rest
	}
	name, target, _ := strings.Cut(s, "/")
	m, ok := chat.Lookup(name)
	if !ok {
		return sel, fmt.Errorf("%s: no such model", name)
	}
	sel.model = m
	switch {
	case target == "":
	case target == groupAMD, target == groupApple, target == groupCoreML:
		sel.group = target
	case target == chat.CoreAITarget, strings.HasPrefix(target, chat.CoreAITarget+"-"):
		sel.group, sel.which = groupApple, target
	default:
		return sel, fmt.Errorf("%s: %s is not amd-xdna2, apple, coreai, coreai-ARCH or coreml", s, target)
	}
	return sel, nil
}

// selectRows are the rows of the selectors (every model's when none), as the layers hold them.
func selectRows(ctx context.Context, args []string) ([]*row, error) {
	ls := layers()
	var sels []selector
	for _, a := range args {
		s, err := parseSelector(a)
		if err != nil {
			return nil, err
		}
		if s.layer >= len(ls) {
			return nil, fmt.Errorf("%s: no layer %d (hwnpu-dev models hf plan lists 0 to %d)", a, s.layer, len(ls)-1)
		}
		sels = append(sels, s)
	}
	if len(sels) == 0 {
		for _, m := range chat.Catalog {
			sels = append(sels, selector{layer: -1, model: m})
		}
	}
	var out []*row
	for _, s := range sels {
		for _, r := range rows(ctx, ls, s.layer, s.model, false) {
			if s.group != "" && r.group != s.group {
				continue
			}
			switch {
			case s.which == chat.CoreAITarget:
				r.prebuilt = nil
			case s.which != "":
				r.entries = slices.DeleteFunc(r.entries, func(f found) bool { return f.got.Kind == chat.CoreAIKind })
				r.prebuilt = slices.DeleteFunc(r.prebuilt, func(f found) bool { return f.got.Target != s.which })
			}
			out = append(out, r)
		}
	}
	return out, nil
}

// hfState says what the model's repository on Hugging Face has of the row at the commit
// models.yaml pins for its group: "-" for no pin, up to date, or what differs.
func hfState(ctx context.Context, r *row) (string, bool) {
	repo := HubRepoName(r.model)
	pin := ""
	for _, h := range r.model.Hub {
		if h.Repo == repo {
			pin = h.Pins[r.group]
		}
	}
	if pin == "" {
		return "-", false
	}
	if len(r.entries) == 0 {
		return "pinned " + short(pin), false
	}
	have := 0
	for _, f := range r.entries {
		b, err := shared.HubRead(ctx, repo, pin, f.rel()+"/"+shared.EntryFile)
		var e shared.Entry
		if err == nil && json.Unmarshal(b, &e) == nil && chat.Takes(f.want, e) {
			have++
		}
	}
	if have == len(r.entries) {
		return "up to date", true
	}
	return fmt.Sprintf("old (%d/%d)", have, len(r.entries)), false
}

func short(c string) string { return c[:min(len(c), 8)] }

// printLayers writes the layers hwnpu-dev models hf takes from and where push puts the conversions.
func printLayers(w io.Writer) {
	fmt.Fprintln(w, "layers:")
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	for i, s := range layers() {
		what := "a source"
		switch {
		case i == 0:
			what = "this machine's cache"
		case s.Writeback:
			what = "a source, written back to"
		}
		fmt.Fprintf(tw, "  %d\t%s\t%s\n", i, s, what)
	}
	fmt.Fprintf(tw, "  -\thf:%s\twhere hwnpu-dev models hf push puts them (hub.push of %s)\n", chat.HubPush, chat.HubPushFile)
	tw.Flush()
}

// printRows writes the rows as hwnpu-dev models hf plan shows them, the ones that can go first.
func printRows(ctx context.Context, w io.Writer, rs []*row) {
	type line struct {
		r        *row
		state    string
		uptodate bool
	}
	var ls []line
	for _, r := range rs {
		st, ok := hfState(ctx, r)
		ls = append(ls, line{r, st, ok})
	}
	rank := func(l line) int {
		switch {
		case len(l.r.why) == 0 && !l.uptodate:
			return 0
		case len(l.r.why) == 0:
			return 1
		}
		return 2
	}
	slices.SortStableFunc(ls, func(a, b line) int {
		return cmp.Or(cmp.Compare(rank(a), rank(b)), strings.Compare(a.r.model.Name, b.r.model.Name), strings.Compare(a.r.group, b.r.group))
	})
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "candidate\tlicense\tpinned\tversion\tchips\tGB\thf\tnext")
	for _, l := range ls {
		r := l.r
		pinned := "no"
		if chat.HubCommitPinned(r.model) {
			pinned = "yes"
		}
		conv := cache.AMDTarget
		chips := "-"
		switch r.group {
		case groupApple:
			conv = chat.CoreAITarget
			chips = fmt.Sprintf("%d/%d", len(r.chips), len(prebuiltChips))
		case groupCoreML:
			conv = chat.CoreMLTarget
		}
		version := "-"
		if v := r.model.Targets[conv]; v > 0 {
			version = fmt.Sprintf("v%d", v)
		}
		next := "push"
		switch {
		case len(r.why) > 0:
			next = r.why[0]
		case l.uptodate:
			next = "-"
		}
		gb := "-"
		if len(r.entries) > 0 {
			gb = fmt.Sprintf("%.2f", float64(r.bytes())/1e9)
		}
		fmt.Fprintf(tw, "%s:%s/%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", r.layerLabel(), r.model.Name, r.group,
			cmp.Or(r.model.License.ID, "-"), pinned, version, chips, gb, l.state, next)
	}
	tw.Flush()
}

// prebuiltChips are the architectures coreai-build makes programs for (docs/design/apple-coreai.md).
var prebuiltChips = []string{"h13g", "h14c", "h14g", "h15c", "h15g", "h16c", "h16g", "h16s", "h17c", "h17d", "h17g", "h17s"}

// PlanCmd is hwnpu-dev models hf plan.
type PlanCmd struct {
	Sel     []string `arg:"" optional:"" name:"N:MODEL/TARGET" help:"What to show (every model when left out): a model, its target (amd-xdna2, apple, coreai, coreai-ARCH, coreml) after /, the layer it comes from before :."`
	Verbose bool     `short:"v" help:"Show each entry, and the chips that have prebuilt programs."`
}

func (p PlanCmd) Run(d *devopt.Defs) error {
	if err := load(d); err != nil {
		return err
	}
	ctx, stop := signalContext()
	defer stop()
	printLayers(os.Stdout)
	fmt.Println()
	rs, err := selectRows(ctx, p.Sel)
	if err != nil {
		return err
	}
	printRows(ctx, os.Stdout, rs)
	if p.Verbose {
		for _, r := range rs {
			if len(r.entries)+len(r.prebuilt) == 0 && len(r.why) == 0 {
				continue
			}
			fmt.Printf("\n%s/%s:\n", r.model.Name, r.group)
			for _, f := range r.entries {
				note := ""
				switch {
				case f.tools != "":
					note = fmt.Sprintf(" (from %s, made with other tools: %s; check the export, or make it again)", f.from[strings.LastIndex(f.from, "/")+1:], short(f.tools))
				case f.raw == nil:
					note = fmt.Sprintf(" (from %s, the same key)", f.from[strings.LastIndex(f.from, "/")+1:])
				}
				fmt.Printf("  %d  %s%s\n", f.layer, f.rel(), note)
			}
			if len(r.prebuilt) > 0 {
				fmt.Printf("  prebuilt (checked against the export's programs when pushed): %s\n", strings.Join(r.chips, ", "))
				var lack []string
				for _, a := range prebuiltChips {
					if !slices.Contains(r.chips, a) {
						lack = append(lack, a)
					}
				}
				if len(lack) > 0 && r.group == groupApple {
					fmt.Printf("  no prebuilt for: %s (those Macs compile the export)\n", strings.Join(lack, ", "))
				}
			}
			for _, w := range r.why {
				fmt.Printf("  to do: %s\n", w)
			}
		}
	}
	return nil
}

// PushCmd is hwnpu-dev models hf push.
type PushCmd struct {
	Sel    []string `arg:"" optional:"" name:"N:MODEL/TARGET" help:"What to put (when left out: every row hwnpu-dev models hf plan shows can go, asked first)."`
	To     string   `placeholder:"NAMESPACE" help:"The namespace on Hugging Face to put them in, this time (else models.yaml's hub.push)."`
	Write  string   `placeholder:"FILE" help:"The definition file to pin the commits in (else the one that gave hub.push: the source tree's models.yaml for the built-in one)."`
	All    bool     `help:"Do not ask before putting every row that can go (a repository is still made only when asked)."`
	DryRun bool     `name:"dry-run" help:"Show what would go and the lines models.yaml would get; download and put nothing."`
	Keep   bool     `help:"Keep the folder the repository was made in (the cache's work/hf/MODEL)."`
}

func (p PushCmd) Run(d *devopt.Defs) error {
	if err := load(d); err != nil {
		return err
	}
	ctx, stop := signalContext()
	defer stop()
	ns := cmp.Or(p.To, chat.HubPush)
	if ns == "" {
		return errors.New("no namespace to put the conversions in: give models.yaml hub: {push: NAMESPACE}, or --to")
	}
	chat.HubPush = ns
	rs, err := selectRows(ctx, p.Sel)
	if err != nil {
		return err
	}
	// what can go
	var goes []*row
	for _, r := range rs {
		if len(r.why) > 0 {
			if len(p.Sel) > 0 {
				fmt.Fprintf(os.Stderr, "%s/%s does not go: %s\n", r.model.Name, r.group, strings.Join(r.why, "; "))
			}
			continue
		}
		if st, ok := hfState(ctx, r); ok {
			fmt.Fprintf(os.Stderr, "%s/%s: %s at the pinned commit\n", r.model.Name, r.group, st)
			continue
		}
		goes = append(goes, r)
	}
	if len(goes) == 0 {
		return errors.New("nothing to put (hwnpu-dev models hf plan says what to do first)")
	}
	printLayers(os.Stderr)
	printRows(ctx, os.Stderr, goes)
	if p.DryRun {
		for _, r := range goes {
			fmt.Printf("%s/%s -> %s (%.2f GB):\n", r.model.Name, r.group, HubRepoName(r.model), float64(r.bytes())/1e9)
			for _, f := range slices.Concat(r.entries, r.prebuilt) {
				fmt.Printf("  %s  (layer %d: %s)\n", f.rel(), f.layer, f.from)
			}
			fmt.Printf("  then %s: models.%s.hub.%s.%s: COMMIT\n", p.pinFile(), r.model.Name, HubRepoName(r.model), r.group)
		}
		return nil
	}
	if len(p.Sel) == 0 && !p.All && !confirm(fmt.Sprintf("Put these %d in %s on Hugging Face?", len(goes), ns)) {
		return errors.New("nothing put")
	}
	pinFile := p.pinFile()
	if pinFile == "" {
		return errors.New("no models.yaml to pin the commits in: run it in hwnpu's source tree, or give --write FILE")
	}
	tok, who, err := writeToken(ctx, ns)
	if err != nil {
		return err
	}
	hfCLI, err := exec.LookPath(cmp.Or(os.Getenv("HWNPU_HF_CLI"), "hf"))
	if err != nil {
		return errors.New("hf (huggingface_hub's command, with hf_xet) is not on PATH: install it (uv tool install huggingface_hub, or pip install -U huggingface_hub) or set HWNPU_HF_CLI")
	}
	byModel := map[string][]*row{}
	var order []string
	for _, r := range goes {
		if _, ok := byModel[r.model.Name]; !ok {
			order = append(order, r.model.Name)
		}
		byModel[r.model.Name] = append(byModel[r.model.Name], r)
	}
	var failed []string
	for _, name := range order {
		if err := p.pushModel(ctx, tok, who, ns, hfCLI, pinFile, byModel[name]); err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", name, err)
			failed = append(failed, name)
		}
	}
	fmt.Fprintf(os.Stderr, "\npinned in %s: see git diff %s, then go test ./internal/chat (TestBuiltinHub) and put it in a PR\n", pinFile, pinFile)
	if len(failed) > 0 {
		return fmt.Errorf("not put: %s", strings.Join(failed, ", "))
	}
	return nil
}

// pinFile is the definition file the commits are pinned in.
func (p PushCmd) pinFile() string {
	if p.Write != "" {
		return p.Write
	}
	if p.To != "" || chat.HubPushFile != chat.Builtin {
		if p.To != "" {
			return chat.UserModelsFile()
		}
		return chat.HubPushFile
	}
	return sourceTreeModels()
}

// sourceTreeModels is the models.yaml of hwnpu's source tree the current folder is in, "" when
// it is in none.
func sourceTreeModels() string {
	d, err := os.Getwd()
	if err != nil {
		return ""
	}
	for {
		mod, err := os.ReadFile(filepath.Join(d, "go.mod"))
		if err == nil && strings.HasPrefix(string(mod), "module github.com/shibukawa/hwnpu\n") {
			if p := filepath.Join(d, "models.yaml"); exists(p) {
				return p
			}
		}
		up := filepath.Dir(d)
		if up == d {
			return ""
		}
		d = up
	}
}

func exists(p string) bool { _, err := os.Stat(p); return err == nil }

// pushModel makes the repository of one model's rows, uploads it and pins its commit.
func (p PushCmd) pushModel(ctx context.Context, tok string, who whoami, ns, hfCLI, pinFile string, rs []*row) error {
	m := rs[0].model
	repo := HubRepoName(m)
	log := os.Stderr
	// the repository: made public when there is none (asked), gated as the model's own is
	info, ok, err := modelInfo(ctx, tok, repo, "")
	if err != nil {
		return err
	}
	upstream, _, err := modelInfo(ctx, "", m.Repo, "")
	if err != nil {
		return fmt.Errorf("%s: %w", m.Repo, err)
	}
	if !ok {
		if !confirm(fmt.Sprintf("Make the public repository %s on Hugging Face?", repo)) {
			return errors.New("no repository")
		}
		if err := createRepo(ctx, tok, who, ns, m.Name); err != nil {
			return err
		}
		fmt.Fprintf(log, "%s: made\n", repo)
	}
	if g := upstream.gated(); g != "" && info.gated() != g {
		if err := setGated(ctx, tok, repo, g); err != nil {
			return fmt.Errorf("making %s gated (%s) as %s is: %w", repo, g, m.Repo, err)
		}
		fmt.Fprintf(log, "%s: gated (%s), as %s is\n", repo, g, m.Repo)
	}
	shared.SetHubToken(repo, tok)

	// the folder of the repository: what goes, the card, the license and the list
	dir := cache.Sub("work", "hf", m.Name)
	if !p.Keep {
		defer os.RemoveAll(dir)
	}
	ls := layers()
	var all []found
	for _, r := range rs {
		fmt.Fprintf(log, "%s/%s: %.2f GB to %s\n", m.Name, r.group, float64(r.bytes())/1e9, repo)
		for _, f := range r.entries {
			if err := stage(ctx, ls, f, dir, log); err != nil {
				return err
			}
			all = append(all, f)
		}
		// the prebuilt programs of the Core AI exports that go, of their programs
		digests := map[string]string{}
		for _, f := range r.entries {
			if f.got.Kind == chat.CoreAIKind {
				d, err := programsDigest(f, dir)
				if err != nil {
					return fmt.Errorf("%s: its programs: %w", f.rel(), err)
				}
				digests[f.got.Recipe["export"]] = d
			}
		}
		for _, f := range r.prebuilt {
			export, _, _ := chat.IsPrebuilt(f.got)
			if digests[export] == "" || f.got.Recipe["programs"] != digests[export] {
				continue // of other programs than the export's
			}
			if err := stage(ctx, ls, f, dir, log); err != nil {
				return err
			}
			all = append(all, f)
		}
	}
	commit := chat.SourceCommit(ctx, m)
	var old *manifest
	if b, err := shared.HubRead(ctx, repo, "main", "config.json"); err == nil {
		var o manifest
		if json.Unmarshal(b, &o) == nil {
			old = &o
		}
	}
	man := merged(old, m, commit, all)
	b, err := json.MarshalIndent(man, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), append(b, '\n'), 0o644); err != nil {
		return err
	}
	if !license(ctx, m, commit, dir) {
		fmt.Fprintf(log, "%s: no license file in %s; the card links %s\n", m.Name, m.Repo, m.License.URL)
	}
	if err := os.WriteFile(filepath.Join(dir, "README.md"), []byte(card(m, commit, man, upstream.gated())), 0o644); err != nil {
		return err
	}

	// up: hf upload-large-folder goes on where it stopped and sends only what Hugging Face lacks
	cmd := exec.CommandContext(ctx, hfCLI, "upload-large-folder", repo, dir, "--repo-type", "model")
	cmd.Env = append(os.Environ(), "HF_TOKEN="+tok)
	cmd.Stdout, cmd.Stderr = log, log
	fmt.Fprintf(log, "%s: %s upload-large-folder %s %s\n", m.Name, hfCLI, repo, dir)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("hf upload-large-folder: %w", err)
	}
	after, _, err := modelInfo(ctx, tok, repo, "main")
	if err != nil || len(after.SHA) != 40 {
		return fmt.Errorf("the commit of %s after the upload: %v", repo, err)
	}
	// each row's entries are there at that commit: then it is pinned
	for _, r := range rs {
		for _, f := range r.entries {
			b, err := shared.HubRead(ctx, repo, after.SHA, f.rel()+"/"+shared.EntryFile)
			var e shared.Entry
			if err != nil || json.Unmarshal(b, &e) != nil || e.Key != f.got.Key {
				return fmt.Errorf("%s is not at %s of %s after the upload (%v): not pinned", f.rel(), short(after.SHA), repo, err)
			}
		}
		src, err := os.ReadFile(pinFile)
		if err != nil {
			return err
		}
		out, err := pinYAML(src, m.Name, repo, r.group, after.SHA)
		if err != nil {
			return fmt.Errorf("%s: %w (pin models.%s.hub.%s.%s: %s by hand)", pinFile, err, m.Name, repo, r.group, after.SHA)
		}
		var back struct {
			Models map[string]struct {
				Hub map[string]map[string]string `yaml:"hub"`
			} `yaml:"models"`
		}
		if err := yaml.Unmarshal(out, &back); err != nil || back.Models[m.Name].Hub[repo][r.group] != after.SHA {
			return fmt.Errorf("%s would not read models.%s.hub.%s.%s as %s (%v): pin it by hand", pinFile, m.Name, repo, r.group, after.SHA, err)
		}
		if err := os.WriteFile(pinFile, out, 0o644); err != nil {
			return err
		}
		fmt.Fprintf(log, "%s/%s: pinned %s at %s in %s\n", m.Name, r.group, repo, after.SHA, pinFile)
	}
	return nil
}

// writeToken is a token that may write to namespace ns, and its account: HF_TOKEN, or hf auth
// login's, else one pasted at the terminal (kept in memory only).
func writeToken(ctx context.Context, ns string) (string, whoami, error) {
	tok, from := knownToken()
	for try := 0; ; try++ {
		if tok != "" {
			var w whoami
			err := api(ctx, tok, "GET", "/api/whoami-v2", nil, &w)
			if err == nil {
				ok, why := w.canWrite(ns)
				if ok {
					return tok, w, nil
				}
				fmt.Fprintf(os.Stderr, "The token of %s cannot write to %s on Hugging Face: %s.\n", from, ns, why)
			} else {
				fmt.Fprintf(os.Stderr, "Hugging Face does not take the token of %s: %v\n", from, err)
			}
		}
		if try >= 2 || !console.Terminal(os.Stdin) || !console.Terminal(os.Stderr) {
			return "", whoami{}, fmt.Errorf("no token that writes to %s: make one at https://huggingface.co/settings/tokens (fine-grained, writing to %s's repositories) and give it in HF_TOKEN, or hf auth login", ns, ns)
		}
		t, err := console.ReadSecret(fmt.Sprintf("Paste a token of Hugging Face that writes to %s (https://huggingface.co/settings/tokens; kept in memory only): ", ns))
		if err != nil {
			return "", whoami{}, err
		}
		tok, from = strings.TrimSpace(t), "the pasted token"
	}
}

// confirm asks a yes or no question at the terminal; no without one (tests replace it).
var confirm = func(q string) bool {
	if !console.Terminal(os.Stdin) || !console.Terminal(os.Stderr) {
		fmt.Fprintf(os.Stderr, "%s no (no terminal to ask at)\n", q)
		return false
	}
	fmt.Fprintf(os.Stderr, "%s [y/N] ", q)
	line, _ := bufio.NewReader(os.Stdin).ReadString('\n')
	a := strings.ToLower(strings.TrimSpace(line))
	return a == "y" || a == "yes"
}

// CheckCmd is hwnpu-dev models hf check.
type CheckCmd struct {
	Model []string `arg:"" optional:"" help:"Models to check (every model whose hub pins a commit when left out)."`
}

func (k CheckCmd) Run(d *devopt.Defs) error {
	if err := load(d); err != nil {
		return err
	}
	ctx, stop := signalContext()
	defer stop()
	var ms []chat.Model
	if len(k.Model) == 0 {
		for _, m := range chat.Catalog {
			if len(m.Hub) > 0 {
				ms = append(ms, m)
			}
		}
	}
	for _, n := range k.Model {
		m, ok := chat.Lookup(n)
		if !ok {
			return fmt.Errorf("%s: no such model", n)
		}
		ms = append(ms, m)
	}
	problems := 0
	tw := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "model\trepository\ttarget\tcommit\tentry\tstate")
	for _, m := range ms {
		if len(m.Hub) == 0 {
			fmt.Fprintf(tw, "%s\t-\t-\t-\t-\tno hub in models.yaml\n", m.Name)
			problems++
			continue
		}
		for _, h := range m.Hub {
			for _, target := range slices.Sorted(mapKeys(h.Pins)) {
				pin := h.Pins[target]
				n, err := checkPin(ctx, tw, m, h.Repo, target, pin)
				if err != nil {
					fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t-\t%v\n", m.Name, h.Repo, target, short(pin), err)
					n++
				}
				problems += n
			}
		}
	}
	tw.Flush()
	if problems > 0 {
		return fmt.Errorf("%d problems", problems)
	}
	return nil
}

// checkPin checks the entries of a repository at the commit a target reads: each one the
// repository's config.json lists for it is there and is what this hwnpu makes now, and, for
// AMD's NPU, every export the commands look for is there. It returns the problems it wrote.
func checkPin(ctx context.Context, w io.Writer, m chat.Model, repo, target, pin string) (int, error) {
	if pin == "main" {
		fmt.Fprintf(w, "%s\t%s\t%s\tmain\t-\tnot a commit: pin one (hwnpu-dev models hf push writes it)\n", m.Name, repo, target)
	}
	b, err := shared.HubRead(ctx, repo, pin, "config.json")
	if err != nil {
		return 0, fmt.Errorf("no config.json at that commit: %w", err)
	}
	var man manifest
	if err := json.Unmarshal(b, &man); err != nil {
		return 0, fmt.Errorf("config.json: %w", err)
	}
	problems := 0
	report := func(path, state string, bad bool) {
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%s\n", m.Name, repo, target, short(pin), path, state)
		if bad {
			problems++
		}
	}
	ofTarget := func(t string) bool {
		switch target {
		case groupAMD:
			return t == cache.AMDTarget || t == "any"
		case groupCoreML:
			return t == chat.CoreMLTarget
		}
		return t == chat.CoreAITarget || strings.HasPrefix(t, chat.CoreAITarget+"-") && (target == groupApple || t == target) || t == "any"
	}
	read := func(rel string) (shared.Entry, error) {
		b, err := shared.HubRead(ctx, repo, pin, rel+"/"+shared.EntryFile)
		var e shared.Entry
		if err == nil {
			err = json.Unmarshal(b, &e)
		}
		return e, err
	}
	conv := target
	switch target {
	case groupAMD:
	case groupCoreML:
		conv = chat.CoreMLTarget
	default:
		conv = chat.CoreAITarget
	}
	pinnedAMD := target == groupAMD && len(m.Pins[cache.AMDTarget]) > 0
	if m.Targets[conv] == 0 && !pinnedAMD {
		what := "targets." + conv + ".version"
		if target == groupAMD {
			what = "models." + m.Name + ".amd.version of model-recipes.yaml"
		}
		report("-", what+" is not written", true)
	}
	if pinnedAMD {
		for _, export := range slices.Sorted(maps.Keys(m.Pins[cache.AMDTarget])) {
			pin := m.Pins[cache.AMDTarget][export]
			rel := hubRel(chat.PinnedPath(m.Name, cache.AMDTarget, export, pin))
			e, err := read(rel)
			switch {
			case err != nil:
				report(rel, "missing (hwnpu-dev models hf push)", true)
			case e.Digest() != pin.Digest:
				report(rel, "not the pinned digest "+pin.Digest, true)
			case !chat.FormatTaken(e):
				report(rel, "a form this hwnpu does not read (chat.AMDExportFormat): make it and pin it again", true)
			default:
				report(rel, "ok", false)
			}
		}
	} else if target == groupAMD {
		wants, err := amdWants(ctx, m)
		if err != nil {
			report("-", err.Error(), true)
		}
		for _, want := range wants {
			rel := hubRel(want.Path)
			e, err := read(rel)
			switch {
			case err != nil:
				report(rel, "missing (hwnpu-dev models hf push)", true)
			case chat.Takes(want, e) && e.Recipe["tools"] != want.Recipe["tools"]:
				report(rel, "the tools changed since: a version to bump if the export changed", false)
			case chat.Takes(want, e):
				report(rel, "ok", false)
			default:
				report(rel, "not what this hwnpu makes: bump targets.amd-xdna2.version and push again", true)
			}
		}
	}
	current := 0
	for _, me := range man.Hwnpu.Entries {
		if !ofTarget(me.Target) || me.Target == cache.AMDTarget {
			continue
		}
		e, err := read(me.Path)
		if err != nil {
			report(me.Path, "listed in config.json but missing", true)
			continue
		}
		var want shared.Entry
		var ok bool
		switch e.Kind {
		case chat.CoreAIKind:
			want, ok = chat.CoreAIWant(ctx, m, e.Recipe["export"])
		case chat.CoreMLKind:
			want, ok = chat.CoreMLWant(ctx, m, e.Recipe["export"])
		case chat.TablesKind:
			want, ok = chat.TablesWant(ctx, m, e)
		case chat.PrebuiltKind:
			report(me.Path, "ok (prebuilt of its export's programs)", false)
			continue
		}
		switch {
		case !ok:
			report(me.Path, "this hwnpu does not know it", true)
		case want.Path != "converted/"+m.Name+"/"+me.Path:
			report(me.Path, "another version (kept for the hwnpu that reads it); this one reads "+hubRel(want.Path), false)
		case chat.Takes(want, e):
			report(me.Path, "ok", false)
			if e.Kind == chat.CoreAIKind || e.Kind == chat.CoreMLKind {
				current++
			}
		default:
			report(me.Path, "not what this hwnpu makes: bump the version and push again", true)
		}
	}
	switch {
	case target == groupCoreML && current == 0:
		report("-", "no Core ML package of the version this hwnpu reads", true)
	case target != groupAMD && target != groupCoreML && current == 0:
		report("-", "no Core AI export of the version this hwnpu reads", true)
	}
	return problems, nil
}
