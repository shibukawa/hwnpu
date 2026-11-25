package cachecmd

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"os"
	"path"
	"slices"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/modelstore"
)

// CacheCmd is hwnpu cache.
type CacheCmd struct {
	Info  InfoCmd  `cmd:"" default:"1" help:"Where the cache is, what it holds by kind, the disk's free space, the limits and the layers models are read from (the command when none is named)."`
	Ls    LsCmd    `cmd:"" help:"What the cache holds (or what PATTERN names) by model and kind: the units, size, when last used; the least recently used first (--units: each unit)."`
	Prune PruneCmd `cmd:"" help:"Remove the units least recently used until the cache holds --max-size, or those not used for --older-than. Units used in the last hour are kept."`
	Clear ClearCmd `cmd:"" help:"Remove the units PATTERN names (a model's name, OWNER/NAME, a glob: 'Qwen*'), of --kind, or --all. Core AI's compiled programs of the exports in them (~/Library/Caches/coreai-cache) go too, as with prune."`
	Clean CleanCmd `cmd:"" help:"Remove from the sources written back to (HWNPU_MODEL_SOURCE ?writeback=true) what a writeback cut off left: an export's folder with no entry.json, a repository's files its hwnpu-lock.json does not name, the temporary files of uploads, in S3 the uploads in parts cut off over an hour ago. Folders being written now are left."`
}

// open opens hwnpu's store as the commands do, its downloads' log to e.Err.
func (e *Env) open() (*modelstore.Store, error) {
	return modelstore.Open(modelstore.Options{Log: e.Err})
}

// KindHelp names the kinds of units, for the commands' help (kong's ${kinds}).
var KindHelp = strings.Join(cache.Kinds, ", ")

// InfoCmd is hwnpu cache info.
type InfoCmd struct{}

func (c InfoCmd) Run(ctx context.Context) error { return run(ctx, Request{Command: "info"}) }

func (InfoCmd) exec(e *Env) error {
	s, err := e.open()
	if err != nil {
		return err
	}
	u, err := s.Usage()
	if err != nil {
		return err
	}
	fmt.Fprintf(e.Out, "cache:  %s", u.Root)
	if os.Getenv(cache.RootEnv) != "" {
		fmt.Fprintf(e.Out, " (%s)", cache.RootEnv)
	}
	fmt.Fprintln(e.Out)
	free, total, derr := s.DiskSpace()
	if derr == nil {
		fmt.Fprintf(e.Out, "disk:   %s free of %s\n", cache.FormatSize(free), cache.FormatSize(total))
	}
	l, err := s.Limits()
	if err != nil {
		fmt.Fprintf(e.Out, "limits: %v\n", err)
	} else {
		fmt.Fprintf(e.Out, "limits: %s (%s, %s)\n", l, cache.MaxSizeEnv, cache.MinFreeEnv)
		if derr == nil && free < l.MinFreeOf(total) {
			fmt.Fprintf(e.Out, "        %s: downloads stop until hwnpu cache prune or clear frees room\n", "the disk has less free")
		}
	}
	fmt.Fprintf(e.Out, "holds:  %s\n", cache.FormatSize(u.Bytes))
	w := tabwriter.NewWriter(e.Out, 0, 4, 2, ' ', tabwriter.AlignRight)
	by := u.ByKind()
	n := map[string]int{}
	for _, x := range u.Units {
		n[x.Kind]++
	}
	for _, k := range cache.Kinds {
		fmt.Fprintf(w, "  %s\t%s\t%d units\t\n", k, cache.FormatSize(by[k]), n[k])
	}
	var other int64
	var names []string
	for k, b := range u.Other {
		other += b
		if b > 0 {
			names = append(names, k)
		}
	}
	slices.SortFunc(names, func(a, b string) int { return cmp.Compare(u.Other[b], u.Other[a]) })
	var parts []string
	for _, k := range names[:min(4, len(names))] {
		parts = append(parts, fmt.Sprintf("%s %s", k, cache.FormatSize(u.Other[k])))
	}
	fmt.Fprintf(w, "  other\t%s\t\t\n", cache.FormatSize(other))
	w.Flush()
	if len(parts) > 0 {
		fmt.Fprintf(e.Out, "        (other: %s; never removed by prune)\n", strings.Join(parts, ", "))
	}
	fmt.Fprintln(e.Out, "layers (models are read from the first that has them):")
	for i, ly := range s.Layers() {
		what := ""
		switch {
		case ly.Bundled:
			what = " (installed with the application, read only)"
		case ly.Writeback:
			what = " (written back to)"
		}
		fmt.Fprintf(e.Out, "  %d  %s%s\n", i, ly.Name, what)
	}
	return nil
}

// match reports whether unit x is one of the patterns: a glob of its name (original/OWNER/NAME),
// of what follows its kind's folder (OWNER/NAME), or of its last part (NAME), in any case; a
// model's name also names its exports (NAME.amd-npu2-q4).
func match(x modelstore.Unit, patterns []string) bool {
	name := strings.ToLower(x.Name)
	_, rest, _ := strings.Cut(name, "/")
	if x.Kind == cache.KindExport || x.Kind == cache.KindConverted && strings.HasPrefix(name, "converted/") {
		_, rest, _ = strings.Cut(rest, "/") // entries/TARGET/NAME, converted/MODEL/...: past TARGET
	}
	base := path.Base(name)
	export, _, _ := strings.Cut(base, ".")
	model := strings.ToLower(x.Model)
	_, modelName, _ := strings.Cut(model, "/")
	for _, p := range patterns {
		p = strings.ToLower(p)
		for _, s := range []string{name, rest, base, export, model, modelName} {
			if s == "" {
				continue
			}
			if ok, _ := path.Match(p, s); ok {
				return true
			}
		}
		if x.Kind == cache.KindConverted && strings.HasPrefix(name, "converted/") {
			if m, _, _ := strings.Cut(strings.TrimPrefix(name, "converted/"), "/"); m == p {
				return true // converted/MODEL/...
			}
		}
	}
	return false
}

func badKinds(kinds []string) error {
	for _, k := range kinds {
		if !slices.Contains(cache.Kinds, k) {
			return fmt.Errorf("no kind %q: %s", k, KindHelp)
		}
	}
	return nil
}

func used(t time.Time) string {
	if t.IsZero() {
		return "-"
	}
	d := time.Since(t)
	switch {
	case d < time.Hour:
		return fmt.Sprintf("%d min ago", int(d.Minutes()))
	case d < 48*time.Hour:
		return fmt.Sprintf("%d h ago", int(d.Hours()))
	}
	return fmt.Sprintf("%d days ago", int(d.Hours()/24))
}

// modelName is the model column of x.
func modelName(x modelstore.Unit) string {
	switch {
	case x.Leftover:
		return "(record of a removed program)"
	case x.Model != "":
		return x.Model
	case x.Kind == cache.KindKernels:
		return "(IRON kernels, shared)"
	}
	return "(source not recorded)"
}

// table lists each unit.
func table(e *Env, units []modelstore.Unit) {
	w := tabwriter.NewWriter(e.Out, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "model\tkind\tsize\tlast used\tunit")
	var total int64
	for _, x := range units {
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\n", modelName(x), x.Kind, cache.FormatSize(x.Bytes), used(x.Used), x.Name)
		total += x.Bytes
	}
	w.Flush()
	fmt.Fprintf(e.Out, "%d units, %s\n", len(units), cache.FormatSize(total))
}

// group is the units of a model of a kind.
type group struct {
	model, kind string
	units       int
	bytes       int64
	used        time.Time
}

// groups are the units by model and kind, in the order of their last use (the latest of theirs).
func groups(units []modelstore.Unit) []group {
	var out []group
	at := map[[2]string]int{}
	for _, x := range units {
		k := [2]string{modelName(x), x.Kind}
		i, ok := at[k]
		if !ok {
			i = len(out)
			at[k] = i
			out = append(out, group{model: k[0], kind: k[1]})
		}
		g := &out[i]
		g.units++
		g.bytes += x.Bytes
		if x.Used.After(g.used) {
			g.used = x.Used
		}
	}
	slices.SortStableFunc(out, func(a, b group) int { return a.used.Compare(b.used) })
	return out
}

// summary lists the units by model and kind.
func summary(e *Env, gs []group) {
	w := tabwriter.NewWriter(e.Out, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "model\tkind\tunits\tsize\tlast used")
	var n int
	var total int64
	for _, g := range gs {
		fmt.Fprintf(w, "%s\t%s\t%d\t%s\t%s\n", g.model, g.kind, g.units, cache.FormatSize(g.bytes), used(g.used))
		n += g.units
		total += g.bytes
	}
	w.Flush()
	fmt.Fprintf(e.Out, "%d units, %s\n", n, cache.FormatSize(total))
}

// LsCmd is hwnpu cache ls.
type LsCmd struct {
	Patterns []string `arg:"" optional:"" name:"pattern" help:"Units to list: a model's name, OWNER/NAME, a glob (all when left out)."`
	Kind     []string `short:"k" help:"Only units of these kinds (${kinds})." placeholder:"KIND,..."`
	Size     bool     `short:"s" help:"The largest first, instead of the least recently used."`
	Units    bool     `short:"u" help:"Each unit, instead of the units of a model of a kind together."`
}

func (l LsCmd) Run(ctx context.Context) error { return run(ctx, Request{Command: "ls", Ls: l}) }

func (l LsCmd) exec(e *Env) error {
	if err := badKinds(l.Kind); err != nil {
		return err
	}
	s, err := e.open()
	if err != nil {
		return err
	}
	u, err := s.Usage()
	if err != nil {
		return err
	}
	units := pick(u.ByUse(), l.Patterns, l.Kind)
	if l.Units {
		if l.Size {
			slices.SortStableFunc(units, func(a, b modelstore.Unit) int { return cmp.Compare(b.Bytes, a.Bytes) })
		}
		table(e, units)
		e.held(units)
		return nil
	}
	gs := groups(units)
	if l.Size {
		slices.SortStableFunc(gs, func(a, b group) int { return cmp.Compare(b.bytes, a.bytes) })
	}
	summary(e, gs)
	e.held(units)
	return nil
}

func pick(units []modelstore.Unit, patterns, kinds []string) []modelstore.Unit {
	var out []modelstore.Unit
	for _, x := range units {
		if len(kinds) > 0 && !slices.Contains(kinds, x.Kind) || len(patterns) > 0 && !match(x, patterns) {
			continue
		}
		out = append(out, x)
	}
	return out
}

// PruneCmd is hwnpu cache prune.
type PruneCmd struct {
	MaxSize   string   `help:"Remove the units least recently used until the cache holds at most this (200GB; ${max_size_env} when left out)." placeholder:"SIZE"`
	OlderThan string   `help:"Remove the units not used for this long (30d, 12h, 2w)." placeholder:"AGE"`
	Kind      []string `short:"k" help:"Only units of these kinds (${kinds})." placeholder:"KIND,..."`
	DryRun    bool     `short:"n" help:"Show what would be removed, and remove nothing."`
}

func (p PruneCmd) Run(ctx context.Context) error {
	return run(ctx, Request{Command: "prune", Prune: p})
}

func (p PruneCmd) exec(e *Env) error {
	if err := badKinds(p.Kind); err != nil {
		return err
	}
	s, err := e.open()
	if err != nil {
		return err
	}
	o := modelstore.PruneOptions{Kinds: p.Kind}
	if p.MaxSize != "" {
		if o.MaxSize, err = cache.ParseSize(p.MaxSize); err != nil {
			return err
		}
	}
	if p.OlderThan != "" {
		if o.OlderThan, err = parseAge(p.OlderThan); err != nil {
			return err
		}
	}
	if o.MaxSize == 0 && o.OlderThan == 0 {
		l, err := s.Limits()
		if err != nil {
			return err
		}
		if l.MaxSize == 0 {
			return fmt.Errorf("say how much to keep: --max-size, --older-than (or set %s)", cache.MaxSizeEnv)
		}
		o.MaxSize = l.MaxSize
	}
	units, err := s.PlanPrune(o)
	if err != nil {
		return err
	}
	return remove(e, s, e.notHeld(units), p.DryRun, true)
}

// remove removes units, or shows them with dryRun; ask asks first on a terminal unless yes.
func remove(e *Env, s *modelstore.Store, units []modelstore.Unit, dryRun, yes bool) error {
	if len(units) == 0 {
		fmt.Fprintln(e.Out, "nothing to remove")
		return nil
	}
	if len(units) > 30 {
		summary(e, groups(units)) // hwnpu cache ls --units lists each
	} else {
		table(e, units)
	}
	// Core AI's entries of the exports go with them (cache.CoreAIEntries), outside the cache
	entries := map[string]int64{}
	var entryBytes int64
	for _, x := range units {
		if b := cache.CoreAIBytes(cache.CoreAIEntries(x)); b > 0 {
			entries[x.Path] = b
			entryBytes += b
		}
	}
	if entryBytes > 0 {
		fmt.Fprintf(e.Out, "and %s of Core AI's compiled programs of their exports (~/Library/Caches/coreai-cache): compiled again when an export is downloaded again\n", cache.FormatSize(entryBytes))
	}
	if dryRun {
		fmt.Fprintln(e.Out, "(--dry-run: nothing removed)")
		return nil
	}
	if !yes {
		if e.Ask == nil {
			return errors.New("--yes removes them (standard input is not a terminal to ask on)")
		}
		if !e.Ask("remove them? [y/N] ") {
			fmt.Fprintln(e.Out, "nothing removed")
			return nil
		}
		// what was loaded while the question was asked stays too
		if units = e.notHeld(units); len(units) == 0 {
			return nil
		}
	}
	done, err := s.Remove(units)
	var bytes, also int64
	for _, x := range done {
		bytes += x.Bytes
		also += entries[x.Path]
	}
	if also > 0 {
		fmt.Fprintf(e.Out, "removed %d units, %s, and %s of Core AI's compiled programs\n", len(done), cache.FormatSize(bytes), cache.FormatSize(also))
	} else {
		fmt.Fprintf(e.Out, "removed %d units, %s\n", len(done), cache.FormatSize(bytes))
	}
	return err
}

// ClearCmd is hwnpu cache clear.
type ClearCmd struct {
	Patterns []string `arg:"" optional:"" name:"pattern" help:"Units to remove: a model's name (its files, programs and exports), OWNER/NAME, a glob."`
	Kind     []string `short:"k" help:"Only units of these kinds (${kinds})." placeholder:"KIND,..."`
	All      bool     `help:"Every unit (of --kind when given)."`
	Yes      bool     `short:"y" help:"Do not ask."`
	DryRun   bool     `short:"n" help:"Show what would be removed, and remove nothing."`
}

func (c ClearCmd) Run(ctx context.Context) error {
	return run(ctx, Request{Command: "clear", Clear: c})
}

func (c ClearCmd) exec(e *Env) error {
	if err := badKinds(c.Kind); err != nil {
		return err
	}
	if len(c.Patterns) == 0 && len(c.Kind) == 0 && !c.All {
		return errors.New("say what to remove: a pattern, --kind or --all")
	}
	if len(c.Patterns) > 0 && c.All {
		return errors.New("--all takes no pattern")
	}
	s, err := e.open()
	if err != nil {
		return err
	}
	u, err := s.Usage()
	if err != nil {
		return err
	}
	now := time.Now()
	var units []modelstore.Unit
	for _, x := range e.notHeld(pick(u.ByUse(), c.Patterns, c.Kind)) {
		if !cache.Removable(x, now, 0) {
			fmt.Fprintf(e.Err, "%s: kept (a writeback to a source is queued for it)\n", x.Name)
			continue
		}
		if now.Sub(x.Used) < time.Minute {
			fmt.Fprintf(e.Err, "%s: used %s: a command running now may stop\n", x.Name, used(x.Used))
		}
		units = append(units, x)
	}
	return remove(e, s, units, c.DryRun, c.Yes)
}

// parseAge reads an age: a Go duration (12h, 90m) or days (30d) or weeks (2w).
func parseAge(s string) (time.Duration, error) {
	for suffix, unit := range map[string]time.Duration{"d": 24 * time.Hour, "w": 7 * 24 * time.Hour} {
		if v, ok := strings.CutSuffix(s, suffix); ok {
			f, err := strconv.ParseFloat(v, 64)
			if err != nil || f < 0 {
				break
			}
			return time.Duration(f * float64(unit)), nil
		}
	}
	d, err := time.ParseDuration(s)
	if err != nil || d < 0 {
		return 0, fmt.Errorf("%q is not an age (30d, 12h, 2w)", s)
	}
	return d, nil
}

// CleanCmd is hwnpu cache clean.
type CleanCmd struct {
	Yes    bool `short:"y" help:"Do not ask."`
	DryRun bool `short:"n" help:"Show what would be removed, and remove nothing."`
}

func (c CleanCmd) Run(ctx context.Context) error {
	return run(ctx, Request{Command: "clean", Clean: c})
}

func (c CleanCmd) exec(e *Env) error {
	s, err := e.open()
	if err != nil {
		return err
	}
	ctx := e.Ctx
	plans, err := s.PlanClean(ctx)
	if err != nil {
		return err
	}
	if len(plans) == 0 {
		fmt.Fprintln(e.Out, "no source is written back to (HWNPU_MODEL_SOURCE ?writeback=true)")
		return nil
	}
	var folders, files, uploads int
	var bytes int64
	w := tabwriter.NewWriter(e.Out, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "source\tsize\tfiles\twhy\tfolder")
	for _, p := range plans {
		for _, f := range p.Folders {
			why := "left over"
			if f.Whole {
				why = "not whole"
			}
			fmt.Fprintf(w, "%s\t%s\t%d\t%s\t%s\n", p.Source, cache.FormatSize(f.Bytes), len(f.Files), why, f.Path)
			folders++
			files += len(f.Files)
		}
		for _, u := range p.Uploads {
			fmt.Fprintf(w, "%s\t?\t1\tupload cut off %s\t%s\n", p.Source, u.Initiated.Local().Format("2006-01-02 15:04"), u.Key)
			uploads++
		}
		bytes += p.Bytes()
	}
	if folders == 0 && uploads == 0 {
		fmt.Fprintln(e.Out, "nothing to remove: what the sources hold is whole")
		return nil
	}
	w.Flush()
	fmt.Fprintf(e.Out, "%d files in %d folders, %s", files, folders, cache.FormatSize(bytes))
	if uploads > 0 {
		fmt.Fprintf(e.Out, "; %d S3 uploads in parts to abort (their parts, of a size S3 does not list, go too)", uploads)
	}
	fmt.Fprintln(e.Out)
	if c.DryRun {
		fmt.Fprintln(e.Out, "(--dry-run: nothing removed)")
		return nil
	}
	if !c.Yes {
		if e.Ask == nil {
			return errors.New("--yes removes them (standard input is not a terminal to ask on)")
		}
		if !e.Ask("remove them? [y/N] ") {
			fmt.Fprintln(e.Out, "nothing removed")
			return nil
		}
	}
	n, err := s.Clean(ctx, plans)
	if all := files + uploads; n < all && err == nil {
		fmt.Fprintf(e.Out, "removed %d of the %d files and uploads (the rest are in folders a writeback has started in since)\n", n, all)
	} else if err == nil {
		fmt.Fprintf(e.Out, "removed %d files, %s", n-uploads, cache.FormatSize(bytes))
		if uploads > 0 {
			fmt.Fprintf(e.Out, ", and aborted %d uploads", uploads)
		}
		fmt.Fprintln(e.Out)
	}
	return err
}
