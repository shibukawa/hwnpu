package main

import (
	"bufio"
	"cmp"
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"path"
	"slices"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/modelstore"
)

// CacheCmd is hwnpu cache.
type CacheCmd struct {
	Info  InfoCmd  `cmd:"" default:"1" help:"Where the cache is, what it holds by kind, the disk's free space, the limits and the layers models are read from (the command when none is named)."`
	Ls    LsCmd    `cmd:"" help:"List the units of the cache (or those PATTERN names): kind, size, when last used; the least recently used first."`
	Prune PruneCmd `cmd:"" help:"Remove the units least recently used until the cache holds --max-size, or those not used for --older-than. Units used in the last hour are kept."`
	Clear ClearCmd `cmd:"" help:"Remove the units PATTERN names (a model's name, OWNER/NAME, a glob: 'Qwen*'), of --kind, or --all."`
	Clean CleanCmd `cmd:"" help:"Remove from the sources written back to (HWNPU_MODEL_SOURCE ?writeback=true) what a writeback cut off left: an export's folder with no entry.json, a repository's files its hwnpu-lock.json does not name, the temporary files of uploads. Folders being written now are left."`
}

func open() (*modelstore.Store, error) {
	return modelstore.Open(modelstore.Options{Log: os.Stderr})
}

// view is a Store as status reads it.
type view struct{ *modelstore.Store }

// LayerNames are the names of the layers, in order.
func (v view) LayerNames() []string {
	var n []string
	for _, l := range v.Layers() {
		s := l.Name
		switch {
		case l.Bundled:
			s += " (bundled, read only)"
		case l.Writeback:
			s += " (written back to)"
		}
		n = append(n, s)
	}
	return n
}

var kindHelp = strings.Join(cache.Kinds, ", ")

// InfoCmd is hwnpu cache info.
type InfoCmd struct{}

func (InfoCmd) Run() error {
	s, err := open()
	if err != nil {
		return err
	}
	u, err := s.Usage()
	if err != nil {
		return err
	}
	fmt.Printf("cache:  %s", u.Root)
	if os.Getenv(cache.RootEnv) != "" {
		fmt.Printf(" (%s)", cache.RootEnv)
	}
	fmt.Println()
	free, total, derr := s.DiskSpace()
	if derr == nil {
		fmt.Printf("disk:   %s free of %s\n", cache.FormatSize(free), cache.FormatSize(total))
	}
	l, err := s.Limits()
	if err != nil {
		fmt.Printf("limits: %v\n", err)
	} else {
		fmt.Printf("limits: %s (%s, %s)\n", l, cache.MaxSizeEnv, cache.MinFreeEnv)
		if derr == nil && free < l.MinFreeOf(total) {
			fmt.Printf("        %s: downloads stop until hwnpu cache prune or clear frees room\n", "the disk has less free")
		}
	}
	fmt.Printf("holds:  %s\n", cache.FormatSize(u.Bytes))
	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', tabwriter.AlignRight)
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
		fmt.Printf("        (other: %s; never removed by prune)\n", strings.Join(parts, ", "))
	}
	fmt.Println("layers (models are read from the first that has them):")
	for i, ly := range s.Layers() {
		what := ""
		switch {
		case ly.Bundled:
			what = " (installed with the application, read only)"
		case ly.Writeback:
			what = " (written back to)"
		}
		fmt.Printf("  %d  %s%s\n", i, ly.Name, what)
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
	for _, p := range patterns {
		p = strings.ToLower(p)
		for _, s := range []string{name, rest, base, export} {
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
			return fmt.Errorf("no kind %q: %s", k, kindHelp)
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

func table(units []modelstore.Unit) {
	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "kind\tsize\tlast used\tunit")
	var total int64
	for _, x := range units {
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\n", x.Kind, cache.FormatSize(x.Bytes), used(x.Used), x.Name)
		total += x.Bytes
	}
	w.Flush()
	fmt.Printf("%d units, %s\n", len(units), cache.FormatSize(total))
}

// LsCmd is hwnpu cache ls.
type LsCmd struct {
	Patterns []string `arg:"" optional:"" name:"pattern" help:"Units to list: a model's name, OWNER/NAME, a glob (all when left out)."`
	Kind     []string `short:"k" help:"Only units of these kinds (${kinds})." placeholder:"KIND,..."`
	Size     bool     `short:"s" help:"The largest first, instead of the least recently used."`
}

func (l LsCmd) Run() error {
	if err := badKinds(l.Kind); err != nil {
		return err
	}
	s, err := open()
	if err != nil {
		return err
	}
	u, err := s.Usage()
	if err != nil {
		return err
	}
	units := pick(u.ByUse(), l.Patterns, l.Kind)
	if l.Size {
		slices.SortStableFunc(units, func(a, b modelstore.Unit) int { return cmp.Compare(b.Bytes, a.Bytes) })
	}
	table(units)
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

func (p PruneCmd) Run() error {
	if err := badKinds(p.Kind); err != nil {
		return err
	}
	s, err := open()
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
	return remove(s, units, p.DryRun, true)
}

// remove removes units, or shows them with dryRun; ask asks first on a terminal unless yes.
func remove(s *modelstore.Store, units []modelstore.Unit, dryRun, yes bool) error {
	if len(units) == 0 {
		fmt.Println("nothing to remove")
		return nil
	}
	table(units)
	if dryRun {
		fmt.Println("(--dry-run: nothing removed)")
		return nil
	}
	if !yes {
		if !console.Terminal(os.Stdin) {
			return errors.New("--yes removes them (standard input is not a terminal to ask on)")
		}
		fmt.Print("remove them? [y/N] ")
		line, _ := bufio.NewReader(os.Stdin).ReadString('\n')
		if a := strings.ToLower(strings.TrimSpace(line)); a != "y" && a != "yes" {
			fmt.Println("nothing removed")
			return nil
		}
	}
	done, err := s.Remove(units)
	var bytes int64
	for _, x := range done {
		bytes += x.Bytes
	}
	fmt.Printf("removed %d units, %s\n", len(done), cache.FormatSize(bytes))
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

func (c ClearCmd) Run() error {
	if err := badKinds(c.Kind); err != nil {
		return err
	}
	if len(c.Patterns) == 0 && len(c.Kind) == 0 && !c.All {
		return errors.New("say what to remove: a pattern, --kind or --all")
	}
	if len(c.Patterns) > 0 && c.All {
		return errors.New("--all takes no pattern")
	}
	s, err := open()
	if err != nil {
		return err
	}
	u, err := s.Usage()
	if err != nil {
		return err
	}
	now := time.Now()
	var units []modelstore.Unit
	for _, x := range pick(u.ByUse(), c.Patterns, c.Kind) {
		if !cache.Removable(x, now, 0) {
			fmt.Fprintf(os.Stderr, "%s: kept (a writeback to a source is queued for it)\n", x.Name)
			continue
		}
		if now.Sub(x.Used) < time.Minute {
			fmt.Fprintf(os.Stderr, "%s: used %s: a command running now may stop\n", x.Name, used(x.Used))
		}
		units = append(units, x)
	}
	return remove(s, units, c.DryRun, c.Yes)
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

func (c CleanCmd) Run() error {
	s, err := open()
	if err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	plans, err := s.PlanClean(ctx)
	if err != nil {
		return err
	}
	if len(plans) == 0 {
		fmt.Println("no source is written back to (HWNPU_MODEL_SOURCE ?writeback=true)")
		return nil
	}
	var folders, files int
	var bytes int64
	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
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
		bytes += p.Bytes()
	}
	if folders == 0 {
		fmt.Println("nothing to remove: what the sources hold is whole")
		return nil
	}
	w.Flush()
	fmt.Printf("%d files in %d folders, %s\n", files, folders, cache.FormatSize(bytes))
	if c.DryRun {
		fmt.Println("(--dry-run: nothing removed)")
		return nil
	}
	if !c.Yes {
		if !console.Terminal(os.Stdin) {
			return errors.New("--yes removes them (standard input is not a terminal to ask on)")
		}
		fmt.Print("remove them? [y/N] ")
		line, _ := bufio.NewReader(os.Stdin).ReadString('\n')
		if a := strings.ToLower(strings.TrimSpace(line)); a != "y" && a != "yes" {
			fmt.Println("nothing removed")
			return nil
		}
	}
	n, err := s.Clean(ctx, plans)
	if n < files {
		fmt.Printf("removed %d of the %d files (the rest are in folders a writeback has started in since)\n", n, files)
	} else {
		fmt.Printf("removed %d files, %s\n", n, cache.FormatSize(bytes))
	}
	return err
}
