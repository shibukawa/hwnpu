package hfcmd

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path"
	"slices"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/goccy/go-yaml"

	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/devopt"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	shared "github.com/shibukawa/hwnpu/internal/store"
)

// PinCmd is hwnpu-dev models pin: it pins in models.yaml (exports) the conversions the sources
// hold, by their folder and digest, so that every hwnpu of that models.yaml takes those bytes
// whatever tools made them (docs/prebuilt.md 5.3).
type PinCmd struct {
	Models []string `arg:"" optional:"" help:"MODEL or MODEL/TARGET (amd-xdna2, the only target pinned yet); all the models the layers hold exports of when none is named."`
	Layer  int      `default:"-1" help:"Take the entries of this layer only (hwnpu-dev models hf plan's numbers; 0, this machine's cache, only when named)."`
	Write  string   `help:"The definition file to write (default: models.yaml of hwnpu's source tree the current folder is in)."`
	DryRun bool     `name:"dry-run" help:"Show the pins, and write nothing."`
}

// pinRow is an export to pin: the entry chosen and the pin it gives.
type pinRow struct {
	model  chat.Model
	target string
	export string
	pin    chat.Pin
	entry  shared.Entry
	layer  int
	note   string
}

func (c PinCmd) Run(d *devopt.Defs) error {
	if err := load(d); err != nil {
		return err
	}
	ctx, stop := signalContext()
	defer stop()
	ls := layers()
	var models []chat.Model
	if len(c.Models) == 0 {
		models = chat.Catalog
	} else {
		for _, s := range c.Models {
			name, target, _ := strings.Cut(s, "/")
			if target != "" && target != cache.AMDTarget {
				return fmt.Errorf("%s: only %s is pinned yet (the Core AI exports: their names in the sources are not all the current ones)", s, cache.AMDTarget)
			}
			m, ok := chat.Lookup(name)
			if !ok {
				return fmt.Errorf("%s: no such model", name)
			}
			models = append(models, m)
		}
	}
	var rows []pinRow
	for _, m := range models {
		rs, err := pinRows(ctx, ls, c.Layer, m, cache.AMDTarget)
		if err != nil {
			return err
		}
		rows = append(rows, rs...)
	}
	if len(rows) == 0 {
		return errors.New("the layers hold no export of those models to pin")
	}
	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "model\ttarget\texport\tfolder\tlayer\tmade\tdigest\tnote")
	for _, r := range rows {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n", r.model.Name, r.target, r.export, r.pin.At, r.layer,
			r.entry.Made.Format(time.DateOnly), strings.TrimPrefix(r.pin.Digest, shared.DigestPrefix)[:12], r.note)
	}
	tw.Flush()
	if c.DryRun {
		return nil
	}
	file := c.Write
	if file == "" {
		file = sourceTreeModels()
	}
	if file == "" {
		return errors.New("no models.yaml of hwnpu's source tree here: name the file with --write")
	}
	src, err := os.ReadFile(file)
	if err != nil {
		return err
	}
	for _, r := range rows {
		if src, err = nestedYAML(src, r.model.Name, "exports", r.target, r.export, r.pin.String()); err != nil {
			return fmt.Errorf("%s: %w", file, err)
		}
	}
	// read back: every pin as written
	var back struct {
		Models map[string]struct {
			Exports map[string]map[string]string `yaml:"exports"`
		} `yaml:"models"`
	}
	if err := yaml.Unmarshal(src, &back); err != nil {
		return fmt.Errorf("%s would not read: %w", file, err)
	}
	for _, r := range rows {
		if got := back.Models[r.model.Name].Exports[r.target][r.export]; got != r.pin.String() {
			return fmt.Errorf("%s would read models.%s.exports.%s.%s as %q, not %q: pin it by hand", file, r.model.Name, r.target, r.export, got, r.pin)
		}
	}
	if err := os.WriteFile(file, src, 0o644); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "pinned %d exports in %s: see git diff %s, run go test ./internal/chat, and put it in a PR\n", len(rows), file, file)
	return nil
}

// pinRows are the exports of model m for target the layers hold, each the entry to pin: of each
// export, the newest entry in a form this hwnpu reads, from the sources (the first layer that
// holds it; this machine's cache only when only is 0).
func pinRows(ctx context.Context, ls []shared.Source, only int, m chat.Model, target string) ([]pinRow, error) {
	held := listed(ctx, ls, only, m)[target]
	byExport := map[string][]string{}
	for dir, i := range held {
		if i == 0 && only != 0 {
			continue // this machine's cache: what no other machine reads
		}
		if e := exportOf(dir); e != "" {
			byExport[e] = append(byExport[e], dir)
		}
	}
	var out []pinRow
	for _, export := range slices.Sorted(mapKeys(byExport)) {
		var best pinRow
		ok := false
		for _, dir := range byExport[export] {
			_, e, err := readEntry(ctx, ls[held[dir]], dir)
			if err != nil || !chat.FormatTaken(e) || e.Model != m.Name {
				continue
			}
			if !ok || e.Made.After(best.entry.Made) {
				best = pinRow{model: m, target: target, export: export, entry: e, layer: held[dir],
					pin: chat.Pin{At: path.Base(dir), Digest: e.Digest()}}
				ok = true
			}
		}
		if !ok {
			continue
		}
		if old, had := chat.PinOf(m, target, export); had {
			if old == best.pin {
				best.note = "pinned already"
			} else {
				best.note = "was " + old.At
			}
		}
		if n := len(byExport[export]); n > 1 {
			best.note = strings.TrimSpace(best.note + fmt.Sprintf(" (newest of %d)", n))
		}
		out = append(out, best)
	}
	return out, nil
}
