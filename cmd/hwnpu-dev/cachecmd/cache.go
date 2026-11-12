// Package cachecmd is hwnpu-dev cache: what is kept here (the cache's root) and in the shared
// layers (docs/cache-sharing.md 7): the layers in effect, what they hold, putting an export in
// them, checking an entry, and finishing the work left in the background.
package cachecmd

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	shared "github.com/shibukawa/hwnpu/internal/store"
)

// Cmd is hwnpu-dev cache.
type Cmd struct {
	Models string `help:"${models_usage}" placeholder:"FILE,..."`

	Layers LayersCmd `cmd:"" help:"Show the cache's root and the shared layers in effect: what each holds, whether it is written or filled, and what set it."`
	Ls     LsCmd     `cmd:"" help:"List the entries the shared layers hold (or one layer: its number of layers)."`
	Push   PushCmd   `cmd:"" help:"Put exports for AMD's NPU that are here in the layers that are written (with entry.json, made for them when missing)."`
	Verify VerifyCmd `cmd:"" help:"Check directories against their entry.json: the sizes, and the SHA-256 with --sums."`
	Flush  FlushCmd  `cmd:"" help:"Do the work the background left in the queue (what goes in the layers) and wait for it."`
}

// Vars are kong's variables of the command.
var Vars = map[string]string{"models_usage": chat.ModelsUsage}

func load(c *Cmd) error { return chat.LoadUserModels(c.Models, os.Stderr) }

func signalContext() (context.Context, context.CancelFunc) {
	return signal.NotifyContext(context.Background(), os.Interrupt)
}

// LayersCmd is hwnpu-dev cache layers.
type LayersCmd struct{}

func (LayersCmd) Run(c *Cmd) error {
	if err := load(c); err != nil {
		return err
	}
	fmt.Printf("root: %s", cache.Root())
	if os.Getenv(cache.RootEnv) != "" {
		fmt.Printf(" (%s)", cache.RootEnv)
	}
	fmt.Printf("\nAMD exports: %s (and read from %s)\nIRON's kernels: %s\n", chat.AMDRoot(), cache.LegacyExports(), chat.IRONCache())
	ls := shared.Layers()
	if len(ls) == 0 {
		fmt.Println("shared layers: none (cache.layers of a definition file, or " + chat.LayersEnv + ")")
		return nil
	}
	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "#\tlayer\tholds\twrite\tfill\tfill_gated\tfrom")
	for i, l := range ls {
		holds := strings.Join(l.Scopes, ",")
		if holds == "" {
			holds = "all"
		}
		fmt.Fprintf(w, "%d\t%s\t%s\t%v\t%v\t%v\t%s\n", i+1, l, holds, l.Write, l.Fill, l.FillGated, l.From)
	}
	return w.Flush()
}

// LsCmd is hwnpu-dev cache ls.
type LsCmd struct {
	Layer int `arg:"" optional:"" help:"The layer's number in hwnpu-dev cache layers (all when left out)."`
}

func (l LsCmd) Run(c *Cmd) error {
	if err := load(c); err != nil {
		return err
	}
	ctx, stop := signalContext()
	defer stop()
	ls := shared.Layers()
	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "layer\tkind\tmodel\ttarget\tGB\tmade\tkey")
	for i, layer := range ls {
		if l.Layer != 0 && l.Layer != i+1 {
			continue
		}
		entries, err := shared.ListEntries(ctx, layer)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", layer, err)
			continue
		}
		for _, k := range entries {
			e, err := shared.ReadLayerEntry(ctx, layer, k[0], k[1])
			if err != nil {
				fmt.Fprintf(w, "%d\t%s\t?\t?\t?\t?\t%s\n", i+1, k[0], k[1][:12])
				continue
			}
			fmt.Fprintf(w, "%d\t%s\t%s\t%s\t%.2f\t%s\t%s\n", i+1, e.Kind, e.Model, e.Target, float64(e.Bytes())/1e9, e.Made.Local().Format(time.DateTime), e.Key[:12])
		}
	}
	return w.Flush()
}

// PushCmd is hwnpu-dev cache push.
type PushCmd struct {
	Model []string `arg:"" help:"Models of the definitions (their exports for AMD's NPU, made by amd_export) to put in the layers."`
}

func (p PushCmd) Run(c *Cmd) error {
	if err := load(c); err != nil {
		return err
	}
	ctx, stop := signalContext()
	defer stop()
	var writes []shared.Layer
	for _, l := range shared.Layers() {
		if l.Write && l.Holds(shared.ScopePortable) {
			writes = append(writes, l)
		}
	}
	if len(writes) == 0 {
		return errors.New("no layer is written: give one write: true (cache.layers) or ?write=1 (" + chat.LayersEnv + ")")
	}
	for _, name := range p.Model {
		m, ok := chat.Lookup(name)
		if !ok {
			return fmt.Errorf("%s is not a model of the definitions", name)
		}
		dir := chat.AMDExportDir(m)
		if dir == "" {
			return fmt.Errorf("%s has no amd_export", m.Name)
		}
		if _, err := os.Stat(filepath.Join(dir, "manifest.json")); err != nil {
			// one made before Root held the exports: under ~/models
			if d := chat.FindAMDExport(m); filepath.Base(d) == filepath.Base(dir) {
				dir = d
			} else {
				return fmt.Errorf("%s: no export %s here", m.Name, filepath.Base(dir))
			}
		}
		recipe := chat.ExportRecipe(ctx, m, "iron_export_lm.py", m.AMDExport)
		if recipe == nil {
			return fmt.Errorf("%s: the commit of its files is not known (pin it, or reach Hugging Face)", m.Name)
		}
		// an export made by hand (or before entries) gets its entry.json: what it would be made
		// from now. Made with other tools than these, it is not this recipe's: --force is not
		// offered, the export is to be made again with this hwnpu
		recipe["kind"] = "iron-lm"
		key := shared.Fingerprint(recipe)
		e, err := shared.ReadEntry(dir)
		if err != nil || e.Key != key {
			fmt.Fprintf(os.Stderr, "%s: describing %s (the SHA-256 of its files)\n", m.Name, dir)
			if e, err = shared.Describe(dir, shared.Entry{Key: key, Kind: "iron-lm", Scope: shared.ScopePortable, Model: m.Name, Target: cache.AMDTarget,
				Recipe: recipe, MadeBy: map[string]string{"note": "described by hwnpu-dev cache push"}}); err != nil {
				return err
			}
		}
		for _, l := range writes {
			fmt.Fprintf(os.Stderr, "%s: putting %.2f GB in %s\n", m.Name, float64(e.Bytes())/1e9, l)
			if err := shared.PutEntry(ctx, l, dir); err != nil {
				return fmt.Errorf("%s: %s: %w", m.Name, l, err)
			}
		}
	}
	return nil
}

// VerifyCmd is hwnpu-dev cache verify.
type VerifyCmd struct {
	Dir  []string `arg:"" help:"Directories with entry.json." type:"existingdir"`
	Sums bool     `help:"Check the SHA-256 of every file too (reads them all)."`
}

func (v VerifyCmd) Run(*Cmd) error {
	bad := 0
	for _, d := range v.Dir {
		if err := shared.Verify(d, v.Sums); err != nil {
			fmt.Printf("%s: %v\n", d, err)
			bad++
			continue
		}
		fmt.Printf("%s: ok\n", d)
	}
	if bad > 0 {
		return fmt.Errorf("%d of %d differ", bad, len(v.Dir))
	}
	return nil
}

// FlushCmd is hwnpu-dev cache flush.
type FlushCmd struct{}

func (FlushCmd) Run(c *Cmd) error {
	if err := load(c); err != nil {
		return err
	}
	ctx, stop := signalContext()
	defer stop()
	shared.Flush(ctx)
	left, _ := filepath.Glob(filepath.Join(cache.Sub("work", "queue"), "*.json"))
	if len(left) > 0 {
		return fmt.Errorf("%d pieces of work failed and are left for the next run: see %s", len(left), cache.Sub("work", "queue.log"))
	}
	fmt.Println("the queue is empty")
	return nil
}
