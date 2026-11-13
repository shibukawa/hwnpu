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
	"github.com/shibukawa/hwnpu/internal/pose"
	shared "github.com/shibukawa/hwnpu/internal/store"
	"github.com/shibukawa/hwnpu/internal/vision/detect"
	"github.com/shibukawa/hwnpu/internal/vision/ironcnn"
	"github.com/shibukawa/hwnpu/internal/vision/segment"
)

// Cmd is hwnpu-dev cache.
type Cmd struct {
	Models string `help:"${models_usage}" placeholder:"FILE,..."`

	Layers LayersCmd `cmd:"" help:"Show the cache's root and the shared layers in effect: what each holds, whether it is written or filled, and what set it."`
	Ls     LsCmd     `cmd:"" help:"List the entries the shared layers hold (or one layer: its number of layers)."`
	Push   PushCmd   `cmd:"" help:"Put exports for AMD's NPU that are here in the layers that are written (with entry.json, made for them when missing)."`
	Verify VerifyCmd `cmd:"" help:"Check directories against their entry.json: the sizes, and the SHA-256 with --sums."`
	Flush  FlushCmd  `cmd:"" help:"Do the work the background left in the queue (what goes in the layers) and wait for it."`
	GC     GCCmd     `cmd:"" name:"gc" help:"Remove entries from a shared layer (by model, kind, key or age) and the blobs only they named; with --orphans the blobs nothing names too. Shows what would go unless --apply."`
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
	Model []string `arg:"" help:"Models of the definitions (their exports for AMD's NPU: made by amd_export, or on the first run of hwnpu-image detect, pose, segment) to put in the layers."`
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
		if tool, version, ok := firstRunExport(m); ok {
			if err := pushFirstRun(ctx, writes, m, tool, version); err != nil {
				return err
			}
			continue
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

// firstRunExport is the script and the version of the export hwnpu-image makes of m on its
// first run on AMD's NPU, if it makes one.
func firstRunExport(m chat.Model) (tool string, version int, ok bool) {
	switch m.Kind {
	case chat.KindDetect:
		return "iron_onnxcnn.py", detect.ExportVersion, true
	case chat.KindPose:
		return "iron_onnxcnn.py", pose.ExportVersion, true
	case chat.KindSegment:
		return "iron_efficientsam.py", segment.ExportVersion, true
	}
	return "", 0, false
}

// pushFirstRun puts the export of a first run (ironcnn.Dir) in the layers writes.
func pushFirstRun(ctx context.Context, writes []shared.Layer, m chat.Model, tool string, version int) error {
	dir := ironcnn.Dir(m.Name)
	if _, err := os.Stat(filepath.Join(dir, "model.json")); err != nil {
		return fmt.Errorf("%s: no export %s here (made by its first run on AMD's NPU)", m.Name, filepath.Base(dir))
	}
	recipe := chat.PinnedRecipe(m, tool, version)
	if recipe == nil {
		return fmt.Errorf("%s: the definitions pin none of its files (sha256)", m.Name)
	}
	kind := chat.ExportKind(tool)
	recipe["kind"] = kind
	key := shared.Fingerprint(recipe)
	e, err := shared.ReadEntry(dir)
	if err != nil || e.Key != key {
		fmt.Fprintf(os.Stderr, "%s: describing %s (the SHA-256 of its files)\n", m.Name, dir)
		if e, err = shared.Describe(dir, shared.Entry{Key: key, Kind: kind, Scope: shared.ScopePortable, Model: m.Name, Target: cache.AMDTarget,
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
	return nil
}

// GCCmd is hwnpu-dev cache gc.
type GCCmd struct {
	Layer     int      `arg:"" help:"The layer's number in hwnpu-dev cache layers."`
	Model     []string `help:"Remove the entries of these models." placeholder:"NAME"`
	Kind      []string `help:"Remove the entries of these kinds (iron-lm, iron-onnxcnn...)." placeholder:"KIND"`
	Key       []string `help:"Remove the entries whose key starts with these (as hwnpu-dev cache ls shows it)." placeholder:"KEY"`
	OlderThan int      `name:"older-than" help:"Remove the entries made more than this many days ago." placeholder:"DAYS"`
	Orphans   bool     `help:"Remove the blobs no entry and no file of a checkpoint names too (not while anything writes to the layer: what is being put there is named last)."`
	Apply     bool     `help:"Remove them (without it, only show what would go)."`
}

func (g GCCmd) Run(c *Cmd) error {
	if err := load(c); err != nil {
		return err
	}
	ctx, stop := signalContext()
	defer stop()
	ls := shared.Layers()
	if g.Layer < 1 || g.Layer > len(ls) {
		return fmt.Errorf("no layer %d: hwnpu-dev cache layers shows %d", g.Layer, len(ls))
	}
	if len(g.Model) == 0 && len(g.Kind) == 0 && len(g.Key) == 0 && g.OlderThan == 0 && !g.Orphans {
		return errors.New("say what to remove: --model, --kind, --key, --older-than or --orphans")
	}
	l := ls[g.Layer-1]
	before := time.Now().AddDate(0, 0, -g.OlderThan)
	drop := func(e shared.Entry) bool {
		for _, m := range g.Model {
			if strings.EqualFold(e.Model, m) {
				return true
			}
		}
		for _, k := range g.Kind {
			if e.Kind == k {
				return true
			}
		}
		for _, k := range g.Key {
			if k != "" && strings.HasPrefix(e.Key, k) {
				return true
			}
		}
		return g.OlderThan > 0 && e.Made.Before(before)
	}
	plan, err := shared.PlanGC(ctx, l, drop, g.Orphans)
	if err != nil {
		return err
	}
	for _, e := range plan.Entries {
		fmt.Printf("entry %s %s %s %.2f GB %s\n", e.Kind, e.Model, e.Key[:12], float64(e.Bytes())/1e9, e.Made.Local().Format(time.DateTime))
	}
	verb := "would remove"
	if g.Apply {
		verb = "removing"
	}
	fmt.Printf("%s: %s %d entries and %d blobs (%.2f GB)\n", l, verb, len(plan.Entries), len(plan.Blobs), float64(plan.Bytes)/1e9)
	if !g.Apply {
		if len(plan.Entries)+len(plan.Blobs) > 0 {
			fmt.Println("(--apply removes them)")
		}
		return nil
	}
	return plan.Apply(ctx, l)
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
