package prebuild

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/backend/coreai"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/recipe"
	shared "github.com/shibukawa/hwnpu/internal/store"
)

// Options say how Run compiles. What they leave out the model's recipe says (model-recipes.yaml,
// internal/recipe).
type Options struct {
	// Model is the model of the definitions whose recipe the exports are built with; "" for the
	// one each export is of (chat.ExportModel).
	Model string
	// Archs are the architectures (when none: the recipe's for a model its recipe prebuilds,
	// apple.prebuilt, and no other model's exports are compiled).
	Archs []string
	// Jobs are the compiles at once (the recipe's when 0, else coreai.DefaultJobs).
	Jobs int
	// Force, DryRun and Packed are coreai.BuildOptions'.
	Force, DryRun, Packed bool
	// Put puts each export's programs for an architecture in the sources written back to once
	// every one of them compiled or is left to the device (packed).
	Put bool
	// AnyTools builds with a coreai-build of another version than the recipes'.
	AnyTools bool
	Log      io.Writer
}

// Run compiles the Core AI programs at paths (exports, or programs of them) ahead of time as
// their models' recipes say: for the recipe's architectures, its jobs at once, but the programs
// a rule leaves to the device. A program that fails, which no rule names, fails the run: what
// its export has for that architecture is not put (a device would compile the program, and
// another build put it), and the rule that would leave it to the device is written.
func Run(ctx context.Context, paths []string, o Options) error {
	recipes, err := recipe.Builtin()
	if err != nil {
		return err
	}
	var version string
	if !o.DryRun {
		v, err := coreai.BuildToolVersion()
		if err != nil {
			return err
		}
		version = v
		if want := recipes.Tools.CoreAIBuild; v != want {
			if !o.AnyTools {
				return fmt.Errorf("coreai-build %s here, the recipes were checked with %s (model-recipes.yaml tools): install that Metal Toolchain, or --any-tools to build with this one (to check it, then write its version there)", v, want)
			}
			fmt.Fprintf(o.Log, "coreai-build %s here, not the recipes' %s (--any-tools)\n", v, want)
		}
	}
	// the paths by their model, in their order
	var models []string
	byModel := map[string][]string{}
	for _, p := range paths {
		name := o.Model
		if name == "" {
			export := p
			if e, ok := coreai.ExportOf(filepath.Join(p, "x")); ok {
				export = e
			}
			m, ok := chat.ExportModel(export)
			if !ok {
				return fmt.Errorf("%s: of no model of the definitions (its entry.json names none, nor its folder): name it (-model)", p)
			}
			name = m.Name
		}
		if _, ok := byModel[name]; !ok {
			models = append(models, name)
		}
		byModel[name] = append(byModel[name], p)
	}
	var failed []string
	for _, name := range models {
		a := recipes.AppleOf(name)
		archs := o.Archs
		if len(archs) == 0 {
			// no architecture named: the recipe's, for a model the recipe prebuilds
			if !a.Prebuilt {
				fmt.Fprintf(o.Log, "%s: not prebuilt (model-recipes.yaml gives it no prebuilt: a Mac loads its programs slower from prebuilt ones than it compiles them); name architectures to compile it all the same\n", name)
				continue
			}
			archs = a.Archs
		}
		jobs := o.Jobs
		if jobs == 0 {
			jobs = a.Jobs
		}
		fmt.Fprintf(o.Log, "%s: for %s, %d at once (model-recipes.yaml)\n", name, strings.Join(archs, " "), max(jobs, 1))
		bo := coreai.BuildOptions{Archs: archs, Jobs: jobs, Force: o.Force, DryRun: o.DryRun, Log: o.Log, Packed: o.Packed || o.Put,
			Tool: version, Model: name,
			Left: func(asset, arch string) string {
				export, _ := coreai.ExportOf(asset)
				if r, ok := a.Left(filepath.Base(export), filepath.Base(asset), arch); ok {
					return fmt.Sprintf("%s, %s", r.Why, r.Checked)
				}
				return ""
			}}
		t0 := time.Now()
		results, berr := coreai.Build(ctx, byModel[name], bo)
		if o.DryRun {
			if berr != nil {
				return berr
			}
			continue
		}
		built, skipped, reused, left := 0, 0, 0, 0
		var code, weights int64
		bad := map[[2]string]bool{}
		for _, r := range results {
			export, _ := coreai.ExportOf(r.Asset)
			switch {
			case r.Err != nil:
				bad[[2]string{export, r.Arch}] = true
				failed = append(failed, fmt.Sprintf("%s %s", filepath.Base(r.Asset), r.Arch))
			case r.Left != "":
				left++
			case r.Skipped:
				skipped++
			case r.Reused != "":
				reused++
			default:
				built++
				code += r.Code
				weights += r.Weights
			}
		}
		fmt.Fprintf(o.Log, "%s: %d compiled in %s (%.0f MB of compiled regions beside %.1f GB of weights), %d prebuilt already, %d taken from other exports, %d left to the device, %d failed\n",
			name, built, time.Since(t0).Round(time.Second), float64(code)/1e6, float64(weights)/1e9, skipped, reused, left, len(bad))
		today := time.Now().Format("2006-01-02")
		for _, r := range results {
			if r.Err != nil && !errors.Is(r.Err, context.Canceled) {
				fmt.Fprintf(o.Log, "%s %s failed: %v\n  fix the program, or leave it to the device in model-recipes.yaml:\n%s", filepath.Base(r.Asset), r.Arch, r.Err,
					indent(recipe.Suggest(name, filepath.Base(r.Asset), r.Arch, r.Err.Error(), today), "    "))
			}
		}
		if o.Put {
			// an export's programs for an architecture, when none failed
			var puts [][2]string
			for _, r := range results {
				export, ok := coreai.ExportOf(r.Asset)
				k := [2]string{export, r.Arch}
				if ok && !bad[k] && r.Left == "" && !slices.Contains(puts, k) {
					puts = append(puts, k)
				}
			}
			for k := range bad {
				fmt.Fprintf(o.Log, "%s for %s: not put (a program failed)\n", filepath.Base(k[0]), k[1])
			}
			// the model is the recipe's: an export converted here with no source to put it in
			// has no entry.json to name it
			pctx := ctx
			if m, ok := chat.Lookup(name); ok {
				pctx = chat.WithModel(ctx, m)
			}
			for _, k := range puts {
				if err := chat.PutPrebuilt(pctx, k[0], k[1]); err != nil {
					return err
				}
			}
		}
		if berr != nil && ctx.Err() != nil {
			return berr
		}
	}
	if len(failed) > 0 {
		return fmt.Errorf("%d programs failed (%s): the rules above would leave them to the device", len(failed), strings.Join(failed, ", "))
	}
	return nil
}

func indent(s, by string) string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	for i := range lines {
		lines[i] = by + lines[i]
	}
	return strings.Join(lines, "\n") + "\n"
}

// Mark writes what a device made of the prebuilt programs of the exports for archs (status: ok,
// or the failure's name) in the records of their builds (coreai.MarkDevice), the programs those
// whose names match one of programs (all when none). A build then takes a failed one for no
// later export, and says the rule that leaves it to the device.
func Mark(exports, archs, programs []string, status, function string, log io.Writer) error {
	d := coreai.DeviceCheck{Status: status, Checked: time.Now().Format("2006-01-02"), Function: function, By: "hwnpu-dev prebuild -mark"}
	for _, e := range exports {
		e = filepath.Clean(e)
		if _, ok := coreai.ExportOf(filepath.Join(e, "x")); !ok {
			return fmt.Errorf("%s is not an export (NAME.coreai)", e)
		}
		for _, arch := range archs {
			marked, err := coreai.MarkDevice(e, arch, programs, d)
			if err != nil {
				return err
			}
			fmt.Fprintf(log, "%s %s: %s %s\n", filepath.Base(e), arch, status, strings.Join(marked, " "))
		}
	}
	return nil
}

// Record writes the records of the builds of prebuilt programs made before records
// (coreai.Backfill): of the exports' given, else of every export in the cache and every entry of
// prebuilt programs in the sources written back to, each fetched, given its record and put again
// (its revision, store.Entry.Revision). One with a record is left. A build then takes their
// programs for later exports of the same source as those built since. dry lists what would be
// written.
func Record(ctx context.Context, exports []string, dry bool, log io.Writer) error {
	recipes, err := recipe.Builtin()
	if err != nil {
		return err
	}
	today := time.Now().Format("2006-01-02")
	jobsOf := func(model string) int { return recipes.AppleOf(model).Jobs }
	sources := len(exports) == 0
	if len(exports) == 0 {
		ps, _ := filepath.Glob(filepath.Join(cache.Root(), "models", "*", "*"+coreai.PackedSuffix))
		ns, _ := filepath.Glob(filepath.Join(cache.Root(), "models", "*", "*.coreai", "*"+coreai.PackedSuffix))
		for _, p := range append(ps, ns...) {
			exports = append(exports, strings.TrimSuffix(p, coreai.PackedSuffix))
		}
	}
	for _, e := range exports {
		e = filepath.Clean(e)
		model := ""
		if m, ok := chat.ExportModel(e); ok {
			model = m.Name
		}
		for _, arch := range coreai.Archs {
			root := coreai.PackedRoot(e, arch)
			if fi, err := os.Stat(root); err != nil || !fi.IsDir() {
				continue
			}
			if dry {
				if rec, _ := coreai.ReadRecord(root); rec == nil {
					fmt.Fprintf(log, "%s %s: would be given a record\n", filepath.Base(e), arch)
				}
				continue
			}
			keys, err := coreai.Backfill(root, arch, filepath.Base(e), model, jobsOf(model), today)
			if err != nil {
				return err
			}
			if len(keys) > 0 {
				fmt.Fprintf(log, "%s %s: recorded %d programs\n", filepath.Base(e), arch, len(keys))
			}
		}
	}
	to := shared.Writebacks()
	if !sources || len(to) == 0 {
		if sources {
			fmt.Fprintln(log, "no source is written back to (HWNPU_MODEL_SOURCE...?writeback=true): the cache's alone")
		}
		return nil
	}
	for _, s := range to {
		rels, err := shared.ListEntries(ctx, s)
		if err != nil {
			return fmt.Errorf("%s: %w", s, err)
		}
		n := 0
		for _, rel := range rels {
			if !strings.Contains(rel, "/"+cache.CoreAITarget+"-") {
				continue // converted/MODEL/coreai-ARCH/EXPORT/VERSION
			}
			e, err := shared.ReadEntryAt(ctx, s, rel)
			if err != nil || e.Kind != chat.PrebuiltKind || e.Revision != "" {
				continue
			}
			n++
			if dry {
				fmt.Fprintf(log, "%s: %s: would be given a record\n", s, rel)
				continue
			}
			if err := recordEntry(ctx, s, rel, e, jobsOf(e.Model), today, log); err != nil {
				return fmt.Errorf("%s: %s: %w", s, rel, err)
			}
		}
		fmt.Fprintf(log, "%s: %d entries of prebuilt programs without a record\n", s, n)
	}
	return nil
}

// recordEntry fetches the entry of prebuilt programs at rel of s, gives its programs a record
// (coreai.Backfill) and puts it again.
func recordEntry(ctx context.Context, s shared.Source, rel string, e shared.Entry, jobs int, today string, log io.Writer) error {
	arch, export := e.Recipe["arch"], e.Recipe["export"]
	if arch == "" || export == "" {
		return errors.New("its recipe names no arch or export")
	}
	tmp, err := os.MkdirTemp(cache.Root(), ".record-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmp)
	dir := filepath.Join(tmp, "entry")
	if _, err := shared.FetchEntry(ctx, []shared.Source{s}, rel, func(shared.Entry) bool { return true }, dir, nil); err != nil {
		return err
	}
	root := filepath.Join(tmp, arch)
	bundle := filepath.Join(dir, coreai.BundleFile)
	if err := coreai.Unbundle(bundle, root); err != nil {
		return err
	}
	keys, err := coreai.Backfill(root, arch, filepath.Base(export), e.Model, jobs, today)
	if err != nil {
		return err
	}
	if err := coreai.Bundle(root, bundle); err != nil {
		return err
	}
	got, err := shared.ReadEntry(dir)
	if err != nil {
		return err
	}
	got.Revision = coreai.RecordDigest(root)
	if got.Revision == "" {
		return errors.New("no record was written (no prebuilt program in it?)")
	}
	if _, err := shared.Describe(dir, got); err != nil {
		return err
	}
	if err := shared.PutEntry(ctx, s, dir); err != nil {
		return err
	}
	fmt.Fprintf(log, "%s: %s: recorded %d programs, put again\n", s, rel, len(keys))
	return nil
}
