package modelscmd

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"os"
	"os/signal"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/devopt"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/sourcescmd"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/backend/coreai"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/prep"
	"github.com/shibukawa/hwnpu/internal/recipe"
	shared "github.com/shibukawa/hwnpu/internal/store"
)

// PackageCmd is hwnpu-dev models package: each model made ready for a target as the apps make it
// on their first run there (airole.Role.Prepare), on this machine whatever its NPU: the programs
// are converted (or taken from the sources), not loaded. For coreai-ARCH the Core AI exports are
// then compiled for that architecture of the Neural Engine with coreai-build (compile, macOS).
// What is made goes in the sources written back to, as the first run's does.
type PackageCmd struct {
	Sel    []string `arg:"" name:"MODEL[/TARGET]" help:"Models of the definitions, each for a target (amd-xdna2; coreai, the generic exports; coreai-ARCH, those compiled for an architecture of the Neural Engine, h14g..., coreai-all for every one) or for the targets its definition gives versions for (targets), else what its recipe makes (model-recipes.yaml: the Core AI exports prebuilt for its architectures). A Core AI export folder or program (DIR[/coreai-ARCH]) is compiled alone, for the architectures of its model's recipe when none is named."`
	Memory []int    `help:"Only the forms of machines of these GB (hwnpu-chat --memory); every form the definitions give when left out." placeholder:"GB"`
	Chip   []string `help:"For coreai, only the exports of Macs of these chips (M1, M2, ...); one of each kind of chip whose exports differ when left out (chat.ExportChips: M1, M2 for the later ones)." placeholder:"CHIP"`
	compileFlags
}

// targets are the targets package makes conversions for; coreai-ARCH is coreai's compiled.
var targets = []string{cache.AMDTarget, cache.CoreAITarget}

// job is a model to make ready for a target; for coreai, archs are the architectures its exports
// are compiled for (none: the generic exports alone).
type job struct {
	model  chat.Model
	target string
	archs  []string
}

// prepared is what a job's preparation of a form gave (airole.Files), as a Mac of chip converts
// ("" for another target's, of no chip).
type prepared struct {
	job   job
	files airole.Files
	chip  string
}

// folder is Core AI export folders and programs to compile for archs (none: the recipe's) as
// model's recipe says ("": each export's model's).
type folder struct {
	paths []string
	archs []string
	model string
}

func (x PackageCmd) Run(d *devopt.Defs) error {
	if err := d.Load(); err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	jobs, folders, err := parseSel(x.Sel)
	if err != nil {
		return err
	}
	if len(folders) > 0 || slices.ContainsFunc(jobs, func(j job) bool { return len(j.archs) > 0 }) {
		if err := canCompile(); err != nil {
			return err
		}
	}
	if len(shared.Writebacks()) == 0 {
		fmt.Fprintf(os.Stderr, "no source is written back to (%s ?writeback=true): what is made stays here\n", shared.SourceEnv)
	}
	done, err := prepare(ctx, jobs, x.Memory, x.Chip, nil)
	if err != nil {
		return err
	}
	for _, p := range done {
		if len(p.job.archs) > 0 {
			if paths := coreAIPaths(p.files.Paths); len(paths) > 0 {
				// the exports made as a chip converts, for the architectures that take them: a
				// chip's own export is not compiled for the others, nor theirs for it (FLUX.2
				// klein's for an M1, whose programs of the others' draw noise)
				archs := slices.DeleteFunc(slices.Clone(p.job.archs), func(arch string) bool { return p.chip != "" && !chat.TakesExportsOf(arch, p.chip) })
				if len(archs) > 0 {
					folders = append(folders, folder{paths, archs, p.job.model.Name})
				} else {
					fmt.Fprintf(os.Stderr, "%s: its exports as on the %s are compiled for none of %s (another chip's)\n", p.job.model.Name, p.chip, strings.Join(p.job.archs, " "))
				}
			}
		}
	}
	for _, j := range jobs {
		if len(j.archs) > 0 && !slices.ContainsFunc(done, func(p prepared) bool { return p.job.model.Name == j.model.Name && len(coreAIPaths(p.files.Paths)) > 0 }) {
			return fmt.Errorf("%s has no Core AI exports to compile", j.model.Name)
		}
	}
	for _, f := range folders {
		if err := compile(ctx, f.paths, f.archs, f.model, x.compileFlags); err != nil {
			return err
		}
	}
	return nil
}

// parseSel reads MODEL[/TARGET]s and Core AI export folders (DIR[/coreai-ARCH]).
func parseSel(sel []string) ([]job, []folder, error) {
	var models []string
	var folders []folder
	for _, s := range sel {
		dir, target := s, ""
		if i := strings.LastIndexByte(s, '/'); i >= 0 && strings.HasPrefix(s[i+1:], cache.CoreAITarget+"-") {
			dir, target = s[:i], s[i+1:]
		}
		if name, _, _ := strings.Cut(s, "/"); !isPath(dir) || isModel(name) && !exists(dir) {
			models = append(models, s)
			continue
		}
		var archs []string
		if target != "" {
			var err error
			if archs, err = archsOf(target); err != nil {
				return nil, nil, fmt.Errorf("%s: %w", s, err)
			}
		}
		if target == "" {
			// a folder alone: for the recipe's architectures, not only this Mac's
			archs = nil
		}
		folders = append(folders, folder{[]string{dir}, archs, ""})
	}
	jobs, err := parseJobs(models)
	return jobs, folders, err
}

// archsOf are the architectures target coreai-ARCH names (coreai-all: every one, coreai.Archs).
func archsOf(target string) ([]string, error) {
	arch, ok := strings.CutPrefix(target, cache.CoreAITarget+"-")
	switch {
	case !ok:
		return nil, fmt.Errorf("no target %q", target)
	case arch == "all":
		return coreai.Archs, nil
	case slices.Contains(coreai.Archs, arch):
		return []string{arch}, nil
	}
	return nil, fmt.Errorf("no architecture %q of the Neural Engine (%s, all)", arch, strings.Join(coreai.Archs, ", "))
}

// parseJobs reads MODEL[/TARGET]s: a model's coreai-ARCHs are one job of coreai.
func parseJobs(sel []string) ([]job, error) {
	var jobs []job
	for _, s := range sel {
		name, target, _ := strings.Cut(s, "/")
		m, ok := chat.Lookup(name)
		if !ok {
			return nil, fmt.Errorf("%s is not a model of the definitions (hwnpu-chat list, hwnpu-audio list, hwnpu-image list)", name)
		}
		var ts []string
		var archs []string
		switch {
		case strings.HasPrefix(target, cache.CoreAITarget+"-"):
			var err error
			if archs, err = archsOf(target); err != nil {
				return nil, fmt.Errorf("%s: %w", s, err)
			}
			ts = []string{cache.CoreAITarget}
		case recipe.GenerationOf(target) != "":
			// an export for AMD's NPU of a generation: one its recipe makes, and hwnpu runs
			if err := amdGeneration(m, recipe.GenerationOf(target)); err != nil {
				return nil, fmt.Errorf("%s: %w", s, err)
			}
			ts = []string{target}
		case target != "":
			if !slices.Contains(targets, target) {
				return nil, fmt.Errorf("%s: no target %q (%s, %s-ARCH)", s, target, strings.Join(targets, ", "), cache.CoreAITarget)
			}
			ts = []string{target}
		default:
			for _, t := range targets {
				// (AMD's from its recipe, below, which gives its version too)
				if _, ok := m.Targets[t]; ok && recipe.GenerationOf(t) == "" {
					ts = append(ts, t)
				}
			}
			recipes, err := recipe.Builtin()
			if err != nil {
				return nil, err
			}
			// and what its recipe makes: the exports for AMD's NPU of its generations (but
			// on a Mac, which has no IRON)
			if _, ok := recipes.AMDOf(m.Recipe); ok {
				gens, skips := recipes.AMDGenerations(m.Recipe)
				for _, sk := range skips {
					fmt.Fprintf(os.Stderr, "%s: not made for %s (model-recipes.yaml skip: %s, %s)\n", m.Name, sk.Generation, sk.Why, sk.Checked)
				}
				for _, gen := range gens {
					t := recipe.AMDTarget(gen)
					switch err := amdGeneration(m, gen); {
					case err != nil:
						fmt.Fprintf(os.Stderr, "%s: %v: skipped\n", m.Name, err)
					case slices.Contains(ts, t):
					case canExportAMD() != nil:
						fmt.Fprintf(os.Stderr, "%s: its recipe's exports for AMD's NPU (%s/%s) are made on Linux or Windows with IRON: skipped here\n", m.Name, m.Name, t)
					default:
						ts = append(ts, t)
					}
				}
			}
			// The Core AI exports are among ts already (the recipe's apple.version is the model's
			// target's version); a model its recipe prebuilds (apple.prebuilt: the image
			// generators) has them compiled for the recipe's architectures too
			if a := recipes.AppleOf(m.Recipe); a.Prebuilt && len(a.Archs) > 0 {
				if err := canCompile(); err != nil {
					fmt.Fprintf(os.Stderr, "%s: its recipe's prebuilt programs (%s/%s-ARCH) are compiled on a Mac: skipped here\n", m.Name, m.Name, cache.CoreAITarget)
				} else {
					if !slices.Contains(ts, cache.CoreAITarget) {
						ts = append(ts, cache.CoreAITarget)
					}
					archs = a.Archs
				}
			}
			if len(ts) == 0 {
				return nil, fmt.Errorf("%s: neither its definition nor its recipe (model-recipes.yaml) gives versions of targets this machine makes: name one (%s/%s)", m.Name, m.Name, strings.Join(targets, ", "+m.Name+"/"))
			}
		}
		for _, t := range ts {
			archs := archs
			if t != cache.CoreAITarget { // the architectures are the Core AI exports'
				archs = nil
			}
			i := slices.IndexFunc(jobs, func(j job) bool { return j.model.Name == m.Name && j.target == t })
			if i < 0 {
				jobs = append(jobs, job{m, t, archs})
				continue
			}
			for _, a := range archs {
				if !slices.Contains(jobs[i].archs, a) {
					jobs[i].archs = append(jobs[i].archs, a)
				}
			}
		}
	}
	return jobs, nil
}

// prepare makes the jobs' models ready, each form of each (forms), and waits for what goes in
// the sources. It returns the files each form's preparation gave (airole.Files), in order. Given
// missing, nothing is converted (llm.WithFetchOnly): what is neither here nor in the sources is
// added to it, and the rest goes on.
func prepare(ctx context.Context, jobs []job, memory []int, chips []string, missing *[]string) ([]prepared, error) {
	if missing != nil {
		ctx = llm.WithFetchOnly(ctx)
	}
	// skip is whether err is of what is not in the sources, then noted, when nothing is converted
	skip := func(what string, err error) bool {
		var nf *llm.NotFetchedError
		if missing == nil || !errors.As(err, &nf) {
			return false
		}
		fmt.Fprintf(os.Stderr, "%s: %v\n", what, err)
		*missing = append(*missing, what+": "+filepath.Base(nf.What))
		return true
	}
	for _, c := range chips {
		if !slices.Contains(chat.Chips, c) {
			return nil, fmt.Errorf("no chip %q (%s)", c, strings.Join(chat.Chips, ", "))
		}
	}
	log := console.Log(os.Stderr)
	var out []prepared
	for _, j := range jobs {
		if j.target == cache.AMDTarget && j.model.Kind == chat.KindChat {
			// IRON's exports of chat models, every form of their recipe (sourcescmd.AMDForms)
			if len(j.model.AMDExport) == 0 {
				return out, fmt.Errorf("%s has no recipe of iron_export_lm.py (model-recipes.yaml)", j.model.Name)
			}
			if missing == nil {
				if err := (sourcescmd.AMDExport{Model: []string{j.model.Name}}).Make(ctx, memory); err != nil {
					return out, err
				}
				continue
			}
			for _, m := range sourcescmd.AMDFormsOn(j.model, memory) {
				if _, err := chat.PrepareAMD(chat.WithModel(ctx, m), m, log); err != nil && !skip(m.Name+" (amd-xdna2)", err) {
					return out, fmt.Errorf("%s: %w", m.Name, err)
				}
			}
			continue
		}
		// what package makes every form with (prep), else the library's role (airole)
		name, makeReady := "", (func(airole.PrepareEnv) (airole.Files, error))(nil)
		if p := prep.For(j.model, j.target); p != nil {
			name, makeReady = p.Name, p.Prepare
		} else if r := airole.For(j.model); r != nil {
			name, makeReady = r.Name, r.Prepare
		} else {
			return out, fmt.Errorf("%s: hwnpu-dev models package does not make %s models yet (run its command once on %s)", j.model.Name, j.model.Kind, j.target)
		}
		npu, err := airole.ForTarget(j.target)
		if err != nil {
			return out, err
		}
		// a Mac's exports differ by its chip: each kind's (chat.ExportChips), as a Mac of it
		// converts; another target's are of no chip
		on := []string{""}
		if j.target == cache.CoreAITarget {
			if on = chips; len(on) == 0 {
				on = chat.ExportChips
			}
		}
		for _, chip := range on {
			if chip != "" && slices.Contains(j.model.NotOn, chip) {
				fmt.Fprintf(os.Stderr, "%s: not run on the %s (not_on)\n", j.model.Name, chip)
				continue
			}
			for _, m := range forms(j.model, memory) {
				what := j.target
				if chip != "" {
					what += " as on the " + chip
				}
				start := time.Now()
				fmt.Fprintf(os.Stderr, "%s: %s for %s%s\n", m.Name, name, what, formNote(m, j.model))
				undo := chat.ActAsChip(chip)
				f, err := makeReady(airole.PrepareEnv{Ctx: chat.WithModel(ctx, m), Model: m, Base: m, Dir: chat.ModelDir(m), NPU: npu, Log: log})
				undo()
				if err != nil {
					if skip(m.Name+" ("+what+")", err) {
						continue
					}
					return out, fmt.Errorf("%s (%s): %w", m.Name, what, err)
				}
				fmt.Fprintf(os.Stderr, "%s: ready for %s in %s\n", m.Name, what, time.Since(start).Round(time.Second))
				if missing == nil {
					putHere(f)
				}
				out = append(out, prepared{j, f, chip})
			}
		}
	}
	shared.Flush(ctx)
	return out, ctx.Err()
}

// amdGeneration is why model m's export for generation gen of AMD's NPU is not made: its recipe
// skips gen (model-recipes.yaml skip), or hwnpu does not run that generation's exports yet
// (recipe.Built); nil when it is made.
func amdGeneration(m chat.Model, gen string) error {
	recipes, err := recipe.Builtin()
	if err != nil {
		return err
	}
	_, skips := recipes.AMDGenerations(m.Recipe)
	for _, sk := range skips {
		if sk.Generation == gen {
			return fmt.Errorf("its recipe makes no export for %s (model-recipes.yaml skip: %s, %s)", gen, sk.Why, sk.Checked)
		}
	}
	if !slices.Contains(recipe.Built, gen) {
		return fmt.Errorf("hwnpu does not run the exports of %s (%s) yet: none is made", gen, recipe.AMDTarget(gen))
	}
	return nil
}

// forms are model m as machines of each memory run it (Model.On), once for each way to run it:
// the model itself and each variant whose Run differs. Given memory, those of its GB alone.
func forms(m chat.Model, memory []int) []chat.Model {
	gbs := memory
	if len(gbs) == 0 {
		gbs = append([]int{0}, slices.Sorted(maps.Keys(m.Variants))...)
	}
	sort.Ints(gbs)
	var out []chat.Model
	for _, gb := range gbs {
		v := m.On(gb)
		if slices.ContainsFunc(out, func(o chat.Model) bool { return o.Run == v.Run }) {
			continue
		}
		out = append(out, v)
	}
	return out
}

// formNote says how form f of m runs when it is not as m's definition gives.
func formNote(f, m chat.Model) string {
	if f.Run == m.Run {
		return ""
	}
	return fmt.Sprintf(" (as on a machine of more memory: %+v)", f.Run)
}

// putHere puts the entries the files of a preparation are in (a folder with entry.json: the
// path, or one above it up to two levels) in the sources written back to, in the background:
// one that was here already (made or fetched before) is not put by the preparation, and a source
// that has its key is left (store.PutEntry).
func putHere(f airole.Files) {
	seen := map[string]bool{}
	for _, ps := range f.Paths {
		for _, p := range ps {
			for d, i := p, 0; i < 3 && d != "." && d != string(filepath.Separator); d, i = filepath.Dir(d), i+1 {
				if _, err := os.Stat(filepath.Join(d, shared.EntryFile)); err != nil {
					continue
				}
				if !seen[d] {
					seen[d] = true
					for _, s := range shared.Writebacks() {
						shared.PutLater(s, d)
					}
				}
				break
			}
		}
	}
}
