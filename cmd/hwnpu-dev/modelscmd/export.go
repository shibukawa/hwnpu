package modelscmd

import (
	"context"
	"fmt"
	"maps"
	"os"
	"os/signal"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/cachecmd"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/devopt"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	shared "github.com/shibukawa/hwnpu/internal/store"
)

// ExportCmd is hwnpu-dev models export: each model made ready for a target as the apps make it
// on their first run there (airole.Role.Prepare), on this machine whatever its NPU: the programs
// are converted (or taken from the sources), not loaded. What is made goes in the sources
// written back to, as the first run's does.
type ExportCmd struct {
	Sel    []string `arg:"" name:"MODEL[/TARGET]" help:"Models of the definitions, each for a target (amd-xdna2, coreai) or for the targets its definition gives versions for (targets)."`
	Memory []int    `help:"Only the forms of machines of these GB (hwnpu-chat --memory); every form the definitions give when left out." placeholder:"GB"`
	Chip   []string `help:"For coreai, only the exports of Macs of these chips (M1, M2, ...); one of each kind of chip whose exports differ when left out (chat.ExportChips: M1, M2 for the later ones)." placeholder:"CHIP"`
}

// targets are what export makes conversions for.
var targets = []string{cache.AMDTarget, cache.CoreAITarget}

// job is a model to make ready for a target.
type job struct {
	model  chat.Model
	target string
}

func (x ExportCmd) Run(d *devopt.Defs) error {
	if err := d.Load(); err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	jobs, err := parseJobs(x.Sel)
	if err != nil {
		return err
	}
	if len(shared.Writebacks()) == 0 {
		fmt.Fprintf(os.Stderr, "no source is written back to (%s ?writeback=true): what is made stays here\n", shared.SourceEnv)
	}
	_, err = prepare(ctx, jobs, x.Memory, x.Chip)
	return err
}

// parseJobs reads MODEL[/TARGET]s.
func parseJobs(sel []string) ([]job, error) {
	var jobs []job
	for _, s := range sel {
		name, target, _ := strings.Cut(s, "/")
		m, ok := chat.Lookup(name)
		if !ok {
			return nil, fmt.Errorf("%s is not a model of the definitions (hwnpu-chat list, hwnpu-audio list, hwnpu-image list)", name)
		}
		var ts []string
		switch {
		case target != "":
			if !slices.Contains(targets, target) {
				return nil, fmt.Errorf("%s: no target %q (%s)", s, target, strings.Join(targets, ", "))
			}
			ts = []string{target}
		default:
			for _, t := range targets {
				if _, ok := m.Targets[t]; ok {
					ts = append(ts, t)
				}
			}
			if len(ts) == 0 {
				return nil, fmt.Errorf("%s: its definition gives no versions of targets: name one (%s/%s)", m.Name, m.Name, strings.Join(targets, ", "+m.Name+"/"))
			}
		}
		for _, t := range ts {
			jobs = append(jobs, job{m, t})
		}
	}
	return jobs, nil
}

// prepare makes the jobs' models ready, each form of each (forms), and waits for what goes in
// the sources. It returns the files each form's preparation gave (airole.Files), in order.
func prepare(ctx context.Context, jobs []job, memory []int, chips []string) ([]airole.Files, error) {
	for _, c := range chips {
		if !slices.Contains(chat.Chips, c) {
			return nil, fmt.Errorf("no chip %q (%s)", c, strings.Join(chat.Chips, ", "))
		}
	}
	log := console.Log(os.Stderr)
	var out []airole.Files
	for _, j := range jobs {
		if j.target == cache.AMDTarget && j.model.Kind == chat.KindChat {
			// IRON's exports of chat models, every form of amd_export (cachecmd.AMDForms)
			if len(j.model.AMDExport) == 0 {
				return out, fmt.Errorf("%s has no amd_export", j.model.Name)
			}
			if err := (cachecmd.ExportCmd{Model: []string{j.model.Name}}).Make(ctx, memory); err != nil {
				return out, err
			}
			continue
		}
		role := airole.For(j.model)
		if role == nil {
			return out, fmt.Errorf("%s: no role of the library makes a %s model ready (run its command once on %s)", j.model.Name, j.model.Kind, j.target)
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
				fmt.Fprintf(os.Stderr, "%s: %s for %s%s\n", m.Name, role.Name, what, formNote(m, j.model))
				undo := chat.ActAsChip(chip)
				f, err := role.Prepare(airole.PrepareEnv{Ctx: chat.WithModel(ctx, m), Model: m, Dir: chat.ModelDir(m), NPU: npu, Log: log})
				undo()
				if err != nil {
					return out, fmt.Errorf("%s (%s): %w", m.Name, what, err)
				}
				fmt.Fprintf(os.Stderr, "%s: ready for %s in %s\n", m.Name, what, time.Since(start).Round(time.Second))
				out = append(out, f)
			}
		}
	}
	shared.Flush(ctx)
	return out, ctx.Err()
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
