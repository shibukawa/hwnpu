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
	"strings"

	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/devopt"
	"github.com/shibukawa/hwnpu/internal/backend/coreai"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	shared "github.com/shibukawa/hwnpu/internal/store"
)

// FetchCmd is hwnpu-dev models fetch: a model brought here without converting anything, to try
// conversions on it after. Its checkpoint (the files the sources' lock of its repository names,
// else what a conversion reads) and the conversions for each target that are in the sources,
// as package would take them; what none has is listed (package makes it). What comes from Hugging
// Face goes in the sources written back to too, as every download does.
type FetchCmd struct {
	Sel          []string `arg:"" name:"MODEL[/TARGET]" help:"Models of the definitions, each for a target (amd-xdna2, coreai, coreai-ARCH, coreai-all: as package) or for the targets its definition gives versions for (targets); a model with neither, its checkpoint alone."`
	Memory       []int    `help:"Only the forms of machines of these GB (as package)." placeholder:"GB"`
	Chip         []string `help:"For coreai, only the exports of Macs of these chips (as package)." placeholder:"CHIP"`
	NoCheckpoint bool     `help:"Not the checkpoint: the conversions alone."`
}

func (f FetchCmd) Run(d *devopt.Defs) error {
	if err := d.Load(); err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	var jobs []job
	var models []chat.Model
	for _, s := range f.Sel {
		js, err := parseJobs([]string{s})
		if err != nil && !strings.Contains(s, "/") {
			// a model with no targets: its checkpoint alone
			m, ok := chat.Lookup(s)
			if !ok {
				return err
			}
			models = append(models, m)
			continue
		}
		if err != nil {
			return err
		}
		jobs = append(jobs, js...)
		if !slices.ContainsFunc(models, func(m chat.Model) bool { return m.Name == js[0].model.Name }) {
			models = append(models, js[0].model)
		}
	}
	if f.NoCheckpoint && len(jobs) == 0 {
		return errors.New("nothing to fetch: the models give no targets, and --no-checkpoint leaves out their checkpoints")
	}
	if !f.NoCheckpoint {
		for _, m := range models {
			if err := fetchCheckpoint(ctx, m); err != nil {
				return fmt.Errorf("%s: %w", m.Name, err)
			}
		}
	}
	var missing []string
	done, err := prepare(ctx, jobs, f.Memory, f.Chip, &missing)
	if err != nil {
		return err
	}
	// coreai-ARCH: the programs compiled for each architecture, of each export there is here
	for _, p := range done {
		for _, export := range coreAIPaths(p.files.Paths) {
			if !strings.HasSuffix(export, ".coreai") {
				continue
			}
			for _, arch := range p.job.archs {
				chat.FetchPrebuilt(ctx, export, arch)
				if _, err := os.Stat(coreai.PackedRoot(export, arch)); err != nil {
					missing = append(missing, fmt.Sprintf("%s (%s-%s): %s", p.job.model.Name, cache.CoreAITarget, arch, filepath.Base(export)))
				}
			}
		}
	}
	if len(missing) > 0 {
		fmt.Fprintf(os.Stderr, "\nnot in the sources (hwnpu-dev models package makes them):\n")
		for _, m := range missing {
			fmt.Fprintf(os.Stderr, "  %s\n", m)
		}
	}
	return nil
}

// fetchCheckpoint brings model m's checkpoint here (chat.RepoDir): every file the first source
// with a lock of its repository at its commit names (what was fetched of it before), else, of
// a chat model, its config.json, its tokenizer and its weights. Another model's comes with the
// conversion that reads it (prepare), when one is not in the sources.
func fetchCheckpoint(ctx context.Context, m chat.Model) error {
	if m.Repo == "" {
		return nil
	}
	for _, s := range shared.Sources() {
		l, err := shared.ReadLock(ctx, s, m.Repo)
		if err != nil || m.Revision != "" && l.Commit != "" && l.Commit != m.Revision {
			continue
		}
		var files []chat.File
		for _, name := range slices.Sorted(maps.Keys(l.Files)) {
			files = append(files, chat.File{Repo: m.Source(), Name: name})
		}
		fmt.Fprintf(os.Stderr, "%s: the checkpoint, %d files of %s (the lock of %s)\n", m.Name, len(files), m.Repo, s)
		return chat.FetchFiles(ctx, files, console.Log(os.Stderr))
	}
	if m.Kind != chat.KindChat && m.Kind != chat.KindEmbed {
		fmt.Fprintf(os.Stderr, "%s: no source has a lock of %s: its checkpoint comes with a conversion that reads it\n", m.Name, m.Repo)
		return nil
	}
	log := console.Log(os.Stderr)
	names := append([]string{"config.json"}, chat.TokenizerFiles(m.Arch)...)
	for _, n := range names {
		if _, err := chat.Fetch(ctx, m.Source(), n, log); err != nil && !chat.NotFound(err) {
			return err
		}
	}
	_, err := chat.FetchWeights(ctx, m.Source(), log)
	return err
}
