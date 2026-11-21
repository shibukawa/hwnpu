package modelscmd

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"slices"
	"time"

	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/devopt"
	"github.com/shibukawa/hwnpu/internal/backend/coreai"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
)

func (p PrebuildCmd) Run(d *devopt.Defs) error {
	if err := d.Load(); err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	var paths []string
	var jobs []job
	for _, s := range p.Sel {
		if isPath(s) {
			paths = append(paths, s)
			continue
		}
		js, err := parseJobs([]string{s + "/" + cache.CoreAITarget})
		if err != nil {
			return err
		}
		jobs = append(jobs, js...)
	}
	if len(jobs) > 0 {
		files, err := prepare(ctx, jobs, p.Memory, p.Chip)
		if err != nil {
			return err
		}
		for _, f := range files {
			paths = append(paths, coreAIPaths(f.Paths)...)
		}
		if len(paths) == 0 {
			return errors.New("the models have no Core AI exports to compile")
		}
	}
	o := coreai.BuildOptions{Jobs: p.Jobs, Force: p.Force, DryRun: p.DryRun, Log: os.Stderr, Packed: p.Packed || !p.NoPut}
	if slices.Contains(p.Arch, "all") {
		o.Archs = coreai.Archs
	} else {
		o.Archs = p.Arch
	}
	t0 := time.Now()
	results, err := coreai.Build(ctx, paths, o)
	built, skipped, failed := 0, 0, 0
	var code, weights int64
	for _, r := range results {
		switch {
		case r.Err != nil:
			failed++
		case r.Skipped:
			skipped++
		default:
			built++
			code += r.Code
			weights += r.Weights
		}
	}
	if !p.DryRun {
		fmt.Fprintf(os.Stderr, "%d compiled in %s (%.0f MB of compiled regions beside %.1f GB of weights), %d prebuilt already, %d failed\n", built, time.Since(t0).Round(time.Second), float64(code)/1e6, float64(weights)/1e9, skipped, failed)
	}
	if !p.NoPut && !p.Packed && !p.DryRun {
		// what failed is left out (the devices compile it): the rest is put all the same
		puts := map[[2]string]bool{}
		for _, r := range results {
			if export, ok := coreai.ExportOf(r.Asset); ok && r.Err == nil {
				puts[[2]string{export, r.Arch}] = true
			}
		}
		for k := range puts {
			if perr := chat.PutPrebuilt(k[0], k[1]); perr != nil {
				return perr
			}
		}
	}
	if err != nil {
		return err
	}
	if len(results) > 0 && !p.DryRun {
		where := "beside their sources, as " + filepath.Base(results[0].Out)
		if o.Packed {
			where = "packed in " + filepath.Dir(filepath.Dir(results[0].Out))
		}
		fmt.Fprintf(os.Stderr, "the prebuilt programs lie %s\n", where)
	}
	return nil
}
