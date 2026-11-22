package modelscmd

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu/internal/backend/coreai"
	"github.com/shibukawa/hwnpu/internal/chat"
)

// canCompile is nil: coreai-build is asked for when it runs (coreai.Build).
func canCompile() error { return nil }

// compile compiles the Core AI exports and programs at paths for archs (none: this Mac's) and
// puts them in the sources written back to (compileFlags).
func compile(ctx context.Context, paths, archs []string, p compileFlags) error {
	o := coreai.BuildOptions{Jobs: p.Jobs, Force: p.Force, DryRun: p.DryRun, Log: os.Stderr, Packed: p.Packed || !p.NoPush}
	o.Archs = archs
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
	if !p.NoPush && !p.Packed && !p.DryRun {
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
