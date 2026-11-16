package chat

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu/internal/cache"
	shared "github.com/shibukawa/hwnpu/internal/store"
)

// AMDNamed is the export for AMD's NPU of model m under the first of names (NAME.amd-npu2, ...)
// that is here (cache.ExportRoots: a directory have takes) or in the sources, "" when neither
// has one. One in the sources is an export made by hand that hwnpu-dev cache import, or the
// first use of it on another machine (ShareMadeByHand), put there (ImportKind): it is
// downloaded into AMDRoot once, and the runs after take that copy without asking the sources.
// One here made by hand goes in the sources written back to (ShareMadeByHand).
func AMDNamed(ctx context.Context, m Model, names []string, have func(dir string) bool, log io.Writer) string {
	if log == nil {
		log = io.Discard
	}
	for _, n := range names {
		if d := cache.FindExport(n, have); d != "" {
			ShareMadeByHand(m, d, log)
			return d
		}
	}
	for _, n := range names {
		out := filepath.Join(AMDRoot(), n)
		if fetchImported(ctx, m, out, log) && have(out) {
			return out
		}
	}
	return ""
}

// fetchImported makes out, an export for AMD's NPU of model m named as out is, from the first
// source that has one made by hand under that name (ImportedRecipe). It reports whether it did.
func fetchImported(ctx context.Context, m Model, out string, log io.Writer) bool {
	if len(shared.Sources()) == 0 {
		return false
	}
	p := sharedPlace(ImportKind, ImportedRecipe(m, cache.AMDTarget, filepath.Base(out)), out)
	unlock, err := cache.Lock(ctx, out+":layers", "the export "+filepath.Base(out), log)
	if err != nil {
		return false
	}
	defer unlock()
	if e, err := shared.ReadEntry(out); err == nil && p.takes(e) {
		return true // another process fetched it meanwhile
	}
	if !fetchFrom(ctx, p, out, log) {
		return false
	}
	os.WriteFile(out+".done", nil, 0o644)
	return true
}

// ShareMadeByHand puts dir, an export for AMD's NPU of model m made here by hand (it has no
// entry.json: it was neither fetched nor made by a first run), in the sources written back to, as
// hwnpu-dev cache import does: described once (the SHA-256 of its files, which entry.json then
// keeps) and put in the background (store.PutLater; a source that holds one of its name already is
// left as it is). Only exports in AMDRoot: ~/models is read, not written (hwnpu-dev cache import
// puts one from there).
func ShareMadeByHand(m Model, dir string, log io.Writer) {
	to := shared.Writebacks()
	if len(to) == 0 {
		return
	}
	if _, err := os.Stat(filepath.Join(dir, shared.EntryFile)); err == nil {
		return
	}
	if rel, err := filepath.Rel(AMDRoot(), dir); err != nil || !filepath.IsLocal(rel) || strings.Contains(rel, string(filepath.Separator)) {
		return
	}
	if log == nil {
		log = io.Discard
	}
	name := filepath.Base(dir)
	e := PlaceEntry(ImportKind, m, cache.AMDTarget, name, ImportedRecipe(m, cache.AMDTarget, name))
	e.MadeBy = map[string]string{"host": hostName(), "note": "made by hand, described by its first use"}
	fmt.Fprintf(log, "%s: made by hand here; describing it (the SHA-256 of its files, once) to put it in the sources\n", name)
	if _, err := shared.Describe(dir, e); err != nil {
		fmt.Fprintf(log, "%s: %s: %v\n", name, shared.EntryFile, err)
		return
	}
	for _, s := range to {
		fmt.Fprintf(log, "putting %s in %s in the background\n", name, s)
		shared.PutLater(s, dir)
	}
}
