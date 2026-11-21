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
	name := filepath.Base(dir)
	ShareMadeHere(m, dir, ImportKind, ImportedRecipe(m, cache.AMDTarget, name), log)
}

// ShareMadeHere is ShareMadeByHand for an export described as kind and recipe's (as hwnpu-dev
// cache import describes one of a first run's tool: ExportKind and PinnedRecipe), nothing when
// recipe is nil.
func ShareMadeHere(m Model, dir, kind string, recipe map[string]string, log io.Writer) {
	to := shared.Writebacks()
	if len(to) == 0 || recipe == nil {
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
	e := PlaceEntry(kind, m, cache.AMDTarget, name, recipe)
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

// AMDPictureExport is the export of a model of pictures m (a depth model, a detector, an image
// generator) on AMD's NPU for pictures of size x size (0: a model of one size;
// AMDPictureNames), here (entries/amd-xdna2 under the cache's root, or ~/models:
// cache.ExportRoots) or in the sources, downloaded once into the cache's entries/amd-xdna2
// (AMDNamed): an export made by hand and put there (hwnpu-dev cache import, or its first use on
// the machine that made it). tool is the script that writes it, for the error when there is
// none.
func AMDPictureExport(ctx context.Context, m Model, tool string, size int, log io.Writer) (string, error) {
	names := AMDPictureNames(m, size)
	if d := AMDNamed(ctx, m, names, isDirectory, log); d != "" {
		return d, nil
	}
	flags := ""
	if m.File != "" {
		flags = " --dit " + m.File
	}
	if size != 0 && size != 1024 {
		flags += fmt.Sprintf(" --size %d", size)
	}
	out := filepath.Join(AMDRoot(), names[0])
	if home, err := os.UserHomeDir(); err == nil {
		if rel, err := filepath.Rel(home, out); err == nil && filepath.IsLocal(rel) {
			out = "~/" + filepath.ToSlash(rel)
		}
	}
	return "", fmt.Errorf("AMD's NPU runs exports of %s, and neither this machine (%s) nor the sources have %s for %s: download %s and run\n  python %s ~/models/%s %s%s\n(its first run here puts it in the sources written back to; see docs/amd-npu-models.md)",
		tool, strings.Join(cache.ExportRoots(), ", "), names[0], m.Name, strings.TrimPrefix(m.Source(), "hf:"), tool, filepath.Base(m.Repo), out, flags)
}

// AMDPictureNames are the names the export of model m for AMD's NPU has, for pictures of size
// x size, in lower case (what tools/amd/iron_*.py are told to write, and the sources hold):
// NAME.amd-npu2 or NAME-1024.amd-npu2 at 1024 and for a model of one size (0),
// NAME-SIZE.amd-npu2 otherwise.
func AMDPictureNames(m Model, size int) []string {
	name := strings.ToLower(m.Name)
	if size == 0 || size == 1024 {
		return []string{name + ".amd-npu2", name + "-1024.amd-npu2"}
	}
	return []string{fmt.Sprintf("%s-%d.amd-npu2", name, size)}
}

// AMDPictureExportHere is the export of model m for AMD's NPU here (AMDPictureNames at no
// size), "" for none, without asking the sources.
func AMDPictureExportHere(m Model) string {
	for _, n := range AMDPictureNames(m, 0) { // NAME.amd-npu2 first: the export of every aspect ratio
		if d := cache.FindExport(n, isDirectory); d != "" {
			return d
		}
	}
	return ""
}

func isDirectory(path string) bool {
	fi, err := os.Stat(path)
	return err == nil && fi.IsDir()
}
