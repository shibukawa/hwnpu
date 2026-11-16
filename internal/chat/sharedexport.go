package chat

import (
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu/internal/cache"
	shared "github.com/shibukawa/hwnpu/internal/store"
	amdtools "github.com/shibukawa/hwnpu/tools/amd"
)

// place is where a conversion goes in the sources (docs/cache-sharing.md 7): its folder
// converted/MODEL/TARGET/EXPORT/VERSION and the entry it is described by (no files yet).
type place struct {
	entry shared.Entry
}

// placeOf is the place of the conversion of kind of model m for target that recipe r describes,
// named export here. VERSION is vN for the version the definitions give the target (targets),
// vN-HASH for a model in development, v0-HASH for one with no version.
func placeOf(kind string, m Model, target, export string, r map[string]string) place {
	if target == "" {
		target = "any"
	}
	key := shared.Fingerprint(r)
	v := m.Targets[target]
	rel := shared.EntryPath(m.Name, target, export, shared.VersionDir(v, m.Develop, key))
	return place{shared.Entry{Key: key, Kind: kind, Scope: shared.ScopePortable, Model: m.Name, Target: target,
		Version: v, Develop: m.Develop, Path: rel, Recipe: r}}
}

// released reports whether the place is a published version (vN): taken whatever tools made it.
func (p place) released() bool { return p.entry.Version > 0 && !p.entry.Develop }

// takes reports whether e, found at the place, is what is wanted: for vN, the same version and
// the same recipe but for the tools' hash (a change of the tools that changes what is made bumps
// the version); else the same key.
func (p place) takes(e shared.Entry) bool {
	if !p.released() {
		return e.Key == p.entry.Key
	}
	if e.Version != p.entry.Version || e.Kind != p.entry.Kind {
		return false
	}
	a, b := maps.Clone(e.Recipe), maps.Clone(p.entry.Recipe)
	delete(a, "tools")
	delete(b, "tools")
	return maps.Equal(a, b)
}

// fetchConverted makes dst from the first source that has the place's entry, and puts it in the
// sources before that one that are written back to. It reports whether it did.
func fetchConverted(ctx context.Context, p place, dst string, log io.Writer) bool {
	ss := shared.Sources()
	if len(ss) == 0 {
		return false
	}
	i, err := shared.FetchEntry(ctx, ss, p.entry.Path, p.takes, dst, log)
	if err != nil {
		if !errors.Is(err, shared.ErrNotFound) && log != nil {
			fmt.Fprintf(log, "the sources: %v; making %s here\n", err, filepath.Base(dst))
		}
		// it is made here next: its files go in the sources written back to as they are written
		for _, s := range shared.Writebacks() {
			shared.StartEarly(s, p.entry.Path, dst)
		}
		return false
	}
	for _, s := range ss[:i] {
		if s.Writeback {
			shared.PutLater(s, dst)
		}
	}
	return true
}

// putConverted describes dir, made here and whole, as the place's entry and puts it in the
// sources that are written back to, in the background.
func putConverted(p place, dir string, log io.Writer) {
	to := shared.Writebacks()
	if len(to) == 0 {
		return
	}
	e := p.entry
	e.MadeBy = map[string]string{"host": hostName()}
	if _, err := shared.Describe(dir, e); err != nil {
		if log != nil {
			fmt.Fprintf(log, "%s: %s: %v\n", filepath.Base(dir), shared.EntryFile, err)
		}
		return
	}
	for _, s := range to {
		if log != nil {
			fmt.Fprintf(log, "putting %s in %s in the background\n", filepath.Base(dir), s)
		}
		shared.PutLater(s, dir)
	}
}

// modelNamed is the model of the definitions named name, or one of that name alone.
func modelNamed(name string) Model {
	for _, m := range Catalog {
		if m.Name == name {
			return m
		}
	}
	return Model{Name: name}
}

// SharedExport makes the export out (a directory) that recipe describes: from the first source
// that has it (docs/cache-sharing.md 7), else with build. What build made gets its entry.json
// and goes in the sources that are written back to. A nil recipe (what it is made from is not
// known: an unpinned model with no lock and Hugging Face out of reach) builds without the
// sources.
func SharedExport(ctx context.Context, kind string, recipe map[string]string, out string, log io.Writer, build func() error) error {
	if recipe == nil {
		return build()
	}
	if log == nil {
		log = io.Discard
	}
	r := maps.Clone(recipe)
	r["kind"] = kind
	p := placeOf(kind, modelNamed(recipe["model"]), recipe["target"], filepath.Base(out), r)
	unlock, err := cache.Lock(ctx, out+":layers", "the export "+filepath.Base(out), log)
	if err != nil {
		return err
	}
	defer unlock()
	if e, err := shared.ReadEntry(out); err == nil && p.takes(e) {
		return nil
	}
	if fetchConverted(ctx, p, out, log) {
		os.WriteFile(out+".done", nil, 0o644)
		return nil
	}
	if err := build(); err != nil {
		return err
	}
	putConverted(p, out, log)
	return nil
}

func hostName() string { h, _ := os.Hostname(); return h }

// sourceCommit is the commit model m's files come from: its revision when that is a commit, else
// the one of the sources' lock (lockedCommit), else the one the hub says its config.json is of;
// "" when none is known.
func sourceCommit(ctx context.Context, m Model) string {
	if c := lockedCommit(ctx, m.Source()); c != "" {
		return c
	}
	_, c := hubFile(ctx, m.Source(), "config.json")
	return c
}

// ExportRecipe is what an export of model m for AMD's NPU with IRON's tools is made from (a key
// of the shared layers): the model, the commit of its files and the SHA-256 the definitions pin,
// the tool and its arguments, the tools' hash. nil when the commit is not known.
func ExportRecipe(ctx context.Context, m Model, tool string, args []string) map[string]string {
	commit := sourceCommit(ctx, m)
	hash, err := amdtools.Hash()
	if commit == "" || err != nil {
		return nil
	}
	base, _, _ := strings.Cut(m.Repo, "@")
	r := map[string]string{"model": m.Name, "target": cache.AMDTarget, "source": base, "commit": commit,
		"tool": tool, "args": strings.Join(args, " "), "tools": hash}
	if m.File != "" {
		r["file"] = m.File
	}
	for f, d := range m.SHA256 {
		r["sha256:"+f] = strings.ToLower(d)
	}
	for role, repo := range m.With {
		r["with:"+role] = repo
	}
	return r
}

// PinnedRecipe is ExportRecipe for a model whose files the definitions pin all of (sha256: the
// ONNX models), made by tool with arguments hwnpu works out in Go: version is the export's
// version there, bumped by hand when what hwnpu passes the tool changes. nil when nothing is
// pinned.
func PinnedRecipe(m Model, tool string, version int) map[string]string {
	hash, err := amdtools.Hash()
	if len(m.SHA256) == 0 || err != nil {
		return nil
	}
	base, _, _ := strings.Cut(m.Repo, "@")
	r := map[string]string{"model": m.Name, "target": cache.AMDTarget, "source": base,
		"tool": tool, "version": fmt.Sprint(version), "tools": hash}
	for f, d := range m.SHA256 {
		r["sha256:"+f] = strings.ToLower(d)
	}
	return r
}

// ExportKind is the kind of entry of an export of tools/amd's script tool: iron_onnxcnn.py,
// iron-onnxcnn.
func ExportKind(tool string) string {
	return "iron-" + strings.TrimSuffix(strings.TrimPrefix(tool, "iron_"), ".py")
}
