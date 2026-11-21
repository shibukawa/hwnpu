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
	"github.com/shibukawa/hwnpu/internal/llm"
	shared "github.com/shibukawa/hwnpu/internal/store"
	amdtools "github.com/shibukawa/hwnpu/tools/amd"
)

// place is where a conversion goes in the sources (docs/cache-sharing-design.md 7): its folder
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

// fetchConverted is fetchFrom, and, when no source has the entry, starts putting dst in the
// sources written back to as it is made here next (startEarly): for a conversion a miss is
// followed by.
func fetchConverted(ctx context.Context, p place, dst string, log io.Writer) bool {
	if fetchFrom(ctx, p, dst, log) {
		return true
	}
	startEarly(p, dst)
	return false
}

// fetchFrom makes dst from the first source that has the place's entry, and puts it in the
// sources before that one that are written back to. It reports whether it did.
func fetchFrom(ctx context.Context, p place, dst string, log io.Writer) bool {
	ss := shared.Sources()
	if len(ss) == 0 {
		return false
	}
	i, err := shared.FetchEntry(ctx, ss, p.entry.Path, p.takes, dst, log)
	if errors.Is(err, shared.ErrNotFound) && p.released() {
		// a published version none has yet: the same key with no version is the same bytes
		// (made before the version was written); it goes in the sources as the version
		if i, err = shared.FetchEntry(ctx, ss, Unversioned(p.entry), func(e shared.Entry) bool { return e.Key == p.entry.Key }, dst, log); err == nil {
			if e, rerr := shared.ReadEntry(dst); rerr == nil {
				rerr = shared.WriteEntry(dst, Released(p.entry, e))
				if rerr == nil {
					i = len(ss) // every source written back to lacks the version
				}
			}
		}
	}
	if err != nil {
		if !errors.Is(err, shared.ErrNotFound) && log != nil {
			fmt.Fprintf(log, "the sources: %v; making %s here\n", err, filepath.Base(dst))
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

// startEarly puts the files of dir, the place's entry being made here next, in the sources
// written back to as they are written (store.StartEarly).
func startEarly(p place, dir string) {
	for _, s := range shared.Writebacks() {
		shared.StartEarly(s, p.entry.Path, dir)
	}
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
// that has it (docs/cache-sharing-design.md 7), else with build. What build made gets its entry.json
// and goes in the sources that are written back to. A nil recipe (what it is made from is not
// known: an unpinned model with no lock and Hugging Face out of reach) builds without the
// sources.
func SharedExport(ctx context.Context, kind string, recipe map[string]string, out string, log io.Writer, build func() error) error {
	return SharedExportOr(ctx, kind, recipe, nil, out, log, build)
}

// SharedRecipe is an entry of the sources by its kind and its recipe.
type SharedRecipe struct {
	Kind   string
	Recipe map[string]string
}

// SharedExportOr is SharedExport that, when no source has recipe's entry, takes the first of
// others that a source has (each at the place its kind and recipe give under out's name: an
// export made by hand that hwnpu-dev models import put there, ImportedRecipe) before it builds.
// What build makes is put as recipe's.
func SharedExportOr(ctx context.Context, kind string, recipe map[string]string, others []SharedRecipe, out string, log io.Writer, build func() error) error {
	if recipe == nil {
		return build()
	}
	if log == nil {
		log = io.Discard
	}
	p := sharedPlace(kind, recipe, out)
	places := []place{p}
	for _, o := range others {
		places = append(places, sharedPlace(o.Kind, o.Recipe, out))
	}
	unlock, err := cache.Lock(ctx, out+":layers", "the export "+filepath.Base(out), log)
	if err != nil {
		return err
	}
	defer unlock()
	if fetchPlaces(ctx, places, out, log) {
		return nil
	}
	if err := llm.MayConvert(ctx, out); err != nil {
		return err
	}
	// made here: its files go in the sources written back to as they are written, as recipe's
	// (not on the misses before: an export of others fetched into out would go too)
	startEarly(p, out)
	if err := build(); err != nil {
		return err
	}
	putConverted(p, out, log)
	return nil
}

// FetchShared is SharedExportOr without the build: it reports whether out is recipe's entry, or
// one of others', here or fetched from a source. A nil recipe is never there.
func FetchShared(ctx context.Context, kind string, recipe map[string]string, others []SharedRecipe, out string, log io.Writer) (bool, error) {
	if recipe == nil {
		return false, nil
	}
	if log == nil {
		log = io.Discard
	}
	places := []place{sharedPlace(kind, recipe, out)}
	for _, o := range others {
		places = append(places, sharedPlace(o.Kind, o.Recipe, out))
	}
	unlock, err := cache.Lock(ctx, out+":layers", "the export "+filepath.Base(out), log)
	if err != nil {
		return false, err
	}
	defer unlock()
	return fetchPlaces(ctx, places, out, log), nil
}

// fetchPlaces reports whether out is the entry of one of places: here, or fetched from the first
// source that has one (out.done written then).
func fetchPlaces(ctx context.Context, places []place, out string, log io.Writer) bool {
	if e, err := shared.ReadEntry(out); err == nil {
		for _, q := range places {
			if q.takes(e) {
				return true
			}
		}
	}
	for _, q := range places {
		if fetchFrom(ctx, q, out, log) {
			os.WriteFile(out+".done", nil, 0o644)
			return true
		}
	}
	return false
}

// sharedPlace is the place of the entry of kind and recipe named as out.
func sharedPlace(kind string, recipe map[string]string, out string) place {
	r := maps.Clone(recipe)
	r["kind"] = kind
	return placeOf(kind, modelNamed(recipe["model"]), recipe["target"], filepath.Base(out), r)
}

// ImportKind is the kind of entry of an export for AMD's NPU made by hand, which hwnpu-dev models
// import puts in the sources.
const ImportKind = "iron-import"

// ImportedRecipe is the recipe hwnpu-dev models import describes an export made by hand with
// (ImportKind): its model, target and name alone, so that a command finds one under the name it
// would make the export with. Its "made" names the command as it was called when the first
// entries were put (hwnpu-dev cache import): it is in the key, so it stays.
func ImportedRecipe(m Model, target, export string) map[string]string {
	return map[string]string{"model": m.Name, "target": target, "export": export, "made": "by hand (hwnpu-dev cache import)"}
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
// the tool and its arguments, the tools' hash (for iron_export_lm.py, of the files of tools/amd
// the export reads: amdtools.LMHash). nil when the commit is not known.
func ExportRecipe(ctx context.Context, m Model, tool string, args []string) map[string]string {
	commit := sourceCommit(ctx, m)
	hash, err := amdtools.Hash()
	if tool == "iron_export_lm.py" {
		hash, err = amdtools.LMHash(args)
	}
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
