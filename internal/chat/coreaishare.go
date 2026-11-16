package chat

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"

	"github.com/shibukawa/hwnpu/internal/backend/coreai"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/coreai/ailower"
	"github.com/shibukawa/hwnpu/internal/llm"
	shared "github.com/shibukawa/hwnpu/internal/store"
)

// CoreAIKind is the kind of entry of a Core AI export in the shared layers.
const CoreAIKind = "coreai"

// CoreAITarget is the target of the Core AI exports: their programs (.aimodel, MLIR bytecode)
// are the same on every Mac (docs/cache-sharing.md 10): what is compiled for the chip stays in
// Core AI's own cache.
const CoreAITarget = "coreai"

func init() { llm.SharedExports = coreAIShares{} }

// coreAIShares fetches the Core AI exports (NAME.coreai, llm.CoreAIExport) of every command from
// the shared layers (docs/cache-sharing.md 7), the folders and S3 buckets the definitions name
// (cache.layers), and puts those made here in the layers that are written. The key is what an
// export is made from (coreAIRecipe): the model's files, the export's name (the revisions and
// formats it is converted with) and the lowering's version.
type coreAIShares struct{}

// sharedLog is where fetching and putting say what they do.
var sharedLog io.Writer = console.Log(os.Stderr)

func (coreAIShares) Fetch(ctx context.Context, export string) bool {
	m, r := coreAIRecipe(ctx, export)
	return fetchShared(ctx, CoreAIKind, m, r, export, r["export"])
}

func (coreAIShares) Put(ctx context.Context, export string) {
	m, r := coreAIRecipe(ctx, export)
	putShared(CoreAIKind, m, r, export, r["export"])
}

// fetchShared makes dst (a directory) from the first source that has the conversion of kind of
// model m that recipe r describes, named export here (nil r: nothing to look for). It reports
// whether it did.
func fetchShared(ctx context.Context, kind string, m Model, r map[string]string, dst, export string) bool {
	if r == nil {
		return false
	}
	return fetchConverted(ctx, placeOf(kind, m, r["target"], export, r), dst, labelLog{sharedLog, filepath.Base(dst), export})
}

// putShared describes dir, made here and whole, as the conversion of kind of model m that
// recipe r describes, and puts it in the sources that are written back to.
func putShared(kind string, m Model, r map[string]string, dir, export string) {
	if r == nil {
		return
	}
	putConverted(placeOf(kind, m, r["target"], export, r), dir, sharedLog)
}

type modelKey struct{}

// WithModel says that the Core AI exports made under ctx are model m's (the key of the shared
// layers, coreAIRecipe). Without it, the model is the one whose directory holds the export.
func WithModel(ctx context.Context, m Model) context.Context {
	return context.WithValue(ctx, modelKey{}, m)
}

// coreAIRecipe is what the Core AI export is made from, the key of the shared layers: the model
// (WithModel, else the one whose directory holds it: modelOfDir), its repository and the files'
// SHA-256 the definitions pin (else the commit of its files), the repositories it reads with it,
// the export's name (with the directory it is in when that is an export too: a size of depth's)
// and the lowering's version. nil when the model is not known, or neither the files' digests nor
// their commit are.
func coreAIRecipe(ctx context.Context, export string) (Model, map[string]string) {
	name, ok := coreAIExportName(export)
	if !ok {
		return Model{}, nil
	}
	m, r := modelRecipe(ctx, export)
	if r == nil {
		return Model{}, nil
	}
	r["kind"], r["target"], r["export"], r["lowering"] = CoreAIKind, CoreAITarget, name, fmt.Sprint(ailower.FormatVersion)
	return m, r
}

// coreAIExportName is the name of the Core AI export in the shared layers: its directory's, with
// the directory it is in when that is an export too (a size of depth's).
func coreAIExportName(export string) (string, bool) {
	name := filepath.Base(export)
	if parent := filepath.Dir(export); filepath.Ext(parent) == ".coreai" {
		return filepath.Base(parent) + "/" + name, true
	}
	return name, filepath.Ext(export) == ".coreai"
}

// PrebuiltKind is the kind of entry of the prebuilt programs of a Core AI export for an
// architecture of the Neural Engine, packed (coreai.PackedRoot): each X.ARCH.aimodelc of
// coreai-build without its weights, which the device makes again from the export's bytecode.
// They are in converted/MODEL/coreai-ARCH/EXPORT/VERSION as one file (coreai.BundleFile;
// docs/cache-sharing.md 7.9).
const PrebuiltKind = "coreai-prebuilt"

// prebuiltFormat is the version of what coreai.Pack writes, part of the key.
const prebuiltFormat = 2

func init() { coreai.FetchPrebuilt = FetchPrebuilt }

// prebuiltPlace is the place of the packed prebuilt programs of the export for arch. Its key is
// what they are compiled from, the programs' bytecode (coreai.ProgramsDigest), not what the
// export was converted from: a machine whose export is the same bytes finds them whatever
// made it. The model is the one of the export's entry.json (from the shared layers, or made
// here and put in them), else the one whose directory holds it.
func prebuiltPlace(export, arch string) (place, bool) {
	name, ok := coreAIExportName(export)
	if !ok {
		return place{}, false
	}
	digest := coreai.ProgramsDigest(export)
	if digest == "" {
		return place{}, false
	}
	var m Model
	if e, err := shared.ReadEntry(export); err == nil && e.Model != "" {
		m = modelNamed(e.Model)
	} else if m, _, ok = modelOfDir(export); !ok {
		return place{}, false
	}
	r := map[string]string{"kind": PrebuiltKind, "export": name, "arch": arch, "programs": digest, "format": fmt.Sprint(prebuiltFormat)}
	return placeOf(PrebuiltKind, m, CoreAITarget+"-"+arch, name, r), true
}

// FetchPrebuilt makes the packed prebuilt programs of the export for arch (coreai.PackedRoot)
// from the first source that has them (coreai.FetchPrebuilt: the backend asks when it would
// compile a program of the export).
func FetchPrebuilt(ctx context.Context, export, arch string) {
	p, ok := prebuiltPlace(export, arch)
	if !ok {
		return
	}
	root := coreai.PackedRoot(export, arch)
	dir := bundleDir(root)
	if e, err := shared.ReadEntry(dir); err == nil && p.takes(e) {
		if _, err := os.Stat(root); err == nil {
			return
		}
	} else if !fetchFrom(ctx, p, dir, labelLog{sharedLog, filepath.Base(dir), filepath.Base(export) + " (prebuilt for " + arch + ")"}) { // nothing is made here after a miss: no early puts
		return
	}
	if err := coreai.Unbundle(filepath.Join(dir, coreai.BundleFile), root); err != nil {
		fmt.Fprintf(sharedLog, "%s (prebuilt for %s): %v; compiled on the device\n", filepath.Base(export), arch, err)
	}
}

// bundleDir is the entry of the shared layers that holds the bundle of the packed prebuilt
// programs at root, beside them (kept: the queue may put it in a source before the one it
// came from).
func bundleDir(root string) string { return root + ".shared" }

// PutPrebuilt puts the packed prebuilt programs of the export for arch (coreai.PackedRoot, made
// here by hwnpu-dev prebuild) in the sources that are written back to, in the background, as
// one file (coreai.Bundle).
func PutPrebuilt(export, arch string) error {
	p, ok := prebuiltPlace(export, arch)
	if !ok {
		return fmt.Errorf("%s: not a Core AI export of a model of the definitions", export)
	}
	if len(shared.Writebacks()) == 0 {
		return errors.New("no source is written back to (HWNPU_MODEL_SOURCE...?writeback=true)")
	}
	root := coreai.PackedRoot(export, arch)
	dir := bundleDir(root)
	os.RemoveAll(dir)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	if err := coreai.Bundle(root, filepath.Join(dir, coreai.BundleFile)); err != nil {
		return err
	}
	putConverted(p, dir, labelLog{sharedLog, filepath.Base(dir), filepath.Base(export) + " (prebuilt for " + arch + ")"})
	return nil
}

// modelRecipe is the model's part of a key of the shared layers for what is made from it in
// path (an export, the tables): the model (WithModel, else the one whose directory holds path:
// modelOfDir), its repository and the files' SHA-256 the definitions pin (else the commit of its
// files) and the repositories it reads with it. nil when the model is not known, or neither the
// files' digests nor their commit are.
func modelRecipe(ctx context.Context, path string) (Model, map[string]string) {
	m, ok := ctx.Value(modelKey{}).(Model)
	if !ok {
		if m, _, ok = modelOfDir(path); !ok {
			return Model{}, nil
		}
	}
	base, _, _ := strings.Cut(m.Repo, "@")
	r := map[string]string{"source": base}
	if m.File != "" {
		r["file"] = m.File
	}
	for f, d := range m.SHA256 {
		r["sha256:"+f] = strings.ToLower(d)
	}
	for role, repo := range m.With {
		r["with:"+role] = repo
	}
	if len(m.SHA256) == 0 {
		commit := commitOf(ctx, m)
		if commit == "" {
			return Model{}, nil
		}
		r["commit"] = commit
	}
	return m, r
}

// coreAICommits caches sourceCommit by the model's source (it may ask the hub).
var coreAICommits sync.Map

func commitOf(ctx context.Context, m Model) string {
	if c, ok := coreAICommits.Load(m.Source()); ok {
		return c.(string)
	}
	c := sourceCommit(ctx, m)
	coreAICommits.Store(m.Source(), c)
	return c
}

// modelOfDir is the model of the catalog whose directory (Model.DirName) holds path, among its
// parents up to three levels (an export, or a directory in one), and that directory. Of several
// models of one directory, the one whose file (Model.File, its extension left out) the name of
// path, or of a directory it is in, starts with (Anima-Turbo's anima-turbo-v1.1-...coreai, not
// Anima-Aesthetic's), else the first by name: the key is the same from every one.
func modelOfDir(path string) (Model, string, bool) {
	d := filepath.Dir(path)
	for range 3 {
		var found []Model
		for _, m := range Catalog {
			if m.DirName() == filepath.Base(d) {
				found = append(found, m)
			}
		}
		if len(found) > 0 {
			slices.SortFunc(found, func(a, b Model) int { return strings.Compare(a.Name, b.Name) })
			if rel, err := filepath.Rel(d, path); err == nil {
				first, _, _ := strings.Cut(filepath.ToSlash(rel), "/")
				for _, m := range found {
					if stem := strings.TrimSuffix(m.File, filepath.Ext(m.File)); stem != "" && strings.HasPrefix(first, stem) {
						return m, d, true
					}
				}
			}
			return found[0], d, true
		}
		if p := filepath.Dir(d); p != d {
			d = p
		}
	}
	return Model{}, "", false
}

// TablesKind is the kind of entry of the tables a command reads rows of (Tables).
const TablesKind = "tables"

// tablesVersion is the version of what llm.WriteTables writes, part of the tables' key.
const tablesVersion = 1

// tablesRecipe is what the tables in out are made from, their key in the shared layers: the
// model's part (modelRecipe), the architecture and the tables' names.
func tablesRecipe(ctx context.Context, out string, a *llm.Arch, names []string) (Model, map[string]string) {
	m, r := modelRecipe(ctx, out)
	if r == nil {
		return Model{}, nil
	}
	sorted := slices.Sorted(slices.Values(names))
	r["kind"], r["arch"], r["tables"], r["version"] = TablesKind, a.Name, strings.Join(sorted, ","), fmt.Sprint(tablesVersion)
	return m, r
}

// fetchTables makes the tables file out from the shared layers (an entry of a directory that
// holds it) and reports whether one had it.
func fetchTables(ctx context.Context, m Model, r map[string]string, out string) bool {
	if r == nil {
		return false
	}
	tmp := out + ".shared"
	defer os.RemoveAll(tmp)
	if !fetchShared(ctx, TablesKind, m, r, tmp, filepath.Base(out)) {
		return false
	}
	return os.Rename(filepath.Join(tmp, filepath.Base(out)), out) == nil
}

// putTables puts the tables file out, written here, in the shared layers that are written: a
// directory of the work area holds it (a hard link, else a copy) until the queue has put it.
func putTables(ctx context.Context, m Model, r map[string]string, out string) {
	if r == nil || len(shared.Writebacks()) == 0 {
		return
	}
	dir := cache.Sub("work", "shared", shared.Fingerprint(r))
	os.RemoveAll(dir)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return
	}
	dst := filepath.Join(dir, filepath.Base(out))
	if err := os.Link(out, dst); err != nil {
		if err := copyStream(dst, out); err != nil {
			fmt.Fprintf(sharedLog, "%s: %v; not put in the shared layers\n", filepath.Base(out), err)
			return
		}
	}
	putShared(TablesKind, m, r, dir, filepath.Base(out))
}

// copyStream copies the file src to dst a piece at a time.
func copyStream(dst, src string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	f, err := os.Create(dst)
	if err != nil {
		return err
	}
	if _, err := io.Copy(f, in); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// labelLog writes to w with the name from in what is said put as to (the downloading line of a
// directory named for the download, not for what it is).
type labelLog struct {
	w        io.Writer
	from, to string
}

func (l labelLog) Write(p []byte) (int, error) {
	if l.from != l.to {
		p = []byte(strings.ReplaceAll(string(p), l.from, l.to))
	}
	if _, err := l.w.Write(p); err != nil {
		return 0, err
	}
	return len(p), nil
}
