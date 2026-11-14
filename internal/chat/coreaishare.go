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
	_, r := coreAIRecipe(ctx, export)
	return fetchShared(ctx, CoreAIKind, r, export, filepath.Base(export))
}

func (coreAIShares) Put(ctx context.Context, export string) {
	m, r := coreAIRecipe(ctx, export)
	putShared(ctx, CoreAIKind, m, r, export, filepath.Base(export))
}

// fetchShared makes dst (a directory) from the first shared layer that has the entry of kind
// that recipe r describes, and reports whether one had it (what says so names it what). The
// layers before it keep a copy when they fill. A nil r (what it is made from is not known) is in
// none.
func fetchShared(ctx context.Context, kind string, r map[string]string, dst, what string) bool {
	layers := shared.Layers()
	if r == nil || !slices.ContainsFunc(layers, func(l shared.Layer) bool { return l.Holds(shared.ScopePortable) }) {
		return false
	}
	i, err := shared.FetchEntry(ctx, layers, shared.ScopePortable, kind, shared.Fingerprint(r), dst, labelLog{sharedLog, filepath.Base(dst), what})
	if err != nil {
		if !errors.Is(err, shared.ErrNotFound) {
			fmt.Fprintf(sharedLog, "the shared layers: %v; making %s here\n", err, what)
		}
		return false
	}
	for _, l := range layers[:i] {
		if l.Fill && l.Holds(shared.ScopePortable) {
			shared.PutLater(l, dst)
		}
	}
	return true
}

// putShared describes dir, made here and whole, as the entry of kind of model m that recipe r
// describes (its entry.json) and puts it in the shared layers that are written, in the
// background (what says so names it what).
func putShared(ctx context.Context, kind string, m Model, r map[string]string, dir, what string) {
	var to []shared.Layer
	for _, l := range shared.Layers() {
		if l.Write && l.Holds(shared.ScopePortable) {
			to = append(to, l)
		}
	}
	if len(to) == 0 || r == nil {
		return
	}
	target := CoreAITarget
	if kind != CoreAIKind {
		target = ""
	}
	e := shared.Entry{Key: shared.Fingerprint(r), Kind: kind, Scope: shared.ScopePortable, Model: m.Name, Target: target,
		Recipe: r, MadeBy: map[string]string{"host": hostName()}}
	if _, err := shared.Describe(dir, e); err != nil {
		fmt.Fprintf(sharedLog, "%s: %s: %v\n", what, shared.EntryFile, err)
		return
	}
	for _, l := range to {
		fmt.Fprintf(sharedLog, "putting %s in %s in the background\n", what, l)
		shared.PutLater(l, dir)
	}
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
	name := filepath.Base(export)
	if parent := filepath.Dir(export); filepath.Ext(parent) == ".coreai" {
		name = filepath.Base(parent) + "/" + name
	} else if filepath.Ext(export) != ".coreai" {
		return Model{}, nil
	}
	m, r := modelRecipe(ctx, export)
	if r == nil {
		return Model{}, nil
	}
	r["kind"], r["target"], r["export"], r["lowering"] = CoreAIKind, CoreAITarget, name, fmt.Sprint(ailower.FormatVersion)
	return m, r
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
// models of one repository (its formats), the first by name: the key is the same from every one.
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
func fetchTables(ctx context.Context, r map[string]string, out string) bool {
	if r == nil {
		return false
	}
	tmp := out + ".shared"
	defer os.RemoveAll(tmp)
	if !fetchShared(ctx, TablesKind, r, tmp, filepath.Base(out)) {
		return false
	}
	return os.Rename(filepath.Join(tmp, filepath.Base(out)), out) == nil
}

// putTables puts the tables file out, written here, in the shared layers that are written: a
// directory of the work area holds it (a hard link, else a copy) until the queue has put it.
func putTables(ctx context.Context, m Model, r map[string]string, out string) {
	if r == nil || !slices.ContainsFunc(shared.Layers(), func(l shared.Layer) bool { return l.Write && l.Holds(shared.ScopePortable) }) {
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
	putShared(ctx, TablesKind, m, r, dir, filepath.Base(out))
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
