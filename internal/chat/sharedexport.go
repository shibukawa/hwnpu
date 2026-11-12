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

// SharedExport makes the export out (a directory) that recipe describes: from the first shared
// layer that has it (docs/cache-sharing.md 7), else with build. What build made gets its
// entry.json and goes in the layers that are written; what a layer had goes in the layers
// before it that are filled. A nil recipe (what it is made from is not known: an unpinned model
// with the hub out of reach) builds without the layers.
func SharedExport(ctx context.Context, kind string, recipe map[string]string, out string, log io.Writer, build func() error) error {
	if recipe == nil {
		return build()
	}
	if log == nil {
		log = io.Discard
	}
	r := maps.Clone(recipe)
	r["kind"] = kind
	key := shared.Fingerprint(r)
	unlock, err := cache.Lock(ctx, out+":layers", "the export "+filepath.Base(out), log)
	if err != nil {
		return err
	}
	defer unlock()
	if e, err := shared.ReadEntry(out); err == nil && e.Key == key {
		return nil
	}
	layers := shared.Layers()
	if i, err := shared.FetchEntry(ctx, layers, shared.ScopePortable, kind, key, out, log); err == nil {
		os.WriteFile(out+".done", nil, 0o644)
		for _, l := range layers[:i] {
			if l.Fill && l.Holds(shared.ScopePortable) {
				shared.PutLater(l, out)
			}
		}
		return nil
	} else if !errors.Is(err, shared.ErrNotFound) {
		fmt.Fprintf(log, "the shared layers: %v; making it here\n", err)
	}
	if err := build(); err != nil {
		return err
	}
	e := shared.Entry{Key: key, Kind: kind, Scope: shared.ScopePortable, Model: recipe["model"], Target: recipe["target"], Recipe: r,
		MadeBy: map[string]string{"host": hostName()}}
	if _, err := shared.Describe(out, e); err != nil {
		fmt.Fprintf(log, "%s: %v\n", shared.EntryFile, err)
		return nil
	}
	for _, l := range layers {
		if l.Write && l.Holds(shared.ScopePortable) {
			fmt.Fprintf(log, "putting %s in %s in the background\n", filepath.Base(out), l)
			shared.PutLater(l, out)
		}
	}
	return nil
}

func hostName() string { h, _ := os.Hostname(); return h }

// sourceCommit is the commit model m's files come from: its revision when that is a commit, else
// the one the hub says its config.json is of; "" when neither is known.
func sourceCommit(ctx context.Context, m Model) string {
	if m.SourceName != "" {
		return ""
	}
	if commitRE.MatchString(m.Revision) {
		return m.Revision
	}
	if c, ok := servedCommit(m.Source()); ok {
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
