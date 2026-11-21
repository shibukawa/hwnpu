package store

import (
	"context"
	"maps"
	"path"
	"slices"
	"strings"
)

// GC is what hwnpu-dev models gc removes from a source: the entries chosen (their folders). What
// a writeback cut off left is hwnpu cache clean's (PlanClean). Nothing is removed until Apply.
type GC struct {
	Entries []Entry  // the entries removed
	Bytes   int64    // what the entries' files held
	keys    []string // every file removed, entry.json first in each folder
}

// PlanGC works out a GC of s: drop chooses the entries to remove.
func PlanGC(ctx context.Context, s Source, drop func(Entry) bool) (GC, error) {
	var g GC
	keys, err := s.Store.List(ctx, "converted/")
	if err != nil {
		return g, err
	}
	entries := map[string]bool{}
	for _, k := range keys {
		if rel, ok := strings.CutSuffix(k, "/"+EntryFile); ok {
			entries[rel] = true
		}
	}
	gone := map[string]bool{}
	for rel := range entries {
		e, err := ReadEntryAt(ctx, s, rel)
		if err != nil {
			continue // one that cannot be read is left
		}
		if e.Path == "" {
			e.Path = rel
		}
		if drop(e) {
			g.Entries = append(g.Entries, e)
			g.Bytes += e.Bytes()
			gone[rel] = true
			g.keys = append(g.keys, rel+"/"+EntryFile)
		}
	}
	for _, k := range keys {
		if strings.HasSuffix(k, "/"+EntryFile) {
			continue
		}
		owner := ""
		for d := path.Dir(k); d != "." && d != "converted"; d = path.Dir(d) {
			if entries[d] {
				owner = d
				break
			}
		}
		if owner != "" && gone[owner] {
			g.keys = append(g.keys, k)
		}
	}
	return g, nil
}

// Apply removes what g names from s: each entry's entry.json first, so that no reader takes a
// folder that is going. Then the folders it emptied go, and those above them left empty (a
// folder of WebDAV stays when its files are deleted: removeFolder).
func (g GC) Apply(ctx context.Context, s Source) error {
	for _, k := range g.keys {
		if err := s.Store.Delete(ctx, k); err != nil {
			return err
		}
	}
	dirs := map[string]bool{}
	for _, e := range g.Entries {
		dirs[e.Path] = true
	}
	for _, d := range slices.Sorted(maps.Keys(dirs)) {
		removeFolder(ctx, s.Store, d)
	}
	return nil
}
