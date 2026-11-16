package store

import (
	"context"
	"path"
	"strings"
)

// GC is what hwnpu-dev cache gc removes from a source: the entries chosen (their folders), and
// with orphans what a writeback cut off left under converted/ (Clean's: the folders with no
// entry.json, the temporary files of uploads; a folder being written is left). Nothing is
// removed until Apply.
type GC struct {
	Entries []Entry  // the entries removed
	Orphans []string // the files removed that no entry has
	Bytes   int64    // what the entries' files held
	keys    []string // every file removed, entry.json first in each folder
}

// PlanGC works out a GC of s: drop chooses the entries to remove.
func PlanGC(ctx context.Context, s Source, drop func(Entry) bool, orphans bool) (GC, error) {
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
	if orphans {
		c, err := PlanClean(ctx, s)
		if err != nil {
			return g, err
		}
		for _, f := range c.Folders {
			if strings.HasPrefix(f.Path, "converted/") && !gone[f.Path] {
				g.Orphans = append(g.Orphans, f.Files...)
				g.keys = append(g.keys, f.Files...)
			}
		}
	}
	return g, nil
}

// Apply removes what g names from s: each entry's entry.json first, so that no reader takes a
// folder that is going.
func (g GC) Apply(ctx context.Context, s Source) error {
	for _, k := range g.keys {
		if err := s.Store.Delete(ctx, k); err != nil {
			return err
		}
	}
	return nil
}
