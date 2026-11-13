package store

import (
	"context"
	"encoding/json"
	"strings"
)

// GC is what hwnpu-dev cache gc removes from a layer: the entries chosen and the blobs that only
// they named, and with orphans the blobs that nothing names (docs/cache-sharing.md 7). Nothing is
// removed until Apply. A blob a push or a fill is putting in the layer at the same time is named
// by nothing until its entry or index is written: run gc with orphans when nothing writes there.
type GC struct {
	Entries []Entry  // the entries removed
	Blobs   []string // the blobs removed, by SHA-256
	Bytes   int64    // what the blobs held (an orphan's size when the store says it)
}

// PlanGC works out a GC of l: drop chooses the entries to remove.
func PlanGC(ctx context.Context, l Layer, drop func(Entry) bool, orphans bool) (GC, error) {
	var g GC
	keep := map[string]bool{}
	dropped := map[string]int64{}
	kinds, err := ListEntries(ctx, l)
	if err != nil {
		return g, err
	}
	for _, kk := range kinds {
		e, err := ReadLayerEntry(ctx, l, kk[0], kk[1])
		if err != nil {
			return g, err // an entry that cannot be read: its blobs are not known, so nothing goes
		}
		if drop(e) {
			g.Entries = append(g.Entries, e)
			for _, f := range e.Files {
				dropped[f.SHA256] = f.Size
			}
			continue
		}
		for _, f := range e.Files {
			keep[f.SHA256] = true
		}
	}
	ups, err := l.Store.List(ctx, "upstream/")
	if err != nil {
		return g, err
	}
	for _, k := range ups {
		b, err := l.Store.Read(ctx, k)
		if err != nil {
			return g, err
		}
		var u upstreamFile
		if json.Unmarshal(b, &u) == nil {
			keep[u.SHA256] = true
		}
	}
	blobs, err := l.Store.List(ctx, "blobs/sha256/")
	if err != nil {
		return g, err
	}
	for _, k := range blobs {
		sum := k[strings.LastIndexByte(k, '/')+1:]
		if keep[sum] || blobKey(sum) != k {
			continue
		}
		if n, ok := dropped[sum]; ok {
			g.Blobs = append(g.Blobs, sum)
			g.Bytes += n
		} else if orphans {
			g.Blobs = append(g.Blobs, sum)
			if n, ok, err := l.Store.Has(ctx, k); err == nil && ok {
				g.Bytes += n
			}
		}
	}
	return g, nil
}

// Apply removes what g names from l: the entries first, so that none names a blob that is gone.
func (g GC) Apply(ctx context.Context, l Layer) error {
	for _, e := range g.Entries {
		if err := l.Store.Delete(ctx, entryKey(e.Kind, e.Key)); err != nil {
			return err
		}
	}
	for _, sum := range g.Blobs {
		if err := l.Store.Delete(ctx, blobKey(sum)); err != nil {
			return err
		}
	}
	return nil
}
