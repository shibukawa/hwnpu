package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"
	"sync"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/console"
)

// What is converted goes in a source as a folder (docs/cache-sharing.md 7):
//
//	converted/MODEL/TARGET/EXPORT/VERSION/FILE...
//	converted/MODEL/TARGET/EXPORT/VERSION/entry.json   written last: a folder with it is whole
//
// EXPORT is the name of the export here (Qwen3.5-0.8B.amd-npu2-q6, LFM3-1.7B.coreai). VERSION
// is vN for version N of the definitions (models.yaml's targets), vN-HASH for a model in
// development (develop: true) and v0-HASH for one with no version: HASH is the first 8 digits of
// the recipe's fingerprint (Entry.Key).

// EntryPath is the folder of an entry in a source.
func EntryPath(model, target, export, version string) string {
	return path.Join("converted", model, target, export, version)
}

// VersionDir is the VERSION of an entry: vN for a version that is published (not develop),
// else vN-HASH of its key.
func VersionDir(version int, develop bool, key string) string {
	if version > 0 && !develop {
		return fmt.Sprintf("v%d", version)
	}
	return fmt.Sprintf("v%d-%s", version, key[:min(8, len(key))])
}

// logf writes a line on log (nil for nowhere).
func logf(log io.Writer) func(string, ...any) {
	if log == nil {
		return nil
	}
	return func(format string, args ...any) { fmt.Fprintf(log, format+"\n", args...) }
}

// bar shows a download from a source on log.
func bar(log io.Writer, what string) (*console.Bar, func(done, total int64)) {
	b := console.NewBar(log, what)
	return b, func(done, total int64) {
		if total > 0 {
			b.Set(float64(done)/float64(total), fmt.Sprintf("%.2f / %.2f GB", float64(done)/1e9, float64(total)/1e9))
		} else {
			b.Set(-1, fmt.Sprintf("%.2f GB", float64(done)/1e9))
		}
	}
}

// FetchEntry looks for the entry at rel (EntryPath) in the sources, in order, that match takes,
// and makes it in dst (through dst.part, renamed when whole and checked) from the first that has
// it. It returns the index of that source; -1 and ErrNotFound when none has it.
func FetchEntry(ctx context.Context, ss []Source, rel string, match func(Entry) bool, dst string, log io.Writer) (int, error) {
	var errs []error
	for i, s := range ss {
		e, err := ReadEntryAt(ctx, s, rel)
		if errors.Is(err, ErrNotFound) {
			continue
		}
		if err == nil && !match(e) {
			continue
		}
		if err == nil {
			err = materialize(ctx, s, e, rel, dst, log)
		}
		if err == nil {
			return i, nil
		}
		errs = append(errs, fmt.Errorf("%s: %w", s, err))
		if log != nil {
			fmt.Fprintf(log, "%s: %v; looking further\n", s, err)
		}
	}
	if len(errs) > 0 {
		return -1, errors.Join(append(errs, ErrNotFound)...)
	}
	return -1, ErrNotFound
}

// ReadEntryAt reads the entry at rel in s.
func ReadEntryAt(ctx context.Context, s Source, rel string) (Entry, error) {
	b, err := s.Store.Read(ctx, rel+"/"+EntryFile)
	if err != nil {
		return Entry{}, err
	}
	return parseEntry(b)
}

// materialize makes the entry e at rel of s in dst.
func materialize(ctx context.Context, s Source, e Entry, rel, dst string, log io.Writer) error {
	part := dst + ".part"
	if err := os.MkdirAll(part, 0o755); err != nil {
		return err
	}
	total := e.Bytes()
	if err := cache.Room(total, dst, logf(log)); err != nil {
		return err
	}
	br, set := bar(log, fmt.Sprintf("downloading %s from %s", filepath.Base(dst), s))
	var done int64
	for _, f := range e.Files {
		p := filepath.Join(part, filepath.FromSlash(f.Path))
		if fi, err := os.Stat(p); err == nil && fi.Size() == f.Size {
			done += f.Size // a download cut off before had it
			continue
		}
		base := done
		if err := getChecked(ctx, s.Store, rel+"/"+f.Path, f.SHA256, p, func(n, _ int64) { set(base+n, total) }); err != nil {
			br.Done("failed")
			return err
		}
		done += f.Size
	}
	if err := WriteEntry(part, e); err != nil {
		br.Done("failed")
		return err
	}
	os.RemoveAll(dst)
	if err := os.Rename(part, dst); err != nil {
		br.Done("failed")
		return err
	}
	br.Done(fmt.Sprintf("%.2f GB in %s", float64(total)/1e9, console.Duration(br.Elapsed())))
	return nil
}

// PutEntry puts the entry of dir (its EntryFile, with its Path) in s: unless s has it (the same
// key), its entry.json is removed first, the files put, and the entry.json written last.
func PutEntry(ctx context.Context, s Source, dir string) error {
	e, err := ReadEntry(dir)
	if err != nil {
		return err
	}
	rel := e.Path
	if !SafeKey(rel) || !strings.HasPrefix(rel, "converted/") {
		return fmt.Errorf("store: %s: %s has no path in a source (converted/...)", dir, EntryFile)
	}
	unlock, err := lockWork(ctx, s.String()+"|"+rel)
	if err != nil {
		return err
	}
	defer unlock()
	if old, err := ReadEntryAt(ctx, s, rel); err == nil && old.Key == e.Key {
		return nil
	}
	if err := s.Store.Delete(ctx, rel+"/"+EntryFile); err != nil {
		return err
	}
	// the files side by side (an export of Core AI is many small ones, each a PUT and a MOVE on
	// WebDAV), entry.json after them all
	err = each(ctx, PutStreams, len(e.Files), func(ctx context.Context, i int) error {
		f := e.Files[i]
		return s.Store.Put(ctx, rel+"/"+f.Path, filepath.Join(dir, filepath.FromSlash(f.Path)))
	})
	if err != nil {
		return err
	}
	b, err := json.MarshalIndent(e, "", "  ")
	if err != nil {
		return err
	}
	return s.Store.Write(ctx, rel+"/"+EntryFile, append(b, '\n'))
}

// ListEntries lists the folders of the entries s holds (EntryPath), in order.
func ListEntries(ctx context.Context, s Source) ([]string, error) {
	keys, err := s.Store.List(ctx, "converted/")
	if err != nil {
		return nil, err
	}
	var out []string
	for _, k := range keys {
		if rel, ok := strings.CutSuffix(k, "/"+EntryFile); ok {
			out = append(out, rel)
		}
	}
	slices.Sort(out)
	return out, nil
}

// PutStreams is how many puts go to a source at a time: the files of an entry, the jobs of the
// queue.
var PutStreams = 4

// each runs f(ctx, 0..n-1), at most k at a time, and returns the first error: the rest are then
// not started, and the ctx of those running ends.
func each(parent context.Context, k, n int, f func(context.Context, int) error) error {
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	var (
		wg    sync.WaitGroup
		mu    sync.Mutex
		first error
	)
	sem := make(chan struct{}, max(k, 1))
	for i := 0; i < n && ctx.Err() == nil; i++ {
		select {
		case sem <- struct{}{}:
		case <-ctx.Done():
			continue
		}
		wg.Add(1)
		go func() {
			defer func() { <-sem; wg.Done() }()
			if err := f(ctx, i); err != nil {
				mu.Lock()
				if first == nil {
					first = err
					cancel()
				}
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	if first == nil {
		first = parent.Err()
	}
	return first
}

// lockWork is the lock of one piece of work on a source (cache.Lock).
func lockWork(ctx context.Context, name string) (func(), error) {
	return cache.Lock(ctx, "store:"+name, name, nil)
}
