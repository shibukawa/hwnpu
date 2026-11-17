package store

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// A conversion made here goes in the sources written back to while it is made: the files of its
// directory that are written and left alone for a while (EarlySettle) are put as they come
// (StartEarly), and PutEntry, when the conversion is whole, puts only those that changed or came
// after, then entry.json (docs/cache-sharing-design.md 7.1). The converters are not told: what is put
// early and changes after is put again, and what is put early and is not in the entry at the
// end (a file the converter removed) is removed. A conversion that stops leaves what was put of it
// without an entry.json, as one cut off (hwnpu-dev cache gc --orphans).

// EarlySettle is how long a file is left alone before it is taken for written; EarlyPoll how
// often the directory is looked at.
var (
	EarlySettle = 3 * time.Second
	EarlyPoll   = 2 * time.Second
)

// stamp is what a file was when it was put: its size and time.
type stamp struct {
	size int64
	mod  time.Time
}

func stampOf(fi os.FileInfo) stamp { return stamp{fi.Size(), fi.ModTime()} }

type early struct {
	s        Source
	rel, dir string
	start    time.Time
	stop     chan struct{}
	done     chan struct{}
	mu       sync.Mutex
	put      map[string]stamp // the files put: slash paths in dir
	// unmark removes the marker that tells Clean the folder at rel is being written (MarkWriting)
	unmark func()
}

var (
	earlyMu sync.Mutex
	earlies = map[string]*early{} // by source and directory
)

func earlyID(s Source, dir string) string { return s.String() + "|" + filepath.Clean(dir) }

// StartEarly puts the files of dir, the conversion of the entry at rel being made here, in s as
// they are written, until PutEntry of dir (or the process ends). It does nothing for a source not
// written back to, an entry that goes packed (Packs: its pack is made when it is whole), or a
// source that holds an entry at rel already (whose files are not put over before PutEntry takes
// its entry.json away).
func StartEarly(s Source, rel, dir string) {
	if !s.Writeback || !SafeKey(rel) || !strings.HasPrefix(rel, "converted/") || Packs(rel) {
		return
	}
	ctx := context.Background()
	if _, ok, err := s.Store.Has(ctx, rel+"/"+EntryFile); ok || err != nil {
		return
	}
	id := earlyID(s, dir)
	earlyMu.Lock()
	defer earlyMu.Unlock()
	if earlies[id] != nil {
		return
	}
	// the folder is not whole until PutEntry writes entry.json, which can be long after the
	// first files (an export for AMD's NPU compiles for minutes): marked meanwhile, so that Clean
	// leaves it
	unmark, err := MarkWriting(ctx, s, rel)
	if err != nil {
		logQueue("%s: not put in %s while it is made: %v", filepath.Base(dir), s, err)
		return
	}
	e := &early{s: s, rel: rel, dir: dir, start: time.Now().Add(-time.Second), stop: make(chan struct{}),
		done: make(chan struct{}), put: map[string]stamp{}, unmark: unmark}
	earlies[id] = e
	go e.run(ctx)
}

// endEarly stops the early puts of dir in s and returns the files they put (nil: none ran).
func endEarly(s Source, dir string) map[string]stamp {
	id := earlyID(s, dir)
	earlyMu.Lock()
	e := earlies[id]
	delete(earlies, id)
	earlyMu.Unlock()
	if e == nil {
		return nil
	}
	close(e.stop)
	<-e.done
	e.unmark() // PutEntry marks the folder from here
	e.mu.Lock()
	defer e.mu.Unlock()
	if n := len(e.put); n > 0 {
		logQueue("%s: %d files put in %s while it was made", filepath.Base(dir), n, s)
	}
	return e.put
}

func (e *early) run(ctx context.Context) {
	defer close(e.done)
	var wg sync.WaitGroup
	defer wg.Wait()
	sem := make(chan struct{}, PutStreams)
	busy := map[string]bool{}
	var busyMu sync.Mutex
	tick := time.NewTicker(EarlyPoll)
	defer tick.Stop()
	for {
		select {
		case <-e.stop:
			return
		case <-tick.C:
		}
		for _, f := range e.settled(busy, &busyMu) {
			select {
			case <-e.stop:
				return
			case sem <- struct{}{}:
			}
			wg.Add(1)
			go func() {
				defer func() { <-sem; wg.Done() }()
				e.putOne(ctx, f)
				busyMu.Lock()
				delete(busy, f.path)
				busyMu.Unlock()
			}()
		}
	}
}

type settledFile struct {
	path string // slash path in dir
	st   stamp
}

// settled are the files of dir written since the start, left alone EarlySettle, not put as they
// are now and not being put (marked busy).
func (e *early) settled(busy map[string]bool, busyMu *sync.Mutex) []settledFile {
	var out []settledFile
	now := time.Now()
	filepath.WalkDir(e.dir, func(p string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		name := d.Name()
		if strings.HasPrefix(name, ".") || strings.HasSuffix(name, ".tmp") || strings.HasSuffix(name, ".part") || name == EntryFile {
			return nil
		}
		fi, err := d.Info()
		if err != nil || !fi.Mode().IsRegular() || fi.ModTime().Before(e.start) || now.Sub(fi.ModTime()) < EarlySettle {
			return nil
		}
		r, err := filepath.Rel(e.dir, p)
		if err != nil {
			return nil
		}
		r = filepath.ToSlash(r)
		st := stampOf(fi)
		e.mu.Lock()
		was, ok := e.put[r]
		e.mu.Unlock()
		busyMu.Lock()
		defer busyMu.Unlock()
		if ok && was == st || busy[r] {
			return nil
		}
		busy[r] = true
		out = append(out, settledFile{r, st})
		return nil
	})
	return out
}

func (e *early) putOne(ctx context.Context, f settledFile) {
	p := filepath.Join(e.dir, filepath.FromSlash(f.path))
	if err := e.s.Store.Put(ctx, e.rel+"/"+f.path, p); err != nil {
		return // PutEntry puts it
	}
	st := f.st
	if fi, err := os.Stat(p); err != nil || stampOf(fi) != st {
		st = stamp{size: -1} // changed or removed while it was put: PutEntry puts it again or removes it
	}
	e.mu.Lock()
	e.put[f.path] = st
	e.mu.Unlock()
}
