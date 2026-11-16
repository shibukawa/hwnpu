package store

import (
	"context"
	"errors"
	"path"
	"slices"
	"strings"
	"time"
)

// Clean is what a writeback cut off left in a source (hwnpu cache clean, docs/cache-sharing-design.md
// 7.1). A folder is whole or it is not: an entry's folder (converted/MODEL/TARGET/EXPORT/VERSION)
// with no entry.json is not, and goes whole; in a repository's folder (original/OWNER/NAME) the
// files its hwnpu-lock.json does not name go, all of them with no lock. The temporary files of
// an upload (.put-) and the markers of writebacks no longer going on (.writing-) go too. A
// folder with a marker written in the last WritingStale, or a file changed then (a writeback of
// a hwnpu before the markers), is being written and is left as it is, as is everything outside
// converted/ and original/. In S3, the multipart uploads under them initiated more than
// WritingStale ago, which a Put cut off left and S3 keeps the parts of unlisted, are aborted.
type Clean struct {
	Source  string // the source's String()
	Folders []CleanFolder
	Uploads []CleanUpload
	src     Source
}

// CleanUpload is a multipart upload in progress Clean aborts.
type CleanUpload struct {
	Key       string // the object it was to make
	ID        string
	Initiated time.Time
}

// uploadCleaner is a Store with uploads in parts that are left when cut off (S3).
type uploadCleaner interface {
	uploadsLeft(ctx context.Context, prefix string, before time.Time) ([]CleanUpload, error)
	abortUpload(ctx context.Context, u CleanUpload) error
}

// CleanFolder is a folder Clean removes files of.
type CleanFolder struct {
	Path  string   // converted/MODEL/TARGET/EXPORT/VERSION or original/OWNER/NAME
	Whole bool     // not whole (no entry.json, no lock): every file goes
	Files []string // the keys removed, entry.json or the lock never among them
	Bytes int64    // what they hold
}

// Bytes is what the files Clean removes hold.
func (c Clean) Bytes() int64 {
	var n int64
	for _, f := range c.Folders {
		n += f.Bytes
	}
	return n
}

// listed is a key of a listing and when its object was last changed (zero: not known).
type listed struct {
	key string
	mod time.Time
}

func keysOf(ls []listed) []string {
	keys := make([]string, len(ls))
	for i, l := range ls {
		keys[i] = l.key
	}
	return keys
}

// fullLister is a Store that lists the temporary files of its uploads too (its List leaves them
// out), and when each object was last changed.
type fullLister interface {
	listAll(ctx context.Context, prefix string) ([]listed, error)
}

func listAll(ctx context.Context, st Store, prefix string) ([]listed, error) {
	if t, ok := st.(fullLister); ok {
		return t.listAll(ctx, prefix)
	}
	keys, err := st.List(ctx, prefix)
	ls := make([]listed, len(keys))
	for i, k := range keys {
		ls[i] = listed{key: k}
	}
	return ls, err
}

// busy reports whether folder d of s, which holds ls, is being written: a marker of it written in
// the last WritingStale, or a file of it changed then (a writeback of a hwnpu before the markers,
// a marker that could not be written).
func busy(ctx context.Context, s Source, ls []listed, now time.Time) (bool, error) {
	var markers []string
	for _, l := range ls {
		if !l.mod.IsZero() && now.Sub(l.mod) < WritingStale {
			return true, nil
		}
		if isMarker(l.key) {
			markers = append(markers, l.key)
		}
	}
	return writing(ctx, s, markers, now)
}

// folderOf is the folder of key under top (converted/ is 5 parts deep, original/ 3), "" for a
// key not in one.
func folderOf(key string) string {
	parts := strings.Split(key, "/")
	n := map[string]int{"converted": 5, "original": 3}[parts[0]]
	if n == 0 || len(parts) <= n {
		return ""
	}
	return strings.Join(parts[:n], "/")
}

// PlanClean works out a Clean of s. Nothing is removed until Apply.
func PlanClean(ctx context.Context, s Source) (Clean, error) {
	c := Clean{Source: s.String(), src: s}
	byFolder := map[string][]listed{}
	for _, top := range []string{"converted/", "original/"} {
		ls, err := listAll(ctx, s.Store, top)
		if err != nil {
			return c, err
		}
		for _, l := range ls {
			if d := folderOf(l.key); d != "" {
				byFolder[d] = append(byFolder[d], l)
			}
		}
	}
	now := time.Now()
	if u, ok := s.Store.(uploadCleaner); ok {
		for _, top := range []string{"converted/", "original/"} {
			ups, err := u.uploadsLeft(ctx, top, now.Add(-WritingStale))
			if err != nil {
				return c, err
			}
			c.Uploads = append(c.Uploads, ups...)
		}
	}
	for _, d := range slices.Sorted(func(yield func(string) bool) {
		for d := range byFolder {
			if !yield(d) {
				return
			}
		}
	}) {
		f, err := planFolder(ctx, s, d, byFolder[d], now)
		if err != nil {
			return c, err
		}
		if len(f.Files) == 0 {
			continue
		}
		for _, k := range f.Files {
			if n, ok, err := s.Store.Has(ctx, k); err == nil && ok {
				f.Bytes += n
			}
		}
		c.Folders = append(c.Folders, f)
	}
	return c, nil
}

// planFolder is what goes of folder d of s, which holds ls; none while it is being written.
func planFolder(ctx context.Context, s Source, d string, ls []listed, now time.Time) (CleanFolder, error) {
	f := CleanFolder{Path: d}
	if b, err := busy(ctx, s, ls, now); err != nil || b {
		return f, err
	}
	keys := keysOf(ls)
	keep := func(string) bool { return false } // the files that stay
	if strings.HasPrefix(d, "converted/") {
		if slices.Contains(keys, d+"/"+EntryFile) {
			keep = func(k string) bool { return !isTemp(k) && !isMarker(k) }
		} else {
			f.Whole = true
		}
	} else {
		repo := strings.TrimPrefix(d, "original/")
		l, err := readLock(ctx, s, repo)
		switch {
		case errors.Is(err, ErrNotFound):
			f.Whole = true
		case err != nil:
			return f, err // one this hwnpu cannot read: left as it is
		default:
			keep = func(k string) bool {
				name := strings.TrimPrefix(k, d+"/")
				_, named := l.Files[name]
				return name == LockName || named && !isTemp(k)
			}
		}
	}
	for _, k := range keys {
		if !keep(k) {
			f.Files = append(f.Files, k)
		}
	}
	return f, nil
}

// Apply removes what c names and returns how many files it removed and uploads it aborted. Each
// folder is looked at again first: one that a writeback has started in since, or that has become
// whole (an entry.json), is left. The markers go last, so that a folder cut off again is still
// taken as one.
func (c Clean) Apply(ctx context.Context) (int, error) {
	n := 0
	s := c.src
	if u, ok := s.Store.(uploadCleaner); ok {
		for _, up := range c.Uploads {
			if err := u.abortUpload(ctx, up); err != nil {
				return n, err
			}
			n++
		}
	}
	for _, f := range c.Folders {
		ls, err := listAll(ctx, s.Store, f.Path+"/")
		if err != nil {
			return n, err
		}
		if b, err := busy(ctx, s, ls, time.Now()); err != nil {
			return n, err
		} else if b || f.Whole && slices.Contains(keysOf(ls), f.Path+"/"+EntryFile) {
			continue
		}
		var last []string
		for _, k := range f.Files {
			if isMarker(k) {
				last = append(last, k)
				continue
			}
			if err := s.Store.Delete(ctx, k); err != nil {
				return n, err
			}
			n++
		}
		for _, k := range last {
			if err := s.Store.Delete(ctx, k); err != nil {
				return n, err
			}
			n++
		}
		if f.Whole {
			removeFolder(ctx, s.Store, f.Path)
		}
	}
	return n, nil
}

// folderRemover is a Store with folders that stay when emptied (WebDAV, a folder): an emptied
// entry's folder is removed, and its parents up to converted/ or original/ that are empty.
type folderRemover interface {
	removeEmpty(ctx context.Context, dir string) bool
}

func removeFolder(ctx context.Context, st Store, dir string) {
	r, ok := st.(folderRemover)
	if !ok {
		return // S3 has no folders
	}
	for d := dir; strings.Contains(d, "/") && r.removeEmpty(ctx, d); d = path.Dir(d) {
	}
}
