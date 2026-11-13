package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/console"
)

// Layer is a shared place in the order things are looked for (docs/cache-sharing.md 7): what it
// holds, by key, in its store:
//
//	blobs/sha256/AB/SHA256                 a file, by its SHA-256
//	entries/KIND/KEY.json                  an entry (Entry): its files are blobs
//	upstream/OWNER/NAME@COMMIT/FILE.json   a file of a checkpoint: its SHA-256 and size
//
// A blob is written before what names it, so that what is named is there whole.
type Layer struct {
	Store Store
	// Write puts what is made here in the layer; Fill puts what is read from a later layer or
	// upstream (Hugging Face), as it is, as a forward proxy keeps what it fetched; FillGated
	// fills the files of gated repositories and of models whose license is asked for too.
	Write, Fill, FillGated bool
	// Scopes are what the layer holds (ScopeUpstream, ScopePortable, ScopeHost); none: all.
	Scopes []string
	// From says what set the layer (a file, HWNPU_CACHE_LAYERS), for hwnpu-dev cache layers.
	From string
}

// Holds reports whether the layer holds what is of scope.
func (l Layer) Holds(scope string) bool {
	return len(l.Scopes) == 0 || slices.Contains(l.Scopes, scope)
}

func (l Layer) String() string { return l.Store.String() }

func blobKey(sum string) string { return "blobs/sha256/" + sum[:2] + "/" + sum }

func entryKey(kind, key string) string { return "entries/" + kind + "/" + key + ".json" }

func upstreamKey(repo, commit, name string) string {
	return "upstream/" + repo + "@" + commit + "/" + name + ".json"
}

// upstreamFile is what a layer keeps of a file of a checkpoint.
type upstreamFile struct {
	SHA256 string `json:"sha256"`
	Size   int64  `json:"size"`
}

// getBlob downloads the blob of sum to dst and checks its SHA-256 (a blob that differs is
// removed from dst and reported).
func getBlob(ctx context.Context, s Store, sum, dst string, bar func(done, total int64)) error {
	if _, err := s.Get(ctx, blobKey(sum), dst, bar); err != nil {
		return err
	}
	got, err := FileSHA256(dst)
	if err != nil {
		return err
	}
	if got != sum {
		os.Remove(dst)
		return fmt.Errorf("store: %s: the blob %s has the SHA-256 %s", s, sum, got)
	}
	return nil
}

// putBlob puts the file src of SHA-256 sum unless the store has it.
func putBlob(ctx context.Context, s Store, sum, src string, size int64) error {
	if n, ok, err := s.Has(ctx, blobKey(sum)); err != nil {
		return err
	} else if ok && n == size {
		return nil
	}
	return s.Put(ctx, blobKey(sum), src)
}

// bar shows a download from a layer on log.
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

// FetchEntry looks for the entry of kind and key in layers (those that hold scope), in order,
// and makes it in dst (through dst.part, renamed when whole and checked) from the first that has
// it. It returns the index of that layer, -1 and ErrNotFound when none has it.
func FetchEntry(ctx context.Context, layers []Layer, scope, kind, key, dst string, log io.Writer) (int, error) {
	var errs []error
	for i, l := range layers {
		if !l.Holds(scope) {
			continue
		}
		b, err := l.Store.Read(ctx, entryKey(kind, key))
		if errors.Is(err, ErrNotFound) {
			continue
		}
		if err == nil {
			err = materialize(ctx, l, b, key, dst, log)
		}
		if err == nil {
			return i, nil
		}
		errs = append(errs, fmt.Errorf("%s: %w", l, err))
		if log != nil {
			fmt.Fprintf(log, "%s: %v; looking further\n", l, err)
		}
	}
	if len(errs) > 0 {
		return -1, errors.Join(append(errs, ErrNotFound)...)
	}
	return -1, ErrNotFound
}

// materialize makes the entry described by b in dst from l's blobs.
func materialize(ctx context.Context, l Layer, b []byte, key, dst string, log io.Writer) error {
	e, err := parseEntry(b)
	if err != nil {
		return err
	}
	if e.Key != key {
		return fmt.Errorf("store: the entry of %s says it is %s", key, e.Key)
	}
	part := dst + ".part"
	if err := os.MkdirAll(part, 0o755); err != nil {
		return err
	}
	br, set := bar(log, fmt.Sprintf("downloading %s from %s", filepath.Base(dst), l))
	var done int64
	total := e.Bytes()
	for _, f := range e.Files {
		p := filepath.Join(part, filepath.FromSlash(f.Path))
		if fi, err := os.Stat(p); err == nil && fi.Size() == f.Size {
			done += f.Size // a download cut off before had it
			continue
		}
		base := done
		if err := getBlob(ctx, l.Store, f.SHA256, p, func(n, _ int64) { set(base+n, total) }); err != nil {
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

// PutEntry puts the entry of dir (its EntryFile) in l: the blobs it lacks, then the entry.
func PutEntry(ctx context.Context, l Layer, dir string) error {
	e, err := ReadEntry(dir)
	if err != nil {
		return err
	}
	if _, ok, err := l.Store.Has(ctx, entryKey(e.Kind, e.Key)); err != nil {
		return err
	} else if ok {
		return nil
	}
	for _, f := range e.Files {
		if err := putBlob(ctx, l.Store, f.SHA256, filepath.Join(dir, filepath.FromSlash(f.Path)), f.Size); err != nil {
			return err
		}
	}
	b, err := json.MarshalIndent(e, "", "  ")
	if err != nil {
		return err
	}
	return l.Store.Write(ctx, entryKey(e.Kind, e.Key), b)
}

// FetchUpstream looks for a file of a checkpoint in layers that hold ScopeUpstream: by its
// SHA-256 when it is known (sum), else by the repository's commit and its name. It downloads it
// to dst and checks it, and returns the index of the layer and its SHA-256; -1 and ErrNotFound
// when no layer has it.
func FetchUpstream(ctx context.Context, layers []Layer, repo, commit, name, sum, dst string, log io.Writer) (int, string, error) {
	for i, l := range layers {
		if !l.Holds(ScopeUpstream) {
			continue
		}
		want := sum
		if want == "" {
			if commit == "" {
				continue
			}
			b, err := l.Store.Read(ctx, upstreamKey(repo, commit, name))
			if err != nil {
				continue
			}
			var u upstreamFile
			if json.Unmarshal(b, &u) != nil || len(u.SHA256) != 64 {
				continue
			}
			want = u.SHA256
		}
		if _, ok, err := l.Store.Has(ctx, blobKey(want)); err != nil || !ok {
			continue
		}
		br, set := bar(log, fmt.Sprintf("downloading %s from %s", filepath.Base(name), l))
		if err := getBlob(ctx, l.Store, want, dst, set); err != nil {
			br.Done("failed")
			if log != nil {
				fmt.Fprintf(log, "%s: %v; looking further\n", l, err)
			}
			continue
		}
		br.Done(console.Duration(br.Elapsed()))
		return i, want, nil
	}
	return -1, "", ErrNotFound
}

// PutUpstream puts a file of a checkpoint (path, of SHA-256 sum) in l: its blob, and with a
// commit, where it is in the repository.
func PutUpstream(ctx context.Context, l Layer, repo, commit, name, path, sum string) error {
	fi, err := os.Stat(path)
	if err != nil {
		return err
	}
	if sum == "" {
		if sum, err = FileSHA256(path); err != nil {
			return err
		}
	}
	if err := putBlob(ctx, l.Store, sum, path, fi.Size()); err != nil {
		return err
	}
	if commit == "" {
		return nil
	}
	b, _ := json.Marshal(upstreamFile{SHA256: sum, Size: fi.Size()})
	return l.Store.Write(ctx, upstreamKey(repo, commit, name), b)
}

// ListEntries lists the entries a layer holds: their kind and key.
func ListEntries(ctx context.Context, l Layer) ([][2]string, error) {
	keys, err := l.Store.List(ctx, "entries/")
	if err != nil {
		return nil, err
	}
	var out [][2]string
	for _, k := range keys {
		rest, ok := strings.CutPrefix(k, "entries/")
		kind, file, ok2 := strings.Cut(rest, "/")
		if ok && ok2 && strings.HasSuffix(file, ".json") {
			out = append(out, [2]string{kind, strings.TrimSuffix(file, ".json")})
		}
	}
	return out, nil
}

// ReadLayerEntry reads an entry a layer holds.
func ReadLayerEntry(ctx context.Context, l Layer, kind, key string) (Entry, error) {
	b, err := l.Store.Read(ctx, entryKey(kind, key))
	if err != nil {
		return Entry{}, err
	}
	return parseEntry(b)
}

// lockWork is the lock of one piece of work on what the layers hold (cache.Lock).
func lockWork(ctx context.Context, name string) (func(), error) {
	return cache.Lock(ctx, "store:"+name, name, nil)
}
