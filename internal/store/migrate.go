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
)

// MigrateReport is what MigrateDir did.
type MigrateReport struct {
	Repos, Files, Entries int
	Skipped               []string // what was left where it was, and why
}

// MigrateDir moves what the folder of a source holds in the layout before this one
// (blobs/sha256/AB/SHA256, upstream/OWNER/NAME@COMMIT/FILE.json, entries/KIND/KEY.json) to this
// one (original/, converted/): the blobs are hard-linked to their new names (copied where links
// fail), so no data moves. Of a repository with files of several commits, the commit with the
// most files is taken. place is where an entry goes (EntryPath), false to leave it. With
// remove, what was moved is removed from the old layout.
func MigrateDir(ctx context.Context, root string, place func(Entry) (string, bool), remove bool, log io.Writer) (MigrateReport, error) {
	var r MigrateReport
	if log == nil {
		log = io.Discard
	}
	blob := func(sum string) string { return filepath.Join(root, "blobs", "sha256", sum[:2], sum) }
	var gone []string // the old files moved

	// upstream/OWNER/NAME@COMMIT/FILE.json
	type up struct{ name, sum, index string }
	byRepo := map[string]map[string][]up{} // repo, commit
	upRoot := filepath.Join(root, "upstream")
	err := filepath.WalkDir(upRoot, func(p string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(p, ".json") {
			if errors.Is(err, os.ErrNotExist) {
				return nil
			}
			return err
		}
		rel, _ := filepath.Rel(upRoot, p)
		parts := strings.SplitN(filepath.ToSlash(rel), "/", 3)
		if len(parts) != 3 {
			return nil
		}
		name, commit, ok := strings.Cut(parts[1], "@")
		if !ok {
			return nil
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		var f struct {
			SHA256 string `json:"sha256"`
		}
		if json.Unmarshal(b, &f) != nil || len(f.SHA256) != 64 {
			r.Skipped = append(r.Skipped, rel+": not an index of a file")
			return nil
		}
		repo := parts[0] + "/" + name
		if byRepo[repo] == nil {
			byRepo[repo] = map[string][]up{}
		}
		byRepo[repo][commit] = append(byRepo[repo][commit], up{strings.TrimSuffix(parts[2], ".json"), f.SHA256, p})
		return nil
	})
	if err != nil {
		return r, err
	}
	d := Dir{Name: "migrate", Root: root}
	src := Source{Store: d}
	for _, repo := range slices.Sorted(mapKeys(byRepo)) {
		if err := ctx.Err(); err != nil {
			return r, err
		}
		commits := byRepo[repo]
		best := ""
		for c, fs := range commits {
			if best == "" || len(fs) > len(commits[best]) || len(fs) == len(commits[best]) && c > best {
				best = c
			}
		}
		for c := range commits {
			if c != best {
				r.Skipped = append(r.Skipped, fmt.Sprintf("%s@%s: %d files of another commit than %s", repo, short(c), len(commits[c]), short(best)))
			}
		}
		l, err := readLock(ctx, src, repo)
		switch {
		case errors.Is(err, ErrNotFound):
			l = Lock{Repo: repo, Commit: best, Files: map[string]LockedFile{}}
		case err != nil:
			return r, err
		case l.Commit != best:
			r.Skipped = append(r.Skipped, fmt.Sprintf("%s: original/ holds %s already, not %s", repo, short(l.Commit), short(best)))
			continue
		}
		for _, f := range commits[best] {
			fi, err := os.Stat(blob(f.sum))
			if err != nil {
				r.Skipped = append(r.Skipped, fmt.Sprintf("%s/%s: no blob %s", repo, f.name, f.sum[:12]))
				continue
			}
			dst := filepath.Join(root, filepath.FromSlash(OriginalKey(repo, f.name)))
			if err := linkInto(blob(f.sum), dst); err != nil {
				return r, err
			}
			l.Files[f.name] = LockedFile{SHA256: f.sum, Size: fi.Size()}
			r.Files++
			gone = append(gone, f.index, blob(f.sum))
		}
		if err := WriteLock(ctx, src, l); err != nil {
			return r, err
		}
		r.Repos++
		fmt.Fprintf(log, "%s@%s: %d files\n", repo, short(best), len(commits[best]))
	}

	// entries/KIND/KEY.json
	files, _ := filepath.Glob(filepath.Join(root, "entries", "*", "*.json"))
	for _, p := range files {
		b, err := os.ReadFile(p)
		if err != nil {
			return r, err
		}
		e, err := parseEntry(b)
		if err != nil {
			r.Skipped = append(r.Skipped, fmt.Sprintf("%s: %v", filepath.Base(p), err))
			continue
		}
		rel, ok := place(e)
		if !ok {
			r.Skipped = append(r.Skipped, fmt.Sprintf("entry %s %s %s: no place for it", e.Kind, e.Model, e.Key[:min(12, len(e.Key))]))
			continue
		}
		dir := filepath.Join(root, filepath.FromSlash(rel))
		missing := false
		for _, f := range e.Files {
			if err := linkInto(blob(f.SHA256), filepath.Join(dir, filepath.FromSlash(f.Path))); err != nil {
				if errors.Is(err, os.ErrNotExist) {
					missing = true
					break
				}
				return r, err
			}
		}
		if missing {
			os.RemoveAll(dir)
			r.Skipped = append(r.Skipped, fmt.Sprintf("entry %s %s: a blob is missing", e.Kind, e.Model))
			continue
		}
		e.Path = rel
		if err := WriteEntry(dir, e); err != nil {
			return r, err
		}
		r.Entries++
		fmt.Fprintf(log, "%s\n", rel)
		gone = append(gone, p)
		for _, f := range e.Files {
			gone = append(gone, blob(f.SHA256))
		}
	}
	if remove {
		for _, p := range gone {
			if err := os.Remove(p); err != nil && !errors.Is(err, os.ErrNotExist) {
				return r, err
			}
			removeEmpty(filepath.Dir(p), root)
		}
	}
	return r, nil
}

// linkInto puts src at dst: a hard link, else a copy.
func linkInto(src, dst string) error {
	if _, err := os.Stat(src); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	if fi, err := os.Stat(dst); err == nil {
		if si, _ := os.Stat(src); os.SameFile(fi, si) {
			return nil
		}
		os.Remove(dst)
	}
	if os.Link(src, dst) == nil {
		return nil
	}
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	_, err = copyTo(context.Background(), in, dst, 0, -1, nil)
	return err
}

// removeEmpty removes dir and the folders above it up to root while they are empty.
func removeEmpty(dir, root string) {
	for dir != root && strings.HasPrefix(dir, root) {
		if os.Remove(dir) != nil {
			return
		}
		dir = filepath.Dir(dir)
	}
}

func mapKeys[K comparable, V any](m map[K]V) func(func(K) bool) {
	return func(yield func(K) bool) {
		for k := range m {
			if !yield(k) {
				return
			}
		}
	}
}
