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
)

// PlaceEntry is the entry (no files yet) of the conversion of kind of model m for target that
// recipe describes, named export here: its Path is its folder in a source (placeOf).
func PlaceEntry(kind string, m Model, target, export string, recipe map[string]string) shared.Entry {
	r := maps.Clone(recipe)
	r["kind"] = kind
	return placeOf(kind, m, target, export, r).entry
}

// MigratePlace is where an entry of the layout before (entries/KIND/KEY.json) goes now, for
// shared.MigrateDir: its export's name here is known for Core AI's exports, the tables, the
// language models' and the first runs' exports for AMD's NPU.
func MigratePlace(e shared.Entry) (string, bool) {
	m := modelNamed(e.Model)
	export := ""
	switch {
	case e.Kind == CoreAIKind:
		export = e.Recipe["export"]
	case e.Kind == TablesKind && len(e.Files) == 1:
		export = e.Files[0].Path
	case e.Kind == "iron-lm":
		export = filepath.Base(AMDExportDir(m))
	case strings.HasPrefix(e.Kind, "iron-"):
		export = strings.ToLower(m.Name) + ".amd-npu2"
	}
	if export == "" || export == "." || shared.Fingerprint(e.Recipe) != e.Key {
		return "", false
	}
	return placeOf(e.Kind, m, e.Target, export, e.Recipe).entry.Path, true
}

// LockedRepos are the repositories s holds files of (their hwnpu-lock.json), in order.
func LockedRepos(ctx context.Context, s shared.Source) ([]string, error) {
	keys, err := s.Store.List(ctx, "original/")
	if err != nil {
		return nil, err
	}
	var out []string
	for _, k := range keys {
		rest, ok := strings.CutSuffix(strings.TrimPrefix(k, "original/"), "/"+shared.LockName)
		if ok && strings.Count(rest, "/") == 1 {
			out = append(out, rest)
		}
	}
	return out, nil
}

// LockUpdate is what UpdateLock did to a repository's files in a source.
type LockUpdate struct {
	From, To               string // the commits
	Same, Changed, Dropped int    // files kept, downloaded again, gone from the repository
}

// UpdateLock moves the files of repo (OWNER/NAME[@REVISION]) in source s to the commit Hugging
// Face serves for the revision (main): each file its lock names that has another SHA-256 there
// is downloaded and put again, one the commit does not have is dropped, and the lock is written
// with the new commit last (hwnpu-dev models update). A source with no lock of repo is left.
func UpdateLock(ctx context.Context, s shared.Source, repo string, log io.Writer) (LockUpdate, error) {
	var u LockUpdate
	if log == nil {
		log = io.Discard
	}
	if !shared.HuggingFace() {
		return u, fmt.Errorf("chat: %s=false: the commit of Hugging Face is not asked for", shared.HuggingFaceEnv)
	}
	base, rev, _ := strings.Cut(repo, "@")
	l, err := shared.ReadLock(ctx, s, base)
	if err != nil {
		return u, err
	}
	u.From = l.Commit
	var first string
	for name := range l.Files {
		if first == "" || name < first {
			first = name
		}
	}
	if first == "" {
		return u, fmt.Errorf("chat: %s: the lock of %s names no file", s, base)
	}
	if commitRE.MatchString(rev) {
		u.To = rev
	} else {
		_, u.To = hubFile(ctx, repo, first)
		if u.To == "" {
			return u, fmt.Errorf("chat: Hugging Face did not say the commit of %s (%s)", repo, first)
		}
	}
	if u.To == l.Commit {
		u.Same = len(l.Files)
		return u, nil
	}
	work := cache.Sub("work", "update", strings.ReplaceAll(base, "/", "--")+"@"+u.To)
	defer os.RemoveAll(work)
	// the files are put before the lock names them: marked, so that hwnpu cache clean leaves them
	done, err := shared.MarkWriting(ctx, s, "original/"+base)
	if err != nil {
		return u, err
	}
	defer done()
	next := shared.Lock{Repo: base, Commit: u.To, Files: map[string]shared.LockedFile{}}
	at := base + "@" + u.To
	for name, f := range l.Files {
		if sum, _ := hubFile(ctx, at, name); sum == f.SHA256 {
			next.Files[name] = f // a large file (LFS) that did not change
			u.Same++
			continue
		}
		p := filepath.Join(work, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return u, err
		}
		err := fetchHF(ctx, at, name, p, log)
		if errors.Is(err, errNotFound) {
			fmt.Fprintf(log, "%s: %s is not in %s; dropped\n", base, name, u.To[:12])
			u.Dropped++
			continue
		}
		if err != nil {
			return u, err
		}
		sum, err := shared.FileSHA256(p)
		if err != nil {
			return u, err
		}
		fi, _ := os.Stat(p)
		if sum == f.SHA256 {
			next.Files[name] = f
			u.Same++
			continue
		}
		if err := s.Store.Put(ctx, shared.OriginalKey(base, name), p); err != nil {
			return u, err
		}
		next.Files[name] = shared.LockedFile{SHA256: sum, Size: fi.Size()}
		u.Changed++
		os.Remove(p)
	}
	return u, shared.WriteLock(ctx, s, next)
}
