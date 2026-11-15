package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu/internal/console"
)

// LockName is the file of a repository's folder in a source that says what its files are: the
// commit of Hugging Face they are of, and each one's SHA-256 and size. A file it does not name
// is not read. It changes only by writeback (a file of the same commit added) and by hwnpu-dev
// cache update (another commit).
const LockName = "hwnpu-lock.json"

// LockFormat is the version of LockName this hwnpu reads and writes.
const LockFormat = 1

// Lock is a repository's LockName.
type Lock struct {
	Format int    `json:"format"`
	Repo   string `json:"repo"` // OWNER/NAME
	// Commit is the commit of Hugging Face the files are of ("" for a repository that is not
	// on Hugging Face: a source's own).
	Commit  string                `json:"commit,omitempty"`
	Files   map[string]LockedFile `json:"files"`
	Updated time.Time             `json:"updated"`
}

// LockedFile is a file of a Lock.
type LockedFile struct {
	SHA256 string `json:"sha256"`
	Size   int64  `json:"size"`
}

var commitRE = regexp.MustCompile(`^[0-9a-f]{40}$`)

// OriginalKey is the key of a file of repository repo (OWNER/NAME) in a source.
func OriginalKey(repo, name string) string { return "original/" + repo + "/" + name }

// ErrOtherCommit is a writeback of a file of another commit than the source's lock.
var ErrOtherCommit = errors.New("the source holds another commit")

// locks are the sources' locks read in this process, by the source and the repository.
var (
	locksMu sync.Mutex
	locks   = map[string]*Lock{} // nil: none there
)

func clearLocks() {
	locksMu.Lock()
	clear(locks)
	locksMu.Unlock()
}

// ReadLock reads the lock of repo in s (once per process); ErrNotFound when there is none.
func ReadLock(ctx context.Context, s Source, repo string) (Lock, error) {
	id := s.String() + "|" + repo
	locksMu.Lock()
	l, ok := locks[id]
	locksMu.Unlock()
	if ok {
		if l == nil {
			return Lock{}, ErrNotFound
		}
		return *l, nil
	}
	got, err := readLock(ctx, s, repo)
	if err != nil && !errors.Is(err, ErrNotFound) {
		return got, err
	}
	locksMu.Lock()
	if err == nil {
		locks[id] = &got
	} else {
		locks[id] = nil
	}
	locksMu.Unlock()
	return got, err
}

func readLock(ctx context.Context, s Source, repo string) (Lock, error) {
	if !SafeKey(repo) || strings.Count(repo, "/") != 1 {
		return Lock{}, fmt.Errorf("store: %q is not OWNER/NAME", repo)
	}
	b, err := s.Store.Read(ctx, OriginalKey(repo, LockName))
	if err != nil {
		return Lock{}, err
	}
	var l Lock
	if err := json.Unmarshal(b, &l); err != nil {
		return l, fmt.Errorf("store: %s: %s of %s: %w", s, LockName, repo, err)
	}
	if l.Format != LockFormat {
		return l, fmt.Errorf("store: %s: %s of %s is of format %d, not the %d this hwnpu reads", s, LockName, repo, l.Format, LockFormat)
	}
	if l.Commit != "" && !commitRE.MatchString(l.Commit) {
		return l, fmt.Errorf("store: %s: %s of %s: commit %q is not 40 hex digits", s, LockName, repo, l.Commit)
	}
	for name, f := range l.Files {
		if !SafeKey(name) || name == LockName || len(f.SHA256) != 64 || strings.ToLower(f.SHA256) != f.SHA256 {
			return l, fmt.Errorf("store: %s: %s of %s: a file %q that is not a path with its SHA-256", s, LockName, repo, name)
		}
	}
	return l, nil
}

// WriteLock writes the lock of a repository in s.
func WriteLock(ctx context.Context, s Source, l Lock) error {
	l.Format, l.Updated = LockFormat, time.Now().UTC()
	b, err := json.MarshalIndent(l, "", "  ")
	if err != nil {
		return err
	}
	if err := s.Store.Write(ctx, OriginalKey(l.Repo, LockName), append(b, '\n')); err != nil {
		return err
	}
	locksMu.Lock()
	locks[s.String()+"|"+l.Repo] = &l
	locksMu.Unlock()
	return nil
}

// FetchOriginal looks for a file of repository repo in the sources, in order: in one whose lock
// names it, of commit (any, for "") and of the SHA-256 sum (any, for ""). It downloads it to dst
// and checks its SHA-256 against the lock. It returns the index of the source and its lock; -1
// and ErrNotFound when none has it.
func FetchOriginal(ctx context.Context, ss []Source, repo, commit, name, sum, dst string, log io.Writer) (int, Lock, error) {
	var errs []error
	for i, s := range ss {
		l, err := ReadLock(ctx, s, repo)
		if errors.Is(err, ErrNotFound) {
			continue
		}
		if err != nil {
			errs = append(errs, err)
			continue
		}
		f, ok := l.Files[name]
		if !ok || commit != "" && l.Commit != commit || sum != "" && !strings.EqualFold(f.SHA256, sum) {
			continue
		}
		br, set := bar(log, fmt.Sprintf("downloading %s from %s", filepath.Base(name), s))
		if err := getChecked(ctx, s.Store, OriginalKey(repo, name), f.SHA256, dst, func(n, _ int64) { set(n, f.Size) }); err != nil {
			br.Done("failed")
			errs = append(errs, fmt.Errorf("%s: %w", s, err))
			if log != nil {
				fmt.Fprintf(log, "%s: %v; looking further\n", s, err)
			}
			continue
		}
		br.Done(console.Duration(br.Elapsed()))
		return i, l, nil
	}
	if len(errs) > 0 {
		return -1, Lock{}, errors.Join(append(errs, ErrNotFound)...)
	}
	return -1, Lock{}, ErrNotFound
}

// getChecked downloads key to dst and checks its SHA-256 (a file that differs is removed from
// dst and reported).
func getChecked(ctx context.Context, s Store, key, sum, dst string, bar func(done, total int64)) error {
	if _, err := s.Get(ctx, key, dst, bar); err != nil {
		return err
	}
	got, err := FileSHA256(dst)
	if err != nil {
		return err
	}
	if got != sum {
		os.Remove(dst)
		return fmt.Errorf("store: %s: %s has the SHA-256 %s, not %s", s, key, got, sum)
	}
	return nil
}

// PutOriginal puts a file of repo of commit (path here, of SHA-256 sum: worked out when "") in s
// and names it in the source's lock. A source whose lock is of another commit is left as it is
// (ErrOtherCommit: hwnpu-dev cache update moves it).
func PutOriginal(ctx context.Context, s Source, repo, commit, name, path, sum string) error {
	unlock, err := lockWork(ctx, s.String()+"|"+repo)
	if err != nil {
		return err
	}
	defer unlock()
	l, err := readLock(ctx, s, repo) // not the one read before: another machine may have added files
	switch {
	case errors.Is(err, ErrNotFound):
		l = Lock{Repo: repo, Commit: commit}
	case err != nil:
		return err
	case l.Commit != commit:
		return fmt.Errorf("%s: %s: %s, not %s: %w", s, repo, short(l.Commit), short(commit), ErrOtherCommit)
	}
	fi, err := os.Stat(path)
	if err != nil {
		return err
	}
	if sum == "" {
		if sum, err = FileSHA256(path); err != nil {
			return err
		}
	}
	if f, ok := l.Files[name]; ok && f.SHA256 == sum {
		return nil
	}
	if err := s.Store.Put(ctx, OriginalKey(repo, name), path); err != nil {
		return err
	}
	l.Files = maps.Clone(l.Files)
	if l.Files == nil {
		l.Files = map[string]LockedFile{}
	}
	l.Files[name] = LockedFile{SHA256: sum, Size: fi.Size()}
	return WriteLock(ctx, s, l)
}

func short(commit string) string {
	if commit == "" {
		return "no commit"
	}
	return commit[:min(12, len(commit))]
}
