package chat

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/console"
	shared "github.com/shibukawa/hwnpu/internal/store"
)

// Where the files of the models come from (docs/cache-sharing.md 7), in layers: the machine's
// own cache (cache.Root: original/OWNER/NAME, with its hwnpu-lock.json), the sources of the
// environment (HWNPU_MODEL_SOURCE, HWNPU_MODEL_SOURCE_1, ...: internal/store), the models an
// application installs with it (store.SetBundles), and Hugging Face last unless
// HWNPU_MODEL_HUGGING_FACE=false. A repository's files in a layer are those its hwnpu-lock.json
// names, of the commit it names: a model that does not pin a commit takes the commit of the first
// lock after the machine's cache, so that every machine of the sources runs the same files
// (hwnpu-dev cache update moves the lock), and the machine's cache follows it. A file read from a
// later layer goes in the machine's cache and in the sources before with ?writeback=true (in the
// background).

var (
	// digests are the SHA-256 of the files of the models the definitions pin (sha256), by the
	// model's Source() and the file's name.
	digests = map[string]map[string]string{}
	// pinned are the files the definitions pin, by their SHA-256 (lower case): the repository
	// (OWNER/NAME) and the name of each, so that a file two models share (MOSS-SoundEffect's
	// text encoder is Qwen3-1.7B) is downloaded once (sameFile).
	pinned = map[string][]pinnedFile{}
)

// safeName reports whether name is a relative path of a source's model that stays inside it.
func safeName(name string) bool {
	if name == "" || strings.HasPrefix(name, "/") || strings.Contains(name, "\\") || strings.Contains(name, ":") {
		return false
	}
	return !slices.Contains(strings.Split(name, "/"), "..")
}

// pinnedFile is a file of a model the definitions pin: its repository (OWNER/NAME) and its name
// in it.
type pinnedFile struct{ repo, name string }

// RepoDir is the folder of the files of repo (OWNER/NAME, or OWNER/NAME@REVISION) here:
// original/OWNER/NAME under the cache's root, as in a source. Whatever the revision, a repository
// has one folder: its hwnpu-lock.json says which commit the files are of.
func RepoDir(repo string) string {
	base, _, _ := strings.Cut(repo, "@")
	return cache.Sub(filepath.FromSlash(shared.OriginalKey(base, "")))
}

// fetch returns the path of a file of a model's repository here (RepoDir), which it puts there
// unless it is there. repo is Model.Source(): OWNER/NAME or OWNER/NAME@REVISION. The file is of the
// commit the model pins, or the first source's lock names, or the one the machine's cache has; it
// is looked for in the sources (shared.Sources), then on Hugging Face (unless
// HWNPU_MODEL_HUGGING_FACE=false). A file the definitions pin (sha256) is checked; one that
// differs is removed. A pinned file another repository of the same SHA-256 has here is linked from
// there first (sameFile).
func fetch(ctx context.Context, repo, name string, log io.Writer) (string, error) {
	if !safeName(name) {
		return "", fmt.Errorf("chat: %q is not a file of %s", name, repo)
	}
	base, rev, _ := strings.Cut(repo, "@")
	path := filepath.Join(RepoDir(base), filepath.FromSlash(name))
	local := shared.Local()
	commit := lockedCommit(ctx, repo)
	want, pinnedSum := digests[repo][name]
	want = strings.ToLower(want)
	if here(ctx, local, base, commit, name, want, path) {
		return path, nil
	}
	// one process downloads a file at a time; one that waited finds it there
	unlock, err := cache.Lock(ctx, path, repo+"/"+name, log)
	if err != nil {
		return "", err
	}
	defer unlock()
	if here(ctx, local, base, commit, name, want, path) {
		return path, nil
	}
	if adopt(ctx, local, repo, commit, name, want, path, log) {
		return path, nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", err
	}
	stop := console.Downloading(repo + "/" + name)
	bytes := int64(-1) // linked from another repository: no download
	defer func() { stop(bytes) }()
	// record puts the file in the machine's lock, of commit c and SHA-256 sum ("": worked out)
	record := func(c, sum string) error {
		if err := verify(repo, name, path); err != nil {
			os.Remove(path)
			return err
		}
		fi, err := os.Stat(path)
		if err != nil {
			return err
		}
		if fi.Mode()&os.ModeSymlink == 0 {
			bytes = fi.Size()
		}
		if sum == "" {
			if sum, err = shared.FileSHA256(path); err != nil {
				return err
			}
		}
		planFinish(path)
		if c != "" || commit == "" {
			noteLockCommit(repo, c)
		}
		return shared.Record(ctx, local, base, c, name, shared.LockedFile{SHA256: sum, Size: fi.Size()}, true, nil)
	}
	if pinnedSum && sameFile(want, name, path, log) {
		bytes = -1
		return path, record(commit, want)
	}
	ss := shared.Sources()
	if commit == "" && rev != "" && rev != "main" && shared.HuggingFace() {
		ss = nil // a branch or a tag whose commit is not known: what the sources hold may be another's
	}
	i, lk, err := shared.FetchOriginal(ctx, ss, base, commit, name, want, path, log)
	if err == nil {
		sum := lk.Files[name].SHA256
		if err := record(lk.Commit, sum); err != nil {
			return "", err
		}
		for _, s := range ss[:i] {
			if s.Writeback {
				shared.FillLater(s, base, lk.Commit, name, path, sum)
			}
		}
		return path, nil
	}
	if !errors.Is(err, shared.ErrNotFound) && log != nil {
		fmt.Fprintf(log, "%v\n", err)
	}
	if !shared.HuggingFace() {
		return "", fmt.Errorf("chat: %s of %s is in none of the sources (%s), and %s=false keeps hwnpu off Hugging Face: %w",
			name, repo, sourceNames(ss), shared.HuggingFaceEnv, errNotFound)
	}
	from := repo
	if commit != "" {
		from = base + "@" + commit // the files of the sources' commit
	}
	if err := fetchHF(ctx, from, name, path, log); err != nil {
		return "", err
	}
	c, _ := servedCommit(from)
	if err := record(c, ""); err != nil {
		return "", err
	}
	for _, s := range shared.Writebacks() {
		if c != "" {
			shared.FillLater(s, base, c, name, path, want)
		}
	}
	return path, nil
}

// here reports whether the file of repo base at path is the one the machine's lock names, of
// commit (any, for "") and of the SHA-256 the definitions pin (want, when not "").
func here(ctx context.Context, local shared.Source, base, commit, name, want, path string) bool {
	l, err := shared.ReadLock(ctx, local, base)
	if err != nil {
		return false
	}
	f, ok := l.Files[name]
	if !ok || commit != "" && l.Commit != commit || want != "" && f.SHA256 != want {
		return false
	}
	fi, err := os.Stat(path)
	if err != nil || fi.Size() != f.Size {
		return false
	}
	if commit == "" {
		noteLockCommit(base, l.Commit)
	}
	return true
}

// adopt takes a file at path that the machine's lock does not name (one of another commit, from
// before the lock moved; or one of an earlier hwnpu's models/NAME/, LegacyModelDir) when it has
// the SHA-256 it should have: the one the definitions pin, a source's lock names or Hugging Face
// says (a file in LFS), and reports whether it did. One that differs is removed. A file whose
// SHA-256 is not known is taken only when hwnpu keeps off Hugging Face (else it comes again: a
// small file, not in LFS).
func adopt(ctx context.Context, local shared.Source, repo, commit, name, want, path string, log io.Writer) bool {
	base, _, _ := strings.Cut(repo, "@")
	if _, err := os.Lstat(path); err != nil {
		old := filepath.Join(LegacyModelDir(base), filepath.FromSlash(name))
		if fi, err := os.Stat(old); err != nil || !fi.Mode().IsRegular() {
			return false
		}
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil || os.Rename(old, path) != nil {
			return false
		}
	}
	sum := want
	if sum == "" && commit != "" {
		for _, s := range shared.Sources() {
			if l, err := shared.ReadLock(ctx, s, base); err == nil && l.Commit == commit {
				if f, ok := l.Files[name]; ok {
					sum = f.SHA256
					break
				}
			}
		}
	}
	c := commit
	if sum == "" && shared.HuggingFace() {
		from := repo
		if commit != "" {
			from = base + "@" + commit
		}
		var hc string
		if sum, hc = hubFile(ctx, from, name); c == "" {
			c = hc
		}
		if sum == "" {
			os.Remove(path)
			return false
		}
	}
	got, err := shared.FileSHA256(path)
	if err != nil {
		return false
	}
	if sum != "" && got != sum {
		if log != nil {
			fmt.Fprintf(log, "%s of %s here is not the one of %s: downloading it again\n", name, base, short(c))
		}
		os.Remove(path)
		return false
	}
	fi, err := os.Stat(path)
	if err != nil {
		return false
	}
	if c == "" {
		if l, err := shared.ReadLock(ctx, local, base); err == nil {
			c = l.Commit // hwnpu off Hugging Face, no source's lock: the commit of what is here
		}
	}
	if shared.Record(ctx, local, base, c, name, shared.LockedFile{SHA256: got, Size: fi.Size()}, true, nil) != nil {
		return false
	}
	noteLockCommit(repo, c)
	planFinish(path)
	return true
}

func short(commit string) string {
	if commit == "" {
		return "the commit asked for"
	}
	return commit[:min(12, len(commit))]
}

// LegacyModelDir is where an earlier hwnpu kept the files of repo base: models/NAME under the
// cache's root. A file asked for that is there is moved to RepoDir when it is the right one.
func LegacyModelDir(base string) string { return cache.Sub("models", path.Base(base)) }

// lockedCommit is the commit fetch takes the files of repo from: the revision the definitions
// pin when it is a commit, else the one this process took them from before; for main, else the
// one of the first source with a lock of the repository, else the one of the machine's cache
// (what is here: Hugging Face is not asked whether main moved; hwnpu-dev cache update moves it);
// for another branch or tag, the one Hugging Face says it is. "" for any (Hugging Face's main).
func lockedCommit(ctx context.Context, repo string) string {
	if c, ok := servedCommit(repo); ok {
		return c
	}
	base, rev, _ := strings.Cut(repo, "@")
	if rev != "" && rev != "main" {
		if len(shared.Sources()) == 0 {
			return "" // nothing to match: Hugging Face serves the revision
		}
		if _, c := hubFile(ctx, repo, "config.json"); c != "" {
			noteLockCommit(repo, c)
			return c
		}
		return "" // what the sources hold of the revision is not known: any lock is taken
	}
	for _, s := range append(shared.Sources(), shared.Local()) {
		if l, err := shared.ReadLock(ctx, s, base); err == nil && l.Commit != "" {
			noteLockCommit(repo, l.Commit)
			return l.Commit
		}
	}
	return ""
}

// noteLockCommit keeps the commit repo's files come from in this process (servedCommit).
func noteLockCommit(repo, commit string) {
	if commit == "" {
		return
	}
	base, rev, _ := strings.Cut(repo, "@")
	if commitRE.MatchString(rev) {
		return
	}
	commitsMu.Lock()
	commits[base+"@"+cmp.Or(rev, "main")] = commit
	commitsMu.Unlock()
}

func sourceNames(ss []shared.Source) string {
	if len(ss) == 0 {
		return "none: " + shared.SourceEnv + " sets them"
	}
	var n []string
	for _, s := range ss {
		n = append(n, s.String())
	}
	return strings.Join(n, ", ")
}

// Pinned is the SHA-256 the definitions pin for a file of repo (as Fetch takes it), if any: a
// download of it is checked against it.
func Pinned(repo, name string) (string, bool) {
	d, ok := digests[repo][name]
	return d, ok
}

// verify checks a fetched file against the SHA-256 the definitions pin for it, if any.
func verify(repo, name, path string) error {
	want, ok := digests[repo][name]
	if !ok {
		return nil
	}
	if err := hasDigest(path, want); err != nil {
		return fmt.Errorf("chat: %s of %s: %w", name, repo, err)
	}
	return nil
}

// hasDigest checks that the file at path has the SHA-256 want.
func hasDigest(path, want string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return err
	}
	if got := hex.EncodeToString(h.Sum(nil)); !strings.EqualFold(got, want) {
		return fmt.Errorf("SHA-256 %s, not the %s of the definitions", got, want)
	}
	return nil
}

// sameFile puts at path (the file name of a repository here) the file of the SHA-256 want
// another repository has here, linked (or copied where links fail), and reports whether it did.
// A copy that does not have the SHA-256 is removed and the next looked at.
func sameFile(want, name, path string, log io.Writer) bool {
	for _, f := range pinned[strings.ToLower(want)] {
		src := filepath.Join(RepoDir(f.repo), filepath.FromSlash(f.name))
		if filepath.Clean(src) == filepath.Clean(path) {
			continue
		}
		if fi, err := os.Stat(src); err != nil || !fi.Mode().IsRegular() {
			continue
		}
		label := fmt.Sprintf("%s of %s (the same SHA-256)", f.name, f.repo)
		if err := linkOrCopy(src, path, label, log); err != nil {
			if log != nil {
				fmt.Fprintf(log, "%s: %v\n", label, err)
			}
			continue
		}
		if err := hasDigest(path, want); err != nil {
			os.Remove(path)
			if log != nil {
				fmt.Fprintf(log, "%s: %v\n", label, err)
			}
			continue
		}
		return true
	}
	return false
}

// linkOrCopy puts src at dst: a hard link where the folders allow it, else a copy (through
// dst.part). what names src in the log.
func linkOrCopy(src, dst, what string, log io.Writer) error {
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	if os.Link(src, dst) == nil {
		if log != nil {
			fmt.Fprintf(log, "linked %s from %s\n", filepath.Base(dst), what)
		}
		return nil
	}
	fi, err := os.Stat(src)
	if err != nil {
		return err
	}
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	part := dst + ".part"
	out, err := os.Create(part)
	if err != nil {
		return err
	}
	pw := &progress{w: out, total: fi.Size(), bar: console.NewBar(log, fmt.Sprintf("copying %s from %s", filepath.Base(dst), what))}
	_, err = io.Copy(pw, in)
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		os.Remove(part)
		return fmt.Errorf("chat: copying %s: %w", src, err)
	}
	pw.bar.Done(fmt.Sprintf("%s to %s", gb(pw.done, -1), filepath.Dir(dst)))
	return os.Rename(part, dst)
}
