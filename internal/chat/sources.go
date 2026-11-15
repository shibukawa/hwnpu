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
	"path/filepath"
	"slices"
	"strings"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/console"
	shared "github.com/shibukawa/hwnpu/internal/store"
)

// Where the files of the models come from (docs/cache-sharing.md 7): the sources of the
// environment (HWNPU_MODEL_SOURCE, HWNPU_MODEL_SOURCE_1, ...: internal/store), in order, and
// Hugging Face last unless HWNPU_MODEL_HUGGING_FACE=false. A repository's files in a source are
// those its hwnpu-lock.json names, of the commit it names: a model that does not pin a commit
// takes the commit of the first source's lock, so that every machine of the sources runs the same
// files (hwnpu-dev cache update moves the lock). A file read from a later source or Hugging Face
// goes in the sources before with ?writeback=true, in the background.

var (
	// digests are the SHA-256 of the files of the models the definitions pin (sha256), by the
	// model's Source() and the file's name.
	digests = map[string]map[string]string{}
	// pinned are the files the definitions pin, by their SHA-256 (lower case): where each is in
	// the models' directory, so that a file two models share (MOSS-SoundEffect's text encoder is
	// Qwen3-1.7B) is downloaded once (sameFile).
	pinned = map[string][]pinnedFile{}
)

// safeName reports whether name is a relative path of a source's model that stays inside it.
func safeName(name string) bool {
	if name == "" || strings.HasPrefix(name, "/") || strings.Contains(name, "\\") || strings.Contains(name, ":") {
		return false
	}
	return !slices.Contains(strings.Split(name, "/"), "..")
}

// pinnedFile is a file of a model the definitions pin: its model's directory (Model.DirName) and
// its name in the repository.
type pinnedFile struct{ dir, name string }

// fetch puts a file of a model's repository at path unless it is there. repo is Model.Source():
// OWNER/NAME or OWNER/NAME@REVISION. The file is looked for in the sources (shared.Sources), then
// on Hugging Face (unless HWNPU_MODEL_HUGGING_FACE=false), of the commit the model pins or the
// first source's lock names. A file the definitions pin (sha256) is checked; one that differs is
// removed. A pinned file another model of the same SHA-256 has in the models' directory is linked
// from there first (sameFile).
func fetch(ctx context.Context, repo, name, path string, log io.Writer) error {
	if _, err := os.Stat(path); err == nil {
		return nil
	}
	// one process downloads a file at a time; one that waited finds it there
	unlock, err := cache.Lock(ctx, path, repo+"/"+name, log)
	if err != nil {
		return err
	}
	defer unlock()
	if _, err := os.Stat(path); err == nil {
		return nil
	}
	stop := console.Downloading(repo + "/" + name)
	bytes := int64(-1) // linked from another model: no download
	defer func() { stop(bytes) }()
	want, pinnedSum := digests[repo][name]
	if pinnedSum && sameFile(want, name, path, log) {
		planFinish(path)
		return nil
	}
	done := func() error {
		if err := verify(repo, name, path); err != nil {
			os.Remove(path)
			return err
		}
		if fi, err := os.Stat(path); err == nil {
			bytes = fi.Size()
		}
		planFinish(path)
		return nil
	}
	base, rev, _ := strings.Cut(repo, "@")
	commit := lockedCommit(ctx, repo)
	ss := shared.Sources()
	if commit == "" && rev != "" && rev != "main" && shared.HuggingFace() {
		ss = nil // a branch or a tag whose commit is not known: what the sources hold may be another's
	}
	i, lk, err := shared.FetchOriginal(ctx, ss, base, commit, name, strings.ToLower(want), path, log)
	if err == nil {
		if err := done(); err != nil {
			return err
		}
		noteLockCommit(repo, lk.Commit)
		for _, s := range ss[:i] {
			if s.Writeback {
				shared.FillLater(s, base, lk.Commit, name, path, lk.Files[name].SHA256)
			}
		}
		return nil
	}
	if !errors.Is(err, shared.ErrNotFound) && log != nil {
		fmt.Fprintf(log, "%v\n", err)
	}
	if !shared.HuggingFace() {
		return fmt.Errorf("chat: %s of %s is in none of the sources (%s), and %s=false keeps hwnpu off Hugging Face: %w",
			name, repo, sourceNames(ss), shared.HuggingFaceEnv, errNotFound)
	}
	from := repo
	if commit != "" {
		from = base + "@" + commit // the files of the sources' commit
	}
	if err := fetchHF(ctx, from, name, path, log); err != nil {
		return err
	}
	if err := done(); err != nil {
		return err
	}
	c, _ := servedCommit(from)
	if c != "" && commit == "" {
		noteLockCommit(repo, c)
	}
	for _, s := range shared.Sources() {
		if s.Writeback && c != "" {
			shared.FillLater(s, base, c, name, path, strings.ToLower(want))
		}
	}
	return nil
}

// lockedCommit is the commit fetch takes the files of repo from: the revision the definitions
// pin when it is a commit, else the one this process took them from before; for main, else the
// one of the first source with a lock of the repository; for another branch or tag, the one
// Hugging Face says it is. "" for any.
func lockedCommit(ctx context.Context, repo string) string {
	if c, ok := servedCommit(repo); ok {
		return c
	}
	if len(shared.Sources()) == 0 {
		return "" // nothing to match: Hugging Face serves the revision
	}
	base, rev, _ := strings.Cut(repo, "@")
	if rev != "" && rev != "main" {
		if _, c := hubFile(ctx, repo, "config.json"); c != "" {
			noteLockCommit(repo, c)
			return c
		}
		return "" // what the sources hold of the revision is not known: any lock is taken
	}
	for _, s := range shared.Sources() {
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

// sameFile puts at path (the file name of a model's directory in the models' directory) the file
// of the SHA-256 want another model has there, linked (or copied where links fail), and reports
// whether it did. A copy that does not have the SHA-256 is removed and the next looked at.
func sameFile(want, name, path string, log io.Writer) bool {
	dir, ok := strings.CutSuffix(filepath.Clean(path), string(filepath.Separator)+filepath.Clean(filepath.FromSlash(name)))
	if !ok {
		return false // not NAME's place in a model's directory: no models' directory to look in
	}
	root := filepath.Dir(dir)
	for _, f := range pinned[strings.ToLower(want)] {
		src := filepath.Join(root, f.dir, filepath.FromSlash(f.name))
		if filepath.Clean(src) == filepath.Clean(path) {
			continue
		}
		if fi, err := os.Stat(src); err != nil || !fi.Mode().IsRegular() {
			continue
		}
		label := fmt.Sprintf("%s of %s (the same SHA-256)", f.name, f.dir)
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

