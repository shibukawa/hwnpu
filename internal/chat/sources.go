package chat

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"sync"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/console"
	shared "github.com/shibukawa/hwnpu/internal/store"
)

// Where the files of the models come from (docs/model-table.md 5): Hugging Face (the source
// "hf", always there), shared folders (path) and S3 (s3, any endpoint: AWS, RustFS, Cloudflare R2; read with
// internal/store, not the aws command). A model's from names its source; a model from hf is looked for in the sources
// of Search in order, so that a mirror of Hugging Face is read first.

// SourceHF is the name of Hugging Face's source.
const SourceHF = "hf"

// Source is a place the files of models are read from.
type Source struct {
	Name string
	Type string // "hf", "path" or "s3"
	// Path is a path source's folder: a model's files are in Path/NAME (for hf models in a
	// search, Path/OWNER/NAME, or Path/OWNER/NAME@REVISION first when pinned).
	Path string
	// The bucket and the prefix of an s3 source, and how it is reached: the endpoint (for
	// RustFS, Cloudflare R2 and the like), the region, the profile of ~/.aws/credentials and
	// ~/.aws/config whose static keys sign, path_style (nil: paths except on AWS's endpoints),
	// and no_sign for a public bucket (shared.S3Config).
	Bucket, Prefix, Endpoint, Region, Profile string
	PathStyle                                 *bool
	NoSign                                    bool
}

// String describes the source for logs and -list.
func (s Source) String() string {
	switch s.Type {
	case "path":
		return fmt.Sprintf("%s (folder %s)", s.Name, s.Path)
	case "s3":
		d := fmt.Sprintf("%s (s3://%s/%s", s.Name, s.Bucket, s.Prefix)
		if s.Endpoint != "" {
			d += " at " + s.Endpoint
		}
		return d + ")"
	}
	return s.Name + " (Hugging Face)"
}

var (
	// Sources are the sources of the definitions by name, Hugging Face's among them.
	Sources = map[string]Source{SourceHF: {Name: SourceHF, Type: "hf"}}
	// Search is the order the sources are looked in for a model from hf (the definitions'
	// search; [hf] when they give none).
	Search = []string{SourceHF}
	// digests are the SHA-256 of the files of the models the definitions pin (sha256), by the
	// model's Source() and the file's name.
	digests = map[string]map[string]string{}
	// pinned are the files the definitions pin, by their SHA-256 (lower case): where each is in
	// the models' directory, so that a file two models share (MOSS-SoundEffect's text encoder is
	// Qwen3-1.7B) is downloaded once (sameFile).
	pinned = map[string][]pinnedFile{}

	// found remembers the source a repository's first file came from, to take the rest from
	// there too.
	foundMu sync.Mutex
	found   = map[string]string{}
)

// sourceEntry is a source in a definition file.
type sourceEntry struct {
	Type      string `yaml:"type"`
	Path      string `yaml:"path"`
	Bucket    string `yaml:"bucket"`
	Prefix    string `yaml:"prefix"`
	Endpoint  string `yaml:"endpoint"`
	Region    string `yaml:"region"`
	Profile   string `yaml:"profile"`
	PathStyle *bool  `yaml:"path_style"`
	NoSign    bool   `yaml:"no_sign"`
}

var sourceName = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_-]*$`)

// toSource checks a source of a definition file.
func toSource(name string, e *sourceEntry) (Source, error) {
	s := Source{Name: name, Type: e.Type, Path: e.Path, Bucket: e.Bucket, Prefix: e.Prefix, Endpoint: e.Endpoint, Region: e.Region, Profile: e.Profile, PathStyle: e.PathStyle, NoSign: e.NoSign}
	if !sourceName.MatchString(name) || name == SourceHF {
		return s, fmt.Errorf("source %q: a name of letters, digits, - and _, not %s", name, SourceHF)
	}
	switch s.Type {
	case "path":
		if s.Path == "" {
			return s, fmt.Errorf("source %s: a path source needs path", name)
		}
		if rest, ok := strings.CutPrefix(s.Path, "~"); ok && (rest == "" || rest[0] == '/' || rest[0] == '\\') {
			home, err := os.UserHomeDir()
			if err != nil {
				return s, err
			}
			s.Path = filepath.Join(home, rest)
		}
	case "s3":
		if s.Bucket == "" {
			return s, fmt.Errorf("source %s: an s3 source needs bucket", name)
		}
		if s.Prefix != "" && !strings.HasSuffix(s.Prefix, "/") {
			s.Prefix += "/"
		}
	default:
		return s, fmt.Errorf("source %s: type %q is not path or s3", name, s.Type)
	}
	return s, nil
}

// splitSource reads a repository as fetch takes it: SOURCE:NAME for a model of a path or s3
// source, else OWNER/NAME[@REVISION], from hf (looked for in Search).
func splitSource(repo string) (source, name string) {
	if s, n, ok := strings.Cut(repo, ":"); ok {
		if _, known := Sources[s]; known && s != SourceHF {
			return s, n
		}
	}
	return "", repo
}

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
// OWNER/NAME or OWNER/NAME@REVISION from hf, looked for in Search in order (the source the
// repository's first file came from first), or SOURCE:NAME from that source alone. A file the
// definitions pin (sha256) is checked; one that differs is removed and the next source tried. A
// pinned file another model of the same SHA-256 has in the models' directory is linked from there
// first (sameFile).
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
	if want, ok := digests[repo][name]; ok && sameFile(want, name, path, log) {
		planFinish(path)
		return nil
	}
	order, key := sourceOrder(repo)
	var errs []error
	for _, sn := range order {
		s := Sources[sn]
		var err error
		fromLayer := false
		switch s.Type {
		case "hf":
			// the shared layers first: a forward proxy of the hub (docs/cache-sharing.md 7.1)
			if fromLayer = fromLayers(ctx, key, name, path, log); !fromLayer {
				err = fetchHF(ctx, key, name, path, log)
			}
		case "path":
			err = fetchPath(s, key, name, path, log)
		case "s3":
			err = fetchS3(ctx, s, key, name, path, log)
		default:
			err = fmt.Errorf("chat: no source %q", sn)
		}
		if err == nil {
			if err = verify(repo, name, path); err == nil {
				if fi, err := os.Stat(path); err == nil {
					bytes = fi.Size()
				}
				planFinish(path)
				foundMu.Lock()
				found[repo] = sn
				foundMu.Unlock()
				if !fromLayer && s.Type == "hf" {
					fillLayers(ctx, key, name, path)
				}
				return nil
			}
			os.Remove(path)
		}
		if errors.Is(err, errNotFound) {
			continue
		}
		errs = append(errs, err)
		if len(order) > 1 && log != nil {
			fmt.Fprintf(log, "%s: %v; trying the next source\n", s.Name, err)
		}
	}
	if len(errs) > 0 {
		return errors.Join(errs...)
	}
	return fmt.Errorf("chat: %s of %s is in none of the sources %s: %w", name, key, strings.Join(order, ", "), errNotFound)
}

// sourceOrder is the sources fetch looks in for a file of repo, in order, and the repository's
// name in them.
func sourceOrder(repo string) (order []string, key string) {
	src, key := splitSource(repo)
	if src != "" {
		return []string{src}, key
	}
	order = slices.Clone(Search)
	foundMu.Lock()
	if f, ok := found[repo]; ok {
		order = append([]string{f}, slices.DeleteFunc(order, func(s string) bool { return s == f })...)
	}
	foundMu.Unlock()
	return order, key
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

// fetchPath copies a file of a path source's model: Path/NAME/FILE, or for a model from hf
// looked for in it, Path/OWNER/NAME@REVISION/FILE and then Path/OWNER/NAME/FILE. A hard link is
// made where the folder allows it.
func fetchPath(s Source, key, name, dst string, log io.Writer) error {
	base, rev, _ := strings.Cut(key, "@")
	if !safeName(base) || !safeName(name) || strings.ContainsAny(rev, "/\\") {
		return fmt.Errorf("chat: %s: %s/%s is not a file inside the source", s.Name, key, name)
	}
	var dirs []string
	if rev != "" {
		dirs = append(dirs, base+"@"+rev)
	}
	dirs = append(dirs, base)
	for _, d := range dirs {
		src := filepath.Join(s.Path, filepath.FromSlash(d), filepath.FromSlash(name))
		fi, err := os.Stat(src)
		if err != nil || fi.IsDir() {
			continue
		}
		return linkOrCopy(src, dst, s.String(), log)
	}
	return fmt.Errorf("chat: %s: %s/%s: %w", s.Name, key, name, errNotFound)
}

// fetchS3 downloads a file of an s3 source's model: s3://BUCKET/PREFIX/NAME/FILE, or for a
// model from hf, PREFIX/OWNER/NAME@REVISION/FILE and then PREFIX/OWNER/NAME/FILE (internal/store:
// signed with the profile's or the environment's keys, or unsigned with no_sign).
func fetchS3(ctx context.Context, s Source, key, name, dst string, log io.Writer) error {
	base, rev, _ := strings.Cut(key, "@")
	if !safeName(base) || !safeName(name) || strings.ContainsAny(rev, "/\\") {
		return fmt.Errorf("chat: %s: %s/%s is not a file inside the source", s.Name, key, name)
	}
	st, err := sourceStore(s)
	if err != nil {
		return err
	}
	var keys []string
	if rev != "" {
		keys = append(keys, base+"@"+rev)
	}
	keys = append(keys, base)
	for _, k := range keys {
		obj := k + "/" + name
		b := console.NewBar(log, fmt.Sprintf("downloading %s from %s", path.Base(name), st))
		n, err := st.Get(ctx, obj, dst, func(done, total int64) {
			if total > 0 {
				b.Set(float64(done)/float64(total), gb(done, total))
			} else {
				b.Set(-1, gb(done, -1))
			}
		})
		if errors.Is(err, shared.ErrNotFound) {
			b.Done("not there")
			continue
		}
		if err != nil {
			b.Done("failed")
			return fmt.Errorf("chat: %s: %s: %w", s.Name, obj, err)
		}
		b.Done(fmt.Sprintf("%s in %s to %s", gb(n, -1), console.Duration(b.Elapsed()), filepath.Dir(dst)))
		return nil
	}
	return fmt.Errorf("chat: %s: %s/%s: %w", s.Name, key, name, errNotFound)
}
