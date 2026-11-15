package store

import (
	"cmp"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/shibukawa/hwnpu/internal/cache"
)

// A source is a place models are looked for in before Hugging Face (docs/cache-sharing.md 7): a
// folder (a mounted NAS, a local disk), an HTTP server (WebDAV, or any server read only) or S3.
// The sources are HWNPU_MODEL_SOURCE, then HWNPU_MODEL_SOURCE_1, _2, ... (every variable that
// starts with HWNPU_MODEL_SOURCE, in the order of its number), each a URL:
//
//	file:///mnt/llm/hwnpu          a folder (file:///C:/models, file://server/share on Windows)
//	https://USER:PASSWORD@host/dav a WebDAV folder; read with GETs, so any HTTP server serves it
//	                               read only (the password: here, ~/.netrc or HWNPU_WEBDAV_PASSWORD;
//	                               ?tls_sha256=HEX trusts a NAS's own certificate)
//	s3://BUCKET/PREFIX?endpoint=http://host:9000&region=R&profile=P&path_style=true&no_sign=true
//
// ?writeback=true puts in the source what is made here and what is read from a later source or
// Hugging Face (a forward proxy). HWNPU_MODEL_HUGGING_FACE=false keeps hwnpu off Hugging Face:
// everything comes from the sources.
//
// What a source holds:
//
//	original/OWNER/NAME/FILE...                      a repository's files, as on Hugging Face
//	original/OWNER/NAME/hwnpu-lock.json              their commit, SHA-256 and sizes (Lock)
//	converted/MODEL/TARGET/EXPORT/VERSION/...        what was converted (Entry): VERSION is vN,
//	converted/MODEL/TARGET/EXPORT/VERSION/entry.json vN-HASH (develop) or v0-HASH (no version)

// SourceEnv is the first source's variable; SourceEnv_1, _2, ... follow it.
const SourceEnv = "HWNPU_MODEL_SOURCE"

// HuggingFaceEnv=false keeps hwnpu off Hugging Face (true, the default: the last source).
const HuggingFaceEnv = "HWNPU_MODEL_HUGGING_FACE"

// Source is a place models are looked for in.
type Source struct {
	Store Store
	// Writeback puts in the source what is made here and what is read from further on.
	Writeback bool
	// From is the variable that set it.
	From string
}

func (s Source) String() string { return s.Store.String() }

// ParseSource reads a source's URL; name names it in logs (its variable).
func ParseSource(name, raw string) (Source, error) {
	s := Source{From: name}
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return s, fmt.Errorf("store: %s: %w", name, err)
	}
	q := u.Query()
	flag := func(k string) (bool, error) {
		v := q.Get(k)
		q.Del(k)
		if v == "" {
			return false, nil
		}
		b, err := strconv.ParseBool(v)
		if err != nil {
			return false, fmt.Errorf("store: %s: %s=%q is not true or false", name, k, v)
		}
		return b, nil
	}
	if s.Writeback, err = flag("writeback"); err != nil {
		return s, err
	}
	switch u.Scheme {
	case "file":
		p, err := filePath(u)
		if err != nil {
			return s, fmt.Errorf("store: %s: %w", name, err)
		}
		s.Store = Dir{Name: name, Root: p}
	case "http", "https":
		c := WebDAVConfig{Name: name, ReadOnly: !s.Writeback, TLSSHA256: q.Get("tls_sha256")}
		q.Del("tls_sha256")
		if u.User != nil {
			c.User = u.User.Username()
			c.Password, _ = u.User.Password()
		}
		v := *u
		v.User, v.RawQuery = nil, ""
		c.URL = v.String()
		if len(q) == 0 {
			st, err := OpenWebDAV(c)
			if err != nil {
				return s, err
			}
			s.Store = st
		}
	case "s3":
		c := S3Config{Name: name, Bucket: u.Host, Prefix: strings.TrimPrefix(u.Path, "/"),
			Endpoint: q.Get("endpoint"), Region: q.Get("region"), Profile: q.Get("profile")}
		q.Del("endpoint")
		q.Del("region")
		q.Del("profile")
		if c.Bucket == "" {
			return s, fmt.Errorf("store: %s: %q has no bucket (s3://BUCKET/PREFIX)", name, raw)
		}
		if q.Has("path_style") {
			b, err := flag("path_style")
			if err != nil {
				return s, err
			}
			c.PathStyle = &b
		}
		if c.NoSign, err = flag("no_sign"); err != nil {
			return s, err
		}
		if len(q) == 0 {
			st, err := OpenS3(c)
			if err != nil {
				return s, err
			}
			s.Store = st
		}
	default:
		return s, fmt.Errorf("store: %s: %q is not a file://, http://, https:// or s3:// URL", name, raw)
	}
	if len(q) > 0 {
		keys := slices.Sorted(func(yield func(string) bool) {
			for k := range q {
				if !yield(k) {
					return
				}
			}
		})
		return s, fmt.Errorf("store: %s: %s is not a setting of a %s source", name, strings.Join(keys, ", "), u.Scheme)
	}
	return s, nil
}

var drive = regexp.MustCompile(`^/[A-Za-z]:`)

// filePath is the folder of a file:// URL.
func filePath(u *url.URL) (string, error) {
	p := u.Path
	switch {
	case u.Host != "" && u.Host != "localhost":
		if runtime.GOOS != "windows" {
			return "", fmt.Errorf("file://%s%s: a host is a share of Windows (file:///PATH for a folder here)", u.Host, p)
		}
		return `\\` + u.Host + filepath.FromSlash(p), nil
	case drive.MatchString(p):
		p = p[1:]
	case strings.HasPrefix(u.Opaque, "~"), strings.HasPrefix(p, "/~/"):
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		p = filepath.Join(home, strings.TrimPrefix(strings.TrimPrefix(cmp.Or(u.Opaque, p), "/"), "~"))
	}
	if p == "" || !filepath.IsAbs(filepath.FromSlash(p)) {
		return "", fmt.Errorf("file:%s is not a whole path (file:///mnt/share)", cmp.Or(u.Opaque, p))
	}
	return filepath.Clean(filepath.FromSlash(p)), nil
}

// FromEnv reads the sources of the environment, in order, and whether Hugging Face is read
// after them.
func FromEnv() ([]Source, bool, error) {
	type named struct {
		name, value string
		n           int // the number after SourceEnv_; -1 for SourceEnv itself
	}
	var vars []named
	for _, kv := range os.Environ() {
		k, v, _ := strings.Cut(kv, "=")
		rest, ok := strings.CutPrefix(k, SourceEnv)
		if !ok || strings.TrimSpace(v) == "" {
			continue
		}
		n := -1
		if rest != "" {
			num, ok := strings.CutPrefix(rest, "_")
			if !ok {
				continue // another variable (HWNPU_MODEL_SOURCES)
			}
			i, err := strconv.Atoi(num)
			if err != nil || i < 0 {
				return nil, false, fmt.Errorf("store: %s: the sources are %s, %s_1, %s_2, ...", k, SourceEnv, SourceEnv, SourceEnv)
			}
			n = i
		}
		vars = append(vars, named{k, v, n})
	}
	slices.SortFunc(vars, func(a, b named) int { return cmp.Or(cmp.Compare(a.n, b.n), cmp.Compare(a.name, b.name)) })
	var out []Source
	var errs []string
	for _, v := range vars {
		s, err := ParseSource(v.name, v.value)
		if err != nil {
			errs = append(errs, err.Error())
			continue
		}
		out = append(out, s)
	}
	hf := true
	if v := strings.TrimSpace(os.Getenv(HuggingFaceEnv)); v != "" {
		b, err := strconv.ParseBool(v)
		if err != nil {
			errs = append(errs, fmt.Sprintf("store: %s=%q is not true or false", HuggingFaceEnv, v))
		} else {
			hf = b
		}
	}
	if len(errs) > 0 {
		return out, hf, fmt.Errorf("%s", strings.Join(errs, "; "))
	}
	return out, hf, nil
}

var (
	sourcesMu   sync.RWMutex
	sources     []Source
	bundles     []Source
	huggingFace = true
)

// SetSources sets the sources of the environment and whether Hugging Face is read after them.
func SetSources(ss []Source, hf bool) {
	sourcesMu.Lock()
	sources, huggingFace = ss, hf
	sourcesMu.Unlock()
	clearLocks()
}

// SetBundles sets the folders of models an application installs with it (read only, used in
// place), read after the sources of the environment and before Hugging Face.
func SetBundles(dirs []string) {
	var bs []Source
	for _, d := range dirs {
		bs = append(bs, Source{Store: Dir{Name: "bundle", Root: d, InPlace: true}, From: "bundle"})
	}
	sourcesMu.Lock()
	bundles = bs
	sourcesMu.Unlock()
	clearLocks()
}

// Sources are the layers after the machine's own cache: the sources of the environment, then
// the bundled models.
func Sources() []Source {
	sourcesMu.RLock()
	defer sourcesMu.RUnlock()
	if len(bundles) == 0 {
		return sources
	}
	return append(slices.Clip(sources), bundles...)
}

// Local is the first layer: the machine's own cache (cache.Root), which holds what is read from
// the others and what is made here, in the layout of a source.
func Local() Source {
	return Source{Store: Dir{Name: "cache", Root: cache.Root()}, Writeback: true, From: cache.RootEnv}
}

// HuggingFace reports whether Hugging Face is read after the sources.
func HuggingFace() bool {
	sourcesMu.RLock()
	defer sourcesMu.RUnlock()
	return huggingFace
}

// Writebacks are the sources that are written back to.
func Writebacks() []Source {
	var out []Source
	for _, s := range Sources() {
		if s.Writeback {
			out = append(out, s)
		}
	}
	return out
}
