package chat

import (
	"cmp"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"

	shared "github.com/shibukawa/hwnpu/internal/store"
)

// The shared layers (docs/cache-sharing.md 7): the folders and S3 buckets of the definitions'
// sources that cache.layers names, in order. What is read is looked for in them before Hugging
// Face, and what is read from further on or made here is put in those that take it (fill,
// write), in the background.

// LayersEnv replaces the layers of the definitions: a comma-separated list of folders and
// s3://BUCKET/PREFIX URLs, each with settings as a query (?write=1&fill=1&fill_gated=1&
// scopes=upstream+portable&endpoint=http://host:9000&region=R&profile=P&path_style=1&no_sign=1),
// or "off" for none.
const LayersEnv = "HWNPU_CACHE_LAYERS"

// LayersWriteEnv=0 keeps every layer read-only: nothing is written or filled.
const LayersWriteEnv = "HWNPU_CACHE_WRITE"

// cacheEntry is the cache section of a definition file.
type cacheEntry struct {
	Layers *[]layerEntry `yaml:"layers"`
	// Prebuilt is designed (auto, off, only: docs/cache-sharing.md 7) but not read yet.
	Prebuilt *string `yaml:"prebuilt"`
}

// layerEntry is a layer in a definition file: a source of sources by name.
type layerEntry struct {
	Use       string   `yaml:"use"`
	Write     bool     `yaml:"write"`
	Fill      bool     `yaml:"fill"`
	FillGated bool     `yaml:"fill_gated"`
	Scopes    []string `yaml:"scopes"`
}

var scopes = []string{shared.ScopeUpstream, shared.ScopePortable, shared.ScopeHost}

// openedStores are the stores of the sources by name, opened once (S3's keys are read then).
var (
	storesMu     sync.Mutex
	openedStores = map[string]shared.Store{}
)

// sourceStore is the Store of a path or s3 source.
func sourceStore(s Source) (shared.Store, error) {
	storesMu.Lock()
	defer storesMu.Unlock()
	ps := ""
	if s.PathStyle != nil {
		ps = strconv.FormatBool(*s.PathStyle)
	}
	id := fmt.Sprintf("%s|%s|%s|%s|%s|%s|%s|%s|%s|%v", s.Name, s.Type, s.Path, s.Bucket, s.Prefix, s.Endpoint, s.Region, s.Profile, ps, s.NoSign)
	if st, ok := openedStores[id]; ok {
		return st, nil
	}
	var st shared.Store
	switch s.Type {
	case "path":
		st = shared.Dir{Name: s.Name, Root: s.Path}
	case "s3":
		s3, err := shared.OpenS3(shared.S3Config{Name: s.Name, Bucket: s.Bucket, Prefix: s.Prefix, Endpoint: s.Endpoint,
			Region: s.Region, Profile: s.Profile, PathStyle: s.PathStyle, NoSign: s.NoSign})
		if err != nil {
			return nil, err
		}
		st = s3
	default:
		return nil, fmt.Errorf("chat: the source %s is not a folder or S3: it is not a layer", s.Name)
	}
	openedStores[id] = st
	return st, nil
}

// buildLayers makes the layers of the definitions (entries over sources), or of LayersEnv when
// it is set. A layer that cannot be opened (no keys for S3) is warned of and left out.
func buildLayers(entries []layerEntry, from string, sources map[string]Source, warn func(string, string, ...any)) ([]shared.Layer, error) {
	if env, ok := os.LookupEnv(LayersEnv); ok && strings.TrimSpace(env) != "" {
		var err error
		entries, sources, err = envLayers(env)
		if err != nil {
			return nil, err
		}
		from = LayersEnv
	}
	readOnly := os.Getenv(LayersWriteEnv) == "0"
	var out []shared.Layer
	for _, e := range entries {
		s, ok := sources[e.Use]
		if !ok {
			return nil, fmt.Errorf("chat: %s: cache.layers uses %q, which sources does not have", from, e.Use)
		}
		for _, sc := range e.Scopes {
			if !slices.Contains(scopes, sc) {
				return nil, fmt.Errorf("chat: %s: the layer %s: scope %q is not one of %s", from, e.Use, sc, strings.Join(scopes, ", "))
			}
		}
		st, err := sourceStore(s)
		if err != nil {
			warn(from, "the layer %s is left out: %v", e.Use, err)
			continue
		}
		l := shared.Layer{Store: st, Write: e.Write, Fill: e.Fill, FillGated: e.FillGated, Scopes: e.Scopes, From: from}
		if readOnly {
			l.Write, l.Fill = false, false
		}
		out = append(out, l)
	}
	return out, nil
}

// envLayers reads LayersEnv.
func envLayers(env string) ([]layerEntry, map[string]Source, error) {
	sources := map[string]Source{}
	var entries []layerEntry
	if strings.TrimSpace(env) == "off" {
		return nil, sources, nil
	}
	for i, item := range strings.Split(env, ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		name := fmt.Sprintf("layer%d", i+1)
		loc, query, _ := strings.Cut(item, "?")
		q, err := url.ParseQuery(query)
		if err != nil {
			return nil, nil, fmt.Errorf("chat: %s: %q: %w", LayersEnv, item, err)
		}
		flag := func(k string) bool { v, _ := strconv.ParseBool(q.Get(k)); return v }
		var s Source
		if rest, ok := strings.CutPrefix(loc, "s3://"); ok {
			bucket, prefix, _ := strings.Cut(rest, "/")
			if bucket == "" {
				return nil, nil, fmt.Errorf("chat: %s: %q has no bucket", LayersEnv, item)
			}
			s = Source{Name: name, Type: "s3", Bucket: bucket, Prefix: prefix, Endpoint: q.Get("endpoint"), Region: q.Get("region"), Profile: q.Get("profile"), NoSign: flag("no_sign")}
			if s.Prefix != "" && !strings.HasSuffix(s.Prefix, "/") {
				s.Prefix += "/"
			}
			if v := q.Get("path_style"); v != "" {
				b, _ := strconv.ParseBool(v)
				s.PathStyle = &b
			}
		} else {
			s = Source{Name: name, Type: "path", Path: loc}
		}
		sources[name] = s
		var sc []string
		if v := q.Get("scopes"); v != "" {
			sc = strings.FieldsFunc(v, func(r rune) bool { return r == ' ' || r == '+' })
		}
		entries = append(entries, layerEntry{Use: name, Write: flag("write"), Fill: flag("fill"), FillGated: flag("fill_gated"), Scopes: sc})
	}
	return entries, sources, nil
}

// SystemModelsFile is the definition file of the machine, which its administrator puts there
// (the organization's sources and layers, its models): read after the built-in one and before
// the user's (docs/cache-sharing.md 7.4).
var SystemModelsFile = func() string {
	switch runtime.GOOS {
	case "windows":
		return filepath.Join(cmp.Or(os.Getenv("ProgramData"), `C:\ProgramData`), "hwnpu", "models.yaml")
	case "darwin":
		return "/Library/Application Support/hwnpu/models.yaml"
	}
	return "/etc/hwnpu/models.yaml"
}

// The upstream files in the layers.

// gatedRepos remembers which repositories are gated (isGated asks the hub), by OWNER/NAME.
var (
	gatedMu    sync.Mutex
	gatedRepos = map[string]bool{}
)

// askedLicense reports whether a model of the catalog from repo asks for its license.
func askedLicense(repo string) bool {
	for _, m := range Catalog {
		if m.License.Accept && strings.EqualFold(m.Repo, repo) {
			return true
		}
	}
	return false
}

// mayFill reports whether a file of repo (OWNER/NAME) goes in layer l: gated repositories and
// models whose license is asked for go only in a layer with fill_gated.
func mayFill(ctx context.Context, l shared.Layer, repo string) bool {
	if !l.Fill || !l.Holds(shared.ScopeUpstream) {
		return false
	}
	if l.FillGated {
		return true
	}
	if askedLicense(repo) {
		return false
	}
	gatedMu.Lock()
	g, ok := gatedRepos[repo]
	gatedMu.Unlock()
	if !ok {
		g = isGated(ctx, repo)
		gatedMu.Lock()
		gatedRepos[repo] = g
		gatedMu.Unlock()
	}
	return !g
}

var sha256RE = regexp.MustCompile(`^[0-9a-f]{64}$`)

// hubFile asks the hub (a HEAD, no download) for a file's SHA-256 (X-Linked-Etag, a file in
// LFS: the large ones) and the commit it is of. Either is "" when it does not say.
func hubFile(ctx context.Context, repo, name string) (sum, commit string) {
	base, rev, _ := strings.Cut(repo, "@")
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, hubURL+base+"/resolve/"+cmp.Or(rev, "main")+"/"+name, nil)
	if err != nil {
		return "", ""
	}
	if tok := hfToken(); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	resp, err := headClient.Do(req)
	if err != nil {
		return "", ""
	}
	resp.Body.Close()
	if resp.StatusCode >= 400 {
		return "", ""
	}
	if e := strings.Trim(resp.Header.Get("X-Linked-Etag"), `"`); sha256RE.MatchString(e) {
		sum = e
	}
	if c := resp.Header.Get("X-Repo-Commit"); commitRE.MatchString(c) {
		commit = c
	}
	return sum, commit
}

// upstreamLayers are the layers that hold checkpoints' files.
func upstreamLayers() []shared.Layer {
	var out []shared.Layer
	for _, l := range shared.Layers() {
		if l.Holds(shared.ScopeUpstream) {
			out = append(out, l)
		}
	}
	return out
}

// fromLayers looks for a file of a checkpoint from hf (key: OWNER/NAME[@REVISION]) in the
// layers: by the SHA-256 the definitions pin, else by the one the hub says (a HEAD), else by
// the commit. It reports whether it put the file at path, and fills the layers before the one
// that had it.
func fromLayers(ctx context.Context, key, name, path string, log io.Writer) bool {
	ls := upstreamLayers()
	if len(ls) == 0 {
		return false
	}
	base, rev, _ := strings.Cut(key, "@")
	sum := digests[key][name]
	commit := ""
	if commitRE.MatchString(rev) {
		commit = rev
	}
	if sum == "" {
		var c string
		sum, c = hubFile(ctx, key, name)
		commit = cmp.Or(commit, c)
		if commit != "" {
			commitsMu.Lock()
			commits[base+"@"+cmp.Or(rev, "main")] = commit
			commitsMu.Unlock()
		}
	}
	if sum == "" && commit == "" {
		return false
	}
	i, got, err := shared.FetchUpstream(ctx, ls, base, commit, name, strings.ToLower(sum), path, log)
	if err != nil {
		return false
	}
	for _, l := range ls[:i] {
		if mayFill(ctx, l, base) {
			shared.FillLater(l, base, commit, name, path, got)
		}
	}
	return true
}

// fillLayers puts a file of a checkpoint just downloaded from a source in the layers that are
// filled (its SHA-256 is worked out in the background).
func fillLayers(ctx context.Context, key, name, path string) {
	ls := upstreamLayers()
	if len(ls) == 0 {
		return
	}
	base, _, _ := strings.Cut(key, "@")
	commit, _ := servedCommit(key)
	sum := strings.ToLower(digests[key][name])
	for _, l := range ls {
		if mayFill(ctx, l, base) {
			shared.FillLater(l, base, commit, name, path, sum)
		}
	}
}
