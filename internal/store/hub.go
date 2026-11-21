package store

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
)

// The converted models on Hugging Face (docs/prebuilt.md 5): repositories that hold, for a
// model, what a source holds under converted/MODEL/ (TARGET/EXPORT/VERSION/...), put there by
// hwnpu-dev hf push. The definitions name each repository with the commit each target reads
// (models.yaml's hub): a Mac reads the commit of coreai-<its architecture>, else of apple; a
// machine with AMD's NPU the commit of amd-xdna2. They are read after the sources and the
// bundled models, and before the models' own repositories, read only.

// HubEnv=false keeps hwnpu from reading the converted models on Hugging Face (they are made
// here, or come from the sources, instead). HuggingFaceEnv=false keeps it off them too.
const HubEnv = "HWNPU_MODEL_HUB"

// HubEndpoint is Hugging Face's address (tests replace it).
var HubEndpoint = "https://huggingface.co"

// HubRepo is a repository of a model's conversions on Hugging Face and the commit each target
// reads in it (models.yaml's hub: amd-xdna2, apple, coreai-ARCH).
type HubRepo struct {
	Repo string            // OWNER/NAME
	Pins map[string]string // by target: a commit (40 hex digits), or main
}

// HubAuth is asked for a token when a repository refuses (401, 403) a request that carried
// sent ("" for none): it returns the token to try again with, or why there is none (a gated
// repository whose terms the person did not agree to, no one at the terminal).
type HubAuth func(ctx context.Context, repo string, status int, sent string) (string, error)

var (
	hubMu     sync.RWMutex
	hubModels map[string][]HubRepo // by model, the repositories in the order they are read
	hubArch   func() string        // this Mac's architecture of the Neural Engine (h14g)
	hubAuth   HubAuth
	hubTokens = map[string]string{}  // by repository, the token it took
	hubStores = map[string]*WebDAV{} // by REPO@COMMIT
)

// SetHub sets the repositories of the models' conversions on Hugging Face (by model, in the
// order they are read), what tells this machine's architecture of the Neural Engine ("" on a
// machine that is not a Mac; asked when a Core AI conversion is first looked for) and what is
// asked for a token when a repository refuses one.
func SetHub(models map[string][]HubRepo, arch func() string, auth HubAuth) {
	hubMu.Lock()
	hubModels, hubArch, hubAuth = models, arch, auth
	hubMu.Unlock()
}

// hubOn reports whether the converted models on Hugging Face are read.
func hubOn() bool {
	if v := strings.TrimSpace(os.Getenv(HubEnv)); v != "" {
		if b, err := strconv.ParseBool(v); err == nil && !b {
			return false
		}
	}
	return true
}

// hubSources are the layers of the converted models on Hugging Face: one for each place a
// repository has in its model's order (the first repository of every model, then the second
// ...), so that a model's second repository is read only when its first does not have what is
// looked for.
func hubSources() []Source {
	hubMu.RLock()
	n := 0
	for _, rs := range hubModels {
		n = max(n, len(rs))
	}
	hubMu.RUnlock()
	var out []Source
	for i := range n {
		out = append(out, Source{Store: Hub{Rank: i}, From: "hub"})
	}
	return out
}

// Hub is a Store of the converted models on Hugging Face (docs/prebuilt.md 5): the object of
// converted/MODEL/TARGET/REST is REST's file in TARGET of the Rank-th repository of MODEL, at
// the commit this machine reads there. Everything else is not there. It is read only.
type Hub struct{ Rank int }

func (h Hub) String() string {
	if h.Rank == 0 {
		return "hub (the converted models on Hugging Face)"
	}
	return fmt.Sprintf("hub %d (the converted models on Hugging Face, the next repository of each)", h.Rank+1)
}

// HubPin is the commit the repository's pins give a target on this machine (arch: its
// architecture of the Neural Engine, "" on a machine that is not a Mac), "" for none: a Mac
// takes coreai-ARCH, else apple, for its Core AI exports, their prebuilt programs and the
// tables; a machine with AMD's NPU takes amd-xdna2.
func HubPin(pins map[string]string, target, arch string) string {
	apple := func() string {
		if arch != "" && pins["coreai-"+arch] != "" {
			return pins["coreai-"+arch]
		}
		return pins["apple"]
	}
	switch {
	case target == "coreai":
		return apple()
	case strings.HasPrefix(target, "coreai-"):
		if pins[target] != "" {
			return pins[target]
		}
		return pins["apple"]
	case target == "any":
		if arch != "" || runtime.GOOS == "darwin" {
			return apple()
		}
		return pins["amd-xdna2"]
	}
	return pins[target]
}

// at is the store and the key in it of key: the repository at its commit, and REST (TARGET/...);
// ok false when this layer does not have it.
func (h Hub) at(key string) (st *WebDAV, repo, rest string, ok bool, err error) {
	parts := strings.SplitN(key, "/", 4)
	if len(parts) < 4 || parts[0] != "converted" || !SafeKey(key) {
		return nil, "", "", false, nil
	}
	hubMu.RLock()
	rs, archOf := hubModels[parts[1]], hubArch
	hubMu.RUnlock()
	if h.Rank >= len(rs) {
		return nil, "", "", false, nil
	}
	r := rs[h.Rank]
	arch := ""
	if archOf != nil && parts[2] != "amd-xdna2" && (runtime.GOOS == "darwin" || parts[2] != "any") {
		arch = archOf()
	}
	commit := HubPin(r.Pins, parts[2], arch)
	if commit == "" {
		return nil, "", "", false, nil
	}
	st, err = hubStore(r.Repo, commit)
	return st, r.Repo, parts[2] + "/" + parts[3], err == nil, err
}

// hubStore is the read-only store of repo at commit (its files' resolve URLs).
func hubStore(repo, commit string) (*WebDAV, error) {
	k := repo + "@" + commit
	hubMu.Lock()
	defer hubMu.Unlock()
	if st, ok := hubStores[k]; ok {
		return st, nil
	}
	st, err := OpenWebDAV(WebDAVConfig{Name: "hub " + repo, URL: strings.TrimSuffix(HubEndpoint, "/") + "/" + repo + "/resolve/" + commit + "/",
		ReadOnly: true, Bearer: func() string { hubMu.RLock(); defer hubMu.RUnlock(); return hubTokens[repo] }})
	if err != nil {
		return nil, err
	}
	hubStores[k] = st
	return st, nil
}

// signed does f, and once more with the token HubAuth gives when the repository refuses it.
func signed(ctx context.Context, repo string, f func() error) error {
	err := f()
	var de *davError
	if !errors.As(err, &de) || de.StatusCode != http.StatusUnauthorized && de.StatusCode != http.StatusForbidden {
		return err
	}
	hubMu.RLock()
	auth, sent := hubAuth, hubTokens[repo]
	hubMu.RUnlock()
	if auth == nil {
		return fmt.Errorf("store: %s on Hugging Face asks to sign in (%s): %w", repo, de.Status, err)
	}
	tok, aerr := auth(ctx, repo, de.StatusCode, sent)
	if aerr != nil {
		return aerr
	}
	hubMu.Lock()
	hubTokens[repo] = tok
	hubMu.Unlock()
	return f()
}

func (h Hub) Get(ctx context.Context, key, dst string, bar func(done, total int64)) (int64, error) {
	st, repo, rest, ok, err := h.at(key)
	if err != nil {
		return 0, err
	}
	if !ok {
		return 0, fmt.Errorf("store: %s: %s: %w", h, key, ErrNotFound)
	}
	var n int64
	err = signed(ctx, repo, func() (err error) { n, err = st.Get(ctx, rest, dst, bar); return err })
	return n, err
}

func (h Hub) Read(ctx context.Context, key string) ([]byte, error) {
	st, repo, rest, ok, err := h.at(key)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("store: %s: %s: %w", h, key, ErrNotFound)
	}
	var b []byte
	err = signed(ctx, repo, func() (err error) { b, err = st.Read(ctx, rest); return err })
	return b, err
}

func (h Hub) Has(ctx context.Context, key string) (int64, bool, error) {
	st, repo, rest, ok, err := h.at(key)
	if err != nil || !ok {
		return 0, false, err
	}
	var n int64
	var has bool
	err = signed(ctx, repo, func() (err error) { n, has, err = st.Has(ctx, rest); return err })
	return n, has, err
}

func (h Hub) readOnly() error {
	return fmt.Errorf("store: %s is read only (hwnpu-dev hf push puts conversions there)", h)
}

func (h Hub) Put(context.Context, string, string) error   { return h.readOnly() }
func (h Hub) Write(context.Context, string, []byte) error { return h.readOnly() }
func (h Hub) Delete(context.Context, string) error        { return h.readOnly() }

// List lists nothing: Hugging Face's files are not listed through the layers (hwnpu-dev hf
// check reads them).
func (h Hub) List(context.Context, string) ([]string, error) { return nil, nil }

// HubRead reads the small file rest of repo at rev (a commit, or main) on Hugging Face, signing
// in as the hub layer does when the repository refuses. ErrNotFound when it is not there.
func HubRead(ctx context.Context, repo, rev, rest string) ([]byte, error) {
	st, err := hubStore(repo, rev)
	if err != nil {
		return nil, err
	}
	var b []byte
	err = signed(ctx, repo, func() (err error) { b, err = st.Read(ctx, rest); return err })
	return b, err
}

// HubGet downloads the file rest of repo at rev to dst, as HubRead.
func HubGet(ctx context.Context, repo, rev, rest, dst string, bar func(done, total int64)) (int64, error) {
	st, err := hubStore(repo, rev)
	if err != nil {
		return 0, err
	}
	var n int64
	err = signed(ctx, repo, func() (err error) { n, err = st.Get(ctx, rest, dst, bar); return err })
	return n, err
}

// SetHubToken makes the requests to repo carry tok (hwnpu-dev hf, which signed in to write).
func SetHubToken(repo, tok string) {
	hubMu.Lock()
	hubTokens[repo] = tok
	hubMu.Unlock()
}
