package chat

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	shared "github.com/shibukawa/hwnpu/internal/store"
	"github.com/shibukawa/hwnpu/internal/store/s3fake"
)

// hub serves files as Hugging Face does (OWNER/NAME/resolve/REV/FILE) and counts the requests.
func hub(t *testing.T, files map[string]string) *atomic.Int32 {
	t.Helper()
	var n atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n.Add(1)
		body, ok := files[strings.TrimPrefix(r.URL.Path, "/")]
		if !ok {
			http.NotFound(w, r)
			return
		}
		fmt.Fprint(w, body)
	}))
	t.Cleanup(srv.Close)
	old := hubURL
	hubURL = srv.URL + "/"
	t.Cleanup(func() { hubURL = old })
	return &n
}

func put(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func digest(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:])
}

func read(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// lockIn puts files of repo at commit in the folder source root, as a writeback would, with
// their lock.
func lockIn(t *testing.T, root, repo, commit string, files map[string]string) {
	t.Helper()
	l := shared.Lock{Format: shared.LockFormat, Repo: repo, Commit: commit, Files: map[string]shared.LockedFile{}}
	for name, body := range files {
		put(t, filepath.Join(root, "original", filepath.FromSlash(repo), filepath.FromSlash(name)), body)
		l.Files[name] = shared.LockedFile{SHA256: digest(body), Size: int64(len(body))}
	}
	b, _ := json.Marshal(l)
	put(t, filepath.Join(root, "original", filepath.FromSlash(repo), shared.LockName), string(b))
}

// fetched fetches a file of repo and returns what it holds.
func fetched(t *testing.T, repo, name string) string {
	t.Helper()
	p, err := fetch(context.Background(), repo, name, nil)
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join(RepoDir(repo), filepath.FromSlash(name)); p != want {
		t.Errorf("%s of %s at %s, not %s", name, repo, p, want)
	}
	return read(t, p)
}

// TestSources checks the order: a file a source's lock names comes from the source, one it
// does not from Hugging Face, of the lock's commit; one neither has is not found (FetchWeights
// then tries the index of shards); a file in the folder the lock does not name is not taken.
// What is fetched is in the machine's cache, named in its lock, and read from there next.
func TestSources(t *testing.T) {
	cacheHere(t)
	share := t.TempDir()
	lockIn(t, share, "LiquidAI/LFM3-1.7B", testCommit, map[string]string{"config.json": "from the share"})
	put(t, filepath.Join(share, "original", "LiquidAI", "LFM3-1.7B", "README.md"), "not locked")
	n := hub(t, map[string]string{"LiquidAI/LFM3-1.7B/resolve/" + testCommit + "/tokenizer.json": "from the hub"})
	if _, err := withSources(t, []string{"file://" + filepath.ToSlash(share)}); err != nil {
		t.Fatal(err)
	}
	m, _ := Lookup("LFM3-1.7B")
	ctx := context.Background()
	if got := fetched(t, m.Source(), "config.json"); got != "from the share" || n.Load() != 0 {
		t.Errorf("config.json %q, %d requests to the hub", got, n.Load())
	}
	if got := fetched(t, m.Source(), "tokenizer.json"); got != "from the hub" {
		t.Errorf("tokenizer.json %q", got)
	}
	if c, _ := servedCommit(m.Source()); c != testCommit {
		t.Errorf("commit %q, not the lock's", c)
	}
	if _, err := fetch(ctx, m.Source(), "model.safetensors", nil); !NotFound(err) {
		t.Errorf("a file none has: %v", err)
	}
	if _, err := fetch(ctx, m.Source(), "README.md", nil); !NotFound(err) {
		t.Errorf("a file the lock does not name: %v", err)
	}
	l, err := shared.ReadLock(ctx, shared.Local(), "LiquidAI/LFM3-1.7B")
	if err != nil || l.Commit != testCommit || len(l.Files) != 2 || l.Files["tokenizer.json"].SHA256 != digest("from the hub") {
		t.Errorf("the machine's lock %+v %v", l, err)
	}
	// read from the machine's cache, even with the source gone
	os.RemoveAll(share)
	clearCommits()
	before := n.Load()
	if got := fetched(t, m.Source(), "tokenizer.json"); got != "from the hub" || n.Load() != before {
		t.Errorf("tokenizer.json %q, %d requests", got, n.Load()-before)
	}
}

// clearCommits forgets the commits this process took files of (a new process).
func clearCommits() {
	commitsMu.Lock()
	clear(commits)
	commitsMu.Unlock()
}

// TestFollowLock checks that the machine's cache follows the sources' lock: when it moves to
// another commit (hwnpu-dev sources update), a file of the old one is fetched again, and one that
// is the same in both is kept (its SHA-256 the new lock's).
func TestFollowLock(t *testing.T) {
	cacheHere(t)
	share := t.TempDir()
	const newCommit = "fedcba9876543210fedcba9876543210fedcba98"
	lockIn(t, share, "lab/m", testCommit, map[string]string{"config.json": "old", "tokenizer.json": "tok"})
	if _, err := withSources(t, []string{"file://" + filepath.ToSlash(share)}); err != nil {
		t.Fatal(err)
	}
	n := hub(t, nil)
	if fetched(t, "lab/m", "config.json") != "old" || fetched(t, "lab/m", "tokenizer.json") != "tok" {
		t.Fatal("not the share's")
	}
	lockIn(t, share, "lab/m", newCommit, map[string]string{"config.json": "new", "tokenizer.json": "tok"})
	if _, err := withSources(t, []string{"file://" + filepath.ToSlash(share)}); err != nil { // a new process
		t.Fatal(err)
	}
	clearCommits()
	tok := filepath.Join(RepoDir("lab/m"), "tokenizer.json")
	before, _ := os.Stat(tok)
	if got := fetched(t, "lab/m", "config.json"); got != "new" {
		t.Errorf("config.json %q after the lock moved", got)
	}
	if got := fetched(t, "lab/m", "tokenizer.json"); got != "tok" {
		t.Errorf("tokenizer.json %q", got)
	}
	if after, _ := os.Stat(tok); !os.SameFile(before, after) {
		t.Error("the file the same in both commits was fetched again")
	}
	l, _ := shared.ReadLock(context.Background(), shared.Local(), "lab/m")
	if l.Commit != newCommit || len(l.Files) != 2 || n.Load() != 0 {
		t.Errorf("lock %+v, %d requests", l, n.Load())
	}
}

// TestLegacyFile checks that a file of an earlier hwnpu's models/NAME is moved into the
// repository's folder when it is the one asked for (the SHA-256 Hugging Face says), and fetched
// again when it is not.
func TestLegacyFile(t *testing.T) {
	cacheHere(t)
	if _, err := withSources(t, nil); err != nil {
		t.Fatal(err)
	}
	counts := fakeHub(t, map[string]string{
		"lab/m/resolve/main/model.safetensors": "the weights",
		"lab/m/resolve/main/vae.safetensors":   "the vae",
	})
	put(t, filepath.Join(LegacyModelDir("lab/m"), "model.safetensors"), "the weights")
	put(t, filepath.Join(LegacyModelDir("lab/m"), "vae.safetensors"), "a stale vae")
	if got := fetched(t, "lab/m", "model.safetensors"); got != "the weights" || counts["GET"] != 0 {
		t.Errorf("model.safetensors %q, %d downloads", got, counts["GET"])
	}
	if _, err := os.Stat(filepath.Join(LegacyModelDir("lab/m"), "model.safetensors")); err == nil {
		t.Error("left in models/")
	}
	if got := fetched(t, "lab/m", "vae.safetensors"); got != "the vae" || counts["GET"] != 1 {
		t.Errorf("vae.safetensors %q, %d downloads", got, counts["GET"])
	}
}

// TestBundle checks that a folder of bundled models (store.SetBundles) is read after the
// sources and used in place: the machine's cache links to it.
func TestBundle(t *testing.T) {
	cacheHere(t)
	if _, err := withSources(t, nil); err != nil {
		t.Fatal(err)
	}
	bundle := t.TempDir()
	lockIn(t, bundle, "lab/m", testCommit, map[string]string{"config.json": "bundled"})
	shared.SetBundles([]string{bundle})
	t.Cleanup(func() { shared.SetBundles(nil) })
	n := hub(t, nil)
	if got := fetched(t, "lab/m", "config.json"); got != "bundled" || n.Load() != 0 {
		t.Errorf("config.json %q, %d requests", got, n.Load())
	}
	a, _ := os.Stat(filepath.Join(RepoDir("lab/m"), "config.json"))
	b, _ := os.Stat(filepath.Join(bundle, "original", "lab", "m", "config.json"))
	if !os.SameFile(a, b) {
		t.Error("copied, not used in place")
	}
}

// TestHuggingFaceOff checks that HWNPU_MODEL_HUGGING_FACE=false never asks Hugging Face.
func TestHuggingFaceOff(t *testing.T) {
	cacheHere(t)
	share := t.TempDir()
	lockIn(t, share, "LiquidAI/LFM3-1.7B", testCommit, map[string]string{"config.json": "x"})
	n := hub(t, map[string]string{"LiquidAI/LFM3-1.7B/resolve/main/tokenizer.json": "x"})
	if _, err := withSources(t, []string{"file://" + filepath.ToSlash(share)}); err != nil {
		t.Fatal(err)
	}
	t.Setenv(shared.HuggingFaceEnv, "false")
	if err := LoadModels(nil, true, nil); err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	if _, err := fetch(ctx, "LiquidAI/LFM3-1.7B", "config.json", nil); err != nil {
		t.Fatal(err)
	}
	_, err := fetch(ctx, "LiquidAI/LFM3-1.7B", "tokenizer.json", nil)
	if !NotFound(err) || n.Load() != 0 || !strings.Contains(err.Error(), shared.HuggingFaceEnv) {
		t.Errorf("%v, %d requests to the hub", err, n.Load())
	}
}

// TestPinnedDigest checks that a source's file that differs from the pinned SHA-256 is passed
// over for Hugging Face, and that a model pinned to a commit does not take another commit's
// files.
func TestPinnedDigest(t *testing.T) {
	cacheHere(t)
	share := t.TempDir()
	lockIn(t, share, "LiquidAI/LFM3-1.7B", testCommit, map[string]string{"config.json": "stale", "tokenizer.json": "tok"})
	hub(t, map[string]string{
		"LiquidAI/LFM3-1.7B/resolve/" + testCommit + "/config.json": "fresh",
		"LiquidAI/LFM3-1.7B/resolve/abc/tokenizer.json":             "tok at abc",
	})
	if _, err := withSources(t, []string{"file://" + filepath.ToSlash(share)}, fmt.Sprintf("version: 1\nmodels:\n  LFM3-1.7B:\n    sha256: {config.json: %s}\n", digest("fresh"))); err != nil {
		t.Fatal(err)
	}
	if got := fetched(t, "LiquidAI/LFM3-1.7B", "config.json"); got != "fresh" {
		t.Errorf("config.json %q", got)
	}
	if got := fetched(t, "LiquidAI/LFM3-1.7B@abc", "tokenizer.json"); got != "tok at abc" {
		t.Errorf("tokenizer.json %q, want the pinned revision's", got)
	}
}

// TestS3Source reads a source of S3 (unsigned, no_sign) in the layout of original/.
func TestS3Source(t *testing.T) {
	cacheHere(t)
	srv := s3fake.New(t)
	l, _ := json.Marshal(shared.Lock{Format: shared.LockFormat, Repo: "LiquidAI/LFM3-1.7B", Commit: testCommit,
		Files: map[string]shared.LockedFile{"config.json": {SHA256: digest("from s3"), Size: 7}}})
	srv.Put("models", "hf/original/LiquidAI/LFM3-1.7B/config.json", []byte("from s3"))
	srv.Put("models", "hf/original/LiquidAI/LFM3-1.7B/"+shared.LockName, l)
	t.Setenv("AWS_ACCESS_KEY_ID", "")
	t.Setenv("AWS_PROFILE", "")
	n := hub(t, nil)
	if _, err := withSources(t, []string{"s3://models/hf?no_sign=true&endpoint=" + srv.URL}); err != nil {
		t.Fatal(err)
	}
	if got := fetched(t, "LiquidAI/LFM3-1.7B", "config.json"); got != "from s3" || n.Load() != 0 {
		t.Errorf("config.json %q, %d requests to the hub", got, n.Load())
	}
	// signed with the profile's keys: no keys is an error that says so, not a silent fall back
	t.Setenv("AWS_SHARED_CREDENTIALS_FILE", filepath.Join(t.TempDir(), "none"))
	if _, err := withSources(t, []string{"s3://models/hf?profile=none&endpoint=" + srv.URL}); err == nil || !strings.Contains(err.Error(), "no keys") {
		t.Errorf("without keys: %v", err)
	}
}

// TestSameFile checks that a pinned file another repository of the same SHA-256 has here is
// linked from there instead of downloaded (MOSS-SoundEffect's text encoder is Qwen3-1.7B), and
// that a copy that differs is not taken.
func TestSameFile(t *testing.T) {
	cacheHere(t)
	weights := "the weights"
	n := hub(t, map[string]string{
		"lab/chat-1b/resolve/main/model.safetensors":         weights,
		"lab/sfx/resolve/abc/text_encoder/model.safetensors": weights,
		"lab/other/resolve/main/model.safetensors":           "other weights",
	})
	if _, err := userFiles(t, fmt.Sprintf(`version: 1
models:
  chat-1b:
    from: hf:lab/chat-1b
    kind: chat
    sha256: {model.safetensors: %[1]s}
  sfx:
    from: hf:lab/sfx@abc
    kind: sfx
    sha256: {text_encoder/model.safetensors: %[1]s}
  other:
    from: hf:lab/other
    kind: chat
    sha256: {model.safetensors: %[2]s}
`, digest(weights), digest("other weights"))); err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	chatFile, err := fetch(ctx, "lab/chat-1b", "model.safetensors", nil)
	if err != nil {
		t.Fatal(err)
	}
	if n.Load() != 1 {
		t.Fatalf("%d requests for the first download", n.Load())
	}
	// the other model's file of the same SHA-256: linked, no request
	var log strings.Builder
	sfxFile, err := fetch(ctx, "lab/sfx@abc", "text_encoder/model.safetensors", &log)
	if err != nil {
		t.Fatal(err)
	}
	if n.Load() != 1 || read(t, sfxFile) != weights || !strings.Contains(log.String(), "linked model.safetensors from model.safetensors of lab/chat-1b") {
		t.Errorf("%d requests, log:\n%s", n.Load(), log.String())
	}
	a, _ := os.Stat(chatFile)
	b, _ := os.Stat(sfxFile)
	if !os.SameFile(a, b) {
		t.Error("a copy, not a link")
	}
	// a model of another SHA-256 is downloaded
	if _, err := fetch(ctx, "lab/other", "model.safetensors", nil); err != nil {
		t.Fatal(err)
	}
	if n.Load() != 2 {
		t.Errorf("%d requests: other's file was not downloaded", n.Load())
	}
	// a file of the same name that was changed is not taken, and stays as it was
	os.Remove(sfxFile)
	if err := os.Remove(chatFile); err != nil {
		t.Fatal(err)
	}
	put(t, chatFile, "changed")
	if _, err := fetch(ctx, "lab/sfx@abc", "text_encoder/model.safetensors", nil); err != nil {
		t.Fatal(err)
	}
	if n.Load() != 3 || read(t, sfxFile) != weights || read(t, chatFile) != "changed" {
		t.Errorf("%d requests, %q, %q", n.Load(), read(t, sfxFile), read(t, chatFile))
	}
}
