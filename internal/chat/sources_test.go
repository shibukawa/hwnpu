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

// TestSources checks the order: a file a source's lock names comes from the source, one it
// does not from Hugging Face, of the lock's commit; one neither has is not found (FetchWeights
// then tries the index of shards); a file in the folder the lock does not name is not taken.
func TestSources(t *testing.T) {
	share := t.TempDir()
	lockIn(t, share, "LiquidAI/LFM3-1.7B", testCommit, map[string]string{"config.json": "from the share"})
	put(t, filepath.Join(share, "original", "LiquidAI", "LFM3-1.7B", "README.md"), "not locked")
	n := hub(t, map[string]string{"LiquidAI/LFM3-1.7B/resolve/" + testCommit + "/tokenizer.json": "from the hub"})
	if _, err := withSources(t, []string{"file://" + filepath.ToSlash(share)}); err != nil {
		t.Fatal(err)
	}
	m, _ := Lookup("LFM3-1.7B")
	dir := t.TempDir()
	ctx := context.Background()
	if err := fetch(ctx, m.Source(), "config.json", filepath.Join(dir, "config.json"), nil); err != nil {
		t.Fatal(err)
	}
	if got := read(t, filepath.Join(dir, "config.json")); got != "from the share" || n.Load() != 0 {
		t.Errorf("config.json %q, %d requests to the hub", got, n.Load())
	}
	if err := fetch(ctx, m.Source(), "tokenizer.json", filepath.Join(dir, "tokenizer.json"), nil); err != nil {
		t.Fatal(err)
	}
	if got := read(t, filepath.Join(dir, "tokenizer.json")); got != "from the hub" {
		t.Errorf("tokenizer.json %q", got)
	}
	if c, _ := servedCommit(m.Source()); c != testCommit {
		t.Errorf("commit %q, not the lock's", c)
	}
	err := fetch(ctx, m.Source(), "model.safetensors", filepath.Join(dir, "model.safetensors"), nil)
	if !NotFound(err) {
		t.Errorf("a file none has: %v", err)
	}
	if err := fetch(ctx, m.Source(), "README.md", filepath.Join(dir, "README.md"), nil); !NotFound(err) {
		t.Errorf("a file the lock does not name: %v", err)
	}
}

// TestHuggingFaceOff checks that HWNPU_MODEL_HUGGING_FACE=false never asks Hugging Face.
func TestHuggingFaceOff(t *testing.T) {
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
	if err := fetch(ctx, "LiquidAI/LFM3-1.7B", "config.json", filepath.Join(t.TempDir(), "config.json"), nil); err != nil {
		t.Fatal(err)
	}
	err := fetch(ctx, "LiquidAI/LFM3-1.7B", "tokenizer.json", filepath.Join(t.TempDir(), "tokenizer.json"), nil)
	if !NotFound(err) || n.Load() != 0 || !strings.Contains(err.Error(), shared.HuggingFaceEnv) {
		t.Errorf("%v, %d requests to the hub", err, n.Load())
	}
}

// TestPinnedDigest checks that a source's file that differs from the pinned SHA-256 is passed
// over for Hugging Face, and that a model pinned to a commit does not take another commit's
// files.
func TestPinnedDigest(t *testing.T) {
	share := t.TempDir()
	lockIn(t, share, "LiquidAI/LFM3-1.7B", testCommit, map[string]string{"config.json": "stale", "tokenizer.json": "tok"})
	hub(t, map[string]string{
		"LiquidAI/LFM3-1.7B/resolve/" + testCommit + "/config.json": "fresh",
		"LiquidAI/LFM3-1.7B/resolve/abc/tokenizer.json":             "tok at abc",
	})
	if _, err := withSources(t, []string{"file://" + filepath.ToSlash(share)}, fmt.Sprintf("version: 1\nmodels:\n  LFM3-1.7B:\n    sha256: {config.json: %s}\n", digest("fresh"))); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := fetch(context.Background(), "LiquidAI/LFM3-1.7B", "config.json", filepath.Join(dir, "config.json"), nil); err != nil {
		t.Fatal(err)
	}
	if got := read(t, filepath.Join(dir, "config.json")); got != "fresh" {
		t.Errorf("config.json %q", got)
	}
	if err := fetch(context.Background(), "LiquidAI/LFM3-1.7B@abc", "tokenizer.json", filepath.Join(dir, "tokenizer.json"), nil); err != nil {
		t.Fatal(err)
	}
	if got := read(t, filepath.Join(dir, "tokenizer.json")); got != "tok at abc" {
		t.Errorf("tokenizer.json %q, want the pinned revision's", got)
	}
}

// TestS3Source reads a source of S3 (unsigned, no_sign) in the layout of original/.
func TestS3Source(t *testing.T) {
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
	dir := t.TempDir()
	if err := fetch(context.Background(), "LiquidAI/LFM3-1.7B", "config.json", filepath.Join(dir, "config.json"), nil); err != nil {
		t.Fatal(err)
	}
	if got := read(t, filepath.Join(dir, "config.json")); got != "from s3" || n.Load() != 0 {
		t.Errorf("config.json %q, %d requests to the hub", got, n.Load())
	}
	// signed with the profile's keys: no keys is an error that says so, not a silent fall back
	t.Setenv("AWS_SHARED_CREDENTIALS_FILE", filepath.Join(t.TempDir(), "none"))
	if _, err := withSources(t, []string{"s3://models/hf?profile=none&endpoint=" + srv.URL}); err == nil || !strings.Contains(err.Error(), "no keys") {
		t.Errorf("without keys: %v", err)
	}
}

// TestSameFile checks that a pinned file another model of the same SHA-256 has in the models'
// directory is linked from there instead of downloaded (MOSS-SoundEffect's text encoder is
// Qwen3-1.7B), and that a copy that differs is not taken.
func TestSameFile(t *testing.T) {
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
	root := t.TempDir()
	for _, d := range []string{"chat-1b", "other", "sfx/text_encoder"} { // as Prepare and sfx.Prepare make them
		if err := os.MkdirAll(filepath.Join(root, d), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	chatFile := filepath.Join(root, "chat-1b", "model.safetensors")
	if err := fetch(context.Background(), "lab/chat-1b", "model.safetensors", chatFile, nil); err != nil {
		t.Fatal(err)
	}
	if n.Load() != 1 {
		t.Fatalf("%d requests for the first download", n.Load())
	}
	// the other model's file of the same SHA-256: linked, no request
	sfxFile := filepath.Join(root, "sfx", "text_encoder", "model.safetensors")
	var log strings.Builder
	if err := fetch(context.Background(), "lab/sfx@abc", "text_encoder/model.safetensors", sfxFile, &log); err != nil {
		t.Fatal(err)
	}
	if n.Load() != 1 || read(t, sfxFile) != weights || !strings.Contains(log.String(), "linked model.safetensors from model.safetensors of chat-1b") {
		t.Errorf("%d requests, log:\n%s", n.Load(), log.String())
	}
	a, _ := os.Stat(chatFile)
	b, _ := os.Stat(sfxFile)
	if !os.SameFile(a, b) {
		t.Error("a copy, not a link")
	}
	// a model of another SHA-256 is downloaded
	if err := fetch(context.Background(), "lab/other", "model.safetensors", filepath.Join(root, "other", "model.safetensors"), nil); err != nil {
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
	if err := fetch(context.Background(), "lab/sfx@abc", "text_encoder/model.safetensors", sfxFile, nil); err != nil {
		t.Fatal(err)
	}
	if n.Load() != 3 || read(t, sfxFile) != weights || read(t, chatFile) != "changed" {
		t.Errorf("%d requests, %q, %q", n.Load(), read(t, sfxFile), read(t, chatFile))
	}
}
