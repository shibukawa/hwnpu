package chat

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
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

// TestSearchMirror checks the search: a model from hf comes from the shared folder first, and from
// Hugging Face when the folder does not have the file.
func TestSearchMirror(t *testing.T) {
	share := t.TempDir()
	put(t, filepath.Join(share, "LiquidAI", "LFM3-1.7B", "config.json"), "from the share")
	n := hub(t, map[string]string{"LiquidAI/LFM3-1.7B/resolve/main/tokenizer.json": "from the hub"})
	if _, err := userFiles(t, fmt.Sprintf("version: 1\nsources:\n  share: {type: path, path: %q}\nsearch: [share, hf]\n", share)); err != nil {
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
	// none has it: not found, as FetchWeights needs to try the index of shards
	err := fetch(ctx, m.Source(), "model.safetensors", filepath.Join(dir, "model.safetensors"), nil)
	if !NotFound(err) {
		t.Errorf("a file none has: %v", err)
	}
}

// TestSearchWithoutHub checks that a search without hf never asks Hugging Face.
func TestSearchWithoutHub(t *testing.T) {
	share := t.TempDir()
	n := hub(t, map[string]string{"LiquidAI/LFM3-1.7B/resolve/main/config.json": "x"})
	if _, err := userFiles(t, fmt.Sprintf("version: 1\nsources:\n  share: {type: path, path: %q}\nsearch: [share]\n", share)); err != nil {
		t.Fatal(err)
	}
	err := fetch(context.Background(), "LiquidAI/LFM3-1.7B", "config.json", filepath.Join(t.TempDir(), "config.json"), nil)
	if !NotFound(err) || n.Load() != 0 {
		t.Errorf("%v, %d requests to the hub", err, n.Load())
	}
}

// TestPinnedDigest checks that a mirror's file that differs from the pinned SHA-256 is dropped
// and the next source tried, and that a pinned revision's folder is read first.
func TestPinnedDigest(t *testing.T) {
	share := t.TempDir()
	put(t, filepath.Join(share, "LiquidAI", "LFM3-1.7B", "config.json"), "stale")
	put(t, filepath.Join(share, "LiquidAI", "LFM3-1.7B", "tokenizer.json"), "tok")
	put(t, filepath.Join(share, "LiquidAI", "LFM3-1.7B@abc", "tokenizer.json"), "tok at abc")
	hub(t, map[string]string{"LiquidAI/LFM3-1.7B/resolve/abc/config.json": "fresh"})
	if _, err := userFiles(t, fmt.Sprintf(`version: 1
sources:
  share: {type: path, path: %q}
search: [share, hf]
models:
  LFM3-1.7B:
    from: hf:LiquidAI/LFM3-1.7B@abc
    sha256: {config.json: %s}
`, share, digest("fresh"))); err != nil {
		t.Fatal(err)
	}
	m, _ := Lookup("LFM3-1.7B")
	dir := t.TempDir()
	var log strings.Builder
	if err := fetch(context.Background(), m.Source(), "config.json", filepath.Join(dir, "config.json"), &log); err != nil {
		t.Fatal(err)
	}
	if got := read(t, filepath.Join(dir, "config.json")); got != "fresh" {
		t.Errorf("config.json %q", got)
	}
	if !strings.Contains(log.String(), "SHA-256") {
		t.Errorf("no word of the stale copy:\n%s", log.String())
	}
	if err := fetch(context.Background(), m.Source(), "tokenizer.json", filepath.Join(dir, "tokenizer.json"), nil); err != nil {
		t.Fatal(err)
	}
	if got := read(t, filepath.Join(dir, "tokenizer.json")); got != "tok at abc" {
		t.Errorf("tokenizer.json %q, want the pinned revision's", got)
	}
	// only the stale copy anywhere: an error, not a wrong file
	if _, err := userFiles(t, fmt.Sprintf("version: 1\nsources:\n  share: {type: path, path: %q}\nsearch: [share]\nmodels:\n  LFM3-1.7B:\n    sha256: {config.json: %s}\n", share, digest("fresh"))); err != nil {
		t.Fatal(err)
	}
	dst := filepath.Join(t.TempDir(), "config.json")
	if err := fetch(context.Background(), "LiquidAI/LFM3-1.7B", "config.json", dst, nil); err == nil {
		t.Error("the stale copy was taken")
	}
	if _, err := os.Stat(dst); err == nil {
		t.Error("the stale copy was left")
	}
}

// TestModelFromSource checks a model whose from names a path source, and a -use of SOURCE:NAME.
func TestModelFromSource(t *testing.T) {
	share := t.TempDir()
	put(t, filepath.Join(share, "team", "distill-0.8b", "config.json"), "distilled")
	n := hub(t, nil)
	if _, err := userFiles(t, fmt.Sprintf(`version: 1
sources:
  share: {type: path, path: %q}
models:
  my-distill:
    like: plamo-3-translate-0.8b
    from: share:team/distill-0.8b
`, share)); err != nil {
		t.Fatal(err)
	}
	m, _ := Lookup("my-distill")
	if m.Source() != "share:team/distill-0.8b" || m.DirName() != "distill-0.8b" {
		t.Fatalf("source %s, directory %s", m.Source(), m.DirName())
	}
	dst := filepath.Join(t.TempDir(), "config.json")
	if err := fetch(context.Background(), m.Source(), "config.json", dst, nil); err != nil || read(t, dst) != "distilled" || n.Load() != 0 {
		t.Errorf("%v, %d requests to the hub", err, n.Load())
	}
	r, err := ResolveFor("share:team/distill-0.8b", KindTranscribe, nil)
	if err != nil || r.Source() != "share:team/distill-0.8b" || r.Name != "distill-0.8b" {
		t.Errorf("-use SOURCE:NAME: %+v, %v", r, err)
	}
	if _, err := ResolveFor("share:../x", KindTranscribe, nil); err == nil {
		t.Error("a name outside the source was taken")
	}
}

// TestS3Source checks an s3 source through a fake aws command: the URL, the endpoint, the
// profile and no_sign it is given, and a missing object as not found.
func TestS3Source(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("the fake aws command is a shell script")
	}
	root := t.TempDir() // the bucket's contents
	put(t, filepath.Join(root, "models", "hf", "LiquidAI", "LFM3-1.7B", "config.json"), "from s3")
	bin := t.TempDir()
	argsLog := filepath.Join(bin, "args")
	script := fmt.Sprintf(`#!/bin/sh
echo "$@" >> %q
while [ "$1" != "--" ]; do shift; done
src=$(echo "$2" | sed 's|^s3://|%s/|')
if [ ! -f "$src" ]; then echo "fatal error: An error occurred (404) when calling the HeadObject operation: Not Found" >&2; exit 1; fi
cp "$src" "$3"
`, argsLog, root)
	put(t, filepath.Join(bin, "aws"), script)
	if err := os.Chmod(filepath.Join(bin, "aws"), 0o755); err != nil {
		t.Fatal(err)
	}
	old := awsCommand
	awsCommand = filepath.Join(bin, "aws")
	t.Cleanup(func() { awsCommand = old })
	n := hub(t, nil)
	if _, err := userFiles(t, `version: 1
sources:
  lab: {type: s3, bucket: models, prefix: hf, endpoint: "http://rustfs.lab:9000", profile: rustfs, no_sign: true}
search: [lab, hf]
`); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := fetch(context.Background(), "LiquidAI/LFM3-1.7B", "config.json", filepath.Join(dir, "config.json"), nil); err != nil {
		t.Fatal(err)
	}
	if got := read(t, filepath.Join(dir, "config.json")); got != "from s3" || n.Load() != 0 {
		t.Errorf("config.json %q, %d requests to the hub", got, n.Load())
	}
	args := read(t, argsLog)
	for _, want := range []string{"s3 cp", "--endpoint-url http://rustfs.lab:9000", "--profile rustfs", "--no-sign-request", "s3://models/hf/LiquidAI/LFM3-1.7B/config.json"} {
		if !strings.Contains(args, want) {
			t.Errorf("aws was not given %q: %s", want, args)
		}
	}
	err := fetch(context.Background(), "LiquidAI/LFM3-1.7B", "tokenizer.json", filepath.Join(dir, "tokenizer.json"), nil)
	if !NotFound(err) {
		t.Errorf("a missing object: %v", err)
	}
	if !strings.Contains(read(t, argsLog), "tokenizer.json") {
		t.Error("the bucket was not asked for tokenizer.json")
	}
	// no aws command: an error that says so, not a silent fall back
	awsCommand = filepath.Join(bin, "no-such-aws")
	err = fetch(context.Background(), "LiquidAI/LFM3-1.7B", "generation_config.json", filepath.Join(dir, "g.json"), nil)
	if err == nil || !strings.Contains(err.Error(), "AWS CLI") {
		t.Errorf("without aws: %v", err)
	}
}
