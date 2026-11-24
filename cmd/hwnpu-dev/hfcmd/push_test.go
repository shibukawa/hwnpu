package hfcmd

import (
	"context"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/devopt"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/recipe"
	shared "github.com/shibukawa/hwnpu/internal/store"
)

// fakeHub is Hugging Face as hwnpu-dev models hf push uses it: the API (whoami, a model's
// information, making a repository) and each repository's files (resolve) at its last commit,
// which the fake hf command (upload-large-folder) copies a folder into.
type fakeHub struct {
	dir  string // REPO/... : the files of each repository
	mu   sync.Mutex
	made map[string]bool
}

func (f *fakeHub) commit(repo string) string {
	h := sha1.New()
	root := filepath.Join(f.dir, repo)
	var names []string
	filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			names = append(names, p)
		}
		return nil
	})
	sort.Strings(names)
	for _, n := range names {
		b, _ := os.ReadFile(n)
		io.WriteString(h, n)
		h.Write(b)
	}
	return hex.EncodeToString(h.Sum(nil))
}

func (f *fakeHub) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	p := r.URL.Path
	switch {
	case p == "/api/whoami-v2":
		if r.Header.Get("Authorization") != "Bearer tok" {
			http.Error(w, "no", http.StatusUnauthorized)
			return
		}
		io.WriteString(w, `{"name":"me","orgs":[],"auth":{"accessToken":{"role":"write"}}}`)
	case p == "/api/repos/create":
		var body struct{ Name string }
		json.NewDecoder(r.Body).Decode(&body)
		f.made["me/"+body.Name] = true
		os.MkdirAll(filepath.Join(f.dir, "me", body.Name), 0o755)
	case strings.HasPrefix(p, "/api/models/"):
		repo, rev, _ := strings.Cut(strings.TrimPrefix(p, "/api/models/"), "/revision/")
		if repo == "openai/whisper-large-v3-turbo" {
			io.WriteString(w, `{"gated":false}`)
			return
		}
		if !f.made[repo] {
			http.NotFound(w, r)
			return
		}
		_ = rev
		json.NewEncoder(w).Encode(map[string]any{"sha": f.commit(repo), "gated": false})
	case strings.Contains(p, "/resolve/"):
		repo, rest, _ := strings.Cut(strings.TrimPrefix(p, "/"), "/resolve/")
		rev, file, _ := strings.Cut(rest, "/")
		if !f.made[repo] || rev != "main" && rev != f.commit(repo) {
			http.NotFound(w, r)
			return
		}
		http.ServeFile(w, r, filepath.Join(f.dir, repo, filepath.FromSlash(file)))
	default:
		http.NotFound(w, r)
	}
}

// hwnpu-dev models hf push puts what the source holds in a repository of its namespace, as the
// commands look for it now (an export made before the version was written goes at the
// version), pins the commit in the definition file, and hf check finds it whole; a machine
// with only the hub then reads the export from there.
func TestPush(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("the fake hf command is a shell script")
	}
	ctx := context.Background()
	work := t.TempDir()
	hub := &fakeHub{dir: filepath.Join(work, "hub"), made: map[string]bool{}}
	srv := httptest.NewServer(hub)
	defer srv.Close()
	defer func(e string) { shared.HubEndpoint = e }(shared.HubEndpoint)
	shared.HubEndpoint = srv.URL
	cli := filepath.Join(work, "hf")
	os.WriteFile(cli, []byte("#!/bin/sh\n[ \"$1\" = upload-large-folder ] && [ \"$HF_TOKEN\" = tok ] || exit 2\nmkdir -p \""+hub.dir+"/$2\" && cp -R \"$3\"/. \""+hub.dir+"/$2/\"\n"), 0o755)
	t.Setenv("HWNPU_HF_CLI", cli)
	t.Setenv("HF_TOKEN", "tok")
	t.Setenv("HF_HOME", filepath.Join(work, "hfhome"))
	t.Setenv("XDG_CONFIG_HOME", filepath.Join(work, "config"))
	t.Setenv("HWNPU_CACHE", filepath.Join(work, "cache"))
	t.Setenv(shared.SourceEnv, "file://"+filepath.ToSlash(filepath.Join(work, "share"))+"?writeback=true")
	t.Setenv(chat.ModelsEnv, "")
	defer func(c func(string) bool) { confirm = c }(confirm)
	confirm = func(string) bool { return true }
	defer chat.LoadModels(nil, false, nil)

	// the export, put in the source before its model had a version
	defs := filepath.Join(work, "models.yaml")
	os.WriteFile(defs, []byte("version: 1\nhub: {push: me}\nmodels:\n  whisper-large-v3-turbo:\n    machine: 8\n"+noPin), 0o644)
	if err := chat.LoadModels([]string{defs}, true, nil); err != nil {
		t.Fatal(err)
	}
	m, _ := chat.Lookup("whisper-large-v3-turbo")
	wants, err := amdWants(ctx, m)
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(work, "export")
	os.MkdirAll(dir, 0o755)
	os.WriteFile(filepath.Join(dir, "manifest.json"), []byte(`{"kind":"whisper"}`), 0o644)
	if _, err := shared.Describe(dir, wants[0]); err != nil {
		t.Fatal(err)
	}
	if err := shared.PutEntry(ctx, shared.Sources()[0], dir); err != nil {
		t.Fatal(err)
	}

	// the version written (the recipe's, models.whisper-large-v3-turbo.amd.version): it goes as vN
	rf, err := recipe.Builtin()
	if err != nil {
		t.Fatal(err)
	}
	rm, _ := rf.AMDOf("whisper-large-v3-turbo")
	v := fmt.Sprintf("v%d", rm.Version)
	os.WriteFile(defs, []byte(fmt.Sprintf("version: 1\nhub: {push: me}\nmodels:\n  whisper-large-v3-turbo:\n    targets: {amd-xdna2: {version: %d}}\n    machine: 8\n", rm.Version)+noPin), 0o644)
	c := &devopt.Defs{Models: defs}
	if err := (PushCmd{Sel: []string{"whisper-large-v3-turbo/amd-xdna2"}, Write: defs}).Run(c); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(filepath.Join(hub.dir, "me/whisper-large-v3-turbo/amd-xdna2/whisper-large-v3-turbo.amd-npu2/"+v+"/entry.json"))
	var e shared.Entry
	if err := json.Unmarshal(b, &e); err != nil || e.Version != rm.Version || e.Pack == nil {
		t.Fatalf("on the hub: %s %v", b, err)
	}
	for _, f := range []string{"config.json", "README.md", e.Pack.Path} {
		if _, err := os.Stat(filepath.Join(hub.dir, "me/whisper-large-v3-turbo", "amd-xdna2/whisper-large-v3-turbo.amd-npu2/"+v, f)); err != nil && f == e.Pack.Path {
			t.Error(err)
		}
		if _, err := os.Stat(filepath.Join(hub.dir, "me/whisper-large-v3-turbo", f)); err != nil && f != e.Pack.Path {
			t.Error(err)
		}
	}
	card, _ := os.ReadFile(filepath.Join(hub.dir, "me/whisper-large-v3-turbo/README.md"))
	if !strings.Contains(string(card), "base_model: openai/whisper-large-v3-turbo") || !strings.Contains(string(card), "base_model_relation: quantized") {
		t.Errorf("card:\n%s", card)
	}
	pinned, _ := os.ReadFile(defs)
	commit := hub.commit("me/whisper-large-v3-turbo")
	if !strings.Contains(string(pinned), "me/whisper-large-v3-turbo:\n        amd-xdna2: "+commit) {
		t.Fatalf("not pinned:\n%s", pinned)
	}

	// hf check: whole at the pinned commit
	if err := (CheckCmd{}).Run(c); err != nil {
		t.Errorf("check: %v", err)
	}
	// pushed again: nothing to put
	if err := (PushCmd{Sel: []string{"whisper-large-v3-turbo/amd-xdna2"}, Write: defs}).Run(c); err == nil {
		t.Error("put again")
	}

	// another machine, no source: the export comes from the hub
	t.Setenv(shared.SourceEnv, "")
	t.Setenv("HWNPU_CACHE", filepath.Join(work, "cache2"))
	if err := chat.LoadModels([]string{defs}, true, nil); err != nil {
		t.Fatal(err)
	}
	m, _ = chat.Lookup("whisper-large-v3-turbo")
	wants, _ = amdWants(ctx, m)
	out := filepath.Join(work, "fetched")
	if _, err := shared.FetchEntry(ctx, shared.Sources(), wants[0].Path, func(x shared.Entry) bool { return chat.Takes(wants[0], x) }, out, nil); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(filepath.Join(out, "manifest.json")); string(b) != `{"kind":"whisper"}` {
		t.Errorf("fetched %q", b)
	}
}

// noPin drops the built-in pin of whisper's export for AMD's NPU (models.yaml's exports): the
// tests of versions put an export of their own.
const noPin = "    exports:\n      amd-xdna2:\n        whisper-large-v3-turbo.amd-npu2: none\n"
