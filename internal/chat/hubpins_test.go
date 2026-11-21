package chat

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	shared "github.com/shibukawa/hwnpu/internal/store"
)

const (
	commitA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	commitB = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	commitC = "cccccccccccccccccccccccccccccccccccccccc"
)

// A model's hub is merged by repository and by target, the repositories of the files read last
// first; hub.push is the last file's; a model like another does not take its hub.
func TestHubDefinitions(t *testing.T) {
	if HubPush != "hwnpu" || HubPushFile != Builtin {
		t.Fatalf("built in: push %q from %q", HubPush, HubPushFile)
	}
	_, err := userFiles(t, `version: 1
models:
  whisper-large-v3-turbo:
    hub:
      hwnpu/whisper-large-v3-turbo: {amd-xdna2: `+commitA+`, apple: `+commitB+`}
`, `version: 1
hub: {push: myorg}
models:
  whisper-large-v3-turbo:
    hub:
      hwnpu/whisper-large-v3-turbo: {apple: `+commitC+`}
      myorg/whisper-large-v3-turbo: {amd-xdna2: `+commitC+`, coreai-h14g: main}
  whisper-like:
    like: whisper-large-v3-turbo
    from: hf:openai/whisper-large-v3-turbo@41f01f3fe87f28c78e2fbf8b568835947dd65ed9
`)
	if err != nil {
		t.Fatal(err)
	}
	if HubPush != "myorg" || filepath.Base(HubPushFile) != "models.yaml" || HubPushFile == Builtin {
		t.Errorf("push %q from %q", HubPush, HubPushFile)
	}
	m, _ := Lookup("whisper-large-v3-turbo")
	if len(m.Hub) != 2 || m.Hub[0].Repo != "myorg/whisper-large-v3-turbo" || m.Hub[1].Repo != "hwnpu/whisper-large-v3-turbo" {
		t.Fatalf("hub %+v", m.Hub)
	}
	if p := m.Hub[1].Pins; p["amd-xdna2"] != commitA || p["apple"] != commitC {
		t.Errorf("the second file changes apple only: %v", p)
	}
	if l, _ := Lookup("whisper-like"); len(l.Hub) != 0 {
		t.Errorf("a model like another takes its hub: %+v", l.Hub)
	}
}

func TestHubErrors(t *testing.T) {
	for _, tc := range []struct{ name, file, want string }{
		{"repo", "version: 1\nmodels:\n  LFM3-1.7B:\n    hub: {hwnpu: {apple: main}}\n", "OWNER/NAME"},
		{"target", "version: 1\nmodels:\n  LFM3-1.7B:\n    hub: {hwnpu/x: {mac: main}}\n", "not a target"},
		{"commit", "version: 1\nmodels:\n  LFM3-1.7B:\n    hub: {hwnpu/x: {apple: abc}}\n", "not a commit"},
		{"push", "version: 1\nhub: {push: \"a/b\"}\n", "namespace"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := userFiles(t, tc.file)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Errorf("got %v, want an error with %q", err, tc.want)
			}
		})
	}
}

// Every model of the built-in definitions with a hub is published as hwnpu-dev models hf push wants
// it: its files' commit is pinned, and each target it pins a commit for has a version.
func TestBuiltinHub(t *testing.T) {
	for _, m := range Catalog {
		if len(m.Hub) == 0 {
			continue
		}
		if !commitRE.MatchString(m.Revision) && len(m.SHA256) == 0 {
			t.Errorf("%s has a hub but its files' commit is not pinned (from: hf:%s@COMMIT)", m.Name, m.Repo)
		}
		if m.Develop {
			t.Errorf("%s has a hub and is in development", m.Name)
		}
		for _, r := range m.Hub {
			for target := range r.Pins {
				conv := "coreai"
				if target == "amd-xdna2" {
					conv = target
				}
				if m.Targets[conv] == 0 {
					t.Errorf("%s: hub %s pins %s, but targets has no version of %s", m.Name, r.Repo, target, conv)
				}
			}
		}
	}
}

// The hub layer reads a model's conversion from its repository at the commit of the machine's
// target, signs in when the repository refuses, and has nothing else.
func TestHubLayer(t *testing.T) {
	var auths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auths = append(auths, r.Header.Get("Authorization"))
		switch r.URL.Path {
		case "/hwnpu/m/resolve/" + commitA + "/amd-xdna2/m.amd-npu2/v1/entry.json":
			io.WriteString(w, "amd")
		case "/hwnpu/gated/resolve/" + commitA + "/amd-xdna2/g/v1/entry.json":
			if r.Header.Get("Authorization") != "Bearer tok" {
				http.Error(w, "sign in", http.StatusUnauthorized)
				return
			}
			io.WriteString(w, "gated")
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	defer func(e string) { shared.HubEndpoint = e }(shared.HubEndpoint)
	shared.HubEndpoint = srv.URL
	shared.SetHub(map[string][]shared.HubRepo{
		"m":     {{Repo: "hwnpu/m", Pins: map[string]string{"amd-xdna2": commitA, "apple": commitB}}},
		"gated": {{Repo: "hwnpu/gated", Pins: map[string]string{"amd-xdna2": commitA}}},
	}, func() string { return "" }, func(ctx context.Context, repo string, status int, sent string) (string, error) {
		if repo != "hwnpu/gated" || status != http.StatusUnauthorized || sent != "" {
			t.Errorf("asked for %s %d %q", repo, status, sent)
		}
		return "tok", nil
	})
	defer setHub(Catalog)
	h := shared.Hub{}
	ctx := context.Background()
	if b, err := h.Read(ctx, "converted/m/amd-xdna2/m.amd-npu2/v1/entry.json"); err != nil || string(b) != "amd" {
		t.Fatalf("%q %v", b, err)
	}
	if auths[0] != "" {
		t.Errorf("a token went to a repository that did not ask: %q", auths[0])
	}
	for _, key := range []string{"original/hwnpu/m/config.json", "converted/other/amd-xdna2/x/v1/entry.json", "converted/m/intel-npu4/x/v1/entry.json"} {
		n := len(auths)
		if _, err := h.Read(ctx, key); err == nil || !strings.Contains(err.Error(), "not there") || len(auths) != n {
			t.Errorf("%s: %v (%d requests)", key, err, len(auths)-n)
		}
	}
	if b, err := h.Read(ctx, "converted/gated/amd-xdna2/g/v1/entry.json"); err != nil || string(b) != "gated" {
		t.Fatalf("gated: %q %v", b, err)
	}
	dst := filepath.Join(t.TempDir(), "x")
	if _, err := h.Get(ctx, "converted/m/amd-xdna2/m.amd-npu2/v1/entry.json", dst, nil); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(dst); string(b) != "amd" {
		t.Errorf("got %q", b)
	}
	if err := h.Put(ctx, "converted/m/amd-xdna2/x", dst); err == nil {
		t.Error("the hub layer took a put")
	}
}

func TestHubPin(t *testing.T) {
	pins := map[string]string{"amd-xdna2": "a", "apple": "b", "coreai-h14g": "c"}
	for _, tc := range []struct{ target, arch, want string }{
		{"amd-xdna2", "", "a"},
		{"coreai", "h13g", "b"},
		{"coreai", "h14g", "c"},
		{"coreai-h14g", "h14g", "c"},
		{"coreai-h15g", "h15g", "b"},
		{"any", "h14g", "c"},
		{"intel-npu4", "", ""},
	} {
		if got := shared.HubPin(pins, tc.target, tc.arch); got != tc.want {
			t.Errorf("%s on %q: %q, want %q", tc.target, tc.arch, got, tc.want)
		}
	}
}

// disk is read by vendor; a vendor it leaves out is not known (not the largest, as bytes).
func TestDisk(t *testing.T) {
	if _, err := userFiles(t, "version: 1\nmodels:\n  LFM3-1.7B:\n    disk: {amd: 2.28e9}\n  Qwen3-0.6B:\n    disk: 1.5e9\n"); err != nil {
		t.Fatal(err)
	}
	m, _ := Lookup("LFM3-1.7B")
	if m.DiskFor("amd") != 2.28e9 || m.DiskFor("apple") != 0 || m.BytesFor("amd") == 0 {
		t.Errorf("amd %v apple %v", m.DiskFor("amd"), m.DiskFor("apple"))
	}
	if q, _ := Lookup("Qwen3-0.6B"); q.DiskFor("apple") != 1.5e9 {
		t.Errorf("one value for every vendor: %v", q.DiskFor("apple"))
	}
}
