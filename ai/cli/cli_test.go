package cli

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/modelstore"
)

type fakeModel struct{}

func (fakeModel) Close() error { return nil }

// testApp is an app of one model whose role writes a file for it, in a cache of its own, with
// only the built-in definitions, on a machine taken to have Intel's NPU.
func testApp(t *testing.T) *ai.App {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	t.Setenv("LOCALAPPDATA", dir)
	t.Setenv(chat.ModelsEnv, "")
	t.Setenv(chat.PolicyEnv, "")
	t.Setenv(chat.MemoryEnv, "8")
	t.Setenv(llm.BackendEnv, llm.BackendIntel)
	saved := chat.SystemModelsFile
	chat.SystemModelsFile = func() string { return "" }
	t.Cleanup(func() {
		chat.SystemModelsFile = saved
		cache.SetRoot("")
		chat.LoadModels(nil, false, nil)
	})
	role := &airole.Role{Name: "transcribe", Kinds: []chat.Kind{chat.KindTranscribe},
		Prepare: func(e airole.PrepareEnv) (f airole.Files, err error) {
			p := filepath.Join(e.Dir, "program.xml")
			if err := os.MkdirAll(e.Dir, 0o755); err != nil {
				return f, err
			}
			f.Set("program", p)
			return f, os.WriteFile(p, nil, 0o644)
		},
		Load: func(context.Context, airole.Files, airole.LoadEnv) (io.Closer, error) { return fakeModel{}, nil }}
	app := ai.NewApp("demo", ai.Define[fakeModel](role, "transcribe"))
	off := false
	app.Store = ai.Store{Models: modelstore.Options{Root: t.TempDir(), Sources: []string{}, HuggingFace: &off}}
	return app
}

func TestCommands(t *testing.T) {
	app := testApp(t)
	ctx := context.Background()
	list := func() string {
		t.Helper()
		var b bytes.Buffer
		if err := List(ctx, app, ListOptions{Out: &b}); err != nil {
			t.Fatal(err)
		}
		return b.String()
	}
	if s := list(); !strings.Contains(s, "demo's models, for this machine's NPU (intel)") || !strings.Contains(s, "missing") || !strings.Contains(s, "1.6 GB to download") ||
		!strings.Contains(s, "a machine of 8 GB has about 3 GB for models") {
		t.Errorf("the list before anything:\n%s", s)
	}
	var out, log bytes.Buffer
	if err := Download(ctx, app, PrepareOptions{Quiet: true, Out: &out, Log: &log}); err != nil {
		t.Fatalf("download: %v", err)
	}
	if !strings.Contains(log.String(), "[1/1] transcribe (whisper-large-v3-turbo): downloading and converting") || !strings.Contains(out.String(), "ready, not compiled") {
		t.Errorf("download wrote:\n%s\n%s", log.String(), out.String())
	}
	if s := list(); !strings.Contains(s, "ready, not compiled") || !strings.Contains(s, "All here;") {
		t.Errorf("the list after the download:\n%s", s)
	}
	if err := Prepare(ctx, app, PrepareOptions{Quiet: true, Out: io.Discard, Log: io.Discard}); err != nil {
		t.Fatalf("prepare: %v", err)
	}
	if s := list(); !strings.Contains(s, "whisper-large-v3-turbo  compiled") || !strings.Contains(s, "All here and compiled") {
		t.Errorf("the list after prepare:\n%s", s)
	}
	var j bytes.Buffer
	if err := List(ctx, app, ListOptions{JSON: true, Out: &j}); err != nil || !strings.Contains(j.String(), `"usable_gb": 3`) || !strings.Contains(j.String(), `"compiled": true`) {
		t.Errorf("the list as JSON: %v\n%s", err, j.String())
	}
}

// TestRun: an app that parses its command line itself hands its arguments over.
func TestRun(t *testing.T) {
	app := testApp(t)
	ctx := context.Background()
	for _, args := range [][]string{nil, {"interpret", "talk.wav"}, {"--mic"}} {
		if handled, err := Run(ctx, app, args); handled || err != nil {
			t.Errorf("%q is the app's own: handled %v, %v", args, handled, err)
		}
	}
	for _, args := range [][]string{{"models"}, {"models", "list", "--json"}, {"models", "download", "-q"}, {"models", "compile", "-q"}, {"models", "--json"}} {
		if handled, err := Run(ctx, app, args); !handled || err != nil {
			t.Errorf("%q: handled %v, %v", args, handled, err)
		}
	}
	if p, err := app.Plan(ctx); err != nil || !p.Compiled() {
		t.Errorf("after \"models compile\": %v, compiled %v", err, p != nil && p.Compiled())
	}
	if handled, err := Run(ctx, app, []string{"models", "remove"}); !handled || err == nil {
		t.Errorf("an unknown command of models: handled %v, %v", handled, err)
	}
	if handled, err := Run(ctx, app, []string{"models", "list", "--no-such-flag"}); !handled || err == nil {
		t.Errorf("an unknown flag: handled %v, %v", handled, err)
	}
	if !strings.Contains(Usage, "models prepare") || !strings.Contains(Usage, "status") {
		t.Errorf("the usage:\n%s", Usage)
	}
}

func TestNetworkLines(t *testing.T) {
	nodes := []flow.NodeInfo{
		{Name: "mic", To: []flow.EdgeInfo{{To: "vad"}}},
		{Name: "vad", To: []flow.EdgeInfo{{To: "transcribe"}}},
		{Name: "transcribe", To: []flow.EdgeInfo{{To: "captions"}, {To: "translate"}}},
		{Name: "captions"},
		{Name: "translate", To: []flow.EdgeInfo{{To: "speech", Policy: "keep 4"}}},
		{Name: "speech", To: []flow.EdgeInfo{{To: "out"}}},
		{Name: "out"},
	}
	want := []string{"mic → vad → transcribe → captions, translate", "translate → speech (keep 4) → out"}
	if got := network(nodes); strings.Join(got, "|") != strings.Join(want, "|") {
		t.Errorf("the network as lines:\n%s\nwant:\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
	}
	if got := network(nil); len(got) != 0 {
		t.Errorf("no network gave %q", got)
	}
}
