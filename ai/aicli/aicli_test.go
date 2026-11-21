package aicli

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/alecthomas/kong"

	"github.com/shibukawa/hwnpu/ai"
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
	var cli struct {
		Models Models `cmd:""`
		Status Status `cmd:""`
	}
	parse := func(args ...string) *kong.Context {
		t.Helper()
		k, err := kong.New(&cli, kong.Name("demo"), Bind(ctx, app))
		if err != nil {
			t.Fatal(err)
		}
		kctx, err := k.Parse(args)
		if err != nil {
			t.Fatal(err)
		}
		return kctx
	}
	plan := func() string {
		t.Helper()
		p, err := app.Plan(ctx)
		if err != nil {
			t.Fatal(err)
		}
		var b bytes.Buffer
		writePlan(&b, app, p)
		return b.String()
	}
	if got := parse("models").Command(); got != "models list" {
		t.Errorf("models runs %q, want the list", got)
	}
	if got := parse("models", "compile").Command(); got != "models prepare" {
		t.Errorf("models compile runs %q, want models prepare", got)
	}
	if s := plan(); !strings.Contains(s, "demo's models, for this machine's NPU (intel)") || !strings.Contains(s, "missing") || !strings.Contains(s, "1.6 GB to download") {
		t.Errorf("the list before anything:\n%s", s)
	}
	if err := parse("models", "list").Run(); err != nil {
		t.Errorf("models list: %v", err)
	}
	if err := parse("models", "download", "-q").Run(); err != nil {
		t.Fatalf("models download: %v", err)
	}
	if s := plan(); !strings.Contains(s, "ready, not compiled") || !strings.Contains(s, "All here;") {
		t.Errorf("the list after the download:\n%s", s)
	}
	if err := parse("models", "prepare", "-q").Run(); err != nil {
		t.Fatalf("models prepare: %v", err)
	}
	if s := plan(); !strings.Contains(s, "whisper-large-v3-turbo  compiled") || !strings.Contains(s, "All here and compiled") {
		t.Errorf("the list after prepare:\n%s", s)
	}
}
