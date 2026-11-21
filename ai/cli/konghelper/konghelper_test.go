package konghelper

import (
	"context"
	"testing"

	"github.com/alecthomas/kong"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/modelstore"
)

func TestCommands(t *testing.T) {
	// an app without models, in a cache of its own: the commands are parsed and reach the app
	app := ai.NewApp("demo")
	off := false
	app.Store = ai.Store{Models: modelstore.Options{Root: t.TempDir(), Sources: []string{}, HuggingFace: &off}}
	var cmds struct {
		Models Models `cmd:""`
		Status Status `cmd:""`
	}
	parse := func(args ...string) *kong.Context {
		t.Helper()
		k, err := kong.New(&cmds, kong.Name("demo"), Bind(context.Background(), app))
		if err != nil {
			t.Fatal(err)
		}
		kctx, err := k.Parse(args)
		if err != nil {
			t.Fatal(err)
		}
		return kctx
	}
	if got := parse("models").Command(); got != "models list" {
		t.Errorf("models runs %q, want the list", got)
	}
	if got := parse("models", "compile", "--accept-license", "-q").Command(); got != "models prepare" {
		t.Errorf("models compile runs %q, want models prepare", got)
	}
	if got := parse("status", "-w", "2s", "--json").Command(); got != "status" || cmds.Status.Watch.Seconds() != 2 || !cmds.Status.JSON {
		t.Errorf("status -w 2s --json: %q, %+v", got, cmds.Status)
	}
	for _, args := range [][]string{{"models", "list", "--json"}, {"models", "download", "-q"}, {"models", "prepare", "-q"}} {
		if err := parse(args...).Run(); err != nil {
			t.Errorf("%q: %v", args, err)
		}
	}
}
