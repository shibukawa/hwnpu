package aihost

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/cachecmd"
	"github.com/shibukawa/hwnpu/modelstore"
)

// heldUses are the Runtimes that hold the fake model in the engine, 0 when none does.
func heldUses() int {
	for _, m := range ai.Held() {
		if m.Role == fakeRole.Name {
			return m.Uses
		}
	}
	return 0
}

// TestSharedLoad: two sessions (two apps' Runtimes) that get the same model load it once; it is
// unloaded when the last lets it go, in either order.
func TestSharedLoad(t *testing.T) {
	serve(t)
	ctx := context.Background()
	use := ai.Define[*fakeModel](fakeRole, "transcribe")
	st := testStore(t)
	app := func(name string) *ai.App {
		a := ai.NewApp(name, use).Memory(8)
		a.Store = st
		return a
	}
	if err := app("prepare").Prepare(ctx, nil); err != nil {
		t.Fatal(err)
	}
	for _, firstOut := range []bool{true, false} {
		loads, closed := counts.loads.Load(), counts.closed.Load()
		rt1, err := app("one").Open(ctx)
		if err != nil {
			t.Fatal(err)
		}
		rt2, err := app("two").Open(ctx)
		if err != nil {
			t.Fatal(err)
		}
		m1, err := use.Get(ctx, rt1)
		if err != nil {
			t.Fatal(err)
		}
		m2, err := use.Get(ctx, rt2)
		if err != nil {
			t.Fatal(err)
		}
		if n := counts.loads.Load() - loads; n != 1 || heldUses() != 2 {
			t.Fatalf("two sessions: %d loads, held by %d", n, heldUses())
		}
		for _, m := range []*fakeModel{m1, m2} {
			if out, err := m.Echo(ctx, "x", nil); err != nil || out != "X" {
				t.Errorf("a shared model: %q %v", out, err)
			}
		}
		first, second := rt1, rt2
		if !firstOut {
			first, second = rt2, rt1
		}
		if err := first.Close(); err != nil {
			t.Fatal(err)
		}
		if counts.closed.Load() != closed || heldUses() != 1 {
			t.Errorf("one let go: closed %d, held by %d", counts.closed.Load()-closed, heldUses())
		}
		if _, err := use.Get(ctx, second); err != nil {
			t.Errorf("the other's Get after: %v", err)
		}
		if err := second.Close(); err != nil {
			t.Fatal(err)
		}
		if counts.closed.Load()-closed != 1 || heldUses() != 0 {
			t.Errorf("the last let go: closed %d, held by %d", counts.closed.Load()-closed, heldUses())
		}
	}
	// a model run otherwise (its role's settings) is a copy of its own
	loads := counts.loads.Load()
	rt1, _ := app("one").Open(ctx)
	rt2, _ := app("two").Open(ctx, ai.LoadSettings(fakeRole.Name, map[string]string{"idle": "1m"}))
	use.Get(ctx, rt1)
	use.Get(ctx, rt2)
	if n := counts.loads.Load() - loads; n != 2 {
		t.Errorf("two settings: %d loads", n)
	}
	rt1.Close()
	rt2.Close()
}

// TestCacheThroughHwnpud: hwnpu cache runs in the engine, which keeps what it holds loaded.
func TestCacheThroughHwnpud(t *testing.T) {
	serve(t)
	root := t.TempDir()
	t.Setenv(cache.RootEnv, root)
	ctx := context.Background()
	use := ai.Define[*fakeModel](fakeRole, "transcribe")
	off := false
	app := ai.NewApp("cache", use).Memory(8)
	app.Store = ai.Store{Models: modelstore.Options{Sources: []string{}, HuggingFace: &off}}
	if err := app.Prepare(ctx, nil); err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(root, "models", "whisper-large-v3-turbo")
	if _, err := os.Stat(dir); err != nil {
		t.Fatal(err)
	}
	rt, err := app.Open(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := use.Get(ctx, rt); err != nil {
		t.Fatal(err)
	}
	if err := (cachecmd.LsCmd{}).Run(ctx); err != nil {
		t.Errorf("ls: %v", err)
	}
	clear := cachecmd.ClearCmd{Patterns: []string{"whisper-large-v3-turbo"}, Yes: true}
	if err := clear.Run(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(dir); err != nil {
		t.Errorf("a model held loaded was removed: %v", err)
	}
	rt.Close()
	if err := clear.Run(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(dir); err == nil {
		t.Error("a model let go was not removed")
	}
	if v, closeFn, err := cachecmd.RemoteStore(ctx); err != nil || v == nil || v.Root() != root {
		t.Errorf("the status's view of hwnpud's cache: %v %v", v, err)
	} else {
		closeFn()
	}
}
