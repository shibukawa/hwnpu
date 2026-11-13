package cache

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestRoot(t *testing.T) {
	d := t.TempDir()
	t.Setenv(RootEnv, d)
	if Root() != d || Sub("models") != filepath.Join(d, "models") || Entries(AMDTarget) != filepath.Join(d, "entries", "amd-xdna2") {
		t.Errorf("%s %s", Root(), Sub("models"))
	}
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	old := filepath.Join(home, "models", "m.amd-npu2")
	if err := os.MkdirAll(old, 0o755); err != nil {
		t.Fatal(err)
	}
	isDir := func(p string) bool { fi, err := os.Stat(p); return err == nil && fi.IsDir() }
	if got := FindExport("m.amd-npu2", isDir); got != old {
		t.Errorf("one in ~/models: %q", got)
	}
	now := filepath.Join(Entries(AMDTarget), "m.amd-npu2")
	os.MkdirAll(now, 0o755)
	if got := FindExport("m.amd-npu2", isDir); got != now {
		t.Errorf("the root's first: %q", got)
	}
}

// TestDone checks that a mark holds its path relative to itself, and that a mark of an earlier
// hwnpu (an absolute path) still reads where the directory is now.
func TestDone(t *testing.T) {
	d := t.TempDir()
	prog := filepath.Join(d, "lfm3-lut8.mlpackage")
	os.MkdirAll(prog, 0o755)
	done := filepath.Join(d, "lfm3-lut8.done")
	if err := WriteDone(done, prog); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(done); string(b) != "lfm3-lut8.mlpackage" {
		t.Errorf("wrote %q", b)
	}
	moved := t.TempDir()
	if err := os.Rename(prog, filepath.Join(moved, filepath.Base(prog))); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(done, filepath.Join(moved, filepath.Base(done))); err != nil {
		t.Fatal(err)
	}
	if p, ok := ReadDone(filepath.Join(moved, "lfm3-lut8.done")); !ok || p != filepath.Join(moved, "lfm3-lut8.mlpackage") {
		t.Errorf("moved: %q %v", p, ok)
	}
	// an earlier hwnpu's absolute path, of a directory since moved
	old := filepath.Join(moved, "old.done")
	os.WriteFile(old, []byte("/elsewhere/lfm3-lut8.mlpackage"), 0o644)
	if p, ok := ReadDone(old); !ok || p != filepath.Join(moved, "lfm3-lut8.mlpackage") {
		t.Errorf("absolute: %q %v", p, ok)
	}
	os.WriteFile(old, []byte("/elsewhere/none.mlpackage"), 0o644)
	if _, ok := ReadDone(old); ok {
		t.Error("a mark of nothing here was read")
	}
}

// TestLock checks that a lock is held until let go, and that a wait ends with its context.
func TestLock(t *testing.T) {
	t.Setenv(RootEnv, t.TempDir())
	ctx := context.Background()
	unlock, err := Lock(ctx, "/x/model", "model", nil)
	if err != nil {
		t.Fatal(err)
	}
	short, cancel := context.WithTimeout(ctx, 300*time.Millisecond)
	defer cancel()
	if _, err := Lock(short, "/x/model", "model", nil); err == nil {
		t.Fatal("taken twice")
	}
	other, err := Lock(ctx, "/x/other", "other", nil)
	if err != nil {
		t.Fatal(err)
	}
	other()
	unlock()
	again, err := Lock(ctx, "/x/model", "model", nil)
	if err != nil {
		t.Fatal(err)
	}
	again()
}
