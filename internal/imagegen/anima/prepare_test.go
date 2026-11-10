package anima

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// TestExportName puts what changes the programs, and the layout version, in the name.
func TestExportName(t *testing.T) {
	o := ConvertOptions{Backend: "coreml", Width: 1024, Height: 1024, Dit: "anima-turbo-v1.1.safetensors", TextFormat: "int8", DitFormat: "int8", Group: 4, QChunk: 512}
	if got, want := ExportName(o), "anima-turbo-v1.1-1024x1024-int8-int8-g4-q512-v3.coreml"; got != want {
		t.Errorf("%s, want %s", got, want)
	}
	o.Backend = "intel"
	if got := ExportName(o); filepath.Ext(got) != ".intel" {
		t.Errorf("intel: %s", got)
	}
}

// TestPrepareDone returns an export whose .done is there without downloading or converting.
func TestPrepareDone(t *testing.T) {
	dir := t.TempDir()
	o := ConvertOptions{Backend: "coreml", Width: 512, Height: 512, Group: 4, QChunk: 512}
	text, dit := Formats("coreml")
	want := filepath.Join(dir, ExportName(ConvertOptions{Backend: "coreml", Width: 512, Height: 512, Dit: "anima-turbo-v1.1.safetensors",
		TextFormat: text, DitFormat: dit, Group: 4, QChunk: 512}))
	if err := os.WriteFile(want+".done", []byte(want), 0o644); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // a download would fail
	got, err := Prepare(ctx, Setup{Dir: dir, ConvertOptions: o})
	if err != nil || got != want {
		t.Fatalf("%q, %v; want %q", got, err, want)
	}
}

func TestDefaultGroup(t *testing.T) {
	for _, c := range []struct {
		backend string
		size    int
		want    int
	}{
		{"coreml", 1024, 4}, {"intel", 512, 4}, {"intel", 768, 1}, {"intel", 1024, 1}, {"intel", 256, 4}, {"intel", 640, 2},
	} {
		if g := DefaultGroup(c.backend, c.size, c.size); g != c.want {
			t.Errorf("%s %d: group %d, want %d", c.backend, c.size, g, c.want)
		}
	}
}
