package anima

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/imagegen/aspect"
)

// TestExportName puts what changes the programs, and the layout version, in the name.
func TestExportName(t *testing.T) {
	o := ConvertOptions{Backend: "coreml", Width: 1024, Height: 1024, Dit: "anima-turbo-v1.1.safetensors", TextFormat: "int8", DitFormat: "int8", Group: 4, QChunk: 512}
	if got, want := ExportName(o), "anima-turbo-v1.1-1024x1024-int8-int8-g4-q512-v4.coreml"; got != want {
		t.Errorf("%s, want %s", got, want)
	}
	// the sizes whose patches are not a multiple of 8 (880 x 1168: 4015) are of the padded
	// cross-attention's version on Apple; 832 x 1248 (4056) keeps the others'
	for _, c := range []struct {
		w, h    int
		backend string
		want    string
	}{
		{880, 1168, "coreai", "anima-turbo-v1.1-880x1168-int8-int8-g4-q512-v6.coreai"},
		{1360, 752, "coreai", "anima-turbo-v1.1-1360x752-int8-int8-g4-q512-v6.coreai"},
		{832, 1248, "coreai", "anima-turbo-v1.1-832x1248-int8-int8-g4-q512-v6.coreai"},
		{880, 1168, "intel", "anima-turbo-v1.1-880x1168-int8-int8-g4-q512-v4.intel"},
	} {
		p := o
		p.Width, p.Height, p.Backend = c.w, c.h, c.backend
		if got := ExportName(p); got != c.want {
			t.Errorf("%s, want %s", got, c.want)
		}
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

// TestExportNameSizes names an export of all the aspect ratios as one (aspect.Tag), whatever size
// is drawn, at the padded version (some of its sizes are padded).
func TestExportNameSizes(t *testing.T) {
	o := ConvertOptions{Backend: "coreai", Dit: "anima-turbo-v1.1.safetensors", TextFormat: "int8", DitFormat: "int8", Group: 4, QChunk: 512,
		Sizes: aspect.SizesOf(aspect.All)}
	var names []string
	for _, a := range aspect.All {
		p := o
		p.Width, p.Height = a.Width, a.Height
		names = append(names, ExportName(p))
	}
	want := "anima-turbo-v1.1-" + aspect.Tag(aspect.SizesOf(aspect.All)) + "-int8-int8-g4-q512-v6.coreai"
	for _, n := range names {
		if n != want {
			t.Errorf("%s, want %s", n, want)
		}
	}
}

// TestManifestAt picks a size's parts of an export of several, and refuses one it lacks.
func TestManifestAt(t *testing.T) {
	m := &ProgramManifest{Width: 1024, Height: 1024, Blocks: []string{"b.aimodel#s1024x1024"}, Final: "f.aimodel#s1024x1024", Vae: []string{"v.aimodel#s1024x1024"},
		Sizes: []SizeParts{
			{1024, 1024, []string{"b.aimodel#s1024x1024"}, "f.aimodel#s1024x1024", []string{"v.aimodel#s1024x1024"}},
			{880, 1168, []string{"b.aimodel#s880x1168"}, "f.aimodel#s880x1168", []string{"v.aimodel#s880x1168"}},
		}}
	if err := m.At(880, 1168); err != nil || m.Width != 880 || m.Final != "f.aimodel#s880x1168" || m.Blocks[0] != "b.aimodel#s880x1168" || m.Vae[0] != "v.aimodel#s880x1168" {
		t.Errorf("At(880, 1168): %v, %+v", err, m)
	}
	if err := m.At(512, 512); err == nil {
		t.Error("At(512, 512) of an export without it: no error")
	}
}
