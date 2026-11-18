package flux3klein

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/shibukawa/hwnpu/internal/imagegen/flux3klein/mil"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
)

func ironTestConfig(w, h int) IronConfig {
	return IronConfig{Width: w, Height: h, TextSlot: 512, TextPad: 80, Hidden: 3072, HeadDim: 128, Latent: 128, Text: 2560,
		TextLayers: mil.TextTaps, Axes: []int{32, 32, 32, 32}, Theta: 10000, ImageRopeScale: mil.ImageRopeScale}
}

// IRON's RoPE tables hold the angles of the Core AI programs' (mil.RopeValues, mil.TextRope), each
// frequency's cos and sin side by side.
func TestIronAngles(t *testing.T) {
	c := ironTestConfig(512, 512)
	d := mil.Dit{Width: 512, Height: 512, HeadDim: 128, Axes: c.Axes, Theta: c.Theta, TextRows: 80}
	img, txt := c.ironIDs()
	if !slices.Equal(img, d.ImageIDs()) || !slices.Equal(txt[:80], d.TextIDs()) {
		t.Fatal("the position ids are not the Core AI programs'")
	}
	check := func(what string, got, cos, sin []float32, rows int) {
		t.Helper()
		D := c.HeadDim
		for r := 0; r < rows; r++ {
			for i := 0; i < D/2; i++ {
				gc, gs := got[r*D+2*i], got[r*D+2*i+1]
				if math.Abs(float64(gc-cos[r*D+i])) > 1e-6 || math.Abs(float64(gs-sin[r*D+i])) > 1e-6 {
					t.Fatalf("%s: row %d frequency %d: (%v, %v), want (%v, %v)", what, r, i, gc, gs, cos[r*D+i], sin[r*D+i])
				}
			}
		}
	}
	cos, sin := mil.RopeValues(d, img)
	check("image", c.ironAngles(img), cos, sin, len(img))
	pos := mil.TextPositions(25, 160)
	tc, ts := mil.TextRope(mil.Text{HeadDim: 128, Theta: 5e6}, pos)
	check("text", ironTextAngles(pos, 128, 5e6), tc, ts, len(pos))
}

// IronSizes reads the first size from the single blocks' manifest and the others from sizes/WxH
// with all of their graphs: dit_in, image_modes and dit_out the first's where the image's padded
// rows are the same.
func TestIronSizes(t *testing.T) {
	dir := t.TempDir()
	put := func(rel string, meta any) {
		t.Helper()
		p := filepath.Join(dir, rel, "manifest.json")
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		b, _ := json.Marshal(map[string]any{"version": iron.ManifestVersion, "kind": "graph", "meta": meta,
			"graph": map[string]any{"elf": "graph.elf", "kernels": []map[string]any{{}}}})
		if err := os.WriteFile(p, b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	put("singles", map[string]any{"config": ironTestConfig(1024, 1024)})
	for _, p := range []string{"dit_in", "image_modes", "singles", "dit_out", "vae"} {
		put(filepath.Join("sizes", "512x512", p), map[string]any{})
	}
	put(filepath.Join("sizes", "768x768", "vae"), map[string]any{}) // not whole
	for _, p := range []string{"singles", "vae"} {
		put(filepath.Join("sizes", "832x1248", p), map[string]any{}) // the first's 4096 rows: its dit_in, image_modes, dit_out
		put(filepath.Join("sizes", "640x640", p), map[string]any{})  // 2048 rows: needs its own
	}
	got := IronSizes(dir)
	if want := [][2]int{{1024, 1024}, {512, 512}, {832, 1248}}; !slices.Equal(got, want) {
		t.Errorf("IronSizes = %v, want %v", got, want)
	}
	first := ironTestConfig(1024, 1024)
	if p := ironPart(dir, first, 832, 1248, "image_modes"); p != "image_modes" {
		t.Errorf("832x1248's image_modes: %q", p)
	}
	if p := ironPart(dir, first, 832, 1248, "singles"); p != filepath.Join("sizes", "832x1248", "singles") {
		t.Errorf("832x1248's singles: %q", p)
	}
	if c := ironTestConfig(880, 1168); c.Tokens() != 4015 || c.ImageRows() != 4096 {
		t.Errorf("880x1168: %d patches, %d rows", c.Tokens(), c.ImageRows())
	}
	if c := ironTestConfig(1024, 1024); c.MaxText() != 480 || c.Tokens() != 4096 {
		t.Errorf("MaxText %d, Tokens %d", c.MaxText(), c.Tokens())
	}
}
