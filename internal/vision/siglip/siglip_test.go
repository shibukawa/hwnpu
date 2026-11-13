package siglip

import (
	"image"
	"image/color"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// TestPatchRows checks the patch rows' order (channel, row, column), the scaling to [-1, 1]
// and the zero rows after the patches.
func TestPatchRows(t *testing.T) {
	m := &Model{Export: Export{Mean: []float32{0.5, 0.5, 0.5}, Std: []float32{0.5, 0.5, 0.5}, PatchIn: 768, Tokens: 8}}
	m.Config.Image, m.Config.Patch = 32, 16
	img := image.NewRGBA(image.Rect(0, 0, 32, 32))
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			img.Set(x, y, color.RGBA{0, 0, 0, 255})
		}
	}
	img.Set(17, 3, color.RGBA{255, 0, 51, 255}) // patch 1, row 3, column 1
	p := m.PatchRows(img)
	row := p[1*768:]
	if row[0*256+3*16+1] != 1 || math.Abs(float64(row[2*256+3*16+1])-(-0.6)) > 1e-6 || row[1*256+3*16+1] != -1 {
		t.Errorf("patch 1 pixel (3, 1): %v %v %v", row[3*16+1], row[256+3*16+1], row[2*256+3*16+1])
	}
	for _, v := range p[4*768:] {
		if v != 0 {
			t.Fatal("rows past the patches are not zeros")
		}
	}
}

// TestTokenIDs checks the texts' ids against Hugging Face's tokenizer (lowercased, <eos>,
// padded to 64), with the tokenizer of an export in ~/models when there is one.
func TestTokenIDs(t *testing.T) {
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", "siglip2-base-patch16-224.amd-npu2")
	if d := os.Getenv("HWNPU_SIGLIP_EXPORT"); d != "" {
		dir = d
	}
	m, err := Open(t.Context(), nil, dir)
	if err != nil {
		t.Skip("no SigLIP 2 export:", err)
	}
	for text, want := range map[string][]int32{
		// GemmaTokenizer of google/siglip2-base-patch16-224 on the lowercased sentences
		"This is a photo of 2 cats.": {883, 603, 476, 2686, 576, 235248, 235284, 19493, 235265, 1},
		"this is a photo of 猫.":      {883, 603, 476, 2686, 576, 39318, 235265, 1},
	} {
		ids, err := m.TokenIDs(text)
		if err != nil {
			t.Fatal(err)
		}
		if len(ids) != 64 || !slices.Equal(ids[:len(want)], want) || slices.ContainsFunc(ids[len(want):], func(id int32) bool { return id != 0 }) {
			t.Errorf("%q: %v, want %v and pads", text, ids, want)
		}
	}
	long := ""
	for range 100 {
		long += "cat "
	}
	ids, _ := m.TokenIDs(long)
	if len(ids) != 64 || ids[63] != 1 {
		t.Errorf("a long text: %d ids, the last %d; want 64 ending in <eos>", len(ids), ids[63])
	}
}

// TestDefaultLabels checks the default list: no comments, no blanks, no repeats.
func TestDefaultLabels(t *testing.T) {
	ls := DefaultLabels()
	if len(ls) < 300 {
		t.Fatalf("%d default labels", len(ls))
	}
	seen := map[string]bool{}
	for _, l := range ls {
		if l == "" || l[0] == '#' || seen[l] {
			t.Errorf("label %q", l)
		}
		seen[l] = true
	}
}

// TestRank checks the probability (sigmoid of the scaled cosine plus the bias) and the order.
func TestRank(t *testing.T) {
	m := &Model{Export: Export{LogitScale: 10, LogitBias: -5}}
	img := []float32{1, 0}
	got := m.Rank(img, []string{"a", "b"}, [][]float32{{0, 1}, {1, 0}})
	if got[0].Label != "b" || math.Abs(got[0].Score-1/(1+math.Exp(-5))) > 1e-12 || math.Abs(got[1].Score-1/(1+math.Exp(5))) > 1e-12 {
		t.Errorf("Rank = %v", got)
	}
}
