package qwenimage

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/imagegen/qwenimage/mil"
	"github.com/shibukawa/hwnpu/internal/llm"
)

// fakeDit is random weights for a small DiT, by name.
type fakeDit struct{ c mil.Dit }

func (f fakeDit) shape(name string) []int {
	E, D, M := f.c.Hidden, f.c.HeadDim, f.c.MLP
	switch {
	case strings.HasSuffix(name, "norm_q.weight"), strings.HasSuffix(name, "norm_k.weight"):
		return []int{D}
	case strings.HasSuffix(name, "gate_layer.weight"), strings.HasSuffix(name, "img_mlp.proj.weight"):
		return []int{M, E}
	case strings.HasSuffix(name, "img_mlp.out.weight"):
		return []int{E, M}
	case name == "dit.img_in.weight":
		return []int{E, f.c.Latent}
	case name == "dit.proj_out.weight":
		return []int{f.c.Latent, E}
	case strings.Contains(name, "attn."):
		return []int{E, E}
	}
	return nil
}

func (f fakeDit) Get(name string) ([]float32, []int, error) {
	sh := f.shape(name)
	if sh == nil {
		return nil, nil, fmt.Errorf("fakeDit: no tensor %s", name)
	}
	n := 1
	for _, d := range sh {
		n *= d
	}
	h := int64(0)
	for _, c := range name {
		h = h*31 + int64(c)
	}
	rng := rand.New(rand.NewSource(h))
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(rng.NormFloat64() * 0.05)
	}
	return v, sh, nil
}

func (f fakeDit) Rows(name string, begin, end int) ([]float32, error) {
	v, sh, err := f.Get(name)
	if err != nil {
		return nil, err
	}
	return v[begin*sh[1] : end*sh[1]], nil
}
func (fakeDit) Source() string { return "fake" }
func (fakeDit) Close() error   { return nil }

// A group's text pass and image blocks in one format are one program whose weights are those of
// one of them (SharedDit): the same layers give the same bytes, lut6 with k-means too.
func TestSharedDitProgram(t *testing.T) {
	c := mil.Dit{Width: 128, Height: 128, Hidden: 256, Heads: 2, HeadDim: 128, Layers: 4, MLP: 1024, Latent: 64,
		Axes: [3]int{16, 56, 56}, Theta: 10000, Eps: 1e-6}
	w := fakeDit{c}
	dir := t.TempDir()
	if d := os.Getenv("HWNPU_QWENIMAGE_SHARED_OUT"); d != "" { // to run on the device (aicheck run DIR/dit_1.aimodel#text)
		dir = d
	}
	opts := llm.Options{Backend: llm.BackendCoreAI, Format: "lut6", Quant: DefaultDitQuant}
	size := func(asset string) int64 {
		st, err := os.Stat(filepath.Join(asset, "main.mlirb"))
		if err != nil {
			t.Fatal(err)
		}
		return st.Size()
	}
	text, blocks := filepath.Join(dir, "text.aimodel"), filepath.Join(dir, "blocks.aimodel")
	if _, err := llm.WritePart(text, mil.TextBlocks(c, 1, 2), w, opts, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := llm.WritePart(blocks, mil.Blocks(c, 1, 2, 0, false), w, opts, nil); err != nil {
		t.Fatal(err)
	}
	both := filepath.Join(dir, "dit_1.aimodel")
	if _, err := llm.WriteAIFunctions(both, w, []llm.AIFunction{{Name: "text", Model: mil.TextBlocks(c, 1, 2), Options: opts},
		{Name: "blocks", Model: mil.Blocks(c, 1, 2, 0, false), Options: opts}}); err != nil {
		t.Fatal(err)
	}
	if os.Getenv("HWNPU_QWENIMAGE_SHARED_OUT") != "" { // the references of aicheck run -cpu -ref
		ref := opts
		ref.Backend = llm.BackendCoreML
		for name, p := range map[string]llm.Model{"text": mil.TextBlocks(c, 1, 2), "blocks": mil.Blocks(c, 1, 2, 0, false)} {
			if _, err := llm.WritePart(filepath.Join(dir, name+".mlpackage"), p, w, ref, nil); err != nil {
				t.Fatal(err)
			}
		}
	}
	// 2 layers of 4 E x E and 3 E x MLP at 6 bits
	weights := int64(2 * (4*c.Hidden*c.Hidden + 3*c.Hidden*c.MLP) * 6 / 8)
	apart := size(text) + size(blocks)
	t.Logf("text pass %d, blocks %d, both %d (the weights about %d)", size(text), size(blocks), size(both), weights)
	if saved := apart - size(both); saved < weights*9/10 {
		t.Errorf("one program saves %d bytes of %d apart, the weights are about %d", saved, apart, weights)
	}
}
