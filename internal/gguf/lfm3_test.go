package gguf

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/safetensors"
)

// Reads the LFM3-1.7B checkpoints when HWNPU_LFM3_DIR points at them and checks the Q6_K and
// Q4_0 decoders against the bf16 weights: the QAD checkpoint is a fine tune of the same model,
// so the rows should correlate strongly even though they are not identical.
func TestLFM3Checkpoints(t *testing.T) {
	dir := os.Getenv("HWNPU_LFM3_DIR")
	if dir == "" {
		t.Skip("HWNPU_LFM3_DIR not set")
	}
	g, err := Open(filepath.Join(dir, "LFM3-1.7B-QAD-Q4_0.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	st, err := safetensors.Open(filepath.Join(dir, "LFM3-1.7B.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	t.Logf("gguf v%d, %d tensors, arch %s; safetensors %d tensors", g.Version, len(g.Names()), g.String("general.architecture"), len(st.Names()))
	for _, n := range st.Names()[:6] {
		x, _ := st.Lookup(n)
		t.Logf("  st %s %s %v", n, x.DType, x.Shape)
	}
	pairs := [][2]string{
		{"token_embd.weight", "model.embed_tokens.weight"},
		{"blk.0.shortconv.in_proj.weight", "model.layers.0.conv.in_proj.weight"},
		{"blk.2.attn_q.weight", "model.layers.2.self_attn.q_proj.weight"},
		{"blk.0.shortconv.conv.weight", "model.layers.0.conv.conv.weight"},
		{"blk.0.attn_norm.weight", "model.layers.0.operator_norm.weight"},
	}
	for _, p := range pairs {
		gt, ok := g.Lookup(p[0])
		if !ok {
			t.Fatalf("gguf: no %s", p[0])
		}
		a, err := g.Float32(p[0])
		if err != nil {
			t.Fatal(err)
		}
		b, err := st.Float32(p[1])
		if err != nil {
			t.Fatal(err)
		}
		if len(a) != len(b) {
			t.Fatalf("%s: %d values, %s: %d values", p[0], len(a), p[1], len(b))
		}
		n := min(len(a), 1<<22)
		var sab, saa, sbb float64
		for i := 0; i < n; i++ {
			sab += float64(a[i]) * float64(b[i])
			saa += float64(a[i]) * float64(a[i])
			sbb += float64(b[i]) * float64(b[i])
		}
		corr := sab / math.Sqrt(saa*sbb)
		t.Logf("%s (%s %v) vs %s: corr %.4f over %d values, rms %.4g vs %.4g", p[0], gt.Type, gt.Shape, p[1], corr, n, math.Sqrt(saa/float64(n)), math.Sqrt(sbb/float64(n)))
		if corr < 0.9 {
			t.Errorf("%s: correlation %.3f with the bf16 weight; the decoder is probably wrong", p[0], corr)
		}
	}
}
