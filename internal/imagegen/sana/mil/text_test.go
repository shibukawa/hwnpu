package mil

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/milcpu"
)

// mapWeights are tensors by name, for a small Gemma 2 of random weights.
type mapWeights map[string]struct {
	v     []float32
	shape []int
}

func (w mapWeights) Get(name string) ([]float32, []int, error) {
	t, ok := w[name]
	if !ok {
		return nil, nil, errors.New("no tensor " + name)
	}
	return append([]float32(nil), t.v...), t.shape, nil
}

func (w mapWeights) Rows(name string, begin, end int) ([]float32, error) {
	t, ok := w[name]
	if !ok {
		return nil, errors.New("no tensor " + name)
	}
	cols := len(t.v) / t.shape[0]
	return append([]float32(nil), t.v[begin*cols:end*cols]...), nil
}

func (w mapWeights) Source() string { return "test" }
func (w mapWeights) Close() error   { return nil }

// smallGemma is a Gemma 2 of layers layers with random weights.
func smallGemma(layers int) (Gemma, mapWeights) {
	c := Gemma{Hidden: 16, Layers: layers, Heads: 4, KVHeads: 2, HeadDim: 4, MLP: 24, Theta: 10000, Eps: 1e-6, QueryScalar: 4}
	rng := rand.New(rand.NewSource(7))
	w := mapWeights{}
	add := func(name string, scale float64, shape ...int) {
		n := 1
		for _, d := range shape {
			n *= d
		}
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64() * scale)
		}
		w[name] = struct {
			v     []float32
			shape []int
		}{v, shape}
	}
	for i := 0; i < layers; i++ {
		p := fmt.Sprintf("gemma.layers.%d.", i)
		for _, n := range []string{"input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"} {
			add(p+n+".weight", 0.1, c.Hidden)
		}
		add(p+"self_attn.q_proj.weight", 0.4, c.Heads*c.HeadDim, c.Hidden)
		add(p+"self_attn.k_proj.weight", 0.4, c.KVHeads*c.HeadDim, c.Hidden)
		add(p+"self_attn.v_proj.weight", 0.4, c.KVHeads*c.HeadDim, c.Hidden)
		add(p+"self_attn.o_proj.weight", 0.4, c.Hidden, c.Heads*c.HeadDim)
		add(p+"mlp.gate_proj.weight", 0.4, c.MLP, c.Hidden)
		add(p+"mlp.up_proj.weight", 0.4, c.MLP, c.Hidden)
		add(p+"mlp.down_proj.weight", 0.4, c.Hidden, c.MLP)
	}
	add("gemma.norm.weight", 0.1, c.Hidden)
	return c, w
}

// runPart writes p as a Core ML package and runs it on x with the CPU interpreter.
func runPart(t *testing.T, p *Part, w llm.Weights, x []float32, shape ...int) []float32 {
	t.Helper()
	path := filepath.Join(t.TempDir(), p.Name+".mlpackage")
	if _, err := llm.WritePart(path, p, w, llm.Options{Backend: llm.BackendCoreML, Format: "fp16"}, nil); err != nil {
		t.Fatal(err)
	}
	prog, closer, err := milcpu.OpenPackage(path)
	if err != nil {
		t.Fatal(err)
	}
	defer closer.Close()
	out, err := prog.Run(map[string]milcpu.Tensor{"x": {Shape: shape, Data: x}}, milcpu.Options{})
	if err != nil {
		t.Fatal(err)
	}
	return out["y"].Data
}

// The rows of a prompt after the held ones, computed from the held rows' keys and values
// (TextKV, TextFrom), are the rows Gemma 2 gives over all of them: in two groups of layers, the
// second ending with the final norm, for several lengths after the held rows.
func TestTextFrom(t *testing.T) {
	c, w := smallGemma(4)
	const held = 6
	groups := [][2]int{{0, 3}, {3, 1}}
	rng := rand.New(rand.NewSource(3))
	for _, rows := range []int{1, 2, 5} {
		L := held + rows
		x := make([]float32, L*c.Hidden)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		// all the rows at once, and the held rows alone
		whole, heldX, fromX := x, x[:held*c.Hidden], x[held*c.Hidden:]
		for _, gr := range groups {
			y := runPart(t, TextKV(c, gr[0], gr[1], L), w, whole, 1, L, c.Hidden)
			if len(y) != TextKVLen(c, gr[1], L) {
				t.Fatalf("TextKV gives %d values, want %d", len(y), TextKVLen(c, gr[1], L))
			}
			whole, _ = SplitTextKV(c, gr[1], L, y)
			var kv [][2][]float32
			heldX, kv = SplitTextKV(c, gr[1], held, runPart(t, TextKV(c, gr[0], gr[1], held), w, heldX, 1, held, c.Hidden))
			if len(kv) != gr[1] || len(kv[0][0]) != c.KVHeads*held*c.HeadDim {
				t.Fatalf("%d layers' keys of %d values", len(kv), len(kv[0][0]))
			}
			fromX = runPart(t, TextFrom(c, gr[0], gr[1], held, rows, kv), w, fromX, 1, rows, c.Hidden)
		}
		want := whole[held*c.Hidden:]
		if len(fromX) != len(want) {
			t.Fatalf("%d rows after %d: %d values, want %d", rows, held, len(fromX), len(want))
		}
		var worst float64
		for i := range want {
			worst = max(worst, math.Abs(float64(fromX[i]-want[i]))/(1+math.Abs(float64(want[i]))))
		}
		// the keys and values are fp16 constants: 0.002 to 0.004 here, where a position off by one
		// or a key seen too early gives 0.3 and more
		if worst > 1e-2 {
			t.Errorf("%d rows after %d held: off by %.2g", rows, held, worst)
		}
		// the held rows are those of the whole too (causal)
		for i, v := range heldX {
			if d := math.Abs(float64(v-whole[i])) / (1 + math.Abs(float64(whole[i]))); d > 1e-4 {
				t.Fatalf("held row value %d: %g, in the whole %g", i, v, whole[i])
			}
		}
	}
}
