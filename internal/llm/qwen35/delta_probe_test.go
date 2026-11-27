package qwen35

import (
	"math/rand/v2"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// TestDeltaChunkRealWeights measures the chunked delta rule's error in fp16 with a real
// checkpoint (HWNPU_TEST_QWEN35_DIR, its config.json and model.safetensors.index.json): the
// hidden state after the first HWNPU_TEST_QWEN35_LAYERS layers (default 3, the first Gated
// DeltaNet layers) of 128 tokens, every operation rounded to fp16, against the float64
// reference.
func TestDeltaChunkRealWeights(t *testing.T) {
	dir := os.Getenv("HWNPU_TEST_QWEN35_DIR")
	if dir == "" {
		t.Skip("HWNPU_TEST_QWEN35_DIR is not set")
	}
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := ParseConfig(raw)
	if err != nil {
		t.Fatal(err)
	}
	w, err := llm.OpenWeights(filepath.Join(dir, "model.safetensors.index.json"), Arch)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	layers := 3
	if n, err := strconv.Atoi(os.Getenv("HWNPU_TEST_QWEN35_LAYERS")); err == nil && n > 0 {
		layers = n
	}
	const n = 128
	seed := 0 // HWNPU_TEST_QWEN35_SEED: another sequence of tokens; a multiple of 100 repeats one token (the delta rule's worst)
	if v, err := strconv.Atoi(os.Getenv("HWNPU_TEST_QWEN35_SEED")); err == nil {
		seed = v
	}
	ids := make([]int32, n)
	for i := range ids {
		ids[i] = int32((i*7919 + 1000 + seed*104729) % 50000)
		if seed != 0 && seed%100 == 0 {
			ids[i] = int32(1000 + seed)
		}
	}
	x := embed16(t, cfg, w, ids)
	p, ins, outs, _ := buildProgram(t, cfg, w, llm.Options{Seq: n, Format: "fp16", Layers: layers, NoHead: true})
	for _, round := range []bool{false, true} {
		res, err := p.Run(map[string]milcpu.Tensor{"x": {Shape: ins[0].Shape, Data: x}}, milcpu.Options{RoundF16: round})
		if err != nil {
			t.Fatal(err)
		}
		got := res[outs[0].Name].Data
		bad := 0
		for _, v := range got {
			if v != v || v > 3e38 || v < -3e38 {
				bad++
			}
		}
		want := reference(t, cfg, w, x, layers-1)
		t.Logf("chunk %d, %d layers, seed %d, fp16 rounding %v: relative error %.5f, %d not finite", deltaChunk, layers, seed, round, relErr64(got, want, cfg.HiddenSize), bad)
	}
}

// TestWriteMLPProbe writes an MLP's beginning alone over (1, 8, HIDDEN) values with random lut8
// weights, for checking coreai-build's programs of it: HWNPU_TEST_QWEN35_MLP_PROBE=FILE.mlpackage
// and _SPEC=INTERxHIDDEN:STAGE, STAGE one of gate (the gate projection), silu (and its silu),
// mul (times the up projection), down (the whole MLP), the forms of the silu (silu4d, sigmoid,
// silu-op, tanh), and share and share-flat: two activations that share kernels. share at
// 1024x256 is the smallest program found whose h13g program has tasks that read past the kernel
// they load (tools/coreai/llirtasks -check: 2 of 16 tasks; docs/known-issues.md 16), share-flat
// the same without them.
func TestWriteMLPProbe(t *testing.T) {
	path := os.Getenv("HWNPU_TEST_QWEN35_MLP_PROBE")
	if path == "" {
		t.Skip("HWNPU_TEST_QWEN35_MLP_PROBE is not set")
	}
	dims, stage, _ := strings.Cut(os.Getenv("HWNPU_TEST_QWEN35_MLP_PROBE_SPEC"), ":")
	is, hs, _ := strings.Cut(dims, "x")
	inter, _ := strconv.Atoi(is)
	hidden, _ := strconv.Atoi(hs)
	if inter < 16 || hidden < 16 {
		t.Fatal("HWNPU_TEST_QWEN35_MLP_PROBE_SPEC=INTERxHIDDEN:STAGE")
	}
	const rows = 8
	pkg, err := mlpackage.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
	m := &builder{Graph: &blocks.Graph{Name: "probe", B: b, Opts: llm.Options{Format: "lut8", Group: 16}}}
	ins := []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: []int{1, rows, hidden}}}
	x := b.Input("x", mlpackage.Float16, 1, rows, hidden)
	rng := rand.New(rand.NewPCG(3, 5))
	weight := func(name string, r, c int) *mlpackage.Var {
		w := make([]float32, r*c)
		for j := range w {
			w[j] = float32(rng.NormFloat64() * 0.02)
		}
		return m.Matrix(name, w, r, c)
	}
	var y *mlpackage.Var
	if err := blocks.Run(func() {
		y = m.Op("linear", map[string]any{"x": x, "weight": weight("gate", inter, hidden)}, 1, rows, inter)
		if stage == "gate" {
			return
		}
		switch stage {
		case "silu4d": // the silu over the values as 16 groups, as the gated norm's over the heads
			y = m.reshape(m.silu(m.reshape(y, 1, rows, 16, inter/16)), 1, rows, inter)
			return
		case "share", "share-flat":
			// a sigmoid over the values as 16 heads (share: a coefficient set a row; share-flat:
			// over the values as they are, one set), then the silu of a second projection (one
			// set), which shares the sigmoid's kernels: with "share", coreai-build's h13g program
			// reads past the shared kernel (llirtasks -check)
			g := y
			if stage == "share" {
				g = m.reshape(m.Sigmoid(m.reshape(y, 1, rows, 16, inter/16)), 1, rows, inter)
			} else {
				g = m.Sigmoid(y)
			}
			up := m.silu(m.Op("linear", map[string]any{"x": x, "weight": weight("up", inter, hidden)}, 1, rows, inter))
			y = m.Op("mul", map[string]any{"x": g, "y": up}, 1, rows, inter)
			return
		case "sigmoid": // x times the Neural Engine's own sigmoid
			y = m.Op("mul", map[string]any{"x": y, "y": m.Op("sigmoid", map[string]any{"x": y}, 1, rows, inter)}, 1, rows, inter)
			return
		case "silu-op": // the silu operation itself
			y = m.Op("silu", map[string]any{"x": y}, 1, rows, inter)
			return
		case "tanh": // x (0.5 tanh(0.5 x) + 0.5)
			h := m.Op("tanh", map[string]any{"x": m.Op("mul", map[string]any{"x": y, "y": m.B.ScalarF16(0.5)}, 1, rows, inter)}, 1, rows, inter)
			h = m.Op("add", map[string]any{"x": m.Op("mul", map[string]any{"x": h, "y": m.B.ScalarF16(0.5)}, 1, rows, inter), "y": m.B.ScalarF16(0.5)}, 1, rows, inter)
			y = m.Op("mul", map[string]any{"x": y, "y": h}, 1, rows, inter)
			return
		}
		y = m.silu(y)
		if stage == "silu" {
			return
		}
		up := m.Op("linear", map[string]any{"x": x, "weight": weight("up", inter, hidden)}, 1, rows, inter)
		y = m.Op("mul", map[string]any{"x": y, "y": up}, 1, rows, inter)
		if stage == "mul" {
			return
		}
		y = m.Op("linear", map[string]any{"x": y, "weight": weight("down", hidden, inter)}, 1, rows, hidden)
		y = m.Op("add", map[string]any{"x": y, "y": x}, 1, rows, hidden)
	}); err != nil {
		t.Fatal(err)
	}
	yo := b.NamedOp("identity", "y", map[string]any{"x": y}, mlpackage.Float16, y.Shape...)
	outs := []mlpackage.Feature{{Name: "y", DType: mlpackage.Float16, Shape: yo.Shape}}
	if err := pkg.Finish(b.Finish(&mlpackage.Var{Name: "y"}), 9, ins, outs); err != nil {
		t.Fatal(err)
	}
}
