package qwen3

import (
	"context"
	"fmt"
	"math"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// tinyQwen2 is a small Qwen2 with what Qwen2.5-Coder-7B has: biases on q, k and v, no norms
// over the heads, no head_dim in the config (hidden_size / heads: 16), grouped query attention
// and an output layer apart from the embeddings (the 0.5B to 3B tie them).
const tinyQwen2 = `{"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "attention_dropout": 0.0,
	"bos_token_id": 151643, "eos_token_id": 151645, "hidden_act": "silu", "hidden_size": 64,
	"intermediate_size": 96, "max_position_embeddings": 32768, "max_window_layers": 3,
	"num_attention_heads": 4, "num_hidden_layers": 3, "num_key_value_heads": 2, "rms_norm_eps": 1e-06,
	"rope_scaling": null, "rope_theta": 1000000.0, "sliding_window": 131072, "tie_word_embeddings": false,
	"torch_dtype": "bfloat16", "use_cache": true, "use_sliding_window": false, "vocab_size": 48}`

func tinyQwen2Config(t *testing.T) *Config {
	t.Helper()
	c, err := ParseQwen2Config([]byte(tinyQwen2))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// archOf is the architecture a test's config is of.
func archOf(c *Config) *llm.Arch {
	if c.qwen2 {
		return Qwen2Arch
	}
	return Arch
}

// Qwen2 in fp16 on the CPU gives the logits of the reference forward pass (modeling_qwen2.py),
// tied and not; each layer on its own gives the reference's next hidden state.
func TestQwen2MatchesReference(t *testing.T) {
	for _, tied := range []bool{false, true} {
		raw := tinyQwen2
		if tied {
			raw = strings.Replace(raw, `"tie_word_embeddings": false`, `"tie_word_embeddings": true`, 1)
		}
		c, err := ParseQwen2Config([]byte(raw))
		if err != nil {
			t.Fatal(err)
		}
		if c.HeadDim != 16 {
			t.Fatalf("head_dim %d, want hidden_size / heads", c.HeadDim)
		}
		w := llm.NewSynthetic(c)
		ref := reference(t, c, w, tokens)
		out := build(t, c, w, llm.Options{Seq: len(tokens), Format: "fp16"}).run(t, map[string][]float32{"x": rows(t, w, c.Embedding(), tokens)})
		if e := relErr(out["y"], ref.logits); !(e <= 1e-4) {
			t.Errorf("tied %v: logits: relative error %g", tied, e)
		} else {
			t.Logf("tied %v: logits: relative error %.2g", tied, e)
		}
		for i := 0; i <= c.NumLayers; i++ {
			p := build(t, c, w, llm.Options{Seq: len(tokens), Format: "fp16", Start: i, Layers: 1, NoHead: i < c.NumLayers})
			feed := map[string][]float32{"x": f32(ref.hidden[i])}
			if i == 0 {
				feed["x"] = rows(t, w, c.Embedding(), tokens)
			}
			want := ref.logits
			if i < c.NumLayers {
				want = ref.hidden[i+1]
			}
			if e := relErr(p.run(t, feed)["y"], want); !(e <= 1e-4) {
				t.Errorf("tied %v: layer %d: relative error %g", tied, i, e)
			}
		}
	}
	tensors := tinyQwen2Config(t).Tensors()
	if _, ok := tensors["model.layers.0.self_attn.q_norm.weight"]; ok {
		t.Error("Qwen2 reads q_norm")
	}
	if _, ok := tensors["model.layers.0.self_attn.v_proj.bias"]; !ok {
		t.Error("Qwen2 does not read v_proj's bias")
	}
}

// Qwen2 split at its attention (every layer's on the host) gives the logits of the whole program.
func TestQwen2PagedMatchesFull(t *testing.T) {
	c := tinyQwen2Config(t)
	w := llm.NewSynthetic(c)
	n := len(tokens)
	full := build(t, c, w, llm.Options{Seq: n, Format: "fp16"}).run(t, map[string][]float32{"x": rows(t, w, c.Embedding(), tokens)})["y"]
	for _, k := range []struct{ q, cache, chunk int }{{1, 12, 4}, {4, 12, 5}} {
		t.Run(fmt.Sprintf("q%d cache%d chunk%d", k.q, k.cache, k.chunk), func(t *testing.T) {
			s, cl, err := runner.OpenSession(context.Background(), nil, convertPaged(t, c, w, k.cache, k.chunk, k.q), c)
			if err != nil {
				t.Fatal(err)
			}
			defer cl.Close()
			worst := 0.0
			for pos := 0; pos < n; {
				v := min(k.q, n-pos)
				logits := step(t, s, c, w, pos, pos+v)
				pos += v
				worst = math.Max(worst, relErr32(logits, full[(pos-1)*c.VocabSize:pos*c.VocabSize]))
			}
			if !(worst <= 5e-3) {
				t.Errorf("relative error %g against the program without a cache", worst)
			}
		})
	}
}

// Qwen2's configs: the settings it does not build are refused, its default format is lut8, and
// the whitelist picks it by model_type.
func TestQwen2Config(t *testing.T) {
	for _, c := range []struct{ old, new, want string }{
		{`"model_type": "qwen2"`, `"model_type": "qwen3"`, "model_type"},
		{`"use_sliding_window": false`, `"use_sliding_window": true`, "use_sliding_window"},
		{`"rope_scaling": null`, `"rope_scaling": {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}`, "rope_scaling"},
		{`"hidden_act": "silu"`, `"hidden_act": "gelu"`, "hidden_act"},
		{`"num_key_value_heads": 2`, `"num_key_value_heads": 3`, "heads"},
		{`"hidden_size": 64,`, `"hidden_size": 64, "head_dim": 32,`, "head_dim"},
	} {
		raw := strings.Replace(tinyQwen2, c.old, c.new, 1)
		if raw == tinyQwen2 {
			t.Fatalf("%q is not in the tiny config", c.old)
		}
		if _, err := ParseQwen2Config([]byte(raw)); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: error %v, want one about %s", c.new, err, c.want)
		}
	}
	if got := Qwen2Arch.DefaultFormat(tinyQwen2Config(t)); got != "lut8" {
		t.Errorf("Qwen2's default format %s, want lut8", got)
	}
	a, err := llm.Detect([]byte(tinyQwen2))
	if err != nil || a != Qwen2Arch {
		t.Errorf("detected %v, %v", a, err)
	}
}

// An intermediate over mlpSplit is computed in pieces: in fp16 the logits of the reference, in
// lut4 those of the MLP in one piece (its pieces cut from the matrices palettized whole).
func TestQwen2SplitMLP(t *testing.T) {
	c := tinyQwen2Config(t)
	w := llm.NewSynthetic(c)
	feed := map[string][]float32{"x": rows(t, w, c.Embedding(), tokens)}
	whole := build(t, c, w, llm.Options{Seq: len(tokens), Format: "lut4"}).run(t, feed)["y"]
	defer func(s int) { mlpSplit = s }(mlpSplit)
	mlpSplit = 48 // intermediate 96: pieces of 64 and 32
	ref := reference(t, c, w, tokens)
	if e := relErr(build(t, c, w, llm.Options{Seq: len(tokens), Format: "fp16"}).run(t, feed)["y"], ref.logits); !(e <= 1e-4) {
		t.Errorf("fp16 in pieces: relative error %g", e)
	}
	p := build(t, c, w, llm.Options{Seq: len(tokens), Format: "lut4"})
	if e := relErr32(p.run(t, feed)["y"], whole); !(e <= 1e-5) {
		t.Errorf("lut4 in pieces: relative error %g against one piece", e)
	}
}

// A Core AI program refuses weights of tables or scales per block of columns, which the Neural
// Engine computes with the first block's alone; whole rows build.
func TestCoreAIRowsOnly(t *testing.T) {
	c := tinyQwen2Config(t)
	w := llm.NewSynthetic(c)
	newB := func() *mlpackage.Builder {
		blobs, err := mlpackage.NewBlobWriter(filepath.Join(t.TempDir(), "w.bin"))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { blobs.Close() })
		return mlpackage.NewBuilder("CoreML8", blobs)
	}
	for _, o := range []llm.Options{{Format: "lut4", ColBlock: 32}, {Format: "int4-g32"}, {Format: "int8-g16"}} {
		o.Seq, o.Backend = 4, llm.BackendCoreAI
		if _, _, err := Build(newB(), c, w, o); err == nil || !strings.Contains(err.Error(), "whole rows only") {
			t.Errorf("%s col block %d: error %v", o.Format, o.ColBlock, err)
		}
		o.Backend = ""
		if _, _, err := Build(newB(), c, w, o); err != nil {
			t.Errorf("%s col block %d for Core ML: %v", o.Format, o.ColBlock, err)
		}
	}
	if _, _, err := Build(newB(), c, w, llm.Options{Seq: 4, Format: "lut4", Backend: llm.BackendCoreAI}); err != nil {
		t.Errorf("whole rows: %v", err)
	}
}
