package qwen35

import (
	"fmt"
	"math"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/shibukawa/bonsaiai/internal/llm"
	"github.com/shibukawa/bonsaiai/internal/milcpu"
)

// tinyMoEJSON is a small Qwen3.5-MoE as its checkpoint describes it (Qwen3.6-35B-A3B's config
// shrunk): 8 experts of 32 x 32, 2 a token, a shared expert of 16, both layer types, GQA and
// partial RoPE.
const tinyMoEJSON = `{"architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe", "tie_word_embeddings": false,
	"text_config": {"model_type": "qwen3_5_moe_text", "hidden_size": 32, "num_hidden_layers": 4,
		"num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 48, "rms_norm_eps": 1e-6,
		"hidden_act": "silu", "attn_output_gate": true, "attention_bias": false, "max_position_embeddings": 4096,
		"layer_types": ["linear_attention", "full_attention", "linear_attention", "full_attention"],
		"linear_conv_kernel_dim": 4, "linear_key_head_dim": 4, "linear_num_key_heads": 8,
		"linear_num_value_heads": 16, "linear_value_head_dim": 4,
		"num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32, "shared_expert_intermediate_size": 16,
		"rope_parameters": {"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.5,
			"mrope_interleaved": true, "mrope_section": [2, 1, 1]}}}`

func tinyMoE(t *testing.T) (*MoE, llm.Synthetic) {
	t.Helper()
	c, err := ParseMoE([]byte(tinyMoEJSON))
	if err != nil {
		t.Fatal(err)
	}
	w := llm.NewSynthetic(c)
	// the experts as the release has them (bf16 there), which the programs do not read: the
	// reference and the test's host (moeHost) run them in float64
	E, H, I := c.NumExperts, c.HiddenSize, c.MoEIntermediate
	for l := range c.NumLayers {
		p := fmt.Sprintf("%slayers.%d.mlp.experts.", c.prefix(), l)
		w[p+"gate_up_proj"] = []int{E, 2 * I, H}
		w[p+"down_proj"] = []int{E, H, I}
	}
	return c, w
}

// moeExperts is layer prefix p's routed experts on a row f in float64 (Qwen3_5MoeExperts: the
// gate the first half of gate_up_proj's rows, silu(gate) * up, down_proj), the experts idx
// weighed by wt.
func moeExperts(t *testing.T, c *MoE, w llm.Weights, p string, f []float64, idx []int, wt []float64) []float64 {
	E, H, I := c.NumExperts, c.HiddenSize, c.MoEIntermediate
	gu, _, err := w.Get(p + "mlp.experts.gate_up_proj")
	if err != nil {
		t.Fatal(err)
	}
	dn, _, err := w.Get(p + "mlp.experts.down_proj")
	if err != nil {
		t.Fatal(err)
	}
	_ = E
	out := make([]float64, H)
	for k, e := range idx {
		act := make([]float64, I)
		for j := range I {
			var g, u float64
			for m := range H {
				g += float64(gu[(e*2*I+j)*H+m]) * f[m]
				u += float64(gu[(e*2*I+I+j)*H+m]) * f[m]
			}
			act[j] = silu(g) * u
		}
		for r := range H {
			var y float64
			for j := range I {
				y += float64(dn[(e*H+r)*I+j]) * act[j]
			}
			out[r] += wt[k] * y
		}
	}
	return out
}

// moeReference is Qwen3_5MoeSparseMoeBlock written out in float64 from modeling_qwen3_5_moe.py
// (transformers 5.22): the router's softmax over all the experts, its top k renormalized, the
// experts, and the shared expert times the sigmoid of its gate. The weights the program stores in
// fp16 are rounded to fp16 (f16); the experts are the host's, not rounded.
func moeReference(t *testing.T, c *MoE, w llm.Weights) func(string, []float64) []float64 {
	return func(p string, f []float64) []float64 {
		H, E, S := c.HiddenSize, c.NumExperts, c.SharedIntermediate
		mv := func(name string, rows int) []float64 {
			m := f16(t, w, name)
			y := make([]float64, rows)
			for r := range y {
				for j := range H {
					y[r] += m[r*H+j] * f[j]
				}
			}
			return y
		}
		logits := mv(p+"mlp.gate.weight", E)
		top := math.Inf(-1)
		for _, v := range logits {
			top = math.Max(top, v)
		}
		probs := make([]float64, E)
		var sum float64
		for e, v := range logits {
			probs[e] = math.Exp(v - top)
			sum += probs[e]
		}
		order := make([]int, E)
		for e := range order {
			order[e] = e
			probs[e] /= sum
		}
		sort.SliceStable(order, func(a, b int) bool { return probs[order[a]] > probs[order[b]] })
		idx := order[:c.TopK]
		wt := make([]float64, c.TopK)
		var tsum float64
		for k, e := range idx {
			wt[k] = probs[e]
			tsum += wt[k]
		}
		for k := range wt {
			wt[k] /= tsum
		}
		out := moeExperts(t, c, w, p, f, idx, wt)
		g, u := mv(p+"mlp.shared_expert.gate_proj.weight", S), mv(p+"mlp.shared_expert.up_proj.weight", S)
		for j := range g {
			g[j] = silu(g[j]) * u[j]
		}
		dn := f16(t, w, p+"mlp.shared_expert.down_proj.weight")
		gate := sigmoid(mv(p+"mlp.shared_expert_gate.weight", 1)[0])
		for r := range H {
			var y float64
			for j := range S {
				y += dn[r*S+j] * g[j]
			}
			out[r] += gate * y
		}
		return out
	}
}

// moeRef is the MoE model's reference: the final hidden state, or with stop >= 0 the hidden
// state after layer stop.
func moeRef(t *testing.T, c *MoE, w llm.Weights, x []float32, stop int) [][]float64 {
	d := c.HiddenSize
	h := make([][]float64, len(x)/d)
	for i := range h {
		h[i] = make([]float64, d)
		for j := range h[i] {
			h[i][j] = float64(x[i*d+j])
		}
	}
	var layers []refLayer
	for l := 0; l < c.NumLayers; l++ {
		layers = append(layers, refLayer{fmt.Sprintf("%slayers.%d.", c.prefix(), l), c.IsLinear(l)})
	}
	return referenceLayers(t, c.Config, w, h, layers, c.prefix()+"norm.weight", stop, moeReference(t, c, w))
}

// moeHost is the test's host for layer's experts: the program's moe_x_N and moe_logits_N (rows of
// rows) into moe_o_N, the experts chosen by the host's own routing (experts.Routing.Choose).
func moeHost(t *testing.T, c *MoE, w llm.Weights, layer int, xs, logits []float32, rows int) []float32 {
	H, E, K := c.HiddenSize, c.NumExperts, c.TopK
	out := make([]float32, rows*H)
	p := fmt.Sprintf("%slayers.%d.", c.prefix(), layer)
	for r := range rows {
		idx, wt := make([]int, K), make([]float32, K)
		c.Routing().Choose(logits[r*E:(r+1)*E], nil, idx, wt)
		f := make([]float64, H)
		for j := range f {
			f[j] = float64(xs[r*H+j])
		}
		w64 := make([]float64, K)
		for k := range wt {
			w64[k] = float64(wt[k])
		}
		for j, v := range moeExperts(t, c, w, p, f, idx, w64) {
			out[r*H+j] = float32(v)
		}
	}
	return out
}

// moeChain runs the parts PlanRouted cuts on the CPU interpreter over x (Seq rows), the host's
// experts between them (moeHost), and returns the logits and the hidden state at each layer's
// end (the residual entering the next layer: y plus moe_o).
func moeChain(t *testing.T, c *MoE, w llm.Weights, x []float32, seq int) (logits []float32, hidden [][]float32) {
	opts := llm.Options{Seq: seq, Format: "fp16", FormatFor: []string{"lm_head=fp16"}}
	parts, err := llm.PlanRouted(c, opts, 1<<40)
	if err != nil {
		t.Fatal(err)
	}
	vals := map[string][]float32{}
	cur := x
	for _, part := range parts {
		o := opts
		o.Start, o.Layers = part.Start, part.Layers
		p, ins, outs, _ := buildProgram(t, c, w, o)
		in := map[string]milcpu.Tensor{}
		for _, f := range ins {
			v := cur
			if f.Name != "x" {
				var ok bool
				if v, ok = vals[f.Name]; !ok {
					t.Fatalf("part %+v: no value for its input %s", part, f.Name)
				}
				if part.Layers != llm.NoLayers && strings.HasPrefix(f.Name, "moe_o_") {
					h := make([]float32, len(cur))
					for i := range h {
						h[i] = cur[i] + v[i]
					}
					hidden = append(hidden, h)
				}
			}
			in[f.Name] = milcpu.Tensor{Shape: f.Shape, Data: v}
		}
		res, err := p.Run(in, milcpu.Options{})
		if err != nil {
			t.Fatalf("part %+v: %v", part, err)
		}
		for _, f := range outs[1:] {
			vals[f.Name] = res[f.Name].Data
		}
		cur = res[outs[0].Name].Data
		for _, f := range outs {
			if n, ok := strings.CutPrefix(f.Name, "moe_logits_"); ok {
				var l int
				fmt.Sscan(n, &l)
				vals["moe_o_"+n] = moeHost(t, c, w, l, vals["moe_x_"+n], vals[f.Name], seq)
			}
		}
	}
	return cur, hidden
}

// The parts, chained with the host's experts, compute the reference's logits, and each layer's
// output on the way.
func TestMoEAgainstFP64(t *testing.T) {
	c, w := tinyMoE(t)
	ids := []int32{3, 1, 4, 1, 5, 9, 2, 6}
	x := embed16(t, c.Config, w, ids)
	got, hidden := moeChain(t, c, w, x, len(ids))
	if len(hidden) != c.NumLayers {
		t.Fatalf("%d layers' outputs, want %d", len(hidden), c.NumLayers)
	}
	for l, h := range hidden {
		if e := relErr64(h, moeRef(t, c, w, x, l), c.HiddenSize); e > 1e-4 {
			t.Errorf("layer %d: relative error %g against the reference", l, e)
		}
	}
	want := logits(t, c.Config, w, moeRef(t, c, w, x, -1))
	e := relErr64(got, want, c.VocabSize)
	if e > 1e-4 {
		t.Errorf("relative error %g against the reference", e)
	}
	t.Logf("relative error %.2g", e)
}

// Split at the full attention layers (Paged) and run a token at a time with the states the host
// keeps (the Gated DeltaNet's, the convolutions' windows, the full attention's keys and values,
// which the test attends over), the parts give the whole sequence's logits.
func TestMoEPagedSteps(t *testing.T) {
	c, w := tinyMoE(t)
	ids := []int32{7, 2, 9, 4, 4, 1}
	x := embed16(t, c.Config, w, ids)
	want := logits(t, c.Config, w, moeRef(t, c, w, x, -1))
	const cache = 16
	opts := llm.Options{Seq: 1, Format: "fp16", FormatFor: []string{"lm_head=fp16"}, Cache: cache, Paged: cache}
	parts, err := llm.PlanRouted(c, opts, 1<<40)
	if err != nil {
		t.Fatal(err)
	}
	type built struct {
		part      llm.Part
		p         *milcpu.Program
		ins, outs []string
		shapes    map[string][]int
	}
	var progs []built
	for _, part := range parts {
		o := opts
		o.Start, o.Layers = part.Start, part.Layers
		p, ins, outs, _ := buildProgram(t, c, w, o)
		b := built{part: part, p: p, shapes: map[string][]int{}}
		for _, f := range ins {
			b.ins = append(b.ins, f.Name)
			b.shapes[f.Name] = f.Shape
		}
		for _, f := range outs {
			b.outs = append(b.outs, f.Name)
		}
		progs = append(progs, b)
	}
	nh, nkv, hd := c.NumHeads, c.NumKVHeads, c.HeadDim
	state := map[string][]float32{}
	keys, values := map[int][][]float32{}, map[int][][]float32{} // by layer: a row (nkv*hd) a position
	d := c.HiddenSize
	for pos := range ids {
		step := c.PagedStepInputs(pos, 1, 1, cache)
		vals := map[string][]float32{}
		cur := x[pos*d : (pos+1)*d]
		for _, b := range progs {
			in := map[string]milcpu.Tensor{}
			for _, name := range b.ins {
				var v []float32
				switch {
				case name == "x":
					v = cur
				case strings.HasPrefix(name, "st_") || strings.HasPrefix(name, "win_"):
					if v = state[name]; v == nil {
						v = make([]float32, count(b.shapes[name]))
					}
				case step[name] != nil:
					v = step[name]
				default:
					var ok bool
					if v, ok = vals[name]; !ok {
						t.Fatalf("part %+v: no value for its input %s", b.part, name)
					}
				}
				in[name] = milcpu.Tensor{Shape: b.shapes[name], Data: v}
			}
			res, err := b.p.Run(in, milcpu.Options{})
			if err != nil {
				t.Fatalf("part %+v: %v", b.part, err)
			}
			cur = res["y"].Data
			for _, name := range b.outs[1:] {
				v := res[name].Data
				switch {
				case strings.HasPrefix(name, "st_"):
					state[strings.TrimSuffix(name, "_new")] = v
				case strings.HasPrefix(name, "win_"):
					// the window of the last rows: of K-1+1 rows, the last K-1
					n := strings.TrimSuffix(name, "_new")
					state[n] = v[len(v)-count(b.shapes[n]):]
				case strings.HasPrefix(name, "kv_k_"):
					var l int
					fmt.Sscanf(name, "kv_k_%d_new", &l)
					keys[l] = append(keys[l], v)
				case strings.HasPrefix(name, "kv_v_"):
					var l int
					fmt.Sscanf(name, "kv_v_%d_new", &l)
					values[l] = append(values[l], v)
				default:
					vals[name] = v
				}
			}
			for _, name := range b.outs {
				if n, ok := strings.CutPrefix(name, "moe_logits_"); ok {
					var l int
					fmt.Sscan(n, &l)
					vals["moe_o_"+n] = moeHost(t, c, w, l, vals["moe_x_"+n], vals[name], 1)
				}
				if n, ok := strings.CutPrefix(name, "attn_q_"); ok {
					// the host's attention: causal over every position so far, kv heads
					// repeated (repeat_kv), the queries (1, nh, 1, hd)
					var l int
					fmt.Sscan(n, &l)
					q := vals[name]
					o := make([]float32, nh*hd)
					for hh := range nh {
						g := hh / (nh / nkv)
						sc := make([]float64, len(keys[l]))
						top := math.Inf(-1)
						for j, k := range keys[l] {
							for e := range hd {
								sc[j] += float64(q[hh*hd+e]) * float64(k[g*hd+e])
							}
							sc[j] /= math.Sqrt(float64(hd))
							top = math.Max(top, sc[j])
						}
						var sum float64
						for j := range sc {
							sc[j] = math.Exp(sc[j] - top)
							sum += sc[j]
						}
						for j, v := range values[l] {
							for e := range hd {
								o[hh*hd+e] += float32(sc[j] / sum * float64(v[g*hd+e]))
							}
						}
					}
					vals["attn_o_"+n] = o
				}
			}
		}
		if e := relErr64(cur, want[pos:pos+1], c.VocabSize); e > 1e-3 {
			t.Errorf("position %d: relative error %g against the reference", pos, e)
		}
	}
}

func TestParseMoE(t *testing.T) {
	if _, err := ParseConfig([]byte(tinyMoEJSON)); err == nil {
		t.Error("the dense architecture took a qwen3_5_moe config")
	}
	if _, err := ParseMoE([]byte(tinyJSON)); err == nil {
		t.Error("the MoE architecture took a dense qwen3_5 config")
	}
	for _, tc := range []struct{ from, to, want string }{
		{`"num_experts": 8,`, `"num_experts": 8, "mlp_only_layers": [1],`, "mlp_only_layers"},
		{`"num_experts": 8,`, `"num_experts": 8, "decoder_sparse_step": 2,`, "decoder_sparse_step"},
		{`"num_experts": 8,`, `"num_experts": 8, "norm_topk_prob": false,`, "norm_topk_prob"},
		{`"moe_intermediate_size": 32`, `"moe_intermediate_size": 48`, "MXFP4"},
		{`"num_experts_per_tok": 2`, `"num_experts_per_tok": 9`, "experts a token"},
		{`"shared_expert_intermediate_size": 16`, `"shared_expert_intermediate_size": 0`, "shared expert"},
		{`"rope_type": "default"`, `"rope_type": "yarn"`, "rope type"},
	} {
		raw := strings.Replace(tinyMoEJSON, tc.from, tc.to, 1)
		if raw == tinyMoEJSON {
			t.Fatalf("%q is not in the config", tc.from)
		}
		if _, err := ParseMoE([]byte(raw)); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: %v, want an error about %s", tc.to, err, tc.want)
		}
	}
	a, err := llm.Detect([]byte(tinyMoEJSON))
	if err != nil {
		t.Fatal(err)
	}
	m, err := a.LoadConfig([]byte(tinyMoEJSON))
	if err != nil {
		t.Fatal(err)
	}
	if a != MoEArch {
		t.Errorf("detected %s", a.Name)
	}
	if _, ok := m.(llm.HostRouted); !ok {
		t.Error("not HostRouted")
	}
}

// A step of several tokens of the MoE model outputs the convolution's window after its valid
// tokens (step_window picks its rows), not the window with every row of the step: three tokens
// in a step of four, the last row padding, leave the window and the recurrent state that three
// steps of a token leave (whose window is a slice of the token's rows).
func TestMoEStepWindow(t *testing.T) {
	c, w := tinyMoE(t)
	ids := []int32{7, 2, 9}
	x := embed16(t, c.Config, w, ids)
	const cache = 16
	d, k := c.HiddenSize, c.ConvKernel
	type built struct {
		p      *milcpu.Program
		shapes map[string][]int
		outs   map[string][]int
	}
	build := func(seq int) built {
		opts := llm.Options{Seq: seq, Format: "fp16", FormatFor: []string{"lm_head=fp16"}, Cache: cache, Paged: cache}
		parts, err := llm.PlanRouted(c, opts, 1<<40)
		if err != nil {
			t.Fatal(err)
		}
		opts.Start, opts.Layers = parts[0].Start, parts[0].Layers
		p, ins, outs, _ := buildProgram(t, c, w, opts)
		b := built{p: p, shapes: map[string][]int{}, outs: map[string][]int{}}
		for _, f := range ins {
			b.shapes[f.Name] = f.Shape
		}
		for _, f := range outs {
			b.outs[f.Name] = f.Shape
		}
		return b
	}
	run := func(b built, rows []float32, step map[string][]float32, state map[string][]float32) map[string]milcpu.Tensor {
		t.Helper()
		in := map[string]milcpu.Tensor{}
		for name, sh := range b.shapes {
			v := step[name]
			switch {
			case name == "x":
				v = rows
			case strings.HasPrefix(name, "st_") || strings.HasPrefix(name, "win_"):
				if v = state[name]; v == nil {
					v = make([]float32, count(sh))
				}
			}
			if len(v) != count(sh) {
				t.Fatalf("input %s %v: %d values", name, sh, len(v))
			}
			in[name] = milcpu.Tensor{Shape: sh, Data: v}
		}
		res, err := b.p.Run(in, milcpu.Options{})
		if err != nil {
			t.Fatal(err)
		}
		return res
	}
	one, four := build(1), build(4)
	ch := one.shapes["win_conv_0"][2]
	for _, b := range []built{one, four} {
		if got := b.outs["win_conv_0_new"]; !slices.Equal(got, []int{1, k - 1, ch}) {
			t.Fatalf("win_conv_0_new is %v, want the window (1, %d, %d)", got, k-1, ch)
		}
	}
	if _, ok := one.shapes["step_window"]; ok {
		t.Error("the program of a token a step takes step_window")
	}
	if got := four.shapes["step_window"]; !slices.Equal(got, []int{1, k - 1, 4 + k - 1}) {
		t.Fatalf("step_window is %v", got)
	}
	state := map[string][]float32{}
	for pos := range ids {
		res := run(one, x[pos*d:(pos+1)*d], c.PagedStepInputs(pos, 1, 1, cache), state)
		state["win_conv_0"], state["st_rec_0"] = res["win_conv_0_new"].Data, res["st_rec_0_new"].Data
	}
	rows := append(slices.Clone(x[:3*d]), x[2*d:3*d]...) // the padding repeats the last token
	res := run(four, rows, c.PagedStepInputs(0, 3, 4, cache), nil)
	for _, name := range []string{"win_conv_0", "st_rec_0"} {
		got, want := res[name+"_new"].Data, state[name]
		if len(got) != len(want) {
			t.Fatalf("%s: %d values, want %d", name, len(got), len(want))
		}
		var num, den float64
		for i := range want {
			num += math.Pow(float64(got[i]-want[i]), 2)
			den += math.Pow(float64(want[i]), 2)
		}
		if e := math.Sqrt(num / den); e > 1e-3 || den == 0 {
			t.Errorf("%s after 3 tokens in a step of 4: relative error %g against a token a step", name, e)
		}
	}
}
