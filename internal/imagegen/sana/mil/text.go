package mil

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/imagegen/milpart"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// TextGroups splits Gemma 2's layers into the programs of Text: at most per layers each, as
// even as they go (26 by 6: 6, 5, 5, 5, 5). At int8 a layer is 78 MB and a program past about
// 1 GB of weights is placed on the CPU (design.md 16.85).
func TextGroups(layers, per int) [][2]int {
	n := (layers + per - 1) / per
	var out [][2]int
	first := 0
	for i := 0; i < n; i++ {
		k := layers / n
		if i < layers%n {
			k++
		}
		out = append(out, [2]int{first, k})
		first += k
	}
	return out
}

// Text is Gemma 2's layers first..first+count-1 over the prompt's TextLen rows, causal: input x
// (1, TextLen, hidden), the embedding rows times sqrt(hidden) for the first group, else the
// residual stream; output its residual stream after them, past the last layer the final norm's
// (Gemma 2's last hidden state). The attention's logit softcapping (50) is left out, as on AMD's
// NPU (the rows Sana keeps come out the same, design.md 16.119).
func Text(c Gemma, first, count int) *Part {
	return part(fmt.Sprintf("text_%d", first), func(g *graph) {
		L := TextLen
		x := g.Input("x", 1, L, c.Hidden)
		causal := g.CausalMask("gemma_causal", L, 0)
		cos, sin := g.RopeTables("gemma", milpart.InvFreq(c.HeadDim, c.Theta), L, c.HeadDim)
		for i := first; i < first+count; i++ {
			x, _, _ = gemmaLayer(g, c, i, x, causal, cos, sin, nil, nil)
		}
		if first+count == c.Layers {
			x = gemmaNorm(g, x, "gemma.norm.weight", c.Eps)
		}
		g.Output(x)
	})
}

// TextHeld is how many of the instruction's rows the programs of TextFrom hold as keys and
// values. Sana's prompt follows an instruction of 208 tokens (sana.Instruction), whose rows are
// the same for every prompt: Gemma 2 is causal. The last of its tokens may merge with the
// prompt's first, so not all of them are held (192 on AMD's NPU too, design.md 16.138).
const TextHeld = 192

// TextFromRows are the rows of TextFrom's functions, the shortest first: after the held rows, the
// instruction's last 16 and a prompt of up to 48 tokens, or of any length (the instruction and
// the prompt are 504 tokens at most).
//
// Not 128 rows: on an M1 every run of that function failed after a minute or two ("Request
// cancelled"), 0.55 GiB stayed on the Neural Engine after the process and the next process's
// first run failed the same way; 64 and 320 rows run (design.md 16.224). A length added here is
// one to run on the device first.
var TextFromRows = []int{64, 320}

// TextKVLen is the values TextKV gives for rows rows: the residual stream, then the layers' keys
// and values.
func TextKVLen(c Gemma, count, rows int) int {
	return rows*c.Hidden + 2*count*c.KVHeads*rows*c.HeadDim
}

// TextKV is Text over rows rows from position 0 that also gives the attention's keys and values:
// what the host computes once of the instruction's held rows (milcpu), for TextFrom to hold. The
// output y (1, TextKVLen) is the residual stream (rows, hidden), then for each layer its keys
// (rotated by RoPE) and its values, each (kv heads, rows, head dim).
func TextKV(c Gemma, first, count, rows int) *Part {
	return part(fmt.Sprintf("text_kv_%d", first), func(g *graph) {
		x := g.Input("x", 1, rows, c.Hidden)
		causal := g.CausalMask("gemma_causal", rows, 0)
		cos, sin := g.RopeTables("gemma", milpart.InvFreq(c.HeadDim, c.Theta), rows, c.HeadDim)
		n := c.KVHeads * rows * c.HeadDim
		var kvs []*mlpackage.Var
		for i := first; i < first+count; i++ {
			var k, v *mlpackage.Var
			x, k, v = gemmaLayer(g, c, i, x, causal, cos, sin, nil, nil)
			kvs = append(kvs, g.Reshape(k, 1, n), g.Reshape(v, 1, n))
		}
		if first+count == c.Layers {
			x = gemmaNorm(g, x, "gemma.norm.weight", c.Eps)
		}
		g.Output(g.Concat(1, append([]*mlpackage.Var{g.Reshape(x, 1, rows*c.Hidden)}, kvs...)...))
	})
}

// SplitTextKV splits TextKV's output y: the residual stream, and each layer's keys and values.
func SplitTextKV(c Gemma, count, rows int, y []float32) (x []float32, kv [][2][]float32) {
	x, y = y[:rows*c.Hidden], y[rows*c.Hidden:]
	n := c.KVHeads * rows * c.HeadDim
	for i := 0; i < count; i++ {
		kv = append(kv, [2][]float32{y[:n], y[n : 2*n]})
		y = y[2*n:]
	}
	return x, kv
}

// TextFrom is Text for the rows of a prompt after the held ones: input x (1, rows, hidden), the
// rows of positions held to held+rows-1; kv[i] are layer first+i's keys and values of the held
// positions before them (TextKV's, (kv heads, held, head dim) each), constants of the program.
// A row attends to them and to the rows up to itself, as it does in Text: the output is Text's
// rows from held on. The program is named as Text's, to be a function of the same file.
func TextFrom(c Gemma, first, count, held, rows int, kv [][2][]float32) *Part {
	return part(fmt.Sprintf("text_%d", first), func(g *graph) {
		x := g.Input("x", 1, rows, c.Hidden)
		// a row sees every held key, and of its own those up to itself
		mask := make([]float32, rows*(held+rows))
		for q := 0; q < rows; q++ {
			for k := q + 1; k < rows; k++ {
				mask[q*(held+rows)+held+k] = -1e4
			}
		}
		causal := g.F16s(fmt.Sprintf("gemma_causal_from_%d", rows), []int{1, 1, rows, held + rows}, mask)
		cv, sv := blocks.RopeValues(milpart.InvFreq(c.HeadDim, c.Theta), held, rows, c.HeadDim)
		cos := g.F16s(fmt.Sprintf("rope_cos_gemma_from_%d", rows), []int{1, rows, 1, c.HeadDim}, cv)
		sin := g.F16s(fmt.Sprintf("rope_sin_gemma_from_%d", rows), []int{1, rows, 1, c.HeadDim}, sv)
		shape := []int{1, c.KVHeads, held, c.HeadDim}
		for i := first; i < first+count; i++ {
			pk := g.F16s(fmt.Sprintf("gemma_held_k_%d", i), shape, kv[i-first][0])
			pv := g.F16s(fmt.Sprintf("gemma_held_v_%d", i), shape, kv[i-first][1])
			x, _, _ = gemmaLayer(g, c, i, x, causal, cos, sin, pk, pv)
		}
		if first+count == c.Layers {
			x = gemmaNorm(g, x, "gemma.norm.weight", c.Eps)
		}
		g.Output(x)
	})
}

// gemmaNorm is Gemma 2's RMS norm, weighted by 1 + w, of x scaled by MaxScaled.
func gemmaNorm(g *graph, x *mlpackage.Var, name string, eps float64) *mlpackage.Var {
	n := x.Shape[len(x.Shape)-1]
	w := g.Get(name, n)
	for i := range w {
		w[i]++
	}
	return g.RMSNormWith(g.MaxScaled(x, -1), name, w, eps, 1)
}

// gemmaLayer is Gemma 2's layer i on x (1, rows, hidden) under mask (1, 1, rows, keys): the
// residual stream after it, and its own rows' keys (rotated) and values (1, kv heads, rows, head
// dim). With pk and pv, the keys and values of the positions before x's (1, kv heads, n, head
// dim), the attention reads those and then its own: mask's keys are n + rows.
func gemmaLayer(g *graph, c Gemma, i int, x, mask, cos, sin, pk, pv *mlpackage.Var) (y, k, v *mlpackage.Var) {
	p := fmt.Sprintf("gemma.layers.%d.", i)
	L, D, H, G := x.Shape[1], c.HeadDim, c.Heads, c.KVHeads
	h := gemmaNorm(g, x, p+"input_layernorm.weight", c.Eps)
	proj := func(name string, heads int, rope bool) *mlpackage.Var {
		y := g.Reshape(g.Linear(h, p+"self_attn."+name+"_proj.weight", "", heads*D), 1, L, heads, D)
		if rope {
			y = g.ApplyRope(y, cos, sin)
		}
		return g.Transpose(y, 0, 2, 1, 3)
	}
	q, k, v := proj("q", H, true), proj("k", G, true), proj("v", G, false)
	ks, vs := k, v
	if pk != nil {
		ks, vs = g.Concat(2, pk, k), g.Concat(2, pv, v)
	}
	o := g.SDPAGrouped(q, ks, vs, 1/math.Sqrt(c.QueryScalar), func() *mlpackage.Var { return mask })
	o = g.Linear(g.Merge(o), p+"self_attn.o_proj.weight", "", c.Hidden)
	x = g.Add(x, gemmaNorm(g, o, p+"post_attention_layernorm.weight", c.Eps))
	h = gemmaNorm(g, x, p+"pre_feedforward_layernorm.weight", c.Eps)
	gt := geluTanh(g, g.Linear(h, p+"mlp.gate_proj.weight", "", c.MLP))
	up := g.Linear(h, p+"mlp.up_proj.weight", "", c.MLP)
	f := g.Linear(g.Mul(gt, up), p+"mlp.down_proj.weight", "", c.Hidden)
	return g.Add(x, gemmaNorm(g, f, p+"post_feedforward_layernorm.weight", c.Eps)), k, v
}

// geluTanh is the tanh approximation of GELU written as x sigmoid(2 sqrt(2/pi) (x + 0.044715
// x^3)), with exp and real_div as blocks.SiLU: the Neural Engine's own gelu (TANH_APPROXIMATION)
// moved Gemma 2's rows Sana keeps (design.md 16.145).
func geluTanh(g *graph, x *mlpackage.Var) *mlpackage.Var {
	// u = -2 sqrt(2/pi) (x + 0.044715 x^3) = x (a + b x^2)
	a := float32(-2 * math.Sqrt(2/math.Pi))
	u := g.Mul(x, g.AddScalar(g.Scale(g.Mul(x, x), a*0.044715), a))
	// past x = -11 or so exp overflows fp16 to inf and the quotient is 0, as gelu is there
	return g.Div(x, g.AddScalar(g.Unary("exp", u), 1))
}
