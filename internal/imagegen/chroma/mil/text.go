package mil

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// T5Scale divides T5's residual stream: it reaches 3e5 in its outlier channels from layer 7 on
// (tools/amd/iron_chroma_parts.py), past fp16's 65504. The host divides the embedding rows by it,
// and each layer's o and wo carry it.
const T5Scale = 64

// t5WoFold is the part of T5Scale folded into wo's weight (its output reaches 1.4e5, past fp16):
// the rest divides its output.
const t5WoFold = 8

// t5Up multiplies the norms' outputs into the projections (t5Layer).
const t5Up = 16

// T5FFScale divides the gated FF's up projection (wi_1): its product with the GELU of wi_0 is
// what wo reads; wo takes it back.
const T5FFScale = 1

// DitScale divides the DiT's residual streams (FLUX's reach past fp16 in the later single blocks:
// diffusers clips them to 65504 in fp16). context_embedder, x_embedder, every block's output
// projections carry it; the norms do not see it.
const DitScale = 32

// TextGroups splits T5's layers into the programs of Text: at most per layers each, as even as
// they go (24 by 4: 4, 4, 4, 4, 4, 4). At int8 a layer is 193 MB and a program past about 1 GB
// of weights is placed on the CPU.
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

// RelativeBuckets is T5's bidirectional relative position bucket of each (query, key) of rows
// (transformers' _relative_position_bucket), row major.
func RelativeBuckets(rows, buckets, maxDistance int) []int {
	nb := buckets / 2
	exact := nb / 2
	out := make([]int, rows*rows)
	for q := 0; q < rows; q++ {
		for k := 0; k < rows; k++ {
			rel := k - q
			b := 0
			if rel > 0 {
				b = nb
			}
			if rel < 0 {
				rel = -rel
			}
			if rel < exact {
				b += rel
			} else {
				// float32 as transformers computes it
				l := float32(math.Log(float64(float32(rel)/float32(exact)))) / float32(math.Log(float64(maxDistance)/float64(exact))) * float32(nb-exact)
				v := exact + int(l)
				if v > nb-1 {
					v = nb - 1
				}
				b += v
			}
			out[q*rows+k] = b
		}
	}
	return out
}

// T5Bias is the position bias (heads, rows, rows) from layer 0's relative_attention_bias table
// (buckets, heads), shared by every layer.
func T5Bias(table []float32, c T5, rows int) []float32 {
	bk := RelativeBuckets(rows, c.Buckets, c.MaxDistance)
	out := make([]float32, c.Heads*rows*rows)
	for h := 0; h < c.Heads; h++ {
		for i, b := range bk {
			out[h*rows*rows+i] = table[b*c.Heads+h]
		}
	}
	return out
}

const t5BiasName = "t5.encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"

// Text is T5's layers first..first+count-1 over the prompt's rows (rows: TextRows, or fewer):
// input x (1, rows, hidden), the embedding rows divided by T5Scale for the first group, else the
// residual stream; mask (1, 1, 1, rows), 0 for the prompt's tokens (its </s> too) and -10000 past
// them. Output the residual stream after them; past the last layer, the final norm's rows through
// the DiT's context_embedder, divided by DitScale (dit, the DiT's shape): the DiT's text rows.
func Text(c T5, dit Dit, first, count, rows int) *Part {
	return part(fmt.Sprintf("text_%d", first), func(g *graph) {
		L := rows
		x := g.Input("x", 1, L, c.Hidden)
		mask := g.Input("mask", 1, 1, 1, L)
		bias := g.F16s(fmt.Sprintf("t5_bias_%d", L), []int{1, c.Heads, L, L}, T5Bias(g.Get(t5BiasName, c.Buckets, c.Heads), c, L))
		// the bias and the mask together, once for every layer
		bm := g.Add(bias, mask)
		for i := first; i < first+count; i++ {
			x = t5Layer(g, c, i, x, bm)
		}
		if first+count == c.Layers {
			x = t5Norm(g, x, "t5.encoder.final_layer_norm.weight", c.Eps)
			// 1/DitScale after the projection (t5Layer's o)
			x = g.Scale(g.LinearBias(x, "dit.context_embedder.weight", "dit.context_embedder.bias", "", dit.Hidden), 1.0/DitScale)
		}
		g.Output(x)
	})
}

func t5Layer(g *graph, c T5, i int, x, bm *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("t5.encoder.block.%d.layer.", i)
	D, H := c.HeadDim, c.Heads
	// the norms' outputs t5Up times larger into the projections, which divide their outputs by
	// it: the products of the small channels stay over fp16's normal range
	h := g.Scale(t5Norm(g, x, p+"0.layer_norm.weight", c.Eps), t5Up)
	a := p + "0.SelfAttention."
	proj := func(name string) *mlpackage.Var {
		return g.Heads(g.Scale(g.Linear(h, a+name+".weight", "", H*D), 1.0/t5Up), H, D)
	}
	q, k, v := proj("q"), proj("k"), proj("v")
	// no 1/sqrt(d) (T5's scores are not scaled); the bias and the mask added
	o := g.SDPAScale(q, k, v, 1, func() *mlpackage.Var { return bm })
	// the residual's 1/T5Scale after the projection, not in its weight: the Neural Engine loses
	// the digits of products under fp16's normal range, and the weight divided by 64 made them
	// (the prompt rows with T5's largest outliers came out at a cosine of 0.36)
	x = g.Add(x, g.Scale(g.Linear(g.Merge(o), a+"o.weight", "", c.Hidden), 1.0/T5Scale))
	h = g.Scale(t5Norm(g, x, p+"1.layer_norm.weight", c.Eps), t5Up)
	f := p + "1.DenseReluDense."
	gt := GeluTanh(g, g.Scale(g.Linear(h, f+"wi_0.weight", "", c.MLP), 1.0/t5Up))
	up := g.Scale(g.Linear(h, f+"wi_1.weight", "", c.MLP), 1.0/(t5Up*T5FFScale))
	// wo's output reaches 1.4e5: 1/t5WoFold of T5Scale in its weight, the rest after it
	y := g.LinearS(g.Mul(gt, up), f+"wo.weight", "", c.Hidden, float32(T5FFScale)/t5WoFold)
	return g.Add(x, g.Scale(y, float32(t5WoFold)/T5Scale))
}

// t5Norm is T5's RMS norm (the named weight) of x, whose rows hold outlier channels thousands
// of times the others. Its mean square is taken of x divided by each row's largest magnitude (so
// that the squares fit fp16), but the output is x itself times rsqrt(that mean) / the largest
// magnitude: x divided first (milpart's ScaledRMSNorm) put the other channels near fp16's
// subnormals, and T5's rows on the Neural Engine came out at a cosine of 0.95 against float32
// (the CPU interpreter 0.99996).
func t5Norm(g *graph, x *mlpackage.Var, name string, eps float64) *mlpackage.Var {
	n := x.Shape[len(x.Shape)-1]
	neg := g.Scale(x, -1)
	abs := g.Op("maximum", map[string]any{"x": x, "y": neg}, x.Shape...)
	m := g.Reduce("max", abs, -1)
	m = g.Op("maximum", map[string]any{"x": m, "y": g.B.ScalarF16(1e-3)}, m.Shape...)
	xs := g.Div(x, m)
	ms := g.Reduce("mean", g.Mul(xs, xs), -1)
	r := g.Op("rsqrt", map[string]any{"x": ms, "epsilon": g.B.ScalarF16(float32(eps))}, ms.Shape...)
	w := g.F16s(name, []int{n}, g.Get(name, n))
	return g.Mul(g.Mul(x, g.Div(r, m)), w)
}

// GeluTanh is the tanh approximation of GELU written as x sigmoid(2 sqrt(2/pi) (x + 0.044715
// x^3)), with exp and real_div: the Neural Engine's own gelu (TANH_APPROXIMATION) drifts over many
// layers (design.md 16.145).
func GeluTanh(g *graph, x *mlpackage.Var) *mlpackage.Var {
	a := float32(-2 * math.Sqrt(2/math.Pi))
	u := g.Mul(x, g.AddScalar(g.Scale(g.Mul(x, x), a*0.044715), a))
	return g.Div(x, g.AddScalar(g.Unary("exp", u), 1))
}
