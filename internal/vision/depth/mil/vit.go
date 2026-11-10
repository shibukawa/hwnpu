package mil

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

const vitPrefix = "model.backbone.pretrained."

// Vit is the patch embedding and the blocks: input patches (1, tokens, PatchIn), row 0 zeros
// (the class token's), then each patch's 3 x 14 x 14 normalized values (channel, row, column);
// output (1, tokens, 4 FeatDim), the four features after the final norm side by side.
func Vit(c Config) *Part {
	return part("vit", func(g *graph) {
		E, T := c.Hidden, c.Tokens()
		x := g.Input("patches", 1, T, PatchIn)
		pw := g.Get(vitPrefix+"patch_embed.proj.weight", E, 3, Patch, Patch)
		wt := make([]float32, E*PatchIn)
		for o := 0; o < E; o++ {
			copy(wt[o*PatchIn:], pw[o*3*Patch*Patch:(o+1)*3*Patch*Patch])
		}
		pos := PosEmbed(g.Get(vitPrefix+"pos_embed"), E, c.GH(), c.GW())
		bias := g.Get(vitPrefix + "patch_embed.proj.bias")
		cls := g.Get(vitPrefix + "cls_token")
		for t := 0; t < T; t++ {
			for j := 0; j < E; j++ {
				if t == 0 {
					pos[j] += cls[j]
				} else {
					pos[t*E+j] += bias[j]
				}
			}
		}
		x = g.Add(g.LinearConst(x, "patch_embed", wt, E, nil), g.F16s("pos", []int{1, T, E}, pos))
		var ropes [2]rope // local, flat (the global blocks')
		if c.RopeStart >= 0 {
			for f := range ropes {
				t := RopeTables(c, f == 1)
				for j, v := range t {
					ropes[f][j] = g.F16s(fmt.Sprintf("rope_%d_%d", f, j), []int{1, T, 1, 32}, v)
				}
			}
		}
		var feats []*mlpackage.Var
		local := x
		for i := 0; i < c.Layers; i++ {
			glob := c.AltStart >= 0 && i >= c.AltStart && i%2 == 1
			if c.AltStart >= 0 && i == c.AltStart {
				// the reference view's camera token in the class token's row
				cam := g.Get(vitPrefix + "camera_token")[:E]
				x = g.Concat(1, g.F16s("camera", []int{1, 1, E}, cam), g.Slice(x, 1, 1, T))
			}
			var rope rope
			if c.RopeStart >= 0 && i >= c.RopeStart {
				f := 0
				if glob {
					f = 1
				}
				rope = ropes[f]
			}
			x = block(g, c, i, x, rope)
			if !glob {
				local = x
			}
			for _, o := range c.OutLayers {
				if o != i {
					continue
				}
				n := g.ScaledLayerNorm(x, 1e-6)
				n = g.Add(g.Mul(n, g.F16s(fmt.Sprintf("norm_w_%d", i), []int{E}, g.Get(vitPrefix+"norm.weight", E))),
					g.F16s(fmt.Sprintf("norm_b_%d", i), []int{E}, g.Get(vitPrefix+"norm.bias", E)))
				if c.CatToken {
					n = g.Concat(-1, local, n)
				}
				feats = append(feats, n)
			}
		}
		g.Output(g.Concat(-1, feats...))
	})
}

// layerNorm is a layer norm over the last axis with the named weight and bias.
func layerNorm(g *graph, x *mlpackage.Var, p string, eps float64) *mlpackage.Var {
	n := x.Shape[len(x.Shape)-1]
	shape := make([]int, len(x.Shape))
	for i := range shape {
		shape[i] = 1
	}
	shape[len(shape)-1] = n
	y := g.ScaledLayerNorm(x, eps)
	return g.Add(g.Mul(y, g.F16s(p+".weight", shape, g.Get(p+".weight", n))), g.F16s(p+".bias", shape, g.Get(p+".bias", n)))
}

// block is a DINOv2 block: layer norm, attention (heads of 64; per-head layer norms of q and k
// and 2-D RoPE from their starts), layer scale; layer norm, GELU MLP, layer scale.
func block(g *graph, c Config, i int, x *mlpackage.Var, rope rope) *mlpackage.Var {
	p := fmt.Sprintf("%sblocks.%d.", vitPrefix, i)
	E, H, T := c.Hidden, c.Heads, x.Shape[1]
	D := E / H
	h := layerNorm(g, x, p+"norm1", 1e-6)
	qkvw, qkvb := g.Get(p+"attn.qkv.weight", 3*E, E), g.Get(p+"attn.qkv.bias", 3*E)
	head := func(j int, norm string) *mlpackage.Var {
		y := g.LinearConst(h, fmt.Sprintf("%sattn.qkv.%d", p, j), qkvw[j*E*E:(j+1)*E*E], E, qkvb[j*E:(j+1)*E])
		y = g.Reshape(y, 1, T, H, D)
		if norm != "" && c.QKNormStart >= 0 && i >= c.QKNormStart {
			y = layerNorm(g, y, p+"attn."+norm, 1e-5)
		}
		if rope[0] != nil && norm != "" {
			// the first 32 dims turn by the row, the last 32 by the column (rotate_half in each)
			a := g.ApplyRope(g.Slice(y, -1, 0, 32), rope[0], rope[1])
			b := g.ApplyRope(g.Slice(y, -1, 32, 64), rope[2], rope[3])
			y = g.Concat(-1, a, b)
		}
		return g.Transpose(y, 0, 2, 1, 3)
	}
	q, k, v := head(0, "q_norm"), head(1, "k_norm"), head(2, "")
	o := g.QChunkAttention(q, k, v, 1/math.Sqrt(float64(D)), nil, 512)
	ls1, ls2 := g.Get(p+"ls1.gamma", E), g.Get(p+"ls2.gamma", E)
	scaled := func(name string, rows, cols int, ls []float32) ([]float32, []float32) {
		w, b := g.Get(name+".weight", rows, cols), g.Get(name+".bias", rows)
		for r := 0; r < rows; r++ {
			for k := 0; k < cols; k++ {
				w[r*cols+k] *= ls[r]
			}
			b[r] *= ls[r]
		}
		return w, b
	}
	ow, ob := scaled(p+"attn.proj", E, E, ls1)
	x = g.Add(x, g.LinearConst(g.Merge(o), p+"attn.proj", ow, E, ob))
	h = layerNorm(g, x, p+"norm2", 1e-6)
	f := g.LinearConst(h, p+"mlp.fc1", g.Get(p+"mlp.fc1.weight", c.MLP, E), c.MLP, g.Get(p+"mlp.fc1.bias", c.MLP))
	f = g.Op("gelu", map[string]any{"x": f, "mode": g.Str("EXACT")}, f.Shape...)
	fw, fb := scaled(p+"mlp.fc2", E, c.MLP, ls2)
	return g.Add(x, g.LinearConst(f, p+"mlp.fc2", fw, E, fb))
}

// rope is a 2-D RoPE's tables (1, tokens, 1, 32): the rows' cos and sin, the columns'.
type rope [4]*mlpackage.Var

// RopeTables is RotaryPositionEmbedding2D's (frequency 100) cos and sin of each token's row and
// column, (tokens, 32) each, cat(f, f) of 16 frequencies: the patches at (row + 1, column +
// 1), the class (camera) token at (0, 0); flat, every patch at (1, 1) (the global blocks'
// positions of one picture).
func RopeTables(c Config, flat bool) [4][]float32 {
	T, gw := c.Tokens(), c.GW()
	var out [4][]float32
	for j := range out {
		out[j] = make([]float32, T*32)
	}
	for t := 1; t < T; t++ {
		y, x := (t-1)/gw+1, (t-1)%gw+1
		if flat {
			y, x = 1, 1
		}
		for k := 0; k < 16; k++ {
			inv := 1 / math.Pow(100, float64(2*k)/32)
			for a, pos := range []int{y, x} {
				ang := float64(pos) * inv
				out[2*a][t*32+k], out[2*a][t*32+16+k] = float32(math.Cos(ang)), float32(math.Cos(ang))
				out[2*a+1][t*32+k], out[2*a+1][t*32+16+k] = float32(math.Sin(ang)), float32(math.Sin(ang))
			}
		}
	}
	for k := 0; k < 32; k++ { // position (0, 0): no turn
		out[0][k], out[2][k] = 1, 1
	}
	return out
}
