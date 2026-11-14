package music

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// The depthformer of mrt2_small (MagentaRT2ModelSmall): the encoder embeds the conditioning (256),
// the temporal decoder (12 layers of 1024, 8 heads of 128, window 41, cross-attention to the
// encoder) steps once a frame, the depth decoder (2 layers of 768, 6 heads of 128) once a
// codebook.
const (
	ReservedTokens = 6
	Vocab          = Codebooks*CodebookSize + ReservedTokens // 12294
	condChannels   = 144                                     // MusicCoCa 12, piano roll 128, drums 1, CFG 2 + 1
	encDim         = 256
	tDim           = 1024
	tLayers        = 12
	tHeads         = 8
	headDim        = 128
	tFFN           = 4096
	Window         = 41 // past frames each attention sees, besides the current one
	dDim           = 768
	dLayers        = 2
	dHeads         = 6
	dFFN           = 3072
	normEps        = 1e-6
	SoftCap        = 30.0
)

const lmPrefix = "depthformer/"

// Conditioning are the conditioning tokens of a frame (the encoder's input, before the offset
// of 7): MusicCoCa's 12 codes, the piano roll's 128 states, the drums', and the classifier-free
// guidance scales binned (musiccoca, notes; drums); -1 for masked.
type Conditioning [condChannels]int

// Encode is the encoder on the host: the conditioning's embedding (256).
func Encode(w llm.Weights, c Conditioning) ([]float32, error) {
	const off = ReservedTokens + 1
	row := func(name string, i int) ([]float32, error) { return w.Rows(name, i, i+1) }
	// MusicCoCa: each level's code + the level * 1031, summed (768), projected to 256
	sum := make([]float64, 768)
	for l := range 12 {
		r, err := row(lmPrefix+"encoder/body/layers_1/branched_mulan_embedder/mulan_embedder/mulan_dequantizer/embedding", c[l]+off+l*1031)
		if err != nil {
			return nil, err
		}
		for i, v := range r {
			sum[i] += float64(v)
		}
	}
	k, _, err := w.Get(lmPrefix + "encoder/body/layers_1/branched_mulan_embedder/mulan_embedder/depth_input_adapter/kernel")
	if err != nil {
		return nil, err
	}
	mulan := make([]float64, encDim)
	for i, s := range sum {
		for j := range encDim {
			mulan[j] += s * float64(k[i*encDim+j])
		}
	}
	// the other channels, each with its own range of rows, averaged
	var sizes []int
	for range 128 {
		sizes = append(sizes, 11)
	}
	sizes = append(sizes, 9, 47, 47, 15)
	regular := make([]float64, encDim)
	at := 0
	for ch, n := range sizes {
		r, err := row(lmPrefix+"encoder/body/layers_1/branched_regular_embedder/regular_embedder/embedding", at+c[12+ch]+off)
		if err != nil {
			return nil, err
		}
		for i, v := range r {
			regular[i] += float64(v) / float64(len(sizes))
		}
		at += n
	}
	// the branches' mean, then encoder_ln
	x := make([]float64, encDim)
	var mean float64
	for i := range x {
		x[i] = (mulan[i] + regular[i]) / 2
		mean += x[i] / encDim
	}
	var vr float64
	for _, v := range x {
		vr += (v - mean) * (v - mean) / encDim
	}
	scale, _, err := w.Get(lmPrefix + "encoder/body/encoder_ln/scale")
	if err != nil {
		return nil, err
	}
	bias, _, err := w.Get(lmPrefix + "encoder/body/encoder_ln/bias")
	if err != nil {
		return nil, err
	}
	out := make([]float32, encDim)
	for i, v := range x {
		out[i] = float32((v-mean)/math.Sqrt(vr+normEps)*float64(scale[i]) + float64(bias[i]))
	}
	return out, nil
}

// EmbedTokens is the depthformer's embedding of a frame's tokens (the unique codes of the 12
// codebooks, ReservedTokens + level * 1024 + code), each scaled by sqrt(1024), averaged: the
// temporal decoder's input of the next frame. One token alone (n 1) is a depth step's input.
func EmbedTokens(w llm.Weights, tokens []int) ([]float32, error) {
	out := make([]float64, tDim)
	for _, t := range tokens {
		r, err := w.Rows(lmPrefix+"decoder/decoder_embedding/embedding/embedding", t, t+1)
		if err != nil {
			return nil, err
		}
		for i, v := range r {
			out[i] += float64(v) * 32 / float64(len(tokens))
		}
	}
	f := make([]float32, tDim)
	for i, v := range out {
		f[i] = float32(v)
	}
	return f, nil
}

// lgraph builds the depthformer's programs.
type lgraph struct {
	*blocks.Graph
	ins  []mlpackage.Feature
	outs []*mlpackage.Var
}

func (g *lgraph) input(name string, shape ...int) *mlpackage.Var {
	v := g.B.Input(name, mlpackage.Float16, shape...)
	g.ins = append(g.ins, mlpackage.Feature{Name: name, DType: mlpackage.Float16, Shape: shape})
	return v
}

func (g *lgraph) output(v *mlpackage.Var, name string) {
	g.outs = append(g.outs, g.B.NamedOp("identity", name, map[string]any{"x": v}, mlpackage.Float16, v.Shape...))
}

// rms is RMSNormalization (MLX's RMSNorm): x times rsqrt(mean(x²) + eps), times the scale; the
// input scaled down by 1/64 first so that its squares stay within fp16.
func (g *lgraph) rms(x *mlpackage.Var, name string) *mlpackage.Var {
	n := x.Shape[len(x.Shape)-1]
	ms := append(append([]int{}, x.Shape[:len(x.Shape)-1]...), 1)
	xs := g.Op("mul", map[string]any{"x": x, "y": g.B.ScalarF16(1.0 / 64)}, x.Shape...)
	sq := g.Op("mul", map[string]any{"x": xs, "y": xs}, x.Shape...)
	m := g.Op("reduce_mean", map[string]any{"x": sq, "axes": g.B.Ints(-1), "keep_dims": g.B.ScalarBool(true)}, ms...)
	r := g.Op("rsqrt", map[string]any{"x": m, "epsilon": g.B.ScalarF16(float32(normEps / 4096))}, ms...)
	y := g.Op("mul", map[string]any{"x": xs, "y": r}, x.Shape...)
	return g.Op("mul", map[string]any{"x": y, "y": g.F16s(name+"_s", []int{n}, g.Get(name, n))}, x.Shape...)
}

// layerNorm is LayerNormalization with scale and bias.
func (g *lgraph) layerNorm(x *mlpackage.Var, name string) *mlpackage.Var {
	n := x.Shape[len(x.Shape)-1]
	return g.Op("layer_norm", map[string]any{"x": x, "axes": g.B.Ints(-1), "gamma": g.F16s(name+"_g", []int{n}, g.Get(name+"/scale", n)),
		"beta": g.F16s(name+"_b", []int{n}, g.Get(name+"/bias", n)), "epsilon": g.B.ScalarF16(float32(normEps))}, x.Shape...)
}

// dense is x (..., in) times a Flax kernel (in, out), plus its bias when bias.
func (g *lgraph) dense(x *mlpackage.Var, name string, out int, bias bool) *mlpackage.Var {
	in := x.Shape[len(x.Shape)-1]
	k := g.Get(name+"/kernel", in, out)
	wt := make([]float32, out*in)
	for i := range in {
		for o := range out {
			wt[o*in+i] = k[i*out+o]
		}
	}
	// the rows padded with zeros to a multiple of 16, the lookup tables' group, and cut after
	rows := (out + 15) / 16 * 16
	wt = append(wt, make([]float32, (rows-out)*in)...)
	args := map[string]any{"x": x, "weight": g.Weightish(name, wt, rows, in)}
	if bias {
		b := append(g.Get(name+"/bias", out), make([]float32, rows-out)...)
		args["bias"] = g.F16s(name+"_bias", []int{rows}, b)
	}
	lead := x.Shape[:len(x.Shape)-1]
	y := g.Op("linear", args, append(append([]int{}, lead...), rows)...)
	if rows == out {
		return y
	}
	begin, end := make([]int32, len(x.Shape)), make([]int32, len(x.Shape))
	for i, d := range lead {
		end[i] = int32(d)
	}
	end[len(end)-1] = int32(out)
	return g.Op("slice_by_index", map[string]any{"x": y, "begin": g.B.Ints(begin...), "end": g.B.Ints(end...)}, append(append([]int{}, lead...), out)...)
}

// Weightish is an (out, in) matrix in the program's format.
func (g *lgraph) Weightish(name string, w []float32, rows, cols int) *mlpackage.Var {
	return g.Matrix(name, w, rows, cols)
}

// heads projects x (1, 1, in) by a Flax attention kernel (in, heads, 128) to (1, heads, 1, 128).
func (g *lgraph) heads(x *mlpackage.Var, name string, heads int) *mlpackage.Var {
	in := x.Shape[2]
	k := g.Get(name+"/kernel", in, heads, headDim)
	wt := make([]float32, heads*headDim*in)
	for i := range in {
		for o := range heads * headDim {
			wt[o*in+i] = k[i*heads*headDim+o]
		}
	}
	y := g.Op("linear", map[string]any{"x": x, "weight": g.Weightish(name, wt, heads*headDim, in)}, 1, 1, heads*headDim)
	return g.Op("reshape", map[string]any{"x": y, "shape": g.B.Ints(1, int32(heads), 1, headDim)}, 1, heads, 1, headDim)
}

// outProj is the output projection: (1, heads, 1, 128) by a kernel (out, heads, 128) to (1, 1, out).
func (g *lgraph) outProj(o *mlpackage.Var, name string, out int) *mlpackage.Var {
	heads := o.Shape[1]
	o = g.Op("reshape", map[string]any{"x": o, "shape": g.B.Ints(1, 1, int32(heads*headDim))}, 1, 1, heads*headDim)
	k := g.Get(name+"/kernel", out, heads, headDim)
	return g.Op("linear", map[string]any{"x": o, "weight": g.Weightish(name, k, out, heads*headDim)}, 1, 1, out)
}

// queryScale is per_dim_scale's scale of the queries: 1.442695041 / sqrt(128) * softplus(p).
func (g *lgraph) queryScale(name string) []float32 {
	p := g.Get(name, headDim)
	out := make([]float32, headDim)
	for i, v := range p {
		out[i] = float32(1.442695041 / math.Sqrt(headDim) * math.Log1p(math.Exp(float64(v))))
	}
	return out
}

// attend is attention of the query q (1, heads, 1, 128) over keys and values (1, heads, n, 128)
// with an additive mask (1, 1, 1, n) and optionally a sink: its key's logit taken with the
// unscaled query, its value always seen. The queries are scaled per dimension (scale).
func (g *lgraph) attend(q, k, v, mask *mlpackage.Var, scale []float32, sink string) *mlpackage.Var {
	heads, n := q.Shape[1], k.Shape[2]
	qs := g.Op("mul", map[string]any{"x": q, "y": g.F16s("qscale", []int{1, 1, 1, headDim}, scale)}, q.Shape...)
	logits := g.Op("matmul", map[string]any{"x": qs, "y": k, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(true)}, 1, heads, 1, n)
	logits = g.Op("add", map[string]any{"x": logits, "y": mask}, 1, heads, 1, n)
	vals := v
	if sink != "" {
		sk := g.Get(sink+"/sink_key_embeddings", 1, heads, headDim)
		sv := g.Get(sink+"/sink_value_embeddings", 1, heads, headDim)
		skv := g.F16s("sink_k", []int{1, heads, headDim, 1}, sk)
		sl := g.Op("matmul", map[string]any{"x": q, "y": skv, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(false)}, 1, heads, 1, 1)
		logits = g.Op("concat", map[string]any{"values": []*mlpackage.Var{sl, logits}, "axis": g.B.ScalarI32(3), "interleave": g.B.ScalarBool(false)}, 1, heads, 1, n+1)
		vals = g.Op("concat", map[string]any{"values": []*mlpackage.Var{g.F16s("sink_v", []int{1, heads, 1, headDim}, sv), v}, "axis": g.B.ScalarI32(2), "interleave": g.B.ScalarBool(false)}, 1, heads, n+1, headDim)
		n++
	}
	p := g.Op("softmax", map[string]any{"x": logits, "axis": g.B.ScalarI32(-1)}, 1, heads, 1, n)
	return g.Op("matmul", map[string]any{"x": p, "y": vals, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(false)}, 1, heads, 1, headDim)
}

func (g *lgraph) add(x, y *mlpackage.Var) *mlpackage.Var {
	return g.Op("add", map[string]any{"x": x, "y": y}, x.Shape...)
}

// ffn is the residual FFN: pre_norm, dense with tanh-approximated GELU, dense, post_norm.
func (g *lgraph) ffn(x *mlpackage.Var, p string, hidden int) *mlpackage.Var {
	h := g.dense(g.rms(x, p+"ffn/pre_norm/scale"), p+"ffn/ffn_layer1", hidden, true)
	h = g.Op("gelu", map[string]any{"x": h, "mode": g.Str("TANH_APPROXIMATION")}, h.Shape...)
	h = g.dense(h, p+"ffn/ffn_layer2", x.Shape[2], true)
	return g.add(x, g.rms(h, p+"ffn/post_norm/scale"))
}

// cacheAttention is a residual attention block of a step with a KV cache: the step's key and
// value (from kvSrc) output as kv_k_NAME_new / kv_v_NAME_new, concatenated after the cached ones
// (kv_k_NAME, kv_v_NAME, (1, heads, cache, 128)).
func (g *lgraph) cacheAttention(x, kvSrc, mask *mlpackage.Var, p, attn, tag string, heads, cache int, sink bool) *mlpackage.Var {
	dim := x.Shape[2]
	h := g.rms(x, p+attn+"/pre_norm/scale")
	src := h
	if kvSrc != nil {
		src = kvSrc
	}
	a := p + attn + "/attention"
	q := g.heads(h, a+"/query_projection", heads)
	k := g.heads(src, a+"/key_projection", heads)
	v := g.heads(src, a+"/value_projection", heads)
	g.output(k, "kv_k_"+tag+"_new")
	g.output(v, "kv_v_"+tag+"_new")
	ck := g.input("kv_k_"+tag, 1, heads, cache, headDim)
	cv := g.input("kv_v_"+tag, 1, heads, cache, headDim)
	cat := func(c, n *mlpackage.Var) *mlpackage.Var {
		return g.Op("concat", map[string]any{"values": []*mlpackage.Var{c, n}, "axis": g.B.ScalarI32(2), "interleave": g.B.ScalarBool(false)}, 1, heads, cache+1, headDim)
	}
	sinkName := ""
	if sink {
		sinkName = a
	}
	o := g.attend(q, cat(ck, k), cat(cv, v), mask, g.queryScale(a+"/per_dim_scale"), sinkName)
	o = g.outProj(o, p+attn+"/output_projection", dim)
	return g.add(x, g.rms(o, p+attn+"/post_norm/scale"))
}

// Temporal is the temporal decoder's step: inputs x (1, 1, 1024), the mean of the previous
// frame's token embeddings; enc (1, 1, 256), the encoder's output of the frame; the caches of
// each layer's self-attention (kv_k_sN, kv_v_sN) and cross-attention (kv_k_cN, kv_v_cN) of
// Window frames each, with masks self_mask and cross_mask (1, 1, 1, Window + 1), 0 to see and
// -1e4 not (the cache's empty slots), the current frame last. Output y (1, 1, 1024) and the
// frame's keys and values.
type Temporal struct{}

// Depth is the depth decoder's step: inputs x (1, 1, 1024), the temporal output or the previous
// codebook's token embedding; the caches of its 2 layers (kv_k_dN, kv_v_dN) of 11 positions and
// mask (1, 1, 1, 12); output y (1, 1, 12294), the logits before the soft cap, and the step's keys
// and values.
type Depth struct{}

const depthCache = Codebooks - 1

// The llm.Model methods the programs do not use.
func (Temporal) Hidden() int               { return 0 }
func (Temporal) Layers() int               { return 0 }
func (Temporal) Output() (int, bool)       { return 0, false }
func (Temporal) Embedding() string         { return "" }
func (Temporal) Tensors() map[string][]int { return nil }
func (Depth) Hidden() int                  { return 0 }
func (Depth) Layers() int                  { return 0 }
func (Depth) Output() (int, bool)          { return 0, false }
func (Depth) Embedding() string            { return "" }
func (Depth) Tensors() map[string][]int    { return nil }

func buildLM(b *mlpackage.Builder, w llm.Weights, opts llm.Options, name string, body func(g *lgraph) *mlpackage.Var) (ins, outs []mlpackage.Feature, err error) {
	if opts.Format == "" {
		opts.Format = "fp16"
	}
	if opts.Group == 0 {
		opts.Group = 16
	}
	g := &lgraph{Graph: &blocks.Graph{Name: name, B: b, W: w, Opts: opts}}
	var y *mlpackage.Var
	err = blocks.Run(func() {
		y = body(g)
	})
	if err != nil {
		return nil, nil, err
	}
	outs = []mlpackage.Feature{{Name: "y", DType: mlpackage.Float16, Shape: y.Shape}}
	for _, v := range g.outs {
		outs = append(outs, mlpackage.Feature{Name: v.Name, DType: mlpackage.Float16, Shape: v.Shape})
	}
	return g.ins, outs, nil
}

// BuildCached writes the temporal step (llm.WritePart builds a Cached model with it).
func (Temporal) BuildCached(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins, outs []mlpackage.Feature, err error) {
	return buildLM(b, w, opts, "temporal", func(g *lgraph) *mlpackage.Var {
		x := g.input("x", 1, 1, tDim)
		enc := g.input("enc", 1, 1, encDim)
		sm := g.input("self_mask", 1, 1, 1, Window+1)
		cm := g.input("cross_mask", 1, 1, 1, Window+1)
		h := x
		for i := range tLayers {
			p := fmt.Sprintf("%sdecoder/temporal_body/transformer/x_layers_%d/", lmPrefix, i)
			h = g.cacheAttention(h, nil, sm, p, "self_attention", fmt.Sprintf("s%d", i), tHeads, Window, true)
			h = g.cacheAttention(h, enc, cm, p, "cross_attention", fmt.Sprintf("c%d", i), tHeads, Window, true)
			h = g.ffn(h, p, tFFN)
		}
		return g.B.NamedOp("identity", "y", map[string]any{"x": h}, mlpackage.Float16, h.Shape...)
	})
}

// BuildCached writes the depth step.
func (Depth) BuildCached(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins, outs []mlpackage.Feature, err error) {
	return buildLM(b, w, opts, "depth", func(g *lgraph) *mlpackage.Var {
		x := g.input("x", 1, 1, tDim)
		m := g.input("mask", 1, 1, 1, depthCache+1)
		const p0 = lmPrefix + "decoder/depth_body/"
		h := g.dense(x, p0+"depth_input_adapter", dDim, false)
		for i := range dLayers {
			p := fmt.Sprintf("%stransformer/x_layers_%d/", p0, i)
			h = g.cacheAttention(h, nil, m, p, "self_attention", fmt.Sprintf("d%d", i), dHeads, depthCache, false)
			h = g.ffn(h, p, dFFN)
		}
		h = g.layerNorm(h, p0+"final_ln")
		y := g.dense(h, p0+"to_logits", Vocab, true)
		return g.B.NamedOp("identity", "y", map[string]any{"x": y}, mlpackage.Float16, y.Shape...)
	})
}

// Program is a program with named inputs and outputs (WriteProgram).
type Program interface {
	BuildCached(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins, outs []mlpackage.Feature, err error)
}

// WriteProgram writes a program to a Core ML package at path, as a Core AI program with
// opts.Backend coreai (path its .aimodel) or as OpenVINO IR with intel (path its .xml).
func WriteProgram(path string, p Program, w llm.Weights, opts llm.Options) error {
	build := func(b *mlpackage.Builder) ([]mlpackage.Feature, []mlpackage.Feature, error) {
		return p.BuildCached(b, w, opts)
	}
	switch opts.Backend {
	case llm.BackendIntel:
		_, err := llm.WriteIR(path, map[string]string{llm.MetaArch: "magenta-realtime-2"}, build)
		return err
	case llm.BackendCoreAI:
		_, err := llm.WriteAI(path, map[string]string{llm.MetaArch: "magenta-realtime-2"}, build)
		return err
	}
	pkg, err := mlpackage.Create(path)
	if err != nil {
		return err
	}
	pkg.Metadata[llm.MetaArch] = "magenta-realtime-2"
	b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
	ins, outs, err := p.BuildCached(b, w, opts)
	if err != nil {
		return err
	}
	var vars []*mlpackage.Var
	for _, f := range outs {
		vars = append(vars, &mlpackage.Var{Name: f.Name})
	}
	return pkg.Finish(b.Finish(vars...), 9, ins, outs)
}

func (t Temporal) Build(b *mlpackage.Builder, w llm.Weights, opts llm.Options) ([]mlpackage.Feature, mlpackage.Feature, error) {
	ins, outs, err := t.BuildCached(b, w, opts)
	if err != nil {
		return nil, mlpackage.Feature{}, err
	}
	return ins, outs[0], nil
}

func (d Depth) Build(b *mlpackage.Builder, w llm.Weights, opts llm.Options) ([]mlpackage.Feature, mlpackage.Feature, error) {
	ins, outs, err := d.BuildCached(b, w, opts)
	if err != nil {
		return nil, mlpackage.Feature{}, err
	}
	return ins, outs[0], nil
}
