// Package mil describes SigLIP 2 (google/siglip2-*, the fixed-size checkpoints) as MIL programs,
// which hwnpu writes as Core ML packages for Apple's Neural Engine (llm.WritePart): the two
// graphs of tools/amd/iron_siglip2.py, the vision tower with its attention pooling head and
// the text tower on Labels texts a call.
//
// They run in fp16; every layer norm first divides each row by its largest magnitude (milpart's
// ScaledLayerNorm), so that no square overflows, and the tanh GELU is written with exp (the
// Neural Engine's own is inaccurate, design.md 16.145).
package mil

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu/internal/imagegen/milpart"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/safetensors"
)

// TextLen is a text's tokens: SigLIP's texts are padded to 64, as it was trained.
const TextLen = 64

// HeadRows is the rows the pooling head's query is repeated on: the Neural Engine leaves a part
// of one row to the CPU (design.md 16.85); row 0 is the picture's embedding.
const HeadRows = 8

// Part is one program (milpart.Part).
type Part = milpart.Part

type graph = milpart.Graph

func part(name string, body func(g *graph)) *Part {
	return &Part{Name: name, Model: "siglip", Body: body}
}

// Tower is a tower's shape.
type Tower struct {
	Hidden, MLP, Layers, Heads int
	Eps                        float64
}

// Config is a SigLIP 2's shape (iron_siglip2.Config).
type Config struct {
	Vision, Text Tower
	Image, Patch int
	Vocab        int
	Projection   int
	Labels       int // the texts a call of the text program takes
}

// Grid is the patches a side.
func (c Config) Grid() int { return c.Image / c.Patch }

// Patches is the picture's patches: the vision program's rows.
func (c Config) Patches() int { return c.Grid() * c.Grid() }

// PatchIn is a patch's values: 3 x Patch x Patch (channel, row, column).
func (c Config) PatchIn() int { return 3 * c.Patch * c.Patch }

// ReadConfig reads root/config.json (Transformers' SiglipConfig) for labels texts a call.
func ReadConfig(root string, labels int) (Config, error) {
	b, err := os.ReadFile(filepath.Join(root, "config.json"))
	if err != nil {
		return Config{}, err
	}
	type tower struct {
		Hidden   *int     `json:"hidden_size"`
		MLP      *int     `json:"intermediate_size"`
		Layers   *int     `json:"num_hidden_layers"`
		Heads    *int     `json:"num_attention_heads"`
		Eps      *float64 `json:"layer_norm_eps"`
		Act      *string  `json:"hidden_act"`
		Image    *int     `json:"image_size"`
		Patch    *int     `json:"patch_size"`
		Vocab    *int     `json:"vocab_size"`
		Position *int     `json:"max_position_embeddings"`
		Proj     *int     `json:"projection_size"`
	}
	var f struct {
		Type   string `json:"model_type"`
		Vision tower  `json:"vision_config"`
		Text   tower  `json:"text_config"`
	}
	if err := json.Unmarshal(b, &f); err != nil {
		return Config{}, fmt.Errorf("siglip: config.json: %w", err)
	}
	if f.Type != "siglip" {
		return Config{}, fmt.Errorf("siglip: %s: a SigLIP 2 of a fixed size (model_type siglip) is supported, not %q", root, f.Type)
	}
	or := func(p *int, d int) int {
		if p != nil {
			return *p
		}
		return d
	}
	of := func(name string, t tower) (Tower, error) {
		x := Tower{Hidden: or(t.Hidden, 768), MLP: or(t.MLP, 3072), Layers: or(t.Layers, 12), Heads: or(t.Heads, 12), Eps: 1e-6}
		if t.Eps != nil {
			x.Eps = *t.Eps
		}
		switch {
		case x.Hidden <= 0 || x.Heads <= 0 || x.Hidden%x.Heads != 0 || x.Layers <= 0 || x.MLP <= 0:
			return x, fmt.Errorf("siglip: %s: the %s tower's shape %+v", root, name, x)
		case t.Act != nil && *t.Act != "gelu_pytorch_tanh":
			return x, fmt.Errorf("siglip: %s: the %s tower's %s: only gelu_pytorch_tanh", root, name, *t.Act)
		}
		return x, nil
	}
	c := Config{Image: or(f.Vision.Image, 224), Patch: or(f.Vision.Patch, 16), Vocab: or(f.Text.Vocab, 32000), Labels: labels}
	if c.Vision, err = of("vision", f.Vision); err != nil {
		return c, err
	}
	if c.Text, err = of("text", f.Text); err != nil {
		return c, err
	}
	c.Projection = or(f.Text.Proj, c.Text.Hidden)
	switch {
	case or(f.Text.Position, TextLen) != TextLen:
		return c, fmt.Errorf("siglip: %s: texts of %d tokens: only %d", root, *f.Text.Position, TextLen)
	case c.Patch <= 0 || c.Image <= 0 || c.Image%c.Patch != 0:
		return c, fmt.Errorf("siglip: %s: pictures of %d in patches of %d", root, c.Image, c.Patch)
	case c.Projection != c.Vision.Hidden:
		return c, fmt.Errorf("siglip: %s: text embeddings of %d and picture embeddings of %d", root, c.Projection, c.Vision.Hidden)
	case labels <= 0:
		return c, fmt.Errorf("siglip: %d texts a call", labels)
	}
	return c, nil
}

// Weights reads a SigLIP 2's model.safetensors.
type Weights struct {
	f    *safetensors.File
	path string
}

// OpenWeights opens root/model.safetensors.
func OpenWeights(root string) (*Weights, error) {
	p := filepath.Join(root, "model.safetensors")
	f, err := safetensors.Open(p)
	if err != nil {
		return nil, err
	}
	return &Weights{f: f, path: p}, nil
}

// Get returns a tensor as float32 and its shape.
func (w *Weights) Get(name string) ([]float32, []int, error) {
	t, ok := w.f.Lookup(name)
	if !ok {
		return nil, nil, fmt.Errorf("siglip: %s has no tensor %s", w.path, name)
	}
	v, err := w.f.Float32(name)
	return v, t.Shape, err
}

// Rows returns rows [begin, end) of a matrix.
func (w *Weights) Rows(name string, begin, end int) ([]float32, error) {
	t, ok := w.f.Lookup(name)
	if !ok {
		return nil, fmt.Errorf("siglip: %s has no tensor %s", w.path, name)
	}
	cols := 1
	for _, d := range t.Shape[1:] {
		cols *= d
	}
	return w.f.Float32Range(name, begin*cols, end*cols)
}

// Source names the checkpoint.
func (w *Weights) Source() string { return "safetensors " + filepath.Base(w.path) }

// Close closes the file.
func (w *Weights) Close() error { return w.f.Close() }

// EmbedName is the token embedding table (Vocab x Text.Hidden) the host looks up.
const EmbedName = "text_model.embeddings.token_embedding.weight"

// linear is the named linear (weight and bias) of x.
func linear(g *graph, x *mlpackage.Var, p string, out int) *mlpackage.Var {
	in := x.Shape[len(x.Shape)-1]
	return g.LinearConst(x, p, g.Get(p+".weight", out, in), out, g.Get(p+".bias", out))
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

// geluTanh is the tanh approximation of GELU as x sigmoid(2 sqrt(2/pi) (x + 0.044715 x^3)),
// with exp and real_div (sana/mil's): past x = -11 or so exp overflows fp16 to inf and the
// quotient is 0, as gelu is there.
func geluTanh(g *graph, x *mlpackage.Var) *mlpackage.Var {
	a := float32(-2 * math.Sqrt(2/math.Pi))
	u := g.Mul(x, g.AddScalar(g.Scale(g.Mul(x, x), a*0.044715), a))
	return g.Div(x, g.AddScalar(g.Unary("exp", u), 1))
}

// mlp is fc1, the tanh GELU and fc2 under p.
func mlp(g *graph, x *mlpackage.Var, p string, t Tower) *mlpackage.Var {
	return linear(g, geluTanh(g, linear(g, x, p+"fc1", t.MLP)), p+"fc2", t.Hidden)
}

// heads is x (b, n, heads d) as (1, b heads, n, d): the b sequences' heads side by side, so that
// one attention of b x heads heads covers them (each over its own sequence's n tokens).
func heads(g *graph, x *mlpackage.Var, h int) *mlpackage.Var {
	b, n, d := x.Shape[0], x.Shape[1], x.Shape[2]/h
	return g.Reshape(g.Transpose(g.Reshape(x, b, n, h, d), 0, 2, 1, 3), 1, b*h, n, d)
}

// merge is heads' inverse on o (1, b heads, n, d): (b, n, heads d).
func merge(g *graph, o *mlpackage.Var, b int) *mlpackage.Var {
	h, n, d := o.Shape[1]/b, o.Shape[2], o.Shape[3]
	return g.Reshape(g.Transpose(g.Reshape(o, b, h, n, d), 0, 2, 1, 3), b, n, h*d)
}

// block is an encoder layer under p on x (b, n, hidden): layer norm, attention, layer norm, MLP.
func block(g *graph, t Tower, p string, x *mlpackage.Var) *mlpackage.Var {
	E, D := t.Hidden, t.Hidden/t.Heads
	h := layerNorm(g, x, p+"layer_norm1", t.Eps)
	proj := func(n string) *mlpackage.Var { return heads(g, linear(g, h, p+"self_attn."+n, E), t.Heads) }
	o := g.SDPAScale(proj("q_proj"), proj("k_proj"), proj("v_proj"), 1/math.Sqrt(float64(D)), nil)
	x = g.Add(x, linear(g, merge(g, o, x.Shape[0]), p+"self_attn.out_proj", E))
	return g.Add(x, mlp(g, layerNorm(g, x, p+"layer_norm2", t.Eps), p+"mlp.", t))
}

// Vision is the vision tower: input patches (1, Patches, PatchIn), each patch's 3 x P x P
// values scaled to [-1, 1] (channel, row, column); output (1, HeadRows, hidden), every row the
// picture's embedding before normalizing (the pooling head's probe as the query of each).
func Vision(c Config) *Part {
	return part("vision", func(g *graph) {
		v := c.Vision
		E, H, D, N := v.Hidden, v.Heads, v.Hidden/v.Heads, c.Patches()
		const p = "vision_model."
		x := g.Input("patches", 1, N, c.PatchIn())
		pw := g.Get(p+"embeddings.patch_embedding.weight", E, 3, c.Patch, c.Patch)
		pos := g.Get(p+"embeddings.position_embedding.weight", N, E)
		bias := g.Get(p+"embeddings.patch_embedding.bias", E)
		for i := range pos {
			pos[i] += bias[i%E]
		}
		x = g.Add(g.LinearConst(x, p+"embeddings.patch_embedding", pw, E, nil), g.F16s(p+"embeddings.position", []int{1, N, E}, pos))
		for i := 0; i < v.Layers; i++ {
			x = block(g, v, fmt.Sprintf("%sencoder.layers.%d.", p, i), x)
		}
		x = layerNorm(g, x, p+"post_layernorm", v.Eps)
		// the pooling head: torch's MultiheadAttention (in_proj q, k, v) with the probe as its
		// query, whose projection is a constant
		const h = p + "head."
		ip, ipb := g.Get(h+"attention.in_proj_weight", 3*E, E), g.Get(h+"attention.in_proj_bias", 3*E)
		probe := g.Get(h + "probe")
		if len(probe) != E {
			g.Fail("%sprobe: %d values for a hidden of %d", h, len(probe), E)
		}
		q := make([]float32, HeadRows*E) // (rows, heads, d) as (1, heads, rows, d)
		for j := 0; j < E; j++ {
			s := float64(ipb[j])
			for k := 0; k < E; k++ {
				s += float64(probe[k]) * float64(ip[j*E+k])
			}
			for r := 0; r < HeadRows; r++ {
				q[((j/D)*HeadRows+r)*D+j%D] = float32(s)
			}
		}
		k := heads(g, g.LinearConst(x, h+"attention.k", ip[E*E:2*E*E], E, ipb[E:2*E]), H)
		vv := heads(g, g.LinearConst(x, h+"attention.v", ip[2*E*E:], E, ipb[2*E:]), H)
		o := g.SDPAScale(g.F16s(h+"attention.q", []int{1, H, HeadRows, D}, q), k, vv, 1/math.Sqrt(float64(D)), nil)
		y := linear(g, merge(g, o, 1), h+"attention.out_proj", E)
		g.Output(g.Add(y, mlp(g, layerNorm(g, y, h+"layernorm", v.Eps), h+"mlp.", v)))
	})
}

// Text is the text tower on c.Labels texts: input x (Labels, 64, hidden), the token embedding
// rows of each text (its tokens, <eos>, then <pad> to 64; the positions are added here); output
// (Labels, 1, Projection), each text's embedding before normalizing: the head of the last
// token's row after the last layer norm.
func Text(c Config) *Part {
	return part("text", func(g *graph) {
		t := c.Text
		E, L := t.Hidden, c.Labels
		const p = "text_model."
		x := g.Input("x", L, TextLen, E)
		x = g.Add(x, g.F16s(p+"embeddings.position", []int{1, TextLen, E}, g.Get(p+"embeddings.position_embedding.weight", TextLen, E)))
		for i := 0; i < t.Layers; i++ {
			x = block(g, t, fmt.Sprintf("%sencoder.layers.%d.", p, i), x)
		}
		x = layerNorm(g, g.Slice(x, 1, TextLen-1, TextLen), p+"final_layer_norm", t.Eps)
		g.Output(linear(g, x, p+"head", c.Projection))
	})
}
