package mil

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/safetensors"
)

// A SigLIP 2 of random weights, small: two layers a tower, heads of 16.
var tiny = Config{
	Vision: Tower{Hidden: 32, MLP: 64, Layers: 2, Heads: 2, Eps: 1e-6},
	Text:   Tower{Hidden: 48, MLP: 96, Layers: 2, Heads: 3, Eps: 1e-6},
	Image:  32, Patch: 16, Vocab: 16, Projection: 32, Labels: 3,
}

// ref is the towers in float64, written from Transformers' modeling_siglip.py on the
// checkpoint's own tensors: nothing of the programs' (no scaled norms, tanh in the GELU, the
// probe's query computed with the others, a text at a time).
type ref struct{ w map[string][]float64 }

type mat struct {
	r, c int
	v    []float64
}

func (m mat) row(i int) []float64 { return m.v[i*m.c : (i+1)*m.c] }

func (f ref) linear(x mat, wt, bias []float64, out int) mat {
	y := mat{x.r, out, make([]float64, x.r*out)}
	for i := 0; i < x.r; i++ {
		for o := 0; o < out; o++ {
			s := bias[o]
			for k, v := range x.row(i) {
				s += v * wt[o*x.c+k]
			}
			y.v[i*out+o] = s
		}
	}
	return y
}

func (f ref) lin(x mat, p string, out int) mat {
	return f.linear(x, f.w[p+".weight"], f.w[p+".bias"], out)
}

func (f ref) norm(x mat, p string, eps float64) mat {
	g, b := f.w[p+".weight"], f.w[p+".bias"]
	y := mat{x.r, x.c, make([]float64, len(x.v))}
	for i := 0; i < x.r; i++ {
		var mean, vr float64
		for _, v := range x.row(i) {
			mean += v
		}
		mean /= float64(x.c)
		for _, v := range x.row(i) {
			vr += (v - mean) * (v - mean)
		}
		vr /= float64(x.c)
		for j, v := range x.row(i) {
			y.v[i*x.c+j] = (v-mean)/math.Sqrt(vr+eps)*g[j] + b[j]
		}
	}
	return y
}

// attend is scaled dot product attention of heads heads: q (n, E) over k and v (m, E).
func attend(q, k, v mat, heads int) mat {
	d := q.c / heads
	o := mat{q.r, q.c, make([]float64, q.r*q.c)}
	for h := 0; h < heads; h++ {
		for i := 0; i < q.r; i++ {
			sc := make([]float64, k.r)
			mx := math.Inf(-1)
			for j := 0; j < k.r; j++ {
				for e := 0; e < d; e++ {
					sc[j] += q.v[i*q.c+h*d+e] * k.v[j*k.c+h*d+e]
				}
				sc[j] /= math.Sqrt(float64(d))
				mx = max(mx, sc[j])
			}
			var sum float64
			for j := range sc {
				sc[j] = math.Exp(sc[j] - mx)
				sum += sc[j]
			}
			for j := range sc {
				for e := 0; e < d; e++ {
					o.v[i*q.c+h*d+e] += sc[j] / sum * v.v[j*v.c+h*d+e]
				}
			}
		}
	}
	return o
}

func gelu(x mat) mat {
	y := mat{x.r, x.c, make([]float64, len(x.v))}
	for i, v := range x.v {
		y.v[i] = 0.5 * v * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(v+0.044715*v*v*v)))
	}
	return y
}

func add(a, b mat) mat {
	y := mat{a.r, a.c, make([]float64, len(a.v))}
	for i := range y.v {
		y.v[i] = a.v[i] + b.v[i]
	}
	return y
}

func (f ref) mlp(x mat, p string, t Tower) mat {
	return f.lin(gelu(f.lin(x, p+"fc1", t.MLP)), p+"fc2", t.Hidden)
}

func (f ref) encoder(x mat, p string, t Tower) mat {
	for i := 0; i < t.Layers; i++ {
		l := fmt.Sprintf("%sencoder.layers.%d.", p, i)
		h := f.norm(x, l+"layer_norm1", t.Eps)
		o := attend(f.lin(h, l+"self_attn.q_proj", t.Hidden), f.lin(h, l+"self_attn.k_proj", t.Hidden), f.lin(h, l+"self_attn.v_proj", t.Hidden), t.Heads)
		x = add(x, f.lin(o, l+"self_attn.out_proj", t.Hidden))
		x = add(x, f.mlp(f.norm(x, l+"layer_norm2", t.Eps), l+"mlp.", t))
	}
	return x
}

// vision is the pooled output of patches (Patches x PatchIn).
func (f ref) vision(c Config, patches []float64) []float64 {
	v, E := c.Vision, c.Vision.Hidden
	const p = "vision_model."
	x := f.linear(mat{c.Patches(), c.PatchIn(), patches}, f.w[p+"embeddings.patch_embedding.weight"], f.w[p+"embeddings.patch_embedding.bias"], E)
	x = add(x, mat{x.r, E, f.w[p+"embeddings.position_embedding.weight"]})
	x = f.norm(f.encoder(x, p, v), p+"post_layernorm", v.Eps)
	const h = p + "head."
	ip, ipb := f.w[h+"attention.in_proj_weight"], f.w[h+"attention.in_proj_bias"]
	q := f.linear(mat{1, E, f.w[h+"probe"]}, ip[:E*E], ipb[:E], E)
	o := attend(q, f.linear(x, ip[E*E:2*E*E], ipb[E:2*E], E), f.linear(x, ip[2*E*E:], ipb[2*E:], E), v.Heads)
	y := f.lin(o, h+"attention.out_proj", E)
	return add(y, f.mlp(f.norm(y, h+"layernorm", v.Eps), h+"mlp.", v)).v
}

// text is the pooled output of one text's embedding rows (64 x hidden).
func (f ref) text(c Config, rows []float64) []float64 {
	t, E := c.Text, c.Text.Hidden
	const p = "text_model."
	x := add(mat{TextLen, E, rows}, mat{TextLen, E, f.w[p+"embeddings.position_embedding.weight"]})
	x = f.norm(f.encoder(x, p, t), p+"final_layer_norm", t.Eps)
	return f.lin(mat{1, E, x.row(TextLen - 1)}, p+"head", c.Projection).v
}

// checkpoint writes a model.safetensors of random weights for c into dir and returns them.
func checkpoint(t *testing.T, dir string, c Config, rng *rand.Rand) ref {
	t.Helper()
	f := ref{map[string][]float64{}}
	var outs []safetensors.Out
	put := func(name string, scale float64, shape ...int) {
		n := 1
		for _, d := range shape {
			n *= d
		}
		raw := make([]byte, 4*n)
		v := make([]float64, n)
		for i := range v {
			x := float32(rng.NormFloat64() * scale)
			v[i] = float64(x)
			binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(x))
		}
		f.w[name] = v
		outs = append(outs, safetensors.Out{Name: name, DType: "F32", Shape: shape, Size: int64(len(raw)), Data: bytes.NewReader(raw)})
	}
	lin := func(p string, out, in int) {
		put(p+".weight", 1/math.Sqrt(float64(in)), out, in)
		put(p+".bias", 0.1, out)
	}
	norm := func(p string, n int) {
		put(p+".weight", 0.2, n) // around 0: both signs
		put(p+".bias", 0.1, n)
	}
	mlp := func(p string, tw Tower) {
		lin(p+"fc1", tw.MLP, tw.Hidden)
		lin(p+"fc2", tw.Hidden, tw.MLP)
	}
	encoder := func(p string, tw Tower) {
		for i := 0; i < tw.Layers; i++ {
			l := fmt.Sprintf("%sencoder.layers.%d.", p, i)
			norm(l+"layer_norm1", tw.Hidden)
			norm(l+"layer_norm2", tw.Hidden)
			for _, n := range []string{"q_proj", "k_proj", "v_proj", "out_proj"} {
				lin(l+"self_attn."+n, tw.Hidden, tw.Hidden)
			}
			mlp(l+"mlp.", tw)
		}
	}
	v, tx := c.Vision, c.Text
	put("vision_model.embeddings.patch_embedding.weight", 1/math.Sqrt(float64(c.PatchIn())), v.Hidden, 3, c.Patch, c.Patch)
	put("vision_model.embeddings.patch_embedding.bias", 0.1, v.Hidden)
	put("vision_model.embeddings.position_embedding.weight", 0.5, c.Patches(), v.Hidden)
	encoder("vision_model.", v)
	norm("vision_model.post_layernorm", v.Hidden)
	put("vision_model.head.probe", 1, 1, 1, v.Hidden)
	put("vision_model.head.attention.in_proj_weight", 1/math.Sqrt(float64(v.Hidden)), 3*v.Hidden, v.Hidden)
	put("vision_model.head.attention.in_proj_bias", 0.1, 3*v.Hidden)
	lin("vision_model.head.attention.out_proj", v.Hidden, v.Hidden)
	norm("vision_model.head.layernorm", v.Hidden)
	mlp("vision_model.head.mlp.", v)
	put(EmbedName, 1, c.Vocab, tx.Hidden)
	put("text_model.embeddings.position_embedding.weight", 0.5, TextLen, tx.Hidden)
	encoder("text_model.", tx)
	norm("text_model.final_layer_norm", tx.Hidden)
	lin("text_model.head", c.Projection, tx.Hidden)
	if err := safetensors.Write(filepath.Join(dir, "model.safetensors"), outs, nil); err != nil {
		t.Fatal(err)
	}
	return f
}

// The programs of a random SigLIP 2, run by the CPU interpreter, against the float64 reference:
// the picture's embedding on every row of the head, and each text's of a call.
func TestTowers(t *testing.T) {
	dir := t.TempDir()
	rng := rand.New(rand.NewSource(1))
	c := tiny
	f := checkpoint(t, dir, c, rng)
	wt, err := OpenWeights(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer wt.Close()
	run := func(p *Part, in map[string][]float32) []float32 {
		path := filepath.Join(dir, p.Name+".mlpackage")
		if _, err := llm.WritePart(path, p, wt, llm.Options{Backend: llm.BackendCoreML, Format: "fp16"}, nil); err != nil {
			t.Fatal(err)
		}
		prog, err := runner.Open(t.Context(), nil, path)
		if err != nil {
			t.Fatal(err)
		}
		defer prog.Close()
		y, err := prog.Run(t.Context(), in)
		if err != nil {
			t.Fatal(err)
		}
		return y
	}
	random := func(n int) ([]float32, []float64) {
		a, b := make([]float32, n), make([]float64, n)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
			b[i] = float64(a[i])
		}
		return a, b
	}
	check := func(what string, got []float32, want []float64) {
		t.Helper()
		var dot, ng, nw, worst float64
		for i, w := range want {
			g := float64(got[i])
			dot, ng, nw = dot+g*w, ng+g*g, nw+w*w
			worst = max(worst, math.Abs(g-w))
		}
		// fp16 weights and activations against float64
		if cos := dot / math.Sqrt(ng*nw); !(cos > 0.99999) || worst > 0.02*math.Sqrt(nw/float64(len(want))) {
			t.Errorf("%s: cosine %.6f, off by up to %.4g (rms %.4g)", what, cos, worst, math.Sqrt(nw/float64(len(want))))
		}
	}

	p32, p64 := random(c.Patches() * c.PatchIn())
	y := run(Vision(c), map[string][]float32{"patches": p32})
	E := c.Vision.Hidden
	if len(y) != HeadRows*E {
		t.Fatalf("vision: %d values, want %d rows of %d", len(y), HeadRows, E)
	}
	want := f.vision(c, p64)
	for r := 0; r < HeadRows; r++ {
		check(fmt.Sprintf("vision row %d", r), y[r*E:(r+1)*E], want)
	}

	x32, x64 := random(c.Labels * TextLen * c.Text.Hidden)
	y = run(Text(c), map[string][]float32{"x": x32})
	if len(y) != c.Labels*c.Projection {
		t.Fatalf("text: %d values, want %d embeddings of %d", len(y), c.Labels, c.Projection)
	}
	n := TextLen * c.Text.Hidden
	for i := 0; i < c.Labels; i++ {
		check(fmt.Sprintf("text %d", i), y[i*c.Projection:(i+1)*c.Projection], f.text(c, x64[i*n:(i+1)*n]))
	}
}

func TestReadConfig(t *testing.T) {
	dir := t.TempDir()
	write := func(s string) {
		if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(s), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// siglip2-base-patch16-224's: all defaults but the vocabulary
	write(`{"model_type": "siglip", "text_config": {"vocab_size": 256000}, "vision_config": {}}`)
	c, err := ReadConfig(dir, 8)
	if err != nil {
		t.Fatal(err)
	}
	want := Config{Vision: Tower{768, 3072, 12, 12, 1e-6}, Text: Tower{768, 3072, 12, 12, 1e-6}, Image: 224, Patch: 16, Vocab: 256000, Projection: 768, Labels: 8}
	if c != want || c.Patches() != 196 || c.PatchIn() != 768 {
		t.Errorf("%+v, want %+v", c, want)
	}
	// siglip2-large-patch16-256's
	write(`{"model_type": "siglip", "text_config": {"hidden_size": 1024, "intermediate_size": 4096, "num_attention_heads": 16, "num_hidden_layers": 24, "projection_size": 1024, "vocab_size": 256000},
		"vision_config": {"hidden_size": 1024, "image_size": 256, "intermediate_size": 4096, "num_attention_heads": 16, "num_hidden_layers": 24, "patch_size": 16}}`)
	if c, err = ReadConfig(dir, 8); err != nil || c.Vision.Hidden != 1024 || c.Text.Layers != 24 || c.Patches() != 256 || c.Projection != 1024 {
		t.Errorf("large: %+v, %v", c, err)
	}
	for _, bad := range []string{
		`{"model_type": "siglip2"}`, // NaFlex: no fixed size
		`{"model_type": "siglip", "text_config": {"max_position_embeddings": 77}}`,
		`{"model_type": "siglip", "vision_config": {"hidden_act": "quick_gelu"}}`,
		`{"model_type": "siglip", "vision_config": {"image_size": 230}}`,
	} {
		write(bad)
		if _, err := ReadConfig(dir, 8); err == nil {
			t.Errorf("%s: no error", bad)
		}
	}
}
