package plamo

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/ovcpu"
	"github.com/shibukawa/hwnpu/internal/palette"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// tiny is a small PlamoBiModel with multi query attention; matrix rows are multiples of the lut4
// group of 16, also for the parts of the fused projections.
const tiny = `{"architectures": ["PlamoBiModel"], "model_type": "plamo", "vocab_size": 40, "hidden_size": 64,
	"num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 1, "hidden_size_per_head": 16,
	"intermediate_size": 48, "max_position_embeddings": 16, "rms_norm_eps": 1e-6, "n_expert": null,
	"sparse_step": null, "linear_type": "fp8"}`

// weights are synthetic weights with the norm weights around 1, as trained ones are: with the
// synthetic values of a tenth or less, the queries and keys would be so small that attention is
// nearly uniform and a wrong position or mask would hardly show.
type weights struct{ llm.Synthetic }

func normWeight(name string) bool {
	for _, s := range []string{"norm.weight", "norm2.weight", "q_weight", "k_weight"} {
		if strings.HasSuffix(name, s) {
			return true
		}
	}
	return false
}

func (w weights) Get(name string) ([]float32, []int, error) {
	v, shape, err := w.Synthetic.Get(name)
	if err == nil && normWeight(name) {
		for i := range v {
			v[i] = 1 + 8*v[i]
		}
	}
	return v, shape, err
}

func (w weights) Rows(name string, begin, end int) ([]float32, error) {
	v, shape, err := w.Get(name)
	if err != nil {
		return nil, err
	}
	return v[begin*shape[1] : end*shape[1]], nil
}

func newWeights(c *Config) weights { return weights{llm.NewSynthetic(c)} }

// f16 reads a tensor rounded to fp16, as the program stores it.
func f16(t *testing.T, w llm.Weights, name string) []float64 {
	v, _, err := w.Get(name)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(mlpackage.Float16Value(mlpackage.Float16Bits(x)))
	}
	return out
}

// r16 rounds to fp16.
func r16(x float64) float64 {
	return float64(mlpackage.Float16Value(mlpackage.Float16Bits(float32(x))))
}

// embed is the program's input x for the tokens: the embedding rows in fp16, as the host passes them.
func embed(t *testing.T, c *Config, w llm.Weights, ids []int32) []float32 {
	x, err := llm.Embed(w, c, ids)
	if err != nil {
		t.Fatal(err)
	}
	for i := range x {
		x[i] = float32(r16(float64(x[i])))
	}
	return x
}

// reference is PlamoBiModel's forward written out directly from modeling_plamo.py, in float64:
// an oracle independent of the program builder and the interpreter. mask is the attention mask
// (1 token, 0 padding); padded keys are left out of the softmax, as the finfo.min of
// _prepare_4d_causal_attention_mask_for_sdpa does.
func reference(t *testing.T, c *Config, w llm.Weights, ids []int32, mask []float64) []float64 {
	d, s, nh, nkv, hd, f := c.HiddenSize, len(ids), c.NumHeads, c.NumKVHeads, c.HeadDim, c.IntermediateSize
	rmsNorm := func(x []float64, n int, g []float64, eps float64) {
		for r := 0; r < len(x)/n; r++ {
			row := x[r*n : (r+1)*n]
			var ms float64
			for _, a := range row {
				ms += a * a
			}
			inv := 1 / math.Sqrt(ms/float64(n)+eps)
			for j := range row {
				row[j] *= inv
				if g != nil {
					row[j] *= g[j]
				}
			}
		}
	}
	linear := func(x []float64, name string, in, out int) []float64 {
		wt := f16(t, w, name)
		y := make([]float64, s*out)
		for r := 0; r < s; r++ {
			for o := 0; o < out; o++ {
				var acc float64
				for k := 0; k < in; k++ {
					acc += x[r*in+k] * wt[o*in+k]
				}
				y[r*out+o] = acc
			}
		}
		return y
	}
	// RotaryEmbedding: cos and sin of cat(freqs, freqs), and x*cos + rotate_half(x)*sin. The
	// tables are rounded to fp16 like the weights: the program holds them as constants.
	rope := func(x []float64, pos int) {
		half := hd / 2
		out := make([]float64, hd)
		for j := 0; j < hd; j++ {
			a := float64(pos) / math.Pow(10000, float64(2*(j%half))/float64(hd))
			// rotate_half: cat(-x2, x1)
			var rot float64
			if j < half {
				rot = -x[j+half]
			} else {
				rot = x[j-half]
			}
			out[j] = x[j]*r16(math.Cos(a)) + rot*r16(math.Sin(a))
		}
		copy(x, out)
	}
	emb := f16(t, w, c.Embedding())
	h := make([]float64, s*d)
	for r, id := range ids {
		copy(h[r*d:(r+1)*d], emb[int(id)*d:(int(id)+1)*d])
	}
	for i := 0; i < c.NumLayers; i++ {
		p := "layers.layers." + string(rune('0'+i)) + "."
		x := append([]float64(nil), h...)
		rmsNorm(x, d, f16(t, w, p+"norm.weight"), c.NormEps)
		qkv := linear(x, p+"self_attn.qkv_proj.weight", d, (nh+2*nkv)*hd)
		width := (nh + 2*nkv) * hd
		qw, kw := f16(t, w, p+"self_attn.q_weight"), f16(t, w, p+"self_attn.k_weight")
		// q[r][head][j], k and v [r][kv head][j]
		q := make([]float64, s*nh*hd)
		k := make([]float64, s*nkv*hd)
		v := make([]float64, s*nkv*hd)
		for r := 0; r < s; r++ {
			copy(q[r*nh*hd:(r+1)*nh*hd], qkv[r*width:r*width+nh*hd])
			copy(k[r*nkv*hd:(r+1)*nkv*hd], qkv[r*width+nh*hd:r*width+(nh+nkv)*hd])
			copy(v[r*nkv*hd:(r+1)*nkv*hd], qkv[r*width+(nh+nkv)*hd:(r+1)*width])
			for head := 0; head < nh; head++ {
				x := q[(r*nh+head)*hd : (r*nh+head+1)*hd]
				rmsNorm(x, hd, qw[head*hd:(head+1)*hd], 1e-6)
				rope(x, r)
			}
			for head := 0; head < nkv; head++ {
				x := k[(r*nkv+head)*hd : (r*nkv+head+1)*hd]
				rmsNorm(x, hd, kw[head*hd:(head+1)*hd], 1e-6)
				rope(x, r)
			}
		}
		ctx := make([]float64, s*nh*hd)
		for head := 0; head < nh; head++ {
			kh := head / (nh / nkv) // repeat_interleave
			for a := 0; a < s; a++ {
				scores := make([]float64, s)
				m := math.Inf(-1)
				for b := 0; b < s; b++ {
					if mask[b] == 0 {
						continue
					}
					var dot float64
					for j := 0; j < hd; j++ {
						dot += q[(a*nh+head)*hd+j] * k[(b*nkv+kh)*hd+j]
					}
					scores[b] = dot / math.Sqrt(float64(hd))
					m = math.Max(m, scores[b])
				}
				var z float64
				for b := range scores {
					if mask[b] == 0 {
						scores[b] = 0
						continue
					}
					scores[b] = math.Exp(scores[b] - m)
					z += scores[b]
				}
				for b := range scores {
					for j := 0; j < hd; j++ {
						ctx[(a*nh+head)*hd+j] += scores[b] / z * v[(b*nkv+kh)*hd+j]
					}
				}
			}
		}
		o := linear(ctx, p+"self_attn.o_proj.weight", nh*hd, d)
		for j := range h {
			h[j] += o[j]
		}
		x = append(x[:0], h...)
		rmsNorm(x, d, f16(t, w, p+"norm2.weight"), c.NormEps)
		gu := linear(x, p+"mlp.gate_up_proj.weight", d, 2*f)
		act := make([]float64, s*f)
		for r := 0; r < s; r++ {
			for j := 0; j < f; j++ {
				g, u := gu[r*2*f+j], gu[r*2*f+f+j]
				act[r*f+j] = g / (1 + math.Exp(-g)) * u
			}
		}
		down := linear(act, p+"mlp.down_proj.weight", f, d)
		for j := range h {
			h[j] += down[j]
		}
	}
	rmsNorm(h, d, f16(t, w, "norm.weight"), c.NormEps)
	return h
}

// run builds the program and runs it on the CPU.
func run(t *testing.T, c *Config, w llm.Weights, opts llm.Options, x, mask []float32) []float32 {
	t.Helper()
	path := filepath.Join(t.TempDir(), "weight.bin")
	blobs, err := mlpackage.NewBlobWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	b := mlpackage.NewBuilder("CoreML8", blobs)
	ins, out, err := c.Build(b, w, opts)
	if err != nil {
		t.Fatal(err)
	}
	prog := b.Finish(&mlpackage.Var{Name: out.Name})
	if err := blobs.Close(); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	p, err := milcpu.Load(prog, f)
	if err != nil {
		t.Fatal(err)
	}
	res, err := p.Run(map[string]milcpu.Tensor{
		ins[0].Name: {Shape: ins[0].Shape, Data: x},
		ins[1].Name: {Shape: ins[1].Shape, Data: mask},
	}, milcpu.Options{})
	if err != nil {
		t.Fatal(err)
	}
	return res[out.Name].Data
}

func relErr(got []float32, ref []float64) float64 {
	var se, sr float64
	for i := range ref {
		d := float64(got[i]) - ref[i]
		se += d * d
		sr += ref[i] * ref[i]
	}
	return math.Sqrt(se / sr)
}

// The program computes PlamoBiModel's last hidden state with and without padding, for multi
// query and grouped query attention; layer by layer, as hwnpu-check runs it, gives the same.
func TestAgainstReference(t *testing.T) {
	ids := []int32{1, 17, 5, 33, 9, 0, 21}
	for _, c := range []struct {
		name, edit string
		mask       []float64
	}{
		{"no padding", "", []float64{1, 1, 1, 1, 1, 1, 1}},
		{"padding", "", []float64{1, 1, 1, 1, 1, 0, 0}},
		{"left padding", "", []float64{0, 0, 1, 1, 1, 1, 1}},
		{"grouped query", `"num_key_value_heads": 1→"num_key_value_heads": 2`, []float64{1, 1, 1, 1, 1, 1, 0}},
	} {
		t.Run(c.name, func(t *testing.T) {
			js := tiny
			if c.edit != "" {
				from, to, _ := strings.Cut(c.edit, "→")
				js = strings.Replace(js, from, to, 1)
			}
			cfg, err := ParseConfig([]byte(js))
			if err != nil {
				t.Fatal(err)
			}
			w := newWeights(cfg)
			x := embed(t, cfg, w, ids)
			mask := make([]float32, len(c.mask))
			for i, m := range c.mask {
				mask[i] = float32(m)
			}
			want := reference(t, cfg, w, ids, c.mask)
			got := run(t, cfg, w, llm.Options{Seq: len(ids), Format: "fp16"}, x, mask)
			if e := relErr(got, want); e > 1e-4 {
				t.Errorf("relative error %g against the reference", e)
			}
			// one layer at a time, then the final norm alone
			h := x
			for i := 0; i < cfg.NumLayers; i++ {
				h = run(t, cfg, w, llm.Options{Seq: len(ids), Format: "fp16", Start: i, Layers: 1, NoHead: true}, h, mask)
			}
			h = run(t, cfg, w, llm.Options{Seq: len(ids), Format: "fp16", Start: cfg.NumLayers}, h, mask)
			if e := relErr(h, want); e > 1e-4 {
				t.Errorf("layer by layer: relative error %g against the reference", e)
			}
		})
	}
}

// quantized are weights with every linear layer's matrix 4 bit palettized as a whole, the way
// blocks.Weight does with Options' defaults (uniform tables of 16 rows).
type quantized struct{ weights }

func (w quantized) Get(name string) ([]float32, []int, error) {
	v, shape, err := w.weights.Get(name)
	if err == nil && len(shape) == 2 && strings.HasSuffix(name, ".weight") && name != "embed_tokens.weight" {
		v = palette.Quantize(v, shape[0], shape[1], 16, 0, 8, palette.Uniform).Dequantize()
	}
	return v, shape, err
}

// The lut4 program quantizes the parts of the fused projections (qkv_proj, gate_up_proj) as the
// matrices they are part of would be: it matches the reference with every matrix quantized whole.
func TestLUT4(t *testing.T) {
	cfg, err := ParseConfig([]byte(tiny))
	if err != nil {
		t.Fatal(err)
	}
	ids := []int32{1, 17, 5, 33, 9, 0}
	mask := []float64{1, 1, 1, 1, 0, 0}
	w := newWeights(cfg)
	got := run(t, cfg, w, llm.Options{Seq: len(ids)}, embed(t, cfg, w, ids), []float32{1, 1, 1, 1, 0, 0})
	if e := relErr(got, reference(t, cfg, quantized{w}, ids, mask)); e > 1e-4 {
		t.Errorf("lut4: relative error %g against the reference with quantized matrices", e)
	}
}

func TestConfig(t *testing.T) {
	for _, bad := range []string{
		`{"architectures": ["PlamoForCausalLM"], "model_type": "plamo"}`,
		`{"num_attention_heads": 6, "num_key_value_heads": 4}`,
		`{"hidden_size_per_head": 15}`,
		`{"n_expert": 8}`,
		`{"sparse_step": 2}`,
		`{"eval_attention_n_bit": 4}`,
	} {
		if _, err := ParseConfig([]byte(bad)); err == nil {
			t.Errorf("accepted %s", bad)
		}
	}
	cfg, _ := ParseConfig([]byte(tiny))
	b := mlpackage.NewBuilder("CoreML8", nil)
	if _, _, err := cfg.Build(b, newWeights(cfg), llm.Options{Seq: 17}); err == nil {
		t.Error("sequence longer than the positions accepted")
	}
	if a, err := llm.Detect([]byte(tiny)); err != nil || a != Arch {
		t.Errorf("detected %v, %v", a, err)
	}
	if a, err := llm.Detect([]byte(`{"model_type": "plamo", "architectures": ["PlamoForCausalLM"]}`)); err == nil {
		t.Errorf("PLaMo's causal LM detected as %s", a.Name)
	}
}

// plamoEmbedding1B is pfnet/plamo-embedding-1b's config.json.
const plamoEmbedding1B = `{"architectures": ["PlamoBiModel"], "attention_dropout": 0.0,
	"auto_map": {"AutoConfig": "modeling_plamo.PlamoConfig", "AutoModel": "modeling_plamo.PlamoBiModel"},
	"bos_token_id": 1, "capacity_factor": 1.0, "eos_token_id": 1, "eval_attention_n_bit": null,
	"eval_mlp_n_bit": null, "eval_offload_moe": false, "expert_dropout": 0.0, "fp8_accum_dtype": "bfloat16",
	"group_size": 1024, "hidden_size": 2048, "hidden_size_per_head": 128, "initializer_range": 0.02,
	"intermediate_size": 8192, "k_expert": null, "linear_type": "fp8", "max_length": 4096,
	"max_position_embeddings": 4096, "model_type": "plamo", "n_expert": null, "num_attention_heads": 16,
	"num_hidden_layers": 16, "num_key_value_heads": 1, "pad_token_id": 3, "rms_norm_eps": 1e-06,
	"shared_intermediate_size": null, "sparse_intermediate_size": null, "sparse_step": null,
	"tie_word_embeddings": false, "tokenizer_class": "PlamoTokenizer", "torch_dtype": "bfloat16",
	"transformers_version": "4.47.0", "use_cache": false, "vocab_size": 50112}`

// The real model's config lists the 130 tensors of its checkpoint with their shapes (from the
// safetensors header), and fits the Neural Engine's 1 GiB in one part in lut4 and lut8 and two in fp16.
func TestPlamoEmbedding1B(t *testing.T) {
	cfg, err := ParseConfig([]byte(plamoEmbedding1B))
	if err != nil {
		t.Fatal(err)
	}
	tensors := cfg.Tensors()
	if len(tensors) != 130 {
		t.Errorf("%d tensors, the checkpoint has 130", len(tensors))
	}
	for name, want := range map[string][]int{
		"embed_tokens.weight":                       {50112, 2048},
		"norm.weight":                               {2048},
		"layers.layers.15.norm2.weight":             {2048},
		"layers.layers.0.self_attn.qkv_proj.weight": {2304, 2048},
		"layers.layers.0.self_attn.o_proj.weight":   {2048, 2048},
		"layers.layers.0.self_attn.q_weight":        {16, 128},
		"layers.layers.0.self_attn.k_weight":        {1, 128},
		"layers.layers.0.mlp.gate_up_proj.weight":   {16384, 2048},
		"layers.layers.0.mlp.down_proj.weight":      {2048, 8192},
	} {
		if got := tensors[name]; len(got) != len(want) || got[0] != want[0] || len(got) == 2 && got[1] != want[1] {
			t.Errorf("%s: %v, the checkpoint has %v", name, got, want)
		}
	}
	limit := int64(1 << 30)
	budget := int64(llm.SplitMargin * float64(limit))
	for format, want := range map[string]int{"lut4": 1, "lut8": 1, "fp16": 2} {
		parts, err := llm.PlanSplit(cfg, llm.Options{Format: format}, budget, 0)
		if err != nil {
			t.Fatal(err)
		}
		if len(parts) != want {
			t.Errorf("%s: %d parts, expected %d", format, len(parts), want)
		}
	}
}

// runIR writes the program as OpenVINO IR for Intel's NPU and runs it with OpenVINO's CPU plugin
// (internal/ovcpu); the test is skipped without HWNPU_OV_PYTHON.
func runIR(t *testing.T, c *Config, w llm.Weights, opts llm.Options, x, mask []float32) []float32 {
	t.Helper()
	py := ovcpu.Python()
	if py == "" {
		t.Skip("HWNPU_OV_PYTHON is not set")
	}
	opts.Backend = "intel"
	path := filepath.Join(t.TempDir(), "m.xml")
	if _, err := llm.WritePart(path, c, w, opts, nil); err != nil {
		t.Fatal(err)
	}
	res, err := ovcpu.Run(py, path, map[string][]float32{"x": x, "mask": mask}, ovcpu.Options{})
	if err != nil {
		t.Fatal(err)
	}
	return res["y"]
}

// linear are weights with every linear layer's matrix symmetrically quantized per row, as
// blocks.Weight writes the intN formats (with the mse scales, the default). Per row scales make no difference between a fused
// projection quantized whole and in its parts.
type linear struct {
	weights
	bits int
}

func (w linear) Get(name string) ([]float32, []int, error) {
	v, shape, err := w.weights.Get(name)
	if err == nil && len(shape) == 2 && strings.HasSuffix(name, ".weight") && name != "embed_tokens.weight" {
		v = palette.QuantizeLinear(v, shape[0], shape[1], 0, w.bits, true).Dequantize()
	}
	return v, shape, err
}

// The program lowered to OpenVINO IR computes the reference too: in fp16, and in int4 and int8
// against the reference with the matrices quantized the same way. The IR's output is fp16.
func TestIntelIR(t *testing.T) {
	cfg, err := ParseConfig([]byte(tiny))
	if err != nil {
		t.Fatal(err)
	}
	ids := []int32{1, 17, 5, 33, 9, 0, 21}
	maskF := []float64{1, 1, 1, 1, 1, 0, 0}
	mask := []float32{1, 1, 1, 1, 1, 0, 0}
	w := newWeights(cfg)
	x := embed(t, cfg, w, ids)
	if e := relErr(runIR(t, cfg, w, llm.Options{Seq: len(ids), Format: "fp16"}, x, mask), reference(t, cfg, w, ids, maskF)); e > 1e-3 {
		t.Errorf("fp16: relative error %g against the reference", e)
	}
	for _, bits := range []int{4, 8} {
		format := map[int]string{4: "int4", 8: "int8"}[bits]
		got := runIR(t, cfg, w, llm.Options{Seq: len(ids), Format: format}, x, mask)
		if e := relErr(got, reference(t, cfg, linear{w, bits}, ids, maskF)); e > 1e-3 {
			t.Errorf("%s: relative error %g against the reference with quantized matrices", format, e)
		}
	}
}

// The real PLaMo-Embedding-1B (HWNPU_PLAMO_DIR: config.json, tokenizer.model and
// model.safetensors) lowered to OpenVINO IR gives the sentence embeddings (the mean of the rows
// after the BOS, as internal/chat pools them) that the fp16 MIL program gives with milcpu: the
// same in fp16, and with the quantization's error in int8 and int4. Needs HWNPU_OV_PYTHON. The
// lut4 and lut8 programs, which the Neural Engine runs, are measured the same way with milcpu.
func TestIntelIRCheckpoint(t *testing.T) {
	dir, py := os.Getenv("HWNPU_PLAMO_DIR"), ovcpu.Python()
	if dir == "" || py == "" {
		t.Skip("HWNPU_PLAMO_DIR or HWNPU_OV_PYTHON is not set")
	}
	a, m, err := llm.LoadConfig(filepath.Join(dir, "config.json"), "")
	if err != nil {
		t.Fatal(err)
	}
	cfg := m.(*Config)
	w, err := llm.OpenWeights(filepath.Join(dir, "model.safetensors"), a)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	tok, err := tokenizer.LoadSentencePiece(filepath.Join(dir, "tokenizer.model"))
	if err != nil {
		t.Fatal(err)
	}
	bos, ok := tok.ID("<|startoftext|>")
	if !ok {
		t.Fatal("no <|startoftext|> in the tokenizer")
	}
	const seq = 48
	d := cfg.HiddenSize
	type sentence struct {
		n       int
		x, mask []float32
		want    []float64
	}
	var sentences []sentence
	for _, text := range []string{
		"PLaMo-Embedding-1Bは、Preferred Networks, Inc. によって開発された日本語テキスト埋め込みモデルです。",
		"東京は日本の首都です。",
		"How fast is the Neural Engine?",
	} {
		ids := append([]int32{bos}, tok.Encode(text)...)
		if len(ids) > seq {
			t.Fatalf("%q: %d tokens", text, len(ids))
		}
		s := sentence{n: len(ids), mask: make([]float32, seq)}
		for i := range ids {
			s.mask[i] = 1
		}
		for len(ids) < seq {
			ids = append(ids, 3) // pad_token_id
		}
		s.x = embed(t, cfg, w, ids)
		sentences = append(sentences, s)
	}
	pool := func(h []float32, n int) []float64 {
		p := make([]float64, d)
		for r := 1; r < n; r++ {
			for j := range p {
				p[j] += float64(h[r*d+j])
			}
		}
		return p
	}
	cos := func(a, b []float64) float64 {
		var ab, aa, bb float64
		for i := range a {
			ab, aa, bb = ab+a[i]*b[i], aa+a[i]*a[i], bb+b[i]*b[i]
		}
		return ab / math.Sqrt(aa*bb)
	}
	for i, s := range sentences {
		sentences[i].want = pool(run(t, cfg, w, llm.Options{Seq: seq, Format: "fp16"}, s.x, s.mask), s.n)
	}
	for _, f := range []struct{ format, quant string }{{"lut4", "uniform"}, {"lut4", "kmeans-minmax"}, {"lut8", "kmeans-minmax"}} {
		worst := 1.0
		for _, s := range sentences {
			worst = min(worst, cos(pool(run(t, cfg, w, llm.Options{Seq: seq, Format: f.format, Quant: f.quant, Group: 16}, s.x, s.mask), s.n), s.want))
		}
		t.Logf("%s %s (milcpu): lowest cosine %.5f", f.format, f.quant, worst)
	}
	for _, c := range []struct {
		format string
		minCos float64
	}{{"fp16", 0.99999}, {"int8", 0.999}, {"int4", 0.9}, {"int4-g128", 0.9}, {"int4-g32", 0.9}} {
		path := filepath.Join(t.TempDir(), "m.xml")
		if _, err := llm.WritePart(path, cfg, w, llm.Options{Seq: seq, Format: c.format, Backend: "intel"}, nil); err != nil {
			t.Fatal(err)
		}
		worst := 1.0
		for _, s := range sentences {
			res, err := ovcpu.Run(py, path, map[string][]float32{"x": s.x, "mask": s.mask}, ovcpu.Options{})
			if err != nil {
				t.Fatal(err)
			}
			worst = min(worst, cos(pool(res["y"], s.n), s.want))
		}
		t.Logf("%s: lowest cosine %.5f against the fp16 MIL program", c.format, worst)
		if worst < c.minCos {
			t.Errorf("%s: cosine %.5f, expected %g or more", c.format, worst, c.minCos)
		}
		os.RemoveAll(filepath.Dir(path))
	}
}
