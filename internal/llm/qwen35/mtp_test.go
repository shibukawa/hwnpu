package qwen35

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// tinyMTP is tinyJSON with a multi-token prediction layer.
func tinyMTP(t *testing.T) *MTP {
	t.Helper()
	m, err := ParseMTP([]byte(strings.Replace(tinyJSON, `"model_type": "qwen3_5_text",`, `"model_type": "qwen3_5_text", "mtp_num_hidden_layers": 1,`, 1)))
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// mtpReference is the MTP layer's logits in float64, written from vLLM's
// Qwen3_5MultiTokenPredictor.forward: for each row the next token's embedding x and the model's
// final hidden state hid, each through its zero centered norm, concatenated (the embedding
// first), fc, the decoder layer mtp.layers.0 (Qwen3_5DecoderLayer of the full attention kind) at
// positions 0, 1, ..., mtp.norm and the output layer. It also returns the final hidden states.
func mtpReference(t *testing.T, c *MTP, w llm.Weights, x, hid []float32) (logits64, h [][]float64) {
	d := c.HiddenSize
	s := len(x) / d
	norm := func(v []float32, name string) []float64 {
		// 1 + w is rounded to fp16 once, as the program stores it (reference's onePlus)
		g, _, err := w.Get(name)
		if err != nil {
			t.Fatal(err)
		}
		var ms float64
		for _, a := range v {
			ms += float64(a) * float64(a)
		}
		r := 1 / math.Sqrt(ms/float64(len(v))+round16(c.NormEps))
		out := make([]float64, len(v))
		for i, a := range v {
			out[i] = float64(a) * r * round16(1+float64(g[i]))
		}
		return out
	}
	fc := f16(t, w, "mtp.fc.weight")
	in := make([][]float64, s)
	for i := range in {
		cat := append(norm(x[i*d:(i+1)*d], "mtp.pre_fc_norm_embedding.weight"), norm(hid[i*d:(i+1)*d], "mtp.pre_fc_norm_hidden.weight")...)
		in[i] = make([]float64, d)
		for r := range in[i] {
			for j, v := range cat {
				in[i][r] += fc[r*2*d+j] * v
			}
		}
	}
	h = referenceLayers(t, c.Config, w, in, []refLayer{{mtpLayer, false}}, "mtp.norm.weight", -1)
	return logits(t, c.Config, w, h), h
}

// mtpInputs are an MTP program's inputs for the tokens: the embeddings of ids[1:] and the
// model's final hidden states after ids[:len-1], both in fp16.
func mtpInputs(t *testing.T, c *MTP, w llm.Weights, ids []int32) (x, hid []float32) {
	d := c.HiddenSize
	main := reference(t, c.Config, w, embed16(t, c.Config, w, ids), -1)
	x = embed16(t, c.Config, w, ids[1:])
	for _, row := range main[:len(ids)-1] {
		for _, v := range row {
			hid = append(hid, mlpackage.Float16Value(mlpackage.Float16Bits(float32(v))))
		}
	}
	if len(hid) != len(x) || len(x)%d != 0 {
		t.Fatalf("%d hidden values for %d embedding values", len(hid), len(x))
	}
	return x, hid
}

// The MTP program computes the reference's logits.
func TestMTPAgainstFP64(t *testing.T) {
	c := tinyMTP(t)
	w := mtpWeights(c)
	ids := []int32{3, 1, 4, 1, 5, 9, 2, 6, 5, 3}
	x, hid := mtpInputs(t, c, w, ids)
	want, _ := mtpReference(t, c, w, x, hid)
	p, ins, outs, _ := buildProgram(t, c, w, llm.Options{Seq: len(ids) - 1, Format: "fp16"})
	if len(ins) != 2 || ins[0].Name != "x" || ins[1].Name != "hid" {
		t.Fatalf("inputs %v, want x and hid", ins)
	}
	res, err := p.Run(map[string]milcpu.Tensor{"x": {Shape: ins[0].Shape, Data: x}, "hid": {Shape: ins[1].Shape, Data: hid}}, milcpu.Options{})
	if err != nil {
		t.Fatal(err)
	}
	e := relErr64(res[outs[0].Name].Data, want, c.VocabSize)
	if e > 1e-4 {
		t.Errorf("relative error %g against the reference", e)
	}
	t.Logf("relative error %.2g", e)
}

// mtpRows are the MTP session's rows: the embedding followed by hid.
func mtpRows(c *MTP, x, hid []float32) [][]float32 {
	d := c.HiddenSize
	var rows [][]float32
	for i := 0; i < len(x)/d; i++ {
		rows = append(rows, append(append([]float32{}, x[i*d:(i+1)*d]...), hid[i*d:(i+1)*d]...))
	}
	return rows
}

// With a KV cache, the MTP program steps through the rows as the program without one computes
// them all, outputs its final hidden state h, and goes back with Truncate; the model's program
// with hidden-out outputs h of every row as the reference's final hidden states, the logits
// unchanged.
func TestMTPCached(t *testing.T) {
	c := tinyMTP(t)
	w := mtpWeights(c)
	ids := []int32{3, 1, 4, 1, 5, 9, 2, 6, 5, 3}
	x, hid := mtpInputs(t, c, w, ids)
	want, wantH := mtpReference(t, c, w, x, hid)
	rows := mtpRows(c, x, hid)
	for _, q := range []int{1, 4} {
		t.Run(fmt.Sprintf("q%d", q), func(t *testing.T) {
			s, err := decode.New(cpuProgram(t, c, w, llm.Options{Seq: q, Cache: 12, Format: "fp16"}, nil), c)
			if err != nil {
				t.Fatal(err)
			}
			check := func(from int) {
				for pos := from; pos < len(rows); {
					v := min(q, len(rows)-pos)
					logits, err := s.Step(rows[pos : pos+v])
					if err != nil {
						t.Fatal(err)
					}
					i := pos + v - 1
					if e := relErr64(logits, want[i:i+1], c.VocabSize); e > 1e-3 {
						t.Errorf("position %d: relative error %g", i, e)
					}
					if e := relErr64(s.Hidden(), wantH[i:i+1], c.HiddenSize); e > 1e-3 {
						t.Errorf("position %d: h relative error %g", i, e)
					}
					pos += v
				}
			}
			check(0)
			if err := s.Truncate(3); err != nil {
				t.Fatal(err)
			}
			if s.Pos() != 3 {
				t.Fatalf("at %d after Truncate(3)", s.Pos())
			}
			check(3)
		})
	}

	// the model's program with hidden-out
	main := reference(t, c.Config, w, embed16(t, c.Config, w, ids), -1)
	emb := embed16(t, c.Config, w, ids)
	d := c.HiddenSize
	for _, last := range []bool{false, true} {
		opts := llm.Options{Seq: 4, Cache: 12, Format: "fp16", LastRow: last, Params: map[string]string{"hidden-out": "1"}}
		s, err := decode.New(cpuProgram(t, c.Config, w, opts, nil), c.Config)
		if err != nil {
			t.Fatal(err)
		}
		for pos := 0; pos < len(ids); pos += 4 {
			v := min(4, len(ids)-pos)
			var r [][]float32
			for i := pos; i < pos+v; i++ {
				r = append(r, emb[i*d:(i+1)*d])
			}
			logits, err := s.Step(r)
			if err != nil {
				t.Fatal(err)
			}
			hs := s.HiddenRows()
			if len(hs) != v {
				t.Fatalf("LastRow %v: %d hidden rows for %d tokens", last, len(hs), v)
			}
			for i, h := range hs {
				if e := relErr64(h, main[pos+i:pos+i+1], d); e > 1e-3 {
					t.Errorf("LastRow %v: h of position %d: relative error %g", last, pos+i, e)
				}
			}
			ref := logits64(t, c, w, main[pos+v-1])
			if e := relErr64(logits, [][]float64{ref}, c.VocabSize); e > 1e-3 {
				t.Errorf("LastRow %v: logits at %d: relative error %g", last, pos+v-1, e)
			}
		}
	}
}

func logits64(t *testing.T, c *MTP, w llm.Weights, h []float64) []float64 {
	return logits(t, c.Config, w, [][]float64{h})[0]
}

// The MTP program split at its attention (llm.Paged) gives the logits of the one in one piece.
func TestMTPPaged(t *testing.T) {
	c := tinyMTP(t)
	w := mtpWeights(c)
	ids := []int32{3, 1, 4, 1, 5, 9, 2, 6, 5, 3}
	x, hid := mtpInputs(t, c, w, ids)
	want, _ := mtpReference(t, c, w, x, hid)
	rows := mtpRows(c, x, hid)
	man := convertPagedModel(t, c, w, 4, 32, 8)
	var allocated int
	s, err := decode.New(cpuPaged(t, man, &allocated), c)
	if err != nil {
		t.Fatal(err)
	}
	for pos := 0; pos < len(rows); pos += 4 {
		v := min(4, len(rows)-pos)
		logits, err := s.Step(rows[pos : pos+v])
		if err != nil {
			t.Fatal(err)
		}
		i := pos + v - 1
		if e := relErr64(logits, want[i:i+1], c.VocabSize); e > 1e-3 {
			t.Errorf("position %d: relative error %g", i, e)
		}
	}
	if raw, err := json.Marshal(man); err == nil {
		t.Logf("%d parts: %s", len(man.Parts), raw[:min(len(raw), 120)])
	}
}

var _ = hwnpu.EncodeF16

// mtpWeights are synthetic weights of the model and its MTP layer.
func mtpWeights(c *MTP) llm.Synthetic {
	w := llm.NewSynthetic(c.Config)
	for name, shape := range c.Tensors() {
		w[name] = shape
	}
	return w
}

// With HeadRows, the MTP program's logits are those of the rows kept, in their order: the whole
// vocabulary's at those tokens.
func TestMTPHeadRows(t *testing.T) {
	c := tinyMTP(t)
	w := mtpWeights(c)
	ids := []int32{3, 1, 4, 1, 5, 9, 2, 6, 5, 3}
	x, hid := mtpInputs(t, c, w, ids)
	feed := func(ins []mlpackage.Feature) map[string]milcpu.Tensor {
		return map[string]milcpu.Tensor{"x": {Shape: ins[0].Shape, Data: x}, "hid": {Shape: ins[1].Shape, Data: hid}}
	}
	p, ins, outs, _ := buildProgram(t, c, w, llm.Options{Seq: len(ids) - 1, Format: "fp16"})
	all, err := p.Run(feed(ins), milcpu.Options{})
	if err != nil {
		t.Fatal(err)
	}
	var rows []int32
	for r := int32(c.VocabSize) - 1; r >= 0; r -= 3 { // out of order and with gaps
		rows = append(rows, r)
	}
	p, ins, outs2, _ := buildProgram(t, c, w, llm.Options{Seq: len(ids) - 1, Format: "fp16", HeadRows: rows})
	part, err := p.Run(feed(ins), milcpu.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if outs2[0].Shape[len(outs2[0].Shape)-1] != len(rows) {
		t.Fatalf("output %v for %d rows", outs2[0].Shape, len(rows))
	}
	a, b := all[outs[0].Name].Data, part[outs2[0].Name].Data
	for s := range len(ids) - 1 {
		for i, r := range rows {
			if got, want := b[s*len(rows)+i], a[s*c.VocabSize+int(r)]; math.Abs(float64(got-want)) > 1e-3*math.Max(1, math.Abs(float64(want))) {
				t.Fatalf("row %d, token %d: %g, the whole head's %g", s, r, got, want)
			}
		}
	}
}
