package qwen3

import (
	"cmp"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/milcpu"
)

// readF32 reads a file of little endian float32.
func readF32(t *testing.T, path string) []float32 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

// With a real checkpoint (QWEN_CHECKPOINT, a model directory) and transformers' float32 forward
// over it (QWEN_REF, the prefix tools/ref/ref_qwen2.py or ref_logits.py writes), the first
// QWEN_LAYERS layers (3 by default) each on the CPU give the next hidden state, and the head
// after the last layer the logits. QWEN_FORMAT sets the weight format (fp16 by default; lut4
// shows what the tables lose).
func TestRealLayers(t *testing.T) {
	dir, ref := os.Getenv("QWEN_CHECKPOINT"), os.Getenv("QWEN_REF")
	if dir == "" || ref == "" {
		t.Skip("QWEN_CHECKPOINT and QWEN_REF not set")
	}
	a, m, err := llm.LoadConfig(filepath.Join(dir, "config.json"), "")
	if err != nil {
		t.Fatal(err)
	}
	c := m.(*Config)
	w, err := llm.OpenWeights(filepath.Join(dir, "model.safetensors"), a)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	var meta struct{ Seq, Hidden, Layers, Vocab int }
	raw, err := os.ReadFile(ref + ".json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &meta); err != nil {
		t.Fatal(err)
	}
	hidden := readF32(t, ref+".hidden.bin")
	at := func(i int) []float32 { return hidden[i*meta.Seq*meta.Hidden : (i+1)*meta.Seq*meta.Hidden] }
	format := cmp.Or(os.Getenv("QWEN_FORMAT"), "fp16")
	n, _ := strconv.Atoi(cmp.Or(os.Getenv("QWEN_LAYERS"), "3"))
	for _, i := range append(seq(n), c.NumLayers) {
		p := build(t, c, w, llm.Options{Seq: meta.Seq, Format: format, Quant: os.Getenv("QWEN_QUANT"), Codes: os.Getenv("QWEN_CODES"), Start: i, Layers: 1, NoHead: i < c.NumLayers})
		in := at(i)
		out := p.run(t, map[string][]float32{"x": in})["y"]
		var want []float32
		if i < c.NumLayers {
			want = at(i + 1)
		} else {
			want = readF32(t, ref+".logits.bin")
		}
		var se, sw, sd float64
		for j := range want {
			e := float64(out[j] - want[j])
			se += e * e
			sw += float64(want[j]) * float64(want[j])
			if i < c.NumLayers {
				dj := float64(want[j] - in[j])
				sd += dj * dj
			}
		}
		if i == c.NumLayers {
			t.Logf("head (%s): relative error %.3g of the logits", format, math.Sqrt(se/sw))
			continue
		}
		change := math.Sqrt(se / sd)
		t.Logf("layer %d (%s): relative error %.3g of the hidden state, %.3g of the layer's change", i, format, math.Sqrt(se/sw), change)
		if format == "fp16" && !(change <= 0.01) {
			t.Errorf("layer %d: relative error %g of the layer's change", i, change)
		}
	}
}

func seq(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = i
	}
	return out
}

// With QWEN_CHECKPOINT and QWEN_REF as TestRealLayers, the whole model on the CPU a layer at a
// time, each fed the last one's output, in QWEN_FORMAT (lut4 by default): the KL(reference ||
// model) of the logits at each position, its mean and median, and how often the top tokens agree.
// This is what the format loses, before the Neural Engine's own arithmetic.
func TestRealChained(t *testing.T) {
	dir, ref := os.Getenv("QWEN_CHECKPOINT"), os.Getenv("QWEN_REF")
	if dir == "" || ref == "" || os.Getenv("QWEN_CHAIN") == "" {
		t.Skip("QWEN_CHECKPOINT, QWEN_REF and QWEN_CHAIN not set")
	}
	a, m, err := llm.LoadConfig(filepath.Join(dir, "config.json"), "")
	if err != nil {
		t.Fatal(err)
	}
	c := m.(*Config)
	w, err := llm.OpenWeights(filepath.Join(dir, "model.safetensors"), a)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	var meta struct{ Seq, Hidden, Layers, Vocab int }
	raw, err := os.ReadFile(ref + ".json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &meta); err != nil {
		t.Fatal(err)
	}
	format := cmp.Or(os.Getenv("QWEN_FORMAT"), "lut4")
	colBlock, _ := strconv.Atoi(os.Getenv("QWEN_COLBLOCK"))
	var formatFor []string
	if f := os.Getenv("QWEN_FORMAT_FOR"); f != "" {
		formatFor = strings.Split(f, ",")
	}
	h := readF32(t, ref+".hidden.bin")[:meta.Seq*meta.Hidden]
	for i := 0; i <= c.NumLayers; i++ {
		p := build(t, c, w, llm.Options{Seq: meta.Seq, Format: format, Quant: os.Getenv("QWEN_QUANT"), Codes: os.Getenv("QWEN_CODES"), ColBlock: colBlock, FormatFor: formatFor, Start: i, Layers: 1, NoHead: i < c.NumLayers})
		h = p.run(t, map[string][]float32{"x": h})["y"]
	}
	want := readF32(t, ref+".logits.bin")
	kls := make([]float64, meta.Seq)
	agree := 0
	for r := 0; r < meta.Seq; r++ {
		lp := logSoftmax(want[r*meta.Vocab : (r+1)*meta.Vocab])
		lq := logSoftmax(h[r*meta.Vocab : (r+1)*meta.Vocab])
		top, topQ := 0, 0
		for j := range lp {
			kls[r] += math.Exp(lp[j]) * (lp[j] - lq[j])
			if lp[j] > lp[top] {
				top = j
			}
			if lq[j] > lq[topQ] {
				topQ = j
			}
		}
		if top == topQ {
			agree++
		}
	}
	sorted := append([]float64(nil), kls...)
	slices.Sort(sorted)
	var mean float64
	for _, k := range kls {
		mean += k
	}
	mean /= float64(len(kls))
	t.Logf("%s %s cols %d %v: KL mean %.4f, median %.4f, largest %.3f; top token %d of %d the same", format, os.Getenv("QWEN_QUANT"), colBlock, formatFor, mean, sorted[len(sorted)/2], sorted[len(sorted)-1], agree, meta.Seq)
}

func logSoftmax(x []float32) []float64 {
	mx := math.Inf(-1)
	for _, v := range x {
		mx = math.Max(mx, float64(v))
	}
	var z float64
	for _, v := range x {
		z += math.Exp(float64(v) - mx)
	}
	out := make([]float64, len(x))
	lz := math.Log(z) + mx
	for i, v := range x {
		out[i] = float64(v) - lz
	}
	return out
}

// With QWEN_CHECKPOINT and QWEN_REF, each layer in fp16 on the CPU with every operation's result
// rounded to fp16 (as the Neural Engine computes): the operations whose largest value passes
// QWEN_OVER (30000), which overflow or nearly overflow fp16's 65504.
func TestRealF16Overflow(t *testing.T) {
	dir, ref := os.Getenv("QWEN_CHECKPOINT"), os.Getenv("QWEN_REF")
	if dir == "" || ref == "" || os.Getenv("QWEN_F16") == "" {
		t.Skip("QWEN_CHECKPOINT, QWEN_REF and QWEN_F16 not set")
	}
	a, m, err := llm.LoadConfig(filepath.Join(dir, "config.json"), "")
	if err != nil {
		t.Fatal(err)
	}
	c := m.(*Config)
	w, err := llm.OpenWeights(filepath.Join(dir, "model.safetensors"), a)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	var meta struct{ Seq, Hidden int }
	raw, _ := os.ReadFile(ref + ".json")
	if err := json.Unmarshal(raw, &meta); err != nil {
		t.Fatal(err)
	}
	over, _ := strconv.ParseFloat(cmp.Or(os.Getenv("QWEN_OVER"), "30000"), 64)
	hidden := readF32(t, ref+".hidden.bin")
	for i := 0; i < c.NumLayers; i++ {
		p := build(t, c, w, llm.Options{Seq: meta.Seq, Format: "fp16", Start: i, Layers: 1, NoHead: true})
		in := hidden[i*meta.Seq*meta.Hidden : (i+1)*meta.Seq*meta.Hidden]
		var worst float32
		var worstOp string
		_, err := p.p.Run(map[string]milcpu.Tensor{"x": {Shape: []int{1, meta.Seq, meta.Hidden}, Data: in}}, milcpu.Options{RoundF16: true, Trace: func(typ, name string, mx, rms float32) {
			if mx > worst || mx != mx {
				worst, worstOp = mx, typ+" "+name
			}
			if float64(mx) > over || mx != mx {
				t.Logf("layer %d: %s %s max %g", i, typ, name, mx)
			}
		}})
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("layer %d: largest %g at %s", i, worst, worstOp)
	}
}
