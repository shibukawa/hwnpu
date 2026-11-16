package gptoss

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// e2m1 codes decode to their values times the group's power of two, the low nibble first.
func TestDecodeMXFP4(t *testing.T) {
	blocks := make([]byte, 32) // one row of two groups
	for j := range 16 {
		blocks[j] = byte(j) | byte(15-j)<<4
		blocks[16+j] = byte(j)
	}
	got := decodeMXFP4(blocks, []byte{127, 129}, 1, 2)
	// the first group: code j low and 15-j high, times 1; the second: code j low and 0 high, times 4
	want := []float32{0, -6, 0.5, -4, 1, -3, 1.5, -2, 2, -1.5, 3, -1, 4, -0.5, 6, 0, 0, 6, -0.5, 4, -1, 3, -1.5, 2, -2, 1.5, -3, 1, -4, 0.5, -6, 0}
	for j := range 16 {
		want = append(want, fp4[j]*4, 0)
	}
	for i := range want {
		if got[i] != want[i] || math.Signbit(float64(got[i])) != math.Signbit(float64(want[i])) && want[i] != 0 {
			t.Fatalf("value %d: %v, want %v (all %v)", i, got[i], want[i], got)
		}
	}
}

// mxfp4Weights is a checkpoint with MXFP4 experts (random codes and scales) besides synthetic
// tensors, and the same experts as transformers' dense parameters (dense true): the two give
// the same program.
type mxfp4Weights struct {
	llm.Synthetic
	c      *Config
	blocks map[string][]byte // by tensor name: blocks and scales
	dense  map[string][]float32
}

func newMXFP4Weights(c *Config) *mxfp4Weights {
	d, f, nE := c.HiddenSize, c.IntermediateSize, c.Experts
	quant := *c
	quant.Quantization = &struct {
		QuantMethod string `json:"quant_method"`
	}{"mxfp4"}
	w := &mxfp4Weights{Synthetic: llm.NewSynthetic(&quant), c: c, blocks: map[string][]byte{}, dense: map[string][]float32{}}
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < c.NumLayers; i++ {
		p := layerPrefix(i) + "mlp.experts."
		for _, m := range []struct {
			name       string
			rows, cols int
		}{{"gate_up_proj", 2 * f, d}, {"down_proj", d, f}} {
			groups := m.cols / 32
			b := make([]byte, nE*m.rows*groups*16)
			rng.Read(b)
			s := make([]byte, nE*m.rows*groups)
			for j := range s {
				s[j] = byte(120 + rng.Intn(6)) // 2^-7 .. 2^-2
			}
			w.blocks[p+m.name+"_blocks"], w.blocks[p+m.name+"_scales"] = b, s
			// transformers' dense parameter: (experts, in, out)
			dn := make([]float32, nE*m.rows*m.cols)
			for e := range nE {
				v := decodeMXFP4(b[e*m.rows*groups*16:(e+1)*m.rows*groups*16], s[e*m.rows*groups:(e+1)*m.rows*groups], m.rows, groups)
				for out := 0; out < m.rows; out++ {
					for in := 0; in < m.cols; in++ {
						dn[(e*m.cols+in)*m.rows+out] = v[out*m.cols+in]
					}
				}
			}
			w.dense[p+m.name] = dn
		}
	}
	return w
}

func (w *mxfp4Weights) Get(name string) ([]float32, []int, error) {
	if v, ok := w.dense[name]; ok {
		f, d := w.c.IntermediateSize, w.c.HiddenSize
		if strings.HasSuffix(name, "gate_up_proj") {
			return v, []int{w.c.Experts, d, 2 * f}, nil
		}
		return v, []int{w.c.Experts, f, d}, nil
	}
	return w.Synthetic.Get(name)
}

func (w *mxfp4Weights) RawRange(name string, begin, end int64) ([]byte, string, []int, error) {
	b, ok := w.blocks[name]
	if !ok {
		return nil, "", nil, fmt.Errorf("no tensor %s", name)
	}
	shape := w.Synthetic[name]
	return b[begin:end], "U8", shape, nil
}

// The experts read from MXFP4 blocks and scales give the program the dense parameters give.
func TestMXFP4MatchesDense(t *testing.T) {
	dense := tinyConfig(t)
	quant, err := ParseConfig([]byte(strings.Replace(tiny, `"attention_bias": true,`, `"attention_bias": true, "quantization_config": {"quant_method": "mxfp4"},`, 1)))
	if err != nil {
		t.Fatal(err)
	}
	if !quant.mxfp4() || dense.mxfp4() {
		t.Fatal("the configs' quantization")
	}
	w := newMXFP4Weights(dense)
	feed := map[string][]float32{"x": rows(t, w, dense.Embedding(), tokens)}
	a := build(t, dense, w, llm.Options{Seq: len(tokens), Format: "fp16"}).run(t, feed)["y"]
	b := build(t, quant, w, llm.Options{Seq: len(tokens), Format: "fp16"}).run(t, feed)["y"]
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("logit %d: %v from MXFP4, %v from the dense experts", i, b[i], a[i])
		}
	}
	// the reference over the dense experts agrees too: the layout of gate_up_proj (the gate on
	// the even columns) is transformers'
	if e := relErr(b, reference(t, dense, w, tokens).logits); !(e <= 1e-4) {
		t.Errorf("relative error %g against the reference", e)
	}
	// in lut4 (the experts' tables fitted exactly, the attention's lut8), near fp16's; a
	// palette fitting is further off
	ref := relErr(b, reference(t, dense, w, tokens).logits)
	lut := map[string]float64{}
	for _, tables := range []string{"exact", "exact-rows", "palette"} {
		y := build(t, quant, w, llm.Options{Seq: len(tokens), Format: "lut4", Params: map[string]string{"expert-tables": tables}}).run(t, feed)["y"]
		var se, sr float64
		for i := range y {
			d := float64(y[i] - b[i])
			se += d * d
			sr += float64(b[i]) * float64(b[i])
		}
		lut[tables] = math.Sqrt(se / sr)
	}
	t.Logf("lut4 against fp16: %v (fp16 against the reference %.2g)", lut, ref)
	if !(lut["exact"] < 0.3) || !(lut["exact"] < lut["palette"]) { // random codes over six scales: harder than gpt-oss-20b's
		t.Errorf("lut4 relative errors %v", lut)
	}
	if _, ok := quant.Tensors()["model.layers.0.mlp.experts.gate_up_proj_blocks"]; !ok {
		t.Error("the MXFP4 config does not read the blocks")
	}
}

// optimalTable's table is the best of 3 entries for a few values, against every split of them
// into runs (brute force), and holds values as they are when there are no more than entries.
func TestOptimalTable(t *testing.T) {
	vals := []float32{-6, -3, -3, -1, 0, 0, 0, 0.5, 0.5, 1, 2, 4, 4, 4, 6}
	got, ok := optimalTable(vals, 3)
	if !ok {
		t.Fatal("not fitted")
	}
	sq := func(c []float64) float64 {
		var s float64
		for _, v := range vals {
			best := math.Inf(1)
			for _, x := range c {
				best = math.Min(best, (float64(v)-x)*(float64(v)-x))
			}
			s += best
		}
		return s
	}
	distinct := []float64{-6, -3, -1, 0, 0.5, 1, 2, 4, 6}
	mean := func(lo, hi float64) float64 {
		var s, n float64
		for _, v := range vals {
			if float64(v) >= lo && float64(v) <= hi {
				s += float64(v)
				n++
			}
		}
		return s / n
	}
	best := math.Inf(1)
	for i := 1; i < len(distinct)-1; i++ {
		for j := i + 1; j < len(distinct); j++ {
			c := []float64{mean(distinct[0], distinct[i-1]), mean(distinct[i], distinct[j-1]), mean(distinct[j], distinct[len(distinct)-1])}
			best = math.Min(best, sq(c))
		}
	}
	if e := sq(got); math.Abs(e-best) > 1e-9 {
		t.Errorf("table %v: squared error %g, the best %g", got, e, best)
	}
	exact, _ := optimalTable(vals, 16)
	if sq(exact) != 0 {
		t.Errorf("16 entries for 9 values: %v", exact)
	}
	// exactTables packs them as the builder's tables take them: 16 rows of 32, one table
	w := make([]float32, 16*32)
	for i := range w {
		w[i] = vals[i%len(vals)]
	}
	idx, lut, ok := exactTables(w, 16, 32, 16, 0, 4)
	if !ok || len(idx) != 16*32/2 || len(lut) != 16 {
		t.Fatalf("%d index bytes, %d entries", len(idx), len(lut))
	}
	for i, v := range w {
		code := idx[i/2] >> (4 * (i % 2)) & 15
		if got := mlpackage.Float16Value(lut[code]); got != v {
			t.Fatalf("weight %d: %v decoded as %v", i, v, got)
		}
	}
}
