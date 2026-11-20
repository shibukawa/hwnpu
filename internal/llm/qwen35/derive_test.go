package qwen35

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/experts"
	"github.com/shibukawa/hwnpu/internal/mxfp4"
	"github.com/shibukawa/hwnpu/internal/safetensors"
)

// bf16 is v rounded to bf16 (to nearest even), as raw little endian bytes and back as values.
func bf16(v []float32) ([]byte, []float32) {
	b := make([]byte, 2*len(v))
	back := make([]float32, len(v))
	for i, x := range v {
		u := math.Float32bits(x)
		u += 0x7fff + (u>>16)&1
		binary.LittleEndian.PutUint16(b[2*i:], uint16(u>>16))
		back[i] = math.Float32frombits(u >> 16 << 16)
	}
	return b, back
}

// A checkpoint of a small Qwen3.5-MoE in bf16 (two shards, a vision tensor and an MTP one) derives
// into one whose experts the host runs (experts.Host on its MXFP4 tensors) to the bf16 experts'
// outputs, whose other tensors are copied, and which a run cut off finishes.
func TestDeriveMXFP4(t *testing.T) {
	raw := strings.NewReplacer(`"hidden_size": 32`, `"hidden_size": 64`, `"moe_intermediate_size": 32`, `"moe_intermediate_size": 64`).Replace(tinyMoEJSON)
	c, err := ParseMoE([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	src := t.TempDir()
	w := llm.NewSynthetic(c)
	E, H, I := c.NumExperts, c.HiddenSize, c.MoEIntermediate
	for l := range c.NumLayers {
		p := fmt.Sprintf("%slayers.%d.mlp.experts.", c.prefix(), l)
		w[p+"gate_up_proj"] = []int{E, 2 * I, H}
		w[p+"down_proj"] = []int{E, H, I}
	}
	w["model.visual.blocks.0.attn.qkv.weight"] = []int{16, 16}
	w["mtp.fc.weight"] = []int{16, 16}
	vals := map[string][]float32{}
	shardOf := map[string]string{}
	var outs [2][]safetensors.Out
	for _, name := range w.Names() {
		v, shape, _ := w.Get(name)
		if strings.Contains(name, "experts") {
			for i := range v {
				v[i] *= 0.3 // the scale of real experts' weights
			}
		}
		b, back := bf16(v)
		vals[name] = back
		s := 0
		if strings.Contains(name, "layers.2.") || strings.Contains(name, "layers.3.") {
			s = 1
		}
		shardOf[name] = fmt.Sprintf("model-%05d-of-00002.safetensors", s+1)
		outs[s] = append(outs[s], safetensors.Out{Name: name, DType: "BF16", Shape: shape, Size: int64(len(b)), Data: bytes.NewReader(b)})
	}
	for s := range outs {
		if err := safetensors.Write(filepath.Join(src, fmt.Sprintf("model-%05d-of-00002.safetensors", s+1)), outs[s], nil); err != nil {
			t.Fatal(err)
		}
	}
	idx, _ := json.Marshal(map[string]any{"weight_map": shardOf})
	for name, b := range map[string][]byte{"config.json": []byte(raw), "model.safetensors.index.json": idx, "tokenizer.json": []byte("{}")} {
		if err := os.WriteFile(filepath.Join(src, name), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}

	out := t.TempDir()
	var log bytes.Buffer
	if err := DeriveMXFP4(DirSource(src), "test", out, &log); err != nil {
		t.Fatal(err)
	}
	// cut off after layer 1: the rest written again, the files there kept
	if err := os.Remove(filepath.Join(out, "model-layer-02.safetensors")); err != nil {
		t.Fatal(err)
	}
	os.Remove(filepath.Join(out, "config.json"))
	log.Reset()
	if err := DeriveMXFP4(DirSource(src), "test", out, &log); err != nil {
		t.Fatal(err)
	}
	if n := strings.Count(log.String(), "there already"); n != 4 {
		t.Errorf("%d files kept, want 4:\n%s", n, log.String())
	}
	cfg, err := os.ReadFile(filepath.Join(out, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !IsDerived(cfg) || IsDerived([]byte(raw)) {
		t.Error("IsDerived")
	}
	if _, err := os.Stat(filepath.Join(out, "tokenizer.json")); err != nil {
		t.Error(err)
	}

	ow, err := llm.OpenWeights(filepath.Join(out, "model.safetensors.index.json"), MoEArch)
	if err != nil {
		t.Fatal(err)
	}
	defer ow.Close()
	for name := range c.Tensors() {
		got, _, err := ow.Get(name)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		for i := range got {
			if got[i] != vals[name][i] {
				t.Fatalf("%s[%d] = %g, want %g", name, i, got[i], vals[name][i])
			}
		}
	}
	var idxOut struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	ib, _ := os.ReadFile(filepath.Join(out, "model.safetensors.index.json"))
	if err := json.Unmarshal(ib, &idxOut); err != nil {
		t.Fatal(err)
	}
	for name := range idxOut.WeightMap {
		if strings.HasPrefix(name, "model.visual.") || strings.HasPrefix(name, "mtp.") || strings.HasSuffix(name, "_proj") && strings.Contains(name, "experts") {
			t.Errorf("%s in the derived checkpoint", name)
		}
	}

	rec := &experts.HostRecord{Checkpoint: filepath.Join(out, "model.safetensors.index.json"), Act: c.ExpertAct(), Layers: map[int]experts.MXFP4{}}
	for _, l := range c.RoutedLayers() {
		rec.Layers[l] = c.HostExperts(l)
	}
	h, err := experts.OpenHost(rec, c.ExpertShape())
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	rows := [][]float32{make([]float32, H), make([]float32, H)}
	for i := range rows {
		for j := range rows[i] {
			rows[i][j] = float32(math.Sin(float64(7*i+3*j) * 0.37))
		}
	}
	// the experts as MXFP4 holds them, in the checkpoint's layout: the host's outputs but for
	// its int8 vectors, where the bf16 ones also differ by MXFP4's rounding
	deq := bf16Weights{}
	for name, v := range vals {
		if !strings.Contains(name, "mlp.experts.") {
			continue
		}
		d := make([]float32, len(v))
		n := I
		if strings.HasSuffix(name, "gate_up_proj") {
			n = H
		}
		bl, sc := make([]byte, n/2), make([]byte, n/32)
		for r := 0; r < len(v); r += n {
			mxfp4.Encode(bl, sc, v[r:r+n])
			mxfp4.Dequantize(d[r:r+n], bl, sc)
		}
		deq[name] = d
	}
	idxs := [][]int{{1, 6}, {6, 3}}
	ws := [][]float32{{0.7, 0.3}, {0.5, 0.5}}
	for _, l := range []int{0, 3} {
		y := [][]float32{make([]float32, H), make([]float32, H)}
		if err := h.Run(l, rows, idxs, ws, y); err != nil {
			t.Fatal(err)
		}
		p := fmt.Sprintf("%slayers.%d.", c.prefix(), l)
		for r := range rows {
			f := make([]float64, H)
			for j := range f {
				f[j] = float64(rows[r][j])
			}
			wt := []float64{float64(ws[r][0]), float64(ws[r][1])}
			cosine := func(want []float64) float64 {
				var dot, na, nb float64
				for j := range want {
					dot += want[j] * float64(y[r][j])
					na += want[j] * want[j]
					nb += float64(y[r][j]) * float64(y[r][j])
				}
				return dot / math.Sqrt(na*nb)
			}
			exact := cosine(moeExperts(t, c, deq, p, f, idxs[r], wt))
			orig := cosine(moeExperts(t, c, bf16Weights(vals), p, f, idxs[r], wt))
			t.Logf("layer %d row %d: cos %.5f against the MXFP4 experts, %.5f against the bf16 ones", l, r, exact, orig)
			if exact < 0.999 || orig < 0.95 {
				t.Errorf("layer %d row %d: cos %.5f against the MXFP4 experts, %.5f against the bf16 ones", l, r, exact, orig)
			}
		}
	}
}

// bf16Weights are the checkpoint's values (llm.Weights of moeExperts).
type bf16Weights map[string][]float32

func (w bf16Weights) Get(name string) ([]float32, []int, error) {
	v, ok := w[name]
	if !ok {
		return nil, nil, fmt.Errorf("no %s", name)
	}
	return v, nil, nil
}
func (w bf16Weights) Rows(name string, begin, end int) ([]float32, error) { return nil, nil }
func (w bf16Weights) Names() []string                                     { return nil }
func (w bf16Weights) Source() string                                      { return "bf16" }
func (w bf16Weights) Close() error                                        { return nil }
