package gptoss

import (
	"cmp"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
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

// With a real checkpoint (GPTOSS_CHECKPOINT, the directory of openai/gpt-oss-20b) and the
// hidden states transformers' forward gives around its first layers (GPTOSS_REF: hiddenN.bin
// entering layer N and meta.json, from ref_gptoss.py's layers in float32), each layer on the CPU
// gives the next hidden state: the MXFP4 experts, YaRN and the sinks as the checkpoint has them.
// GPTOSS_FORMAT sets the weight format (fp16 by default; lut4 shows what the tables lose) and
// GPTOSS_TABLES expert-tables. The reference: tools/ref/ref_gptoss.py hidden MODEL_DIR OUT.
func TestRealLayers(t *testing.T) {
	dir, ref := os.Getenv("GPTOSS_CHECKPOINT"), os.Getenv("GPTOSS_REF")
	if dir == "" || ref == "" {
		t.Skip("GPTOSS_CHECKPOINT and GPTOSS_REF not set")
	}
	a, m, err := llm.LoadConfig(filepath.Join(dir, "config.json"), "")
	if err != nil {
		t.Fatal(err)
	}
	c := m.(*Config)
	w, err := llm.OpenWeights(filepath.Join(dir, "model.safetensors.index.json"), a)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	var meta struct{ Tokens, Layers int }
	raw, err := os.ReadFile(filepath.Join(ref, "meta.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &meta); err != nil {
		t.Fatal(err)
	}
	format := cmp.Or(os.Getenv("GPTOSS_FORMAT"), "fp16")
	params := map[string]string{}
	if v := os.Getenv("GPTOSS_TABLES"); v != "" {
		params["expert-tables"] = v
	}
	for i := 0; i < meta.Layers; i++ {
		in := readF32(t, filepath.Join(ref, "hidden"+strconv.Itoa(i)+".bin"))
		want := readF32(t, filepath.Join(ref, "hidden"+strconv.Itoa(i+1)+".bin"))
		p := build(t, c, w, llm.Options{Seq: meta.Tokens, Format: format, Quant: os.Getenv("GPTOSS_QUANT"), Params: params, Start: i, Layers: 1, NoHead: true})
		out := p.run(t, map[string][]float32{"x": in})["y"]
		// the layer's own change, h_out - h_in, is what it computes: its error apart from the
		// residual that passes through
		var se, sh, sd float64
		for j := range want {
			e := float64(out[j] - want[j])
			se += e * e
			sh += float64(want[j]) * float64(want[j])
			dj := float64(want[j] - in[j])
			sd += dj * dj
		}
		hidden, change := math.Sqrt(se/sh), math.Sqrt(se/sd)
		t.Logf("layer %d (%s, %s): relative error %.3g of the hidden state, %.3g of the layer's change", i, c.LayerTypes[i], format, hidden, change)
		if format == "fp16" && !(change <= 0.01) {
			t.Errorf("layer %d: relative error %g of the layer's change", i, change)
		}
	}
}
