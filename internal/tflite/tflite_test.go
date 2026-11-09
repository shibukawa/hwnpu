package tflite

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/onnx"
)

func readBin[T float32 | int32](t *testing.T, path string) []T {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var z T
	out := make([]T, len(raw)/binary.Size(z))
	if err := binary.Read(bytes.NewReader(raw), binary.LittleEndian, out); err != nil {
		t.Fatal(err)
	}
	return out
}

func tensor[T float32 | int32](v []T, typ int, shape ...int) *onnx.Tensor {
	t := &onnx.Tensor{Shape: shape, Type: typ}
	for _, x := range v {
		t.F = append(t.F, float64(x))
	}
	return t
}

func relErr(got []float64, want []float32) float64 {
	var se, sw float64
	for i := range want {
		d := got[i] - float64(want[i])
		se += d * d
		sw += float64(want[i]) * float64(want[i])
	}
	return math.Sqrt(se / sw)
}

func run(t *testing.T, m *Model, ins ...*onnx.Tensor) []float64 {
	t.Helper()
	given := map[string]*onnx.Tensor{}
	for i, v := range ins {
		given[m.InputValues[i]] = v
	}
	out, err := m.Graph.Eval(given, m.OutputValues)
	if err != nil {
		t.Fatal(err)
	}
	return out[m.OutputValues[0]].F
}

// MusicCoCa's TFLite models (HWNPU_MUSICCOCA, their directory) evaluated on the host give the
// reference's embedding, mapped embedding and codes (HWNPU_MUSICCOCA_REF, mcc_ref.py's prefix).
func TestMusicCoCa(t *testing.T) {
	dir, ref := os.Getenv("HWNPU_MUSICCOCA"), os.Getenv("HWNPU_MUSICCOCA_REF")
	if dir == "" || ref == "" {
		t.Skip("HWNPU_MUSICCOCA and HWNPU_MUSICCOCA_REF are not set")
	}
	load := func(name string) *Model {
		m, err := Load(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%s: inputs %v, outputs %v, %d nodes", name, m.Inputs, m.Outputs, len(m.Graph.Nodes))
		return m
	}
	te := load("text_encoder.tflite")
	ids, pad := readBin[int32](t, ref+".ids.bin"), readBin[float32](t, ref+".pad.bin")
	var in []*onnx.Tensor
	for _, name := range te.Inputs {
		if bytes.Contains([]byte(name), []byte("ids")) {
			in = append(in, tensor(ids, onnx.Int32, 1, 128))
		} else {
			in = append(in, tensor(pad, onnx.Float, 1, 128))
		}
	}
	t0 := time.Now()
	emb := run(t, te, in...)
	t.Logf("text encoder: %.1f s", time.Since(t0).Seconds())
	if e := relErr(emb, readBin[float32](t, ref+".emb.bin")); e > 1e-4 {
		t.Errorf("text embedding: relative error %g", e)
	} else {
		t.Logf("text embedding: relative error %.3g", e)
	}
	mp := load("mapper.tflite")
	noise := readBin[float32](t, ref+".noise.bin")
	emb32 := make([]float32, len(emb))
	for i, v := range emb {
		emb32[i] = float32(v)
	}
	t0 = time.Now()
	mapped := run(t, mp, tensor(emb32, onnx.Float, 1, 768), tensor(noise, onnx.Float, 1, 768))
	t.Logf("mapper: %.1f s", time.Since(t0).Seconds())
	if e := relErr(mapped, readBin[float32](t, ref+".mapped.bin")); e > 1e-4 {
		t.Errorf("mapped embedding: relative error %g", e)
	} else {
		t.Logf("mapped embedding: relative error %.3g", e)
	}
	var norm float64
	for _, v := range mapped {
		norm += v * v
	}
	norm = math.Sqrt(norm)
	q := make([]float32, len(mapped))
	for i, v := range mapped {
		q[i] = float32(v / norm)
	}
	vq := load("pretrained_vector_quantizer.tflite")
	codes := run(t, vq, tensor(q, onnx.Float, 1, 768))
	var meta struct {
		Tokens []int `json:"tokens"`
	}
	raw, _ := os.ReadFile(ref + ".json")
	json.Unmarshal(raw, &meta)
	for i, c := range meta.Tokens {
		if int(codes[i]) != c {
			t.Errorf("codes %v, want %v", codes[:12], meta.Tokens)
			break
		}
	}
}
