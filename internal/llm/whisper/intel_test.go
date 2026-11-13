package whisper

import (
	"context"
	"fmt"
	"io"
	"math"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/ovcpu"
)

// On OpenVINO's CPU: the three programs converted to OpenVINO IR as hwnpu-audio transcribe
// converts them for Intel's NPU give what the MIL programs give: the encoder's last hidden state,
// the cross program's keys and values, and the decoder with a KV cache (programs of 1 and 3
// tokens per call sharing it) the logits of WhisperDecoder at every step. Needs HWNPU_OV_PYTHON.
func TestIntelIR(t *testing.T) {
	if ovcpu.Python() == "" {
		t.Skip("HWNPU_OV_PYTHON is not set")
	}
	ctx := context.Background()
	dev, _, err := llm.OpenDevice(ctx, "openvino", "")
	if err != nil {
		t.Skip(err)
	}
	defer dev.Close()
	dir := t.TempDir()
	convert := func(a *llm.Arch, m llm.Model, w llm.Weights, name string, opts llm.Options) string {
		t.Helper()
		opts.Backend = llm.BackendIntel
		path, err := llm.Convert(a, m, w, filepath.Join(dir, name), "none", 0, opts, io.Discard)
		if err != nil {
			t.Fatal(err)
		}
		return path
	}
	run := func(path string, x []float32) []float32 {
		t.Helper()
		p, err := runner.Open(ctx, dev, path)
		if err != nil {
			t.Fatal(err)
		}
		defer p.Close()
		y, err := p.Run(ctx, map[string][]float32{"x": x})
		if err != nil {
			t.Fatal(err)
		}
		return y
	}
	t.Run("encoder", func(t *testing.T) {
		e := tinyEncoder(t, "")
		w := llm.NewSynthetic(e)
		mel := f32(randomMel(e, 1))
		want := runEncoder(t, e, w, llm.Options{Format: "fp16"}, mel)
		got := run(convert(EncoderArch, e, w, "enc.xml", llm.Options{Format: "fp16"}), mel)
		if e := relErr32(got, want); !(e <= 2e-3) {
			t.Errorf("relative error %g against the MIL program", e)
		} else {
			t.Logf("relative error %.2g", e)
		}
	})
	d := tinyDecoder(t, "")
	w := weights(d)
	enc := randomEncoding(d.Config, 4)
	kv := crossKV(t, d.Config, w, enc)
	t.Run("cross", func(t *testing.T) {
		got := run(convert(CrossArch, &Cross{d.Config}, w, "cross.xml", llm.Options{Format: "fp16"}), f32(enc))
		var want []float32
		for i := 0; i < d.DecoderLayers; i++ {
			want = append(append(want, kv[fmt.Sprintf("xk_%d", i)]...), kv[fmt.Sprintf("xv_%d", i)]...)
		}
		if e := relErr32(got, want); !(e <= 2e-3) {
			t.Errorf("relative error %g against the MIL program", e)
		} else {
			t.Logf("relative error %.2g", e)
		}
	})
	t.Run("decoder", func(t *testing.T) {
		_, want := decoderRef(t, d, w, tokens, enc)
		var paths []string
		for _, q := range []int{1, 3} {
			paths = append(paths, convert(Arch, d, w, fmt.Sprintf("d-q%d.xml", q), llm.Options{Seq: q, Cache: 16, Format: "fp16"}))
		}
		s, cl, err := runner.OpenSession(ctx, dev, paths, d)
		if err != nil {
			t.Fatal(err)
		}
		defer cl.Close()
		for name, v := range kv {
			b := make([]byte, 2*len(v))
			hwnpu.EncodeF16(b, v)
			if err := s.SetInputF16(name, b); err != nil {
				t.Fatal(err)
			}
		}
		worst := 0.0
		for _, st := range [][2]int{{0, 1}, {1, 3}, {3, 6}, {6, 7}, {7, 9}} {
			var rows [][]float32
			for i := st[0]; i < st[1]; i++ {
				row, err := llm.TokenRow(w, d, tokens[i])
				if err != nil {
					t.Fatal(err)
				}
				rows = append(rows, row)
			}
			logits, err := s.Step(rows)
			if err != nil {
				t.Fatal(err)
			}
			pos := st[1]
			worst = math.Max(worst, relErr(logits, want[(pos-1)*d.VocabSize:pos*d.VocabSize]))
		}
		if !(worst <= 5e-3) {
			t.Errorf("relative error %g against the reference", worst)
		}
		t.Logf("worst relative error %.2g", worst)
	})
}

func relErr32(got, want []float32) float64 {
	var se, sw float64
	for i := range want {
		d := float64(got[i] - want[i])
		se += d * d
		sw += float64(want[i]) * float64(want[i])
	}
	if len(got) != len(want) {
		return math.Inf(1)
	}
	return math.Sqrt(se / math.Max(sw, 1e-30))
}
