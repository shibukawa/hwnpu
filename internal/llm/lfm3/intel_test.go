package lfm3

import (
	"context"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/chain"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/ovcpu"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// writeIRParts is writeParts for Intel's NPU: the parts are OpenVINO IR.
func writeIRParts(t *testing.T, cfg *Config, w llm.Weights, opts llm.Options, n int) string {
	t.Helper()
	opts.Backend = llm.BackendIntel
	plan, err := llm.PlanSplit(cfg, opts, math.MaxInt64, n)
	if err != nil || len(plan) != n {
		t.Fatalf("plan of %d parts: %v, %v", n, plan, err)
	}
	dir := t.TempDir()
	man := &llm.Manifest{Seq: opts.Seq, Cache: opts.Cache, Backend: llm.BackendIntel}
	for i, p := range plan {
		o := opts
		o.Start, o.Layers = p.Start, p.Layers
		rec, err := llm.WritePart(filepath.Join(dir, fmt.Sprintf("m.q%d.part%d.xml", opts.Seq, i+1)), cfg, w, o, nil)
		if err != nil {
			t.Fatal(err)
		}
		man.Parts = append(man.Parts, rec)
	}
	path := filepath.Join(dir, fmt.Sprintf("m.q%d.parts.json", opts.Seq))
	if err := llm.WriteManifest(path, man); err != nil {
		t.Fatal(err)
	}
	return path
}

// The program with a KV cache lowered to OpenVINO IR, run through internal/llm/chain on
// OpenVINO's CPU (the openvino backend, as hwnpu-chat runs it), gives the logits of the MIL
// program without a cache at every position: the IR's arguments bind by the names decode uses,
// the convolution windows, the rings of the sliding layer and the cached keys and values come
// back through the host buffers, in one part and in several, and two programs (prompts and
// generation) share one cache. Needs HWNPU_OV_PYTHON.
func TestIntelIRCached(t *testing.T) {
	if ovcpu.Python() == "" {
		t.Skip("HWNPU_OV_PYTHON is not set")
	}
	ctx := context.Background()
	dev, backend, err := llm.OpenDevice(ctx, "openvino", "")
	if err != nil {
		t.Skip(err)
	}
	defer dev.Close()
	if backend != llm.BackendIntel {
		t.Fatalf("backend %q", backend)
	}
	cfg := tinyConfig()
	w := llm.NewSynthetic(cfg)
	tokens := []int32{3, 1, 4, 1, 5, 9, 2, 6}
	n := len(tokens)
	emb, err := llm.Embed(w, cfg, tokens)
	if err != nil {
		t.Fatal(err)
	}
	full := runCPU(t, cfg, w, Options{Seq: n, Format: "fp16"}, emb)
	row := func(i int) []float32 { return emb[i*cfg.HiddenSize : (i+1)*cfg.HiddenSize] }
	for _, c := range []struct {
		q, gen, cache, parts int
		format               string
		tol                  float64
	}{
		{1, 0, 8, 1, "fp16", 2e-3},
		{3, 0, 9, 2, "fp16", 2e-3},
		{4, 1, 10, 1, "fp16", 2e-3},
		{4, 1, 10, 3, "int8", 0.05},
	} {
		t.Run(fmt.Sprintf("q%d gen%d cache%d parts%d %s", c.q, c.gen, c.cache, c.parts, c.format), func(t *testing.T) {
			opts := llm.Options{Seq: c.q, Cache: c.cache, Format: c.format, Params: map[string]string{"post-scale": "1"}}
			ch, err := chain.Open(ctx, dev, writeIRParts(t, cfg, w, opts, c.parts))
			if err != nil {
				t.Fatal(err)
			}
			defer ch.Close()
			s, err := decode.New(ch.Program(ctx), cfg)
			if err != nil {
				t.Fatal(err)
			}
			if c.gen > 0 {
				opts.Seq = c.gen
				g, err := ch.Share(ctx, dev, writeIRParts(t, cfg, w, opts, c.parts))
				if err != nil {
					t.Fatal(err)
				}
				defer g.Close()
				if err := s.Add(g.Program(ctx)); err != nil {
					t.Fatal(err)
				}
			}
			worst := 0.0
			for pos := 0; pos < n; {
				// the prompt in steps of q, then one token at a time on the program that generates
				v := min(c.q, n-pos)
				if c.gen > 0 && pos >= c.q {
					v = 1
				}
				var step [][]float32
				for i := 0; i < v; i++ {
					step = append(step, row(pos+i))
				}
				logits, err := s.Step(step)
				if err != nil {
					t.Fatal(err)
				}
				pos += v
				want := full[(pos-1)*cfg.VocabSize : pos*cfg.VocabSize]
				worst = math.Max(worst, relErr(logits, want))
			}
			if !(worst <= c.tol) {
				t.Errorf("relative error %g against the program without a cache", worst)
			}
			t.Logf("worst relative error %.2g", worst)
		})
	}
}

// The real LFM3-1.7B (HWNPU_LFM3_DIR: config.json, tokenizer.json and model.safetensors) in
// Intel's weight formats, generating one token per call on OpenVINO's CPU through
// internal/llm/chain: how often the most likely next token is the fp16 program's, and the
// logits' relative error against it, over a Japanese and an English text. Needs HWNPU_OV_PYTHON.
func TestIntelIRCheckpoint(t *testing.T) {
	dir := os.Getenv("HWNPU_LFM3_DIR")
	if dir == "" || ovcpu.Python() == "" {
		t.Skip("HWNPU_LFM3_DIR or HWNPU_OV_PYTHON is not set")
	}
	ctx := context.Background()
	dev, _, err := llm.OpenDevice(ctx, "openvino", "")
	if err != nil {
		t.Skip(err)
	}
	defer dev.Close()
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
	tok, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	ids := tok.Encode("日本の首都は東京です。東京には多くの美術館や公園があり、春には桜が咲きます。" +
		"The Neural Engine and Intel's NPU both run small language models with little power, " +
		"so that a laptop can answer questions without a server.")
	var rows [][]float32
	for _, id := range ids {
		r, err := llm.TokenRow(w, cfg, id)
		if err != nil {
			t.Fatal(err)
		}
		rows = append(rows, r)
	}
	// FORMAT,SUBSTRING=FORMAT... (llm.Options.FormatFor) in IR on OpenVINO; coreml:FORMAT... the
	// Neural Engine's program (lut tables fitted as hwnpu-chat fits them) on the CPU interpreter
	logits := func(format string) [][]float32 {
		dir, err := os.MkdirTemp("", "lfm3-intel")
		if err != nil {
			t.Fatal(err)
		}
		defer os.RemoveAll(dir)
		f, coreml := strings.CutPrefix(format, "coreml:")
		fs := strings.Split(f, ",")
		opts := llm.Options{Seq: 1, Cache: 128, Format: fs[0], FormatFor: fs[1:], Backend: llm.BackendIntel, Params: map[string]string{}}
		out := filepath.Join(dir, "m.xml")
		if coreml {
			opts.Backend, opts.Quant, out = llm.BackendCoreML, "kmeans-minmax", filepath.Join(dir, "m.mlpackage")
		}
		path, err := llm.Convert(a, m, w, out, "none", 0, opts, io.Discard)
		if err != nil {
			t.Fatal(err)
		}
		var s *decode.Session
		if coreml {
			sess, cl, err := runner.OpenSession(ctx, nil, []string{path}, cfg)
			if err != nil {
				t.Fatal(err)
			}
			defer cl.Close()
			s = sess
		} else {
			ch, err := chain.Open(ctx, dev, path)
			if err != nil {
				t.Fatal(err)
			}
			defer ch.Close()
			if s, err = decode.New(ch.Program(ctx), cfg); err != nil {
				t.Fatal(err)
			}
		}
		var res [][]float32
		for _, r := range rows {
			l, err := s.Step([][]float32{r})
			if err != nil {
				t.Fatal(err)
			}
			res = append(res, l)
		}
		return res
	}
	argmax := func(v []float32) int {
		best := 0
		for i, x := range v {
			if x > v[best] {
				best = i
			}
		}
		return best
	}
	ref := logits("fp16")
	formats := []string{"int8", "int4-g128", "int4"}
	if f := os.Getenv("HWNPU_LFM3_FORMATS"); f != "" {
		formats = strings.Split(f, ";")
	}
	for _, format := range formats {
		got := logits(format)
		same, worst, sum := 0, 0.0, 0.0
		for i := range ref {
			if argmax(got[i]) == argmax(ref[i]) {
				same++
			}
			e := relErr(got[i], ref[i])
			worst, sum = math.Max(worst, e), sum+e
		}
		t.Logf("%s: next token as fp16's at %d of %d positions, logits' relative error %.3f on average, %.3f at worst", format, same, len(ref), sum/float64(len(ref)), worst)
	}
}
