package lfm3

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/llm/ovrun"
	"github.com/shibukawa/hwnpu/internal/ovcpu"
)

// The program with a KV cache written as OpenVINO IR for Intel's NPU, stepped through a sequence
// with OpenVINO's CPU plugin (internal/ovcpu), gives the logits of the MIL program without a
// cache at every position, in one part and in several, with the cache in chunks too. Needs
// HWNPU_OV_PYTHON.
func TestIntelCachedMatchesFull(t *testing.T) {
	py := ovcpu.Python()
	if py == "" {
		t.Skip("HWNPU_OV_PYTHON is not set")
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
		q, cache, parts int
		chunked         bool
	}{{1, 8, 1, false}, {3, 9, 2, false}, {3, 9, 1, true}} {
		t.Run(fmt.Sprintf("q%d cache%d parts%d chunked %v", c.q, c.cache, c.parts, c.chunked), func(t *testing.T) {
			if c.chunked {
				defer func(over, n int) { llm.CacheChunkOver, llm.CacheChunkLen = over, n }(llm.CacheChunkOver, llm.CacheChunkLen)
				llm.CacheChunkOver, llm.CacheChunkLen = 5, 4
			}
			opts := llm.Options{Seq: c.q, Cache: c.cache, Format: "fp16", Backend: "intel", Params: map[string]string{"post-scale": "1"}}
			p, err := ovrun.Program(py, writeParts(t, cfg, w, opts, c.parts), ovcpu.Options{})
			if err != nil {
				t.Fatal(err)
			}
			s, err := decode.New(p, cfg)
			if err != nil {
				t.Fatal(err)
			}
			worst := 0.0
			for pos := 0; pos < n; {
				v := min(c.q, n-pos)
				var step [][]float32
				for i := 0; i < v; i++ {
					step = append(step, row(pos+i))
				}
				logits, err := s.Step(step)
				if err != nil {
					t.Fatal(err)
				}
				pos += v
				worst = math.Max(worst, relErr(logits, full[(pos-1)*cfg.VocabSize:pos*cfg.VocabSize]))
			}
			if worst > 2e-3 {
				t.Errorf("relative error %g against the program without a cache", worst)
			}
			t.Logf("worst relative error %.2g", worst)
		})
	}
}

// LFM3-1.7B written as OpenVINO IR in Intel's weight formats, run with OpenVINO's CPU plugin on
// the prompt of tools/ref/ref_logits.py, against the PyTorch logits (ref_bf16.*.bin): the mean KL
// divergence and how many of the next token argmaxes agree. Needs HWNPU_LFM3_DIR (config.json,
// LFM3-1.7B.safetensors and the reference) and HWNPU_OV_PYTHON; HWNPU_LFM3_FORMATS lists the
// formats (default int4,int4-g128,int8). On the Neural Engine lut4 gives KL 0.95 (9 of 24).
func TestIntelIRCheckpoint(t *testing.T) {
	dir, py := os.Getenv("HWNPU_LFM3_DIR"), ovcpu.Python()
	if dir == "" || py == "" {
		t.Skip("HWNPU_LFM3_DIR or HWNPU_OV_PYTHON is not set")
	}
	cfg, err := LoadConfig(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	w, err := OpenWeights(filepath.Join(dir, "LFM3-1.7B.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	ids := readI32(t, filepath.Join(dir, "ref_bf16.ids.bin"))
	ref := readF32(t, filepath.Join(dir, "ref_bf16.logits.bin"))
	seq := len(ids)
	vocab := len(ref) / seq
	x, err := llm.Embed(w, cfg, ids)
	if err != nil {
		t.Fatal(err)
	}
	formats := "int4,int4-g128,int8"
	if f := os.Getenv("HWNPU_LFM3_FORMATS"); f != "" {
		formats = f
	}
	for _, format := range strings.Split(formats, ",") {
		path := filepath.Join(t.TempDir(), "m.xml")
		if _, err := llm.WritePart(path, cfg, w, llm.Options{Seq: seq, Format: format, Backend: "intel"}, nil); err != nil {
			t.Fatal(err)
		}
		res, err := ovcpu.Run(py, path, map[string][]float32{"x": x}, ovcpu.Options{})
		if err != nil {
			t.Fatal(err)
		}
		y := res["y"]
		if len(y) != seq*vocab {
			t.Fatalf("%s: %d logits, the reference has %d", format, len(y), seq*vocab)
		}
		var kl float64
		agree := 0
		for i := 0; i < seq; i++ {
			got, want := y[i*vocab:(i+1)*vocab], ref[i*vocab:(i+1)*vocab]
			kl += klDiv(want, got)
			if argmax(got) == argmax(want) {
				agree++
			}
		}
		t.Logf("%s: KL %.3f against PyTorch, argmax %d of %d", format, kl/float64(seq), agree, seq)
		os.RemoveAll(filepath.Dir(path))
	}
}
