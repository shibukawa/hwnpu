package sfx

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

func textWeights(t *testing.T, dir string) *Weights {
	t.Helper()
	w, err := OpenWeights(nil, map[string][]string{"": {
		filepath.Join(dir, "text_encoder", "model-00001-of-00002.safetensors"),
		filepath.Join(dir, "text_encoder", "model-00002-of-00002.safetensors"),
	}})
	if err != nil {
		t.Fatal(err)
	}
	return w
}

func readI32(t *testing.T, path string) []int32 {
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	v := make([]int32, len(b)/4)
	for i := range v {
		v[i] = int32(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v
}

// The prompt's tokens are the reference's (HWNPU_SFX_REF of tools/ref/ref_moss_sfx.py text, the
// prompt in HWNPU_SFX_PROMPT with --seconds 10).
func TestPromptIDs(t *testing.T) {
	dir, ref := refDir(t)
	prompt := os.Getenv("HWNPU_SFX_PROMPT")
	if prompt == "" {
		t.Skip("HWNPU_SFX_PROMPT is not set")
	}
	tok, err := tokenizer.Load(filepath.Join(dir, "tokenizer", "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	got := tok.Encode(Prompt(prompt, 10))
	want := readI32(t, ref+".ids.bin")
	if !slices.Equal(got, want) {
		t.Fatalf("ids %v, want %v", got, want)
	}
}

// On the Neural Engine (HWNPU_SFX_DEVICE; the format in HWNPU_SFX_TEXT_FORMAT, fp16 by default)
// the text encoder's output is the reference's.
func TestTextOnDevice(t *testing.T) {
	dir, ref := refDir(t)
	prompt := os.Getenv("HWNPU_SFX_PROMPT")
	if prompt == "" || os.Getenv("HWNPU_SFX_DEVICE") == "" {
		t.Skip("HWNPU_SFX_PROMPT and HWNPU_SFX_DEVICE are not set")
	}
	format := os.Getenv("HWNPU_SFX_TEXT_FORMAT")
	if format == "" {
		format = "fp16"
	}
	ctx := context.Background()
	dev, _, err := llm.OpenDevice(ctx, "", "")
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	w := textWeights(t, dir)
	defer w.Close()
	cfg, err := TextConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	out := os.Getenv("HWNPU_SFX_WRITE")
	if out == "" {
		out = t.TempDir()
	}
	base := filepath.Join(out, "text-"+format)
	var paths []string
	if _, err := os.Stat(base + "-0.mlpackage"); err == nil {
		for i := range TextSplit(format) {
			paths = append(paths, base+"-"+string(rune('0'+i))+".mlpackage")
		}
	} else if paths, err = writeText(cfg, w, textOpts(format), base, nil); err != nil {
		t.Fatal(err)
	}
	var parts []Program
	for _, p := range paths {
		t0 := time.Now()
		prog, err := runner.Open(ctx, dev, p)
		if err != nil {
			t.Fatal(err)
		}
		defer prog.Close()
		t.Logf("%s: loaded in %s, on the NPU: %v", filepath.Base(p), time.Since(t0).Round(time.Millisecond), prog.OnNPU())
		parts = append(parts, prog)
	}
	tok, err := tokenizer.Load(filepath.Join(dir, "tokenizer", "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	e := NewTextEncoder(tok, w, cfg, parts)
	t0 := time.Now()
	got, ids, err := e.Encode(ctx, Prompt(prompt, 10))
	if err != nil {
		t.Fatal(err)
	}
	want := readF32(t, ref+".context.bin")
	n := len(ids) * cfg.HiddenSize
	t.Logf("%d tokens in %s: %.3g off the reference (relative RMS)", len(ids), time.Since(t0).Round(time.Millisecond), relRMS(got[:n], want[:n]))
	var cos []string
	for r := range len(ids) {
		d := cfg.HiddenSize
		cos = append(cos, fmt.Sprintf("%.3f", cosine(got[r*d:(r+1)*d], want[r*d:(r+1)*d])))
	}
	t.Logf("cosines by row: %v", cos)
	if e := relRMS(got[:n], want[:n]); !(e < 0.05) {
		t.Fatalf("%.3g off the reference", e)
	}
}

func cosine(a, b []float32) float64 {
	var ab, aa, bb float64
	for i := range a {
		ab += float64(a[i]) * float64(b[i])
		aa += float64(a[i]) * float64(a[i])
		bb += float64(b[i]) * float64(b[i])
	}
	return ab / math.Sqrt(aa*bb)
}

// textOpts are the options of HWNPU_SFX_QUANT and HWNPU_SFX_FORMAT_FOR (comma separated) for
// trying the text encoder's formats.
func textOpts(format string) llm.Options {
	o := TextOptions(format)
	o.Quant = os.Getenv("HWNPU_SFX_QUANT")
	if ff := os.Getenv("HWNPU_SFX_FORMAT_FOR"); ff != "" {
		o.FormatFor = strings.Split(ff, ",")
	}
	return o
}
