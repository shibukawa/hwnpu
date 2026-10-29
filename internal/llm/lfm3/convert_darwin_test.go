package lfm3

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/coreml"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/types"
)

// Converts LFM3-1.7B layer by layer and compares the Neural Engine's hidden states and logits
// with the PyTorch reference written by tools/ref/ref_logits.py. Needs HWNPU_LFM3_DIR with
// config.json, the checkpoint and ref_bf16.*.bin. HWNPU_LFM3_LAYERS lists the truncation points
// to check (default "1,3,24"); HWNPU_LFM3_FORMAT picks lut4 (default) or fp16; HWNPU_LFM3_WEIGHTS
// names the checkpoint file (default LFM3-1.7B.safetensors).
func TestAgainstReference(t *testing.T) {
	dir := os.Getenv("HWNPU_LFM3_DIR")
	if dir == "" {
		t.Skip("HWNPU_LFM3_DIR not set")
	}
	ctx := context.Background()
	infos, err := coreml.Provider{}.Enumerate(ctx)
	if err != nil || len(infos) == 0 {
		t.Skip("no Neural Engine")
	}
	cfg, err := LoadConfig(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	wfile := os.Getenv("HWNPU_LFM3_WEIGHTS")
	if wfile == "" {
		wfile = "LFM3-1.7B.safetensors"
	}
	w, err := OpenWeights(filepath.Join(dir, wfile))
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	format := os.Getenv("HWNPU_LFM3_FORMAT")
	if format == "" {
		format = "lut4"
	}
	ids := readI32(t, filepath.Join(dir, "ref_bf16.ids.bin"))
	seq := len(ids)
	hidden := readF32(t, filepath.Join(dir, "ref_bf16.hidden.bin"))
	logits := readF32(t, filepath.Join(dir, "ref_bf16.logits.bin"))
	if len(hidden) != (cfg.NumLayers+1)*seq*cfg.HiddenSize || len(logits) != seq*cfg.VocabSize {
		t.Fatalf("reference sizes do not match the config: hidden %d, logits %d, seq %d", len(hidden), len(logits), seq)
	}
	emb, err := llm.Embed(w, cfg, ids)
	if err != nil {
		t.Fatal(err)
	}
	report("embeddings (host lookup) vs reference", emb, hidden[:seq*cfg.HiddenSize], t)
	x := make([]byte, 2*len(emb))
	for i, v := range emb {
		binary.LittleEndian.PutUint16(x[2*i:], mlpackage.Float16Bits(v))
	}

	dev, err := coreml.Provider{}.Open(ctx, infos[0], backend.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	layerList := os.Getenv("HWNPU_LFM3_LAYERS")
	if layerList == "" {
		layerList = "1,3,24"
	}
	for _, ls := range strings.Split(layerList, ",") {
		layers, _ := strconv.Atoi(ls)
		opts := Options{Seq: seq, Layers: layers, Format: format, Group: envInt("HWNPU_LFM3_GROUP", 16), ColBlock: envInt("HWNPU_LFM3_COLBLOCK", 0), ConvMode: os.Getenv("HWNPU_LFM3_CONV"), Quant: os.Getenv("HWNPU_LFM3_QUANT"), PostScale: float32(envInt("HWNPU_LFM3_POSTSCALE", 1))}
		if keep := os.Getenv("HWNPU_LFM3_KEEP"); keep != "" {
			opts.KeepFP16 = strings.Split(keep, ",")
		}
		t.Logf("options: %+v", opts)
		out := filepath.Join(t.TempDir(), "m.mlpackage")
		pkg, err := mlpackage.Create(out)
		if err != nil {
			t.Fatal(err)
		}
		b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
		start := time.Now()
		in, y, err := Build(b, cfg, w, opts)
		if err != nil {
			t.Fatal(err)
		}
		if err := pkg.Finish(b.Finish(&mlpackage.Var{Name: y.Name}), 9, []mlpackage.Feature{in}, []mlpackage.Feature{y}); err != nil {
			t.Fatal(err)
		}
		built := time.Since(start)
		start = time.Now()
		g, err := dev.Compile(ctx, types.GraphSource{Path: out})
		if err != nil {
			t.Fatalf("layers %d: compile: %v", layers, err)
		}
		p := g.Placement()
		t.Logf("layers %d (%s): built in %.0fs, compiled in %.0fs, npu ops %d, elsewhere %d %v", layers, format, built.Seconds(), time.Since(start).Seconds(), p.NPUOps, len(p.Ops), head(p.Ops, 6))
		got := run(t, dev, g, x)
		g.Close()
		switch {
		case layers < cfg.NumLayers:
			ref := hidden[layers*seq*cfg.HiddenSize : (layers+1)*seq*cfg.HiddenSize]
			report("hidden after layer "+ls, got, ref, t)
		default:
			report("logits", got, logits, t)
			agree := 0
			var kl float64
			for s := 0; s < seq; s++ {
				a, b := got[s*cfg.VocabSize:(s+1)*cfg.VocabSize], logits[s*cfg.VocabSize:(s+1)*cfg.VocabSize]
				if argmax(a) == argmax(b) {
					agree++
				}
				kl += klDiv(b, a)
			}
			t.Logf("argmax agreement %d/%d, mean KL(ref || npu) %.4f nats", agree, seq, kl/float64(seq))
		}
	}
}

func envInt(name string, def int) int {
	if v, err := strconv.Atoi(os.Getenv(name)); err == nil {
		return v
	}
	return def
}

func head(ops []types.OpPlacement, n int) []types.OpPlacement {
	if len(ops) > n {
		return ops[:n]
	}
	return ops
}

func run(t *testing.T, dev backend.Device, g backend.Graph, x []byte) []float32 {
	req, err := g.NewRequest()
	if err != nil {
		t.Fatal(err)
	}
	defer req.Close()
	var out backend.Buffer
	for _, a := range g.Args() {
		buf, err := dev.AllocBuffer(a.Bytes, types.BufferDefault)
		if err != nil {
			t.Fatal(err)
		}
		defer buf.Close()
		if err := req.Bind(a.Index, buf); err != nil {
			t.Fatal(err)
		}
		if a.Kind == types.ArgInput {
			copy(buf.Bytes(), x)
		} else {
			out = buf
		}
	}
	start := time.Now()
	f, err := req.Submit()
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}
	t.Logf("  first run %.1f ms", float64(time.Since(start).Microseconds())/1000)
	raw := out.Bytes()
	vals := make([]float32, len(raw)/2)
	for i := range vals {
		vals[i] = mlpackage.Float16Value(binary.LittleEndian.Uint16(raw[2*i:]))
	}
	return vals
}

// report logs how close got is to ref: relative RMS error, cosine similarity and the largest
// absolute difference.
func report(what string, got, ref []float32, t *testing.T) {
	if len(got) != len(ref) {
		t.Fatalf("%s: %d values, reference %d", what, len(got), len(ref))
	}
	var se, sr, dot, gg, rr, maxAbs, maxRef float64
	for i := range got {
		d := float64(got[i] - ref[i])
		se += d * d
		sr += float64(ref[i]) * float64(ref[i])
		dot += float64(got[i]) * float64(ref[i])
		gg += float64(got[i]) * float64(got[i])
		rr += float64(ref[i]) * float64(ref[i])
		maxAbs = math.Max(maxAbs, math.Abs(d))
		maxRef = math.Max(maxRef, math.Abs(float64(ref[i])))
	}
	t.Logf("%s: rel RMS err %.4f, cosine %.6f, max |diff| %.3g (max |ref| %.3g)", what, math.Sqrt(se/sr), dot/math.Sqrt(gg*rr), maxAbs, maxRef)
}

func argmax(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}

// klDiv is KL(p || q) of the softmax distributions of two logit vectors.
func klDiv(p, q []float32) float64 {
	lp, lq := logSoftmax(p), logSoftmax(q)
	var kl float64
	for i := range lp {
		kl += math.Exp(lp[i]) * (lp[i] - lq[i])
	}
	return kl
}

func logSoftmax(v []float32) []float64 {
	m := float64(v[argmax(v)])
	var z float64
	for _, x := range v {
		z += math.Exp(float64(x) - m)
	}
	lz := math.Log(z) + m
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x) - lz
	}
	return out
}

func readF32(t *testing.T, path string) []float32 {
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return out
}

func readI32(t *testing.T, path string) []int32 {
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]int32, len(raw)/4)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return out
}
