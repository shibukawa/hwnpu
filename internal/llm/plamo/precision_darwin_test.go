package plamo

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/coreml"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
	"github.com/shibukawa/hwnpu/internal/types"
)

// Runs PLaMo-Embedding-1B on the Neural Engine with the residual stream after every layer as an
// output, compares each with the same program run on the CPU in float32 (milcpu), and wants the
// sentence embeddings to agree. It shows where the Neural Engine's fp16 error grows and where a
// value leaves fp16 (Inf or NaN): without head-scale the last MLP's output (5e4) did, and the
// embedding's cosine with the CPU's was 0.975. Needs HWNPU_PLAMO_DIR; HWNPU_PLAMO_FORMAT picks
// the weights (default lut8), HWNPU_PLAMO_HEADSCALE the head-scale, HWNPU_PLAMO_F16 adds the CPU
// with every result rounded to fp16.
func TestANEPrecision(t *testing.T) {
	dir := os.Getenv("HWNPU_PLAMO_DIR")
	if dir == "" {
		t.Skip("HWNPU_PLAMO_DIR not set")
	}
	ctx := context.Background()
	infos, err := coreml.Provider{}.Enumerate(ctx)
	if err != nil || len(infos) == 0 {
		t.Skip("no Neural Engine")
	}
	format := os.Getenv("HWNPU_PLAMO_FORMAT")
	if format == "" {
		format = "lut8"
	}
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
	tok, err := tokenizer.LoadSentencePiece(filepath.Join(dir, "tokenizer.model"))
	if err != nil {
		t.Fatal(err)
	}
	bos, _ := tok.ID("<|startoftext|>")
	text := "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。"
	ids := append([]int32{bos}, tok.Encode(text)...)
	const seq = 64
	n, d := len(ids), cfg.HiddenSize
	mask := make([]float32, seq)
	for i := range ids {
		mask[i] = 1
	}
	for len(ids) < seq {
		ids = append(ids, 3)
	}
	x := embed(t, cfg, w, ids)

	out := filepath.Join(t.TempDir(), "p.mlpackage")
	pkg, err := mlpackage.Create(out)
	if err != nil {
		t.Fatal(err)
	}
	b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
	opts := llm.Options{Seq: seq, Format: format, Quant: "kmeans-minmax", Group: 16, Layers: cfg.NumLayers}
	hs, _ := strconv.ParseFloat(Arch.Param(llm.Options{Params: paramsFromEnv()}, "head-scale"), 64)
	pb := &builder{Graph: &blocks.Graph{Name: "plamo", B: b, W: w, Opts: opts}, cfg: cfg, seq: seq, d: d, headScale: float32(hs)}
	t.Logf("head-scale %g", hs)
	var names []string
	var outs []*mlpackage.Var
	keep := func(name string, v *mlpackage.Var) *mlpackage.Var {
		names = append(names, name)
		outs = append(outs, b.NamedOp("identity", name, map[string]any{"x": v}, mlpackage.Float16, v.Shape...))
		return v
	}
	err = blocks.Run(func() {
		xv := b.Input("x", mlpackage.Float16, 1, seq, d)
		mv := b.Input("mask", mlpackage.Float16, 1, seq)
		bias := pb.MaskBias(mv)
		cos, sin := pb.RopeTables("plamo", cfg.invFreq(), seq, cfg.HeadDim)
		h := xv
		for i := 0; i < cfg.NumLayers; i++ {
			s := float32(1)
			if i == cfg.NumLayers-1 {
				s = pb.headScale
			}
			h = keep(fmt.Sprintf("h%d", i), pb.layer(i, h, bias, cos, sin, s))
		}
		k := float64(pb.headScale)
		keep("y", pb.RMSNorm(h, "norm.weight", d, cfg.NormEps/(k*k), 1))
	})
	if err != nil {
		t.Fatal(err)
	}
	var feats []mlpackage.Feature
	for _, v := range outs {
		feats = append(feats, mlpackage.Feature{Name: v.Name, DType: mlpackage.Float16, Shape: v.Shape})
	}
	prog := b.Finish(outs...)
	if err := pkg.Finish(prog, 9, []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: []int{1, seq, d}}, {Name: "mask", DType: mlpackage.Float16, Shape: []int{1, seq}}}, feats); err != nil {
		t.Fatal(err)
	}

	// the CPU in float32, from the package's own weights
	wf, err := os.Open(filepath.Join(out, "Data", "com.apple.CoreML", "weights", "weight.bin"))
	if err != nil {
		t.Fatal(err)
	}
	cp, err := milcpu.Load(prog, wf)
	if err != nil {
		t.Fatal(err)
	}
	ins := map[string]milcpu.Tensor{"x": {Shape: []int{1, seq, d}, Data: x}, "mask": {Shape: []int{1, seq}, Data: mask}}
	ref, err := cp.Run(ins, milcpu.Options{})
	if err != nil {
		t.Fatal(err)
	}
	var r16 map[string]milcpu.Tensor
	if os.Getenv("HWNPU_PLAMO_F16") != "" {
		if r16, err = cp.Run(ins, milcpu.Options{RoundF16: true}); err != nil {
			t.Fatal(err)
		}
	}
	wf.Close()

	dev, err := coreml.Provider{}.Open(ctx, infos[0], backend.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	g, err := dev.Compile(ctx, types.GraphSource{Path: out})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	pl := g.Placement()
	t.Logf("%s: npu ops %d, elsewhere %d", format, pl.NPUOps, len(pl.Ops))
	got := runAll(t, dev, g, map[string][]float32{"x": x, "mask": mask})

	for _, name := range names {
		k := len(ref[name].Data) / seq // the BOS and the padding are not pooled
		line := compare(got[name][k:n*k], ref[name].Data[k:n*k])
		line += " | BOS " + compare(got[name][:k], ref[name].Data[:k])
		for i, v := range got[name][:n*k] {
			if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
				line += fmt.Sprintf(" | %v at token %d channel %d", v, i/k, i%k)
				break
			}
		}
		if r16 != nil {
			line += " | f16 cpu " + compare(r16[name].Data[k:n*k], ref[name].Data[k:n*k])
		}
		t.Logf("%-6s %s", name, line)
	}
	pool := func(h []float32) []float64 {
		p := make([]float64, d)
		for r := 1; r < n; r++ {
			for j := range p {
				p[j] += float64(h[r*d+j])
			}
		}
		return p
	}
	gp, rp := pool(got["y"]), pool(ref["y"].Data)
	var ab, aa, bb float64
	for j := range gp {
		ab, aa, bb = ab+gp[j]*rp[j], aa+gp[j]*gp[j], bb+rp[j]*rp[j]
	}
	cos := ab / math.Sqrt(aa*bb)
	t.Logf("embedding cosine, Neural Engine against the CPU: %.5f", cos)
	if cos < 0.999 {
		t.Errorf("embedding cosine %.5f with the CPU's, want 0.999 or more", cos)
	}
}

// compare says how far got is from ref: relative RMS error, RMS and largest magnitude of ref.
func compare(got, ref []float32) string {
	var se, sr, mx float64
	for i := range ref {
		e := float64(got[i] - ref[i])
		se += e * e
		sr += float64(ref[i]) * float64(ref[i])
		mx = math.Max(mx, math.Abs(float64(ref[i])))
	}
	return fmt.Sprintf("rel err %.4f  rms %8.3g  max %8.3g", math.Sqrt(se/sr), math.Sqrt(sr/float64(len(ref))), mx)
}

// runAll runs g once with fp16 inputs by name and returns every output by name.
func runAll(t *testing.T, dev backend.Device, g backend.Graph, in map[string][]float32) map[string][]float32 {
	t.Helper()
	req, err := g.NewRequest()
	if err != nil {
		t.Fatal(err)
	}
	defer req.Close()
	outs := map[string]backend.Buffer{}
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
			for i, v := range in[a.Name] {
				binary.LittleEndian.PutUint16(buf.Bytes()[2*i:], mlpackage.Float16Bits(v))
			}
		} else {
			outs[a.Name] = buf
		}
	}
	f, err := req.Submit()
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}
	res := map[string][]float32{}
	for name, buf := range outs {
		raw := buf.Bytes()
		v := make([]float32, len(raw)/2)
		for i := range v {
			v[i] = mlpackage.Float16Value(binary.LittleEndian.Uint16(raw[2*i:]))
		}
		res[name] = v
	}
	return res
}

// paramsFromEnv reads HWNPU_PLAMO_HEADSCALE into the architecture's settings.
func paramsFromEnv() map[string]string {
	if v := os.Getenv("HWNPU_PLAMO_HEADSCALE"); v != "" {
		return map[string]string{"head-scale": v}
	}
	return nil
}
