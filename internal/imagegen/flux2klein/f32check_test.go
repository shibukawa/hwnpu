package flux2klein

// TestF32Check (opt-in) is the first step's velocity of the DiT, from the inputs of a run of an
// export of Convert, on the CPU interpreter in float32 (Core ML packages of the same parts) and on
// the Neural Engine (the export's programs), to tell how far the export is from float32 and why
// (design.md 16.227):
//
//	HWNPU_KLEIN_F32=cpu HWNPU_KLEIN_FORMAT=fp16   the parts with fp16 weights on the CPU: v_cpufp16.f32
//	HWNPU_KLEIN_F32=cpu HWNPU_KLEIN_FORMAT=lut8   the parts in the export's format on the CPU: v_cpulut8.f32
//	HWNPU_KLEIN_F32=npu                           the export's programs: v_npu.f32, against v_cpufp16.f32
//	HWNPU_KLEIN_F32=group                         each group alone on the CPU's (lut8) input
//
// HWNPU_KLEIN_EXPORT is the export (its default size), HWNPU_KLEIN_REPO the checkpoint's
// repository, HWNPU_KLEIN_OUT a directory with dump/ctx.bf16 of a run with --dump (the text rows:
// both sides take the same), where the files are written. The seed is 1.

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/imagegen/flux2klein/mil"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/milcpu"
)

func readF32s(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

func writeF32s(path string, x []float32) {
	b := make([]byte, 4*len(x))
	for i, v := range x {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
	}
	os.WriteFile(path, b, 0o644)
}

func TestF32Check(t *testing.T) {
	mode := os.Getenv("HWNPU_KLEIN_F32")
	if mode == "" {
		t.Skip()
	}
	exp, root, out := os.Getenv("HWNPU_KLEIN_EXPORT"), os.Getenv("HWNPU_KLEIN_REPO"), os.Getenv("HWNPU_KLEIN_OUT")
	if exp == "" || root == "" || out == "" {
		t.Fatal("HWNPU_KLEIN_EXPORT, HWNPU_KLEIN_REPO and HWNPU_KLEIN_OUT")
	}
	m, err := ReadProgramManifest(exp)
	if err != nil {
		t.Fatal(err)
	}
	te, err := readTimeFile(filepath.Join(exp, TimeFile), m.Dit.Hidden, 256)
	if err != nil {
		t.Fatal(err)
	}
	c := m.Dit
	E := c.Hidden
	ctxRows := fromBF16s(must(os.ReadFile(filepath.Join(out, "dump/ctx.bf16"))))
	for i := range ctxRows {
		ctxRows[i] /= mil.DitScale
	}
	p := &Pipeline{C: c}
	x := p.Noise(1)
	mods := te.modulation(1)
	md := func(lo, hi int) []float32 { return mods[lo*E : hi*E] }
	type grp struct {
		name string
		mk   func() *mil.Part
		in   func(h []float32) map[string][]float32
	}
	dc, err := mil.ReadDit(root, c.Width, c.Height)
	if err != nil {
		t.Fatal(err)
	}
	var gs []grp
	for i, g := range m.Doubles {
		g := g
		first := i == 0
		gs = append(gs, grp{g.Name, func() *mil.Part { return mil.Doubles(dc, g.First, g.Count) }, func(h []float32) map[string][]float32 {
			if first {
				return map[string][]float32{"x": x, "ctx": ctxRows, "mods": md(mil.ModImage, mil.ModSingle)}
			}
			return map[string][]float32{"x": h, "mods": md(mil.ModImage, mil.ModSingle)}
		}})
	}
	for i, g := range m.Singles {
		g := g
		last := i == len(m.Singles)-1
		gs = append(gs, grp{g.Name, func() *mil.Part { return mil.Singles(dc, g.First, g.Count) }, func(h []float32) map[string][]float32 {
			if last {
				return map[string][]float32{"x": h, "mods": md(mil.ModSingle, mil.ModRows)}
			}
			return map[string][]float32{"x": h, "mods": md(mil.ModSingle, mil.ModOut)}
		}})
	}
	ctx := context.Background()
	var h []float32
	switch mode {
	case "cpu":
		fmt := os.Getenv("HWNPU_KLEIN_FORMAT")
		if fmt == "" {
			fmt = "fp16"
		}
		w, err := mil.OpenWeights(root, dc)
		if err != nil {
			t.Fatal(err)
		}
		defer w.Close()
		for _, g := range gs {
			path := filepath.Join(out, "packages-"+fmt, g.name+".mlpackage")
			if _, err := os.Stat(path); err != nil {
				if _, err := llm.WritePart(path, g.mk(), w, llm.Options{Backend: llm.BackendCoreML, Format: fmt}, nil); err != nil {
					t.Fatal(err)
				}
			}
			prog, closer, err := milcpu.OpenPackage(path)
			if err != nil {
				t.Fatal(err)
			}
			in := map[string]milcpu.Tensor{}
			for k, v := range g.in(h) {
				shape := map[string][]int{"x": {1, len(v) / E, E}, "ctx": {1, c.TextRows, E}, "mods": {1, len(v) / E, E}}[k]
				if k == "x" && len(v) == c.Tokens()*c.Latent {
					shape = []int{1, c.Tokens(), c.Latent}
				}
				in[k] = milcpu.Tensor{Shape: shape, Data: v}
			}
			o, err := prog.Run(in, milcpu.Options{})
			closer.Close()
			if err != nil {
				t.Fatal(err)
			}
			h = o["y"].Data
			writeF32s(filepath.Join(out, "g_"+fmt+"_"+g.name+".f32"), h)
			t.Logf("%s on the CPU", g.name)
		}
		writeF32s(filepath.Join(out, "v_cpu"+fmt+".f32"), h)
	case "group":
		dev, _, err := llm.OpenDevice(ctx, llm.BackendCoreAI, cache.Root(), hwnpu.WithPlacement(hwnpu.PlacementStrict))
		if err != nil {
			t.Fatal(err)
		}
		defer dev.Close()
		var prev []float32
		for _, g := range gs {
			prog, err := runner.Open(ctx, dev, filepath.Join(exp, g.name))
			if err != nil {
				t.Fatal(err)
			}
			y, err := prog.Run(ctx, g.in(prev))
			prog.Close()
			if err != nil {
				t.Fatal(err)
			}
			want := readF32s(must(os.ReadFile(filepath.Join(out, "g_lut8_"+g.name+".f32"))))
			var num, den float64
			for i := range want {
				d := float64(y[i] - want[i])
				num, den = num+d*d, den+float64(want[i])*float64(want[i])
			}
			t.Logf("%s alone (the CPU's input): relative RMS %.4f", g.name, math.Sqrt(num/den))
			prev = want
		}
	case "npu":
		dev, _, err := llm.OpenDevice(ctx, llm.BackendCoreAI, cache.Root(), hwnpu.WithPlacement(hwnpu.PlacementStrict))
		if err != nil {
			t.Fatal(err)
		}
		defer dev.Close()
		for _, g := range gs {
			prog, err := runner.Open(ctx, dev, filepath.Join(exp, g.name))
			if err != nil {
				t.Fatal(err)
			}
			h, err = prog.Run(ctx, g.in(h))
			prog.Close()
			if err != nil {
				t.Fatal(err)
			}
		}
		writeF32s(filepath.Join(out, "v_npu.f32"), h)
		cpu := readF32s(must(os.ReadFile(filepath.Join(out, "v_cpufp16.f32"))))
		var num, den, dot, nc, nn float64
		for i := range cpu {
			d := float64(h[i] - cpu[i])
			num, den = num+d*d, den+float64(cpu[i])*float64(cpu[i])
			dot, nc, nn = dot+float64(h[i])*float64(cpu[i]), nc+float64(cpu[i])*float64(cpu[i]), nn+float64(h[i])*float64(h[i])
		}
		t.Logf("the Neural Engine against the CPU (fp16 weights, float32): relative RMS %.4f, cosine %.6f", math.Sqrt(num/den), dot/math.Sqrt(nc*nn))
	}
}

func must(b []byte, err error) []byte {
	if err != nil {
		panic(err)
	}
	return b
}
