package mlpackage

import (
	"context"
	"encoding/binary"
	"math"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/coreml"
	"github.com/shibukawa/hwnpu/internal/types"
)

// Measures the Neural Engine's error on one fp16 linear for inputs of different distributions,
// to learn whether its accuracy depends on the input's dynamic range (as it would if activations
// were quantized with one shared scale per tensor).
func TestActivationRange(t *testing.T) {
	ctx := context.Background()
	infos, err := coreml.Provider{}.Enumerate(ctx)
	if err != nil || len(infos) == 0 {
		t.Skip("no Neural Engine")
	}
	dev, err := coreml.Provider{}.Open(ctx, infos[0], backend.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	const seq, d = 32, 2048
	rng := rand.New(rand.NewSource(7))
	w := make([]float32, d*d)
	for i := range w {
		w[i] = Float16Value(Float16Bits(float32(rng.NormFloat64()) * 0.02))
	}
	dir := filepath.Join(t.TempDir(), "a.mlpackage")
	pkg, err := Create(dir)
	if err != nil {
		t.Fatal(err)
	}
	b := NewBuilder("CoreML8", pkg.Blobs)
	x := b.Input("x", Float16, 1, seq, d)
	wv, _ := b.F16s("w", []int{d, d}, w)
	h := b.Op("linear", map[string]any{"x": x, "weight": wv}, Float16, 1, seq, d)
	// a second op so the plan does not fall back to the CPU for a one op graph
	h = b.Op("add", map[string]any{"x": h, "y": b.ScalarF16(0)}, Float16, 1, seq, d)
	y := b.NamedOp("identity", "y", map[string]any{"x": h}, Float16, 1, seq, d)
	if err := pkg.Finish(b.Finish(y), 9, []Feature{{"x", Float16, x.Shape}}, []Feature{{"y", Float16, y.Shape}}); err != nil {
		t.Fatal(err)
	}
	g, err := dev.Compile(ctx, types.GraphSource{Path: dir})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	p := g.Placement()
	t.Logf("npu ops %d, elsewhere %v", p.NPUOps, p.Ops)

	cases := []struct {
		name          string
		rms, outlier  float32
		outlierEveryN int
	}{
		{"normal rms 1", 1, 0, 0},
		{"normal rms 0.03", 0.03, 0, 0},
		{"rms 0.03 + outliers 0.5 (1 in 64)", 0.03, 0.5, 64},
		{"rms 1 + outliers 16 (1 in 64)", 1, 16, 64},
		{"rms 0.03 + one outlier 2 per row", 0.03, 2, d},
	}
	for _, c := range cases {
		in := make([]float32, seq*d)
		for i := range in {
			in[i] = float32(rng.NormFloat64()) * c.rms
			if c.outlierEveryN > 0 && i%c.outlierEveryN == 7 {
				in[i] = c.outlier
			}
			in[i] = Float16Value(Float16Bits(in[i]))
		}
		ref := make([]float32, seq*d)
		for s := 0; s < seq; s++ {
			for o := 0; o < d; o++ {
				var acc float64
				for k := 0; k < d; k++ {
					acc += float64(in[s*d+k]) * float64(w[o*d+k])
				}
				ref[s*d+o] = float32(acc)
			}
		}
		req, _ := g.NewRequest()
		var out backend.Buffer
		for _, a := range g.Args() {
			buf, _ := dev.AllocBuffer(a.Bytes, types.BufferDefault)
			defer buf.Close()
			req.Bind(a.Index, buf)
			if a.Kind == types.ArgInput {
				for i, v := range in {
					binary.LittleEndian.PutUint16(buf.Bytes()[2*i:], Float16Bits(v))
				}
			} else {
				out = buf
			}
		}
		f, err := req.Submit()
		if err != nil {
			t.Fatal(err)
		}
		if err := f.Wait(ctx); err != nil {
			t.Fatal(err)
		}
		var se, sr float64
		for i := range ref {
			got := float64(Float16Value(binary.LittleEndian.Uint16(out.Bytes()[2*i:])))
			se += (got - float64(ref[i])) * (got - float64(ref[i]))
			sr += float64(ref[i]) * float64(ref[i])
		}
		req.Close()
		t.Logf("%-36s rel RMS err %.5f", c.name, math.Sqrt(se/sr))
	}
}
