package coreai

import (
	"context"
	"math"
	"math/rand"
	"path/filepath"
	"testing"

	"encoding/binary"
	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/types"
)

// The operations of the image models' ONNX graphs (internal/onnx: PP-OCR, RTMDet, RTMO, YuNet)
// and the depth heads that ailower lowers for them: max_pool with pads and without, a 5 x 5
// pool of stride 1, and conv_transpose with a crop at the top and left and zeros past the
// bottom's end, and DPT's 2 x 2 of stride 2. On the Neural Engine with Core AI, against milcpu.
func TestImageOps(t *testing.T) {
	infos, err := Provider{}.Enumerate(context.Background())
	if err != nil || len(infos) == 0 {
		t.Skip("no Core AI:", err)
	}
	t.Setenv("HWNPU_CACHE", t.TempDir())
	dir := filepath.Join(t.TempDir(), "m.mlpackage")
	pkg, err := mlpackage.Create(dir)
	if err != nil {
		t.Fatal(err)
	}
	b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
	f16 := mlpackage.Float16
	rng := rand.New(rand.NewSource(1))
	consts := func(name string, shape ...int) *mlpackage.Var {
		n := 1
		for _, d := range shape {
			n *= d
		}
		v := make([]float32, n)
		for i := range v {
			v[i] = rng.Float32() - 0.5
		}
		c, err := b.F16s(name, shape, v)
		if err != nil {
			t.Fatal(err)
		}
		return c
	}
	x := b.Input("x", f16, 1, 4, 12, 12)
	mp := b.Op("max_pool", map[string]any{"x": x, "kernel_sizes": b.Ints(3, 3), "strides": b.Ints(2, 2),
		"pad_type": b.Str("c", "custom"), "pad": b.Ints(1, 1, 1, 1), "ceil_mode": b.ScalarBool(false)}, f16, 1, 4, 6, 6)
	mp2 := b.Op("max_pool", map[string]any{"x": x, "kernel_sizes": b.Ints(2, 2), "strides": b.Ints(2, 2),
		"pad_type": b.Str("c", "valid"), "ceil_mode": b.ScalarBool(false)}, f16, 1, 4, 6, 6)
	mp5 := b.Op("max_pool", map[string]any{"x": x, "kernel_sizes": b.Ints(5, 5), "strides": b.Ints(1, 1),
		"pad_type": b.Str("c", "custom"), "pad": b.Ints(2, 2, 2, 2), "ceil_mode": b.ScalarBool(false)}, f16, 1, 4, 12, 12)
	h := b.Op("add", map[string]any{"x": mp, "y": mp2}, f16, 1, 4, 6, 6)
	// (Cin 4, Cout 3, 3, 4), stride (2, 3): the full output 13 x 19, 1 cropped at the top and
	// left, 2 rows of zeros (then the bias) past the bottom's end
	ct := b.Op("conv_transpose", map[string]any{"x": h, "weight": consts("w", 4, 3, 3, 4), "bias": consts("bias", 3),
		"strides": b.Ints(2, 3), "pad_type": b.Str("c", "custom"), "pad": b.Ints(1, -2, 1, 0),
		"dilations": b.Ints(1, 1), "groups": b.ScalarI32(1)}, f16, 1, 3, 14, 18)
	up := b.Op("conv_transpose", map[string]any{"x": h, "weight": consts("w2", 4, 8, 2, 2), "bias": consts("bias2", 8),
		"strides": b.Ints(2, 2), "pad_type": b.Str("c", "valid"), "dilations": b.Ints(1, 1), "groups": b.ScalarI32(1)}, f16, 1, 8, 12, 12)
	// PP-OCR's: pixel_shuffle (its detector's upsampling), the exact gelu as erf, relu, clip
	ps := b.Op("pixel_shuffle", map[string]any{"x": x, "upscale_factor": b.ScalarI32(2)}, f16, 1, 1, 24, 24)
	// erf over -8 to 8 (the exact gelu's erf(x / sqrt 2) of PP-OCR's activations)
	er := b.Op("erf", map[string]any{"x": b.Op("mul", map[string]any{"x": x, "y": b.ScalarF16(3)}, f16, 1, 4, 12, 12)}, f16, 1, 4, 12, 12)
	re := b.Op("relu", map[string]any{"x": x}, f16, 1, 4, 12, 12)
	cl := b.Op("clip", map[string]any{"x": x, "alpha": b.ScalarF16(-0.5), "beta": b.ScalarF16(0.75)}, f16, 1, 4, 12, 12)
	// MoGe-2's head: pad replicate; and reflect
	rp := b.Op("pad", map[string]any{"x": x, "pad": b.Ints(1, 2, 2, 1), "mode": b.Str("c", "replicate"), "constant_val": b.ScalarF16(0)}, f16, 1, 4, 15, 15)
	rf := b.Op("pad", map[string]any{"x": x, "pad": b.Ints(2, 1, 1, 2), "mode": b.Str("c", "reflect"), "constant_val": b.ScalarF16(0)}, f16, 1, 4, 15, 15)
	// the audio codecs': elu (alpha 1, all the CPU interpreter knows)
	el := b.Op("elu", map[string]any{"x": x, "alpha": b.ScalarF16(1)}, f16, 1, 4, 12, 12)
	outs := []*mlpackage.Var{
		b.NamedOp("identity", "el", map[string]any{"x": el}, f16, 1, 4, 12, 12),
		b.NamedOp("identity", "rp", map[string]any{"x": rp}, f16, 1, 4, 15, 15),
		b.NamedOp("identity", "rf", map[string]any{"x": rf}, f16, 1, 4, 15, 15),
		b.NamedOp("identity", "ps", map[string]any{"x": ps}, f16, 1, 1, 24, 24),
		b.NamedOp("identity", "er", map[string]any{"x": er}, f16, 1, 4, 12, 12),
		b.NamedOp("identity", "re", map[string]any{"x": re}, f16, 1, 4, 12, 12),
		b.NamedOp("identity", "cl", map[string]any{"x": cl}, f16, 1, 4, 12, 12),
		b.NamedOp("identity", "ct", map[string]any{"x": ct}, f16, 1, 3, 14, 18),
		b.NamedOp("identity", "up", map[string]any{"x": up}, f16, 1, 8, 12, 12),
		b.NamedOp("identity", "mp5", map[string]any{"x": mp5}, f16, 1, 4, 12, 12),
	}
	prog := b.Finish(outs...)
	feats := []mlpackage.Feature{{Name: "el", DType: f16, Shape: []int{1, 4, 12, 12}}, {Name: "rp", DType: f16, Shape: []int{1, 4, 15, 15}}, {Name: "rf", DType: f16, Shape: []int{1, 4, 15, 15}}, {Name: "ps", DType: f16, Shape: []int{1, 1, 24, 24}},
		{Name: "er", DType: f16, Shape: []int{1, 4, 12, 12}}, {Name: "re", DType: f16, Shape: []int{1, 4, 12, 12}},
		{Name: "cl", DType: f16, Shape: []int{1, 4, 12, 12}}, {Name: "ct", DType: f16, Shape: []int{1, 3, 14, 18}},
		{Name: "up", DType: f16, Shape: []int{1, 8, 12, 12}}, {Name: "mp5", DType: f16, Shape: []int{1, 4, 12, 12}}}
	if err := pkg.Finish(prog, 9, []mlpackage.Feature{{Name: "x", DType: f16, Shape: []int{1, 4, 12, 12}}}, feats); err != nil {
		t.Fatal(err)
	}

	ctx := context.Background()
	dev, err := Provider{}.Open(ctx, types.DeviceInfo{}, backend.Options{CacheDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	g, err := dev.Compile(ctx, types.GraphSource{Path: dir})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	req, err := g.NewRequest()
	if err != nil {
		t.Fatal(err)
	}
	in := map[string]milcpu.Tensor{}
	bufs := map[string]backend.Buffer{}
	for _, a := range g.Args() {
		buf, err := dev.AllocBuffer(a.Bytes, 0)
		if err != nil {
			t.Fatal(err)
		}
		defer buf.Close()
		req.Bind(a.Index, buf)
		bufs[a.Name] = buf
		if a.Kind == types.ArgInput {
			v := make([]float32, a.Bytes/2)
			for i := range v {
				v[i] = float32(rng.NormFloat64())
			}
			for i := range v {
				h := mlpackage.Float16Bits(v[i])
				binary.LittleEndian.PutUint16(buf.Bytes()[2*i:], h)
				v[i] = mlpackage.Float16Value(h) // the values the device sees
			}
			in[a.Name] = milcpu.Tensor{Shape: a.Dims, Data: v}
		}
	}
	f, err := req.Submit()
	if err == nil {
		err = f.Wait(ctx)
	}
	if err != nil {
		t.Fatal(err)
	}
	p, c, err := milcpu.OpenPackage(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	want, err := p.Run(in, milcpu.Options{RoundF16: true})
	if err != nil {
		t.Fatal(err)
	}
	for _, o := range feats {
		got := make([]float32, len(want[o.Name].Data))
		for i := range got {
			got[i] = mlpackage.Float16Value(binary.LittleEndian.Uint16(bufs[o.Name].Bytes()[2*i:]))
		}
		var worst float64
		for i, w := range want[o.Name].Data {
			d := math.Abs(float64(got[i] - w))
			if math.IsNaN(d) {
				d = math.Inf(1)
			}
			worst = max(worst, d)
		}
		if testing.Verbose() && o.Name == "er" {
			for i, w := range want[o.Name].Data {
				if d := math.Abs(float64(got[i] - w)); d > 0.01 || math.IsNaN(d) {
					t.Logf("erf(%.3f) = %.4f, want %.4f", 3*in["x"].Data[i], got[i], w)
				}
			}
		}
		if worst > 0.01 {
			t.Errorf("%s: off by up to %g from milcpu", o.Name, worst)
		}
	}
	if !g.Placement().OnNPU() && !OnCPU() {
		t.Error("not all on the Neural Engine")
	}
}
