package ovlower

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/ovcpu"
	"github.com/shibukawa/hwnpu/internal/ovir"
	"github.com/shibukawa/hwnpu/internal/palette"
)

const f16 = mlpackage.Float16

// program is a MIL program under construction for a test, with its inputs' values.
type program struct {
	t      *testing.T
	b      *mlpackage.Builder
	blobs  *mlpackage.BlobWriter
	dir    string
	inputs map[string][]float32
	rng    *rand.Rand
}

func newProgram(t *testing.T) *program {
	dir := t.TempDir()
	blobs, err := mlpackage.NewBlobWriter(filepath.Join(dir, "weight.bin"))
	if err != nil {
		t.Fatal(err)
	}
	return &program{t: t, b: mlpackage.NewBuilder("CoreML8", blobs), blobs: blobs, dir: dir, inputs: map[string][]float32{}, rng: rand.New(rand.NewSource(1))}
}

// rounded returns n values uniform in [lo, hi), rounded to fp16 as the IR's inputs are.
func (p *program) rounded(n int, lo, hi float32) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = mlpackage.Float16Value(mlpackage.Float16Bits(lo + (hi-lo)*p.rng.Float32()))
	}
	return v
}

// input declares an fp16 input with random values in [lo, hi).
func (p *program) input(name string, lo, hi float32, shape ...int) *mlpackage.Var {
	n := 1
	for _, d := range shape {
		n *= d
	}
	p.inputs[name] = p.rounded(n, lo, hi)
	return p.b.Input(name, f16, shape...)
}

func (p *program) consts(name string, lo, hi float32, shape ...int) *mlpackage.Var {
	n := 1
	for _, d := range shape {
		n *= d
	}
	v, err := p.b.F16s(name, shape, p.rounded(n, lo, hi))
	if err != nil {
		p.t.Fatal(err)
	}
	return v
}

func (p *program) bools(v ...bool) *mlpackage.Var {
	x, err := p.b.Bools("c", []int{len(v)}, v)
	if err != nil {
		p.t.Fatal(err)
	}
	return x
}

func (p *program) op(typ string, inputs map[string]any, shape ...int) *mlpackage.Var {
	return p.b.Op(typ, inputs, f16, shape...)
}

// check runs the program with milcpu and, lowered, with OpenVINO's CPU plugin when
// HWNPU_OV_PYTHON is set, and compares every output. It returns the lowered model.
func (p *program) check(outs ...*mlpackage.Var) *ovir.Model {
	t := p.t
	t.Helper()
	prog := p.b.Finish(outs...)
	if err := p.blobs.Close(); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(filepath.Join(p.dir, "weight.bin"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	xmlPath := filepath.Join(p.dir, "m.xml")
	if _, err := WriteFiles(xmlPath, "test", prog.GetFunctions()["main"], f, map[string]string{"com.github.shibukawa.hwnpu.arch": "test"}); err != nil {
		t.Fatal(err)
	}
	xml, err := os.ReadFile(xmlPath)
	if err != nil {
		t.Fatal(err)
	}
	m, err := ovir.Parse(xml)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(xml), `<com.github.shibukawa.hwnpu.arch value="test"/>`) {
		t.Error("the metadata is not in rt_info")
	}
	for i, o := range outs {
		if i >= len(m.Outputs) || m.Outputs[i].Name != o.Name {
			t.Fatalf("outputs %+v, expected %s at %d (the producing layer must carry the output's name)", m.Outputs, o.Name, i)
		}
	}
	py := ovcpu.Python()
	if py == "" {
		t.Log("HWNPU_OV_PYTHON is not set: the IR is not run")
		return m
	}
	mp, err := milcpu.Load(prog, f)
	if err != nil {
		t.Fatal(err)
	}
	in := map[string]milcpu.Tensor{}
	for name, v := range p.inputs {
		in[name] = milcpu.Tensor{Data: v}
		for _, x := range m.Inputs {
			if x.Name == name {
				in[name] = milcpu.Tensor{Shape: x.Dims, Data: v}
			}
		}
	}
	want, err := mp.Run(in, milcpu.Options{})
	if err != nil {
		t.Fatal(err)
	}
	got, err := ovcpu.Run(py, xmlPath, p.inputs, ovcpu.Options{})
	if err != nil {
		t.Fatal(err)
	}
	for _, o := range outs {
		g, w := got[o.Name], want[o.Name].Data
		if len(g) != len(w) {
			t.Fatalf("%s: %d values from OpenVINO, %d from milcpu", o.Name, len(g), len(w))
		}
		var se, sw, worst float64
		for i := range w {
			d := float64(g[i] - w[i])
			se += d * d
			sw += float64(w[i]) * float64(w[i])
			worst = math.Max(worst, math.Abs(d))
		}
		rel := math.Sqrt(se / math.Max(sw, 1e-30))
		t.Logf("%s: relative error %.2g, largest difference %.2g", o.Name, rel, worst)
		if rel > 1e-3 {
			t.Errorf("%s: relative error %.3g between OpenVINO and milcpu", o.Name, rel)
		}
	}
	return m
}

// Elementwise operations, the normalizations and the activations.
func TestElementwise(t *testing.T) {
	p := newProgram(t)
	b := p.b
	sh := []int{1, 4, 16}
	x := p.input("x", -2, 2, sh...)
	z := p.input("z", 0.5, 2, sh...)
	g := p.consts("gamma", 0.5, 1.5, 1, 1, 16)
	beta := p.consts("beta", -0.5, 0.5, 16)
	// RMSNorm as blocks.RMSNorm writes it
	sq := p.op("mul", map[string]any{"x": x, "y": x}, sh...)
	ms := p.op("reduce_mean", map[string]any{"x": sq, "axes": b.Ints(-1), "keep_dims": b.ScalarBool(true)}, 1, 4, 1)
	r := p.op("rsqrt", map[string]any{"x": ms, "epsilon": b.ScalarF16(1e-5)}, 1, 4, 1)
	h := p.op("mul", map[string]any{"x": p.op("mul", map[string]any{"x": x, "y": r}, sh...), "y": g}, sh...)
	h = p.op("add", map[string]any{"x": h, "y": p.op("real_div", map[string]any{"x": x, "y": z}, sh...)}, sh...)
	h = p.op("sub", map[string]any{"x": h, "y": b.ScalarF16(0.25)}, sh...)
	h = p.op("maximum", map[string]any{"x": h, "y": p.op("silu", map[string]any{"x": x}, sh...)}, sh...)
	for _, u := range []string{"sigmoid", "tanh", "relu"} {
		h = p.op("add", map[string]any{"x": h, "y": p.op(u, map[string]any{"x": x}, sh...)}, sh...)
	}
	h = p.op("add", map[string]any{"x": h, "y": p.op("sqrt", map[string]any{"x": z}, sh...)}, sh...)
	h = p.op("add", map[string]any{"x": h, "y": p.op("exp", map[string]any{"x": x}, sh...)}, sh...)
	h = p.op("add", map[string]any{"x": h, "y": p.op("gelu", map[string]any{"x": x, "mode": b.Str("c", "EXACT")}, sh...)}, sh...)
	h = p.op("add", map[string]any{"x": h, "y": p.op("gelu", map[string]any{"x": z, "mode": b.Str("c", "TANH_APPROXIMATION")}, sh...)}, sh...)
	h = p.op("layer_norm", map[string]any{"x": h, "axes": b.Ints(-1), "gamma": g, "beta": beta, "epsilon": b.ScalarF16(1e-5)}, sh...)
	s := p.op("softmax", map[string]any{"x": h, "axis": b.ScalarI32(-1)}, sh...)
	m := p.op("reduce_max", map[string]any{"x": h, "axes": b.Ints(1), "keep_dims": b.ScalarBool(false)}, 1, 16)
	y := b.NamedOp("identity", "y", map[string]any{"x": s}, f16, sh...)
	p.check(y, b.NamedOp("identity", "m", map[string]any{"x": m}, f16, 1, 16))
}

// Moving data: reshape, transpose, tile, slices, concat and pad.
func TestShapes(t *testing.T) {
	p := newProgram(t)
	b := p.b
	x := p.input("x", -1, 1, 1, 6, 8)
	r := p.op("reshape", map[string]any{"x": x, "shape": b.Ints(1, 6, 2, 4)}, 1, 6, 2, 4)
	tr := p.op("transpose", map[string]any{"x": r, "perm": b.Ints(0, 2, 1, 3)}, 1, 2, 6, 4)
	ti := p.op("tile", map[string]any{"x": tr, "reps": b.Ints(1, 2, 1, 1)}, 1, 4, 6, 4)
	// the halves of the last axis, as blocks.ApplyRope takes them
	x1 := p.op("slice_by_index", map[string]any{"x": ti, "begin": b.Ints(0, 0, 0, 0), "end": b.Ints(0, 0, 0, 2),
		"begin_mask": p.bools(true, true, true, true), "end_mask": p.bools(true, true, true, false)}, 1, 4, 6, 2)
	x2 := p.op("slice_by_index", map[string]any{"x": ti, "begin": b.Ints(0, 0, 0, 2), "end": b.Ints(0, 0, 0, 4),
		"begin_mask": p.bools(true, true, true, false), "end_mask": p.bools(true, true, true, true)}, 1, 4, 6, 2)
	c := p.op("concat", map[string]any{"values": []*mlpackage.Var{x2, x1}, "axis": b.ScalarI32(-1), "interleave": b.ScalarBool(false)}, 1, 4, 6, 4)
	// every other row from 1, the last axis squeezed away at index 3
	st := p.op("slice_by_index", map[string]any{"x": c, "begin": b.Ints(0, 0, 1, 3), "end": b.Ints(1, 4, 6, 4), "stride": b.Ints(1, 1, 2, 1),
		"squeeze_mask": p.bools(false, false, false, true)}, 1, 4, 3)
	pd := p.op("pad", map[string]any{"x": st, "pad": b.Ints(2, 0), "mode": b.Str("c", "constant"), "constant_val": b.ScalarF16(0.5)}, 1, 4, 5)
	p.check(b.NamedOp("identity", "y", map[string]any{"x": c}, f16, 1, 4, 6, 4), b.NamedOp("identity", "s", map[string]any{"x": pd}, f16, 1, 4, 5))
}

// Matrix products and linear layers with fp16, int4 (per row and in groups) and int8 weights.
func TestLinear(t *testing.T) {
	p := newProgram(t)
	b := p.b
	const in, out = 256, 32
	x := p.input("x", -1, 1, 1, 3, in)
	k := p.input("k", -1, 1, 1, 2, 5, 8)
	q := p.input("q", -1, 1, 1, 2, 3, 8)
	w := p.rounded(out*in, -0.1, 0.1)
	wf, _ := b.F16s("w", []int{out, in}, w)
	bias := p.consts("bias", -1, 1, out)
	y := p.op("linear", map[string]any{"x": x, "weight": wf, "bias": bias}, 1, 3, out)
	for _, c := range []struct{ bits, block int }{{4, 0}, {4, 128}, {8, 0}} {
		l := palette.QuantizeLinear(w, out, in, c.block, c.bits, true)
		var wq *mlpackage.Var
		var err error
		if c.bits == 4 {
			wq, err = b.Int4Scaled("wq", []int{out, in}, []int{1, l.ColBlock}, l.Data, l.Scales)
		} else {
			wq, err = b.Int8Scaled("wq", []int{out, in}, []int{1, l.ColBlock}, l.Data, l.Scales)
		}
		if err != nil {
			t.Fatal(err)
		}
		y = p.op("add", map[string]any{"x": y, "y": p.op("linear", map[string]any{"x": x, "weight": wq}, 1, 3, out)}, 1, 3, out)
	}
	scores := p.op("matmul", map[string]any{"x": q, "y": k, "transpose_x": b.ScalarBool(false), "transpose_y": b.ScalarBool(true)}, 1, 2, 3, 5)
	o := p.op("matmul", map[string]any{"x": scores, "y": k, "transpose_x": b.ScalarBool(false), "transpose_y": b.ScalarBool(false)}, 1, 2, 3, 8)
	p.check(b.NamedOp("identity", "y", map[string]any{"x": y}, f16, 1, 3, out), b.NamedOp("identity", "o", map[string]any{"x": o}, f16, 1, 2, 3, 8))
}

// A depthwise convolution with a bias, as LFM3's short convolution (ConvMode conv), and a full
// one with explicit padding.
func TestConv(t *testing.T) {
	p := newProgram(t)
	b := p.b
	const d, s, k = 8, 6, 3
	x := p.input("x", -1, 1, 1, d, s)
	padded := p.op("pad", map[string]any{"x": x, "pad": b.Ints(k-1, 0), "mode": b.Str("c", "constant"), "constant_val": b.ScalarF16(0)}, 1, d, s+k-1)
	dw := p.op("conv", map[string]any{"x": padded, "weight": p.consts("dw", -1, 1, d, 1, k), "bias": p.consts("db", -1, 1, d),
		"strides": b.Ints(1), "pad_type": b.Str("c", "valid"), "pad": b.Ints(0, 0), "dilations": b.Ints(1), "groups": b.ScalarI32(d)}, 1, d, s)
	full := p.op("conv", map[string]any{"x": dw, "weight": p.consts("fw", -0.5, 0.5, 4, d, k),
		"strides": b.Ints(1), "pad_type": b.Str("c", "custom"), "pad": b.Ints(1, 1), "dilations": b.Ints(1), "groups": b.ScalarI32(1)}, 1, 4, s)
	p.check(b.NamedOp("identity", "y", map[string]any{"x": full}, f16, 1, 4, s))
}

// Outputs keep their names whatever computes them: an input passed through, a value that is two
// outputs, a value that is also used further on. Lookup table weights are refused.
func TestOutputNames(t *testing.T) {
	p := newProgram(t)
	b := p.b
	x := p.input("x", -1, 1, 1, 4)
	h := p.op("mul", map[string]any{"x": x, "y": b.ScalarF16(2)}, 1, 4)
	y := p.op("add", map[string]any{"x": h, "y": b.ScalarF16(1)}, 1, 4)
	outs := []*mlpackage.Var{
		b.NamedOp("identity", "y", map[string]any{"x": y}, f16, 1, 4),
		b.NamedOp("identity", "h", map[string]any{"x": h}, f16, 1, 4),
		b.NamedOp("identity", "h2", map[string]any{"x": h}, f16, 1, 4),
		b.NamedOp("identity", "x_out", map[string]any{"x": x}, f16, 1, 4),
	}
	prog := b.Finish(outs...)
	p.blobs.Close()
	xml, err := Lower("t", prog.GetFunctions()["main"], nil, &strings.Builder{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	m, err := ovir.Parse(xml)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, o := range m.Outputs {
		names = append(names, o.Name)
	}
	if strings.Join(names, ",") != "y,h,h2,x_out" || len(m.Inputs) != 1 || m.Inputs[0].Name != "x" {
		t.Errorf("inputs %+v, outputs %v", m.Inputs, names)
	}

	p = newProgram(t)
	x = p.input("x", -1, 1, 1, 32)
	l := palette.QuantizeBits(p.rounded(32*32, -1, 1), 32, 32, 16, 0, 4, 1, palette.Uniform)
	wl, err := p.b.LUT("w", 4, []int{32, 32}, []int{16, 32}, l.Indices, l.LUT)
	if err != nil {
		t.Fatal(err)
	}
	y = p.op("linear", map[string]any{"x": x, "weight": wl}, 1, 32)
	prog = p.b.Finish(y)
	p.blobs.Close()
	f, _ := os.Open(filepath.Join(p.dir, "weight.bin"))
	defer f.Close()
	if _, err := Lower("t", prog.GetFunctions()["main"], f, &strings.Builder{}, nil); err == nil || !strings.Contains(err.Error(), "int4") {
		t.Errorf("lut4 weights: %v, expected an error that points to int4", err)
	}
}
