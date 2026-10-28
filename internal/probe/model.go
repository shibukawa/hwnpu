// Package probe generates the models cmd/npuprobe and cmd/powerbench measure: LLM decode shaped
// graphs with random weights, as Core ML ML Programs or OpenVINO IR, without Python.
package probe

import (
	"fmt"
	"io/fs"
	"math/rand/v2"
	"path/filepath"

	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/ovir"
)

// The probe models have the shape of LLM decoding (tools/coreml/gen_decode.py): one token through
// Llama 3.2 1B sized layers with random weights, so every call reads every weight once.
//
//	x (1, 1, d) fp16 -> layers x [norm, q/k/v/o projections, norm, SwiGLU MLP] -> norm -> y
const (
	DModel = 2048
	DFFN   = 8192
	DKV    = 512
)

// LayerParams is the number of weights per layer (about 61 M).
const LayerParams = 2*DModel*DModel + 2*DModel*DKV + 3*DModel*DFFN

// BytesPerParam is the stored size of one weight in each format.
var BytesPerParam = map[string]float64{"fp16": 2, "int8": 1, "lut4": 0.5, "int4": 0.5}

// WeightBytes is the stored size of the weights of a model with the given layers.
func WeightBytes(format string, layers int) int64 {
	return int64(float64(layers) * LayerParams * BytesPerParam[format])
}

// fp16Table maps 16 random bits to an fp16 weight, uniform in [-0.04, 0.04) (std about 0.023,
// close to 1/sqrt(2048)).
var fp16Table = func() []uint16 {
	t := make([]uint16, 1<<16)
	for i := range t {
		t[i] = mlpackage.Float16Bits((float32(i)/(1<<16) - 0.5) * 0.08)
	}
	return t
}()

type weightGen struct{ rng *rand.Rand }

func newWeightGen(seed uint64) *weightGen {
	return &weightGen{rng: rand.New(rand.NewPCG(seed, 0x6877_6e70_7531))}
}

func (g *weightGen) bytes(n int) []byte {
	p := make([]byte, n)
	for i := 0; i < n; i += 8 {
		v := g.rng.Uint64()
		for j := 0; j < 8 && i+j < n; j++ {
			p[i+j] = byte(v >> (8 * j))
		}
	}
	return p
}

func (g *weightGen) halves(n int) []uint16 {
	h := make([]uint16, n)
	for i := 0; i < n; i += 4 {
		v := g.rng.Uint64()
		for j := 0; j < 4 && i+j < n; j++ {
			h[i+j] = fp16Table[uint16(v>>(16*j))]
		}
	}
	return h
}

// scales returns n fp16 quantization scales spread around v. They must not all be equal: Core ML
// on the GPU folds a constant scale tensor to a scalar and then fails an MPSGraph shape assertion.
func (g *weightGen) scales(n int, v float32) []uint16 {
	h := make([]uint16, n)
	for i := range h {
		h[i] = mlpackage.Float16Bits(v * (0.75 + 0.5*g.rng.Float32()))
	}
	return h
}

func constHalves(n int, v float32) []uint16 {
	h := make([]uint16, n)
	for i := range h {
		h[i] = mlpackage.Float16Bits(v)
	}
	return h
}

// WriteMLPackage writes the probe model as an ML Program with weights in the given format:
// fp16, int8 (symmetric, one scale per output row), lut4 (a 16 entry table per 16 rows, the
// layout the Neural Engine runs at full speed) or int4 (one scale per 32 inputs, which Core ML
// runs on the CPU or the GPU only).
func WriteMLPackage(dir, format string, layers int, seed uint64) error {
	pkg, err := mlpackage.Create(dir)
	if err != nil {
		return err
	}
	pkg.Metadata["com.github.shibukawa.hwnpu.probe"] = fmt.Sprintf("decode %d layers %s", layers, format)
	b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
	g := newWeightGen(seed)
	var werr error
	weight := func(rows, cols int) *mlpackage.Var {
		var v *mlpackage.Var
		var err error
		switch format {
		case "fp16":
			v, err = b.F16("w", []int{rows, cols}, g.halves(rows*cols))
		case "int8":
			v, err = b.Int8Scaled("w", []int{rows, cols}, []int{1, cols}, g.bytes(rows*cols), g.scales(rows, 0.04/127))
		case "lut4":
			v, err = b.LUT4("w", []int{rows, cols}, []int{16, cols}, g.bytes(rows*cols/2), g.halves(rows/16*16))
		case "int4":
			v, err = b.Int4Scaled("w", []int{rows, cols}, []int{1, 32}, g.bytes(rows*cols/2), g.scales(rows*cols/32, 0.04/7))
		default:
			err = fmt.Errorf("unknown Core ML format %q", format)
		}
		if err != nil && werr == nil {
			werr = err
		}
		return v
	}
	shape := []int{1, 1, DModel}
	x := b.Input("x", mlpackage.Float16, shape...)
	gamma, err := b.F16("gamma", []int{DModel}, constHalves(DModel, 1))
	if err != nil {
		return err
	}
	beta, err := b.F16("beta", []int{DModel}, constHalves(DModel, 0))
	if err != nil {
		return err
	}
	norm := func(v *mlpackage.Var) *mlpackage.Var {
		return b.Op("layer_norm", map[string]any{"x": v, "axes": b.Ints(-1), "gamma": gamma, "beta": beta, "epsilon": b.ScalarF16(1e-5)}, mlpackage.Float16, shape...)
	}
	linear := func(v *mlpackage.Var, rows, cols int) *mlpackage.Var {
		return b.Op("linear", map[string]any{"x": v, "weight": weight(rows, cols)}, mlpackage.Float16, 1, 1, rows)
	}
	bin := func(op string, p, q *mlpackage.Var) *mlpackage.Var {
		return b.Op(op, map[string]any{"x": p, "y": q}, mlpackage.Float16, p.Shape...)
	}
	h := x
	for range layers {
		n := norm(h)
		q := linear(n, DModel, DModel)
		k := linear(n, DKV, DModel)
		v := linear(n, DKV, DModel)
		// Stands in for attention with one query: every head reads its shared k/v group.
		kv := b.Op("tile", map[string]any{"x": bin("add", k, v), "reps": b.Ints(1, 1, DModel/DKV)}, mlpackage.Float16, shape...)
		h = bin("add", h, linear(bin("mul", q, kv), DModel, DModel))
		n = norm(h)
		gate := b.Op("silu", map[string]any{"x": linear(n, DFFN, DModel)}, mlpackage.Float16, 1, 1, DFFN)
		h = bin("add", h, linear(bin("mul", gate, linear(n, DFFN, DModel)), DModel, DFFN))
		if werr != nil {
			return werr
		}
	}
	y := b.NamedOp("identity", "y", map[string]any{"x": norm(h)}, mlpackage.Float16, shape...)
	return pkg.Finish(b.Finish(y), 9, []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: shape}},
		[]mlpackage.Feature{{Name: "y", DType: mlpackage.Float16, Shape: shape}})
}

// BuildIR returns the probe model as OpenVINO IR with weights in the given format: fp16, int8 or
// int4 (symmetric, one scale per output row, the compressed weight pattern OpenVINO's NNCF writes
// and the NPU compiler recognizes).
func BuildIR(format string, layers int, seed uint64) (xmlData, bin []byte, err error) {
	b := ovir.NewBuilder(fmt.Sprintf("probe_l%d_%s", layers, format), int(WeightBytes(format, layers))+1<<20)
	g := newWeightGen(seed)
	shape := []int{1, 1, DModel}
	i64 := func(v ...int64) []byte {
		p := make([]byte, 8*len(v))
		for i, x := range v {
			for j := range 8 {
				p[8*i+j] = byte(uint64(x) >> (8 * j))
			}
		}
		return p
	}
	f16bytes := func(h []uint16) []byte {
		p := make([]byte, 2*len(h))
		for i, x := range h {
			p[2*i], p[2*i+1] = byte(x), byte(x>>8)
		}
		return p
	}
	weight := func(rows, cols int) ovir.Value {
		ws := []int{rows, cols}
		var q ovir.Value
		var scale float32
		switch format {
		case "fp16":
			return b.Const("f16", ws, f16bytes(g.halves(rows*cols)))
		case "int8":
			q, scale = b.Const("i8", ws, g.bytes(rows*cols)), 0.04/127
		case "int4":
			q, scale = b.Const("i4", ws, g.bytes(rows*cols/2)), 0.04/7
		default:
			if err == nil {
				err = fmt.Errorf("unknown OpenVINO format %q", format)
			}
			return b.Const("f16", ws, make([]byte, 2*rows*cols))
		}
		c := b.Op("Convert", "opset1", []string{"destination_type", "f16"}, "f16", ws, q)
		s := b.Const("f16", []int{rows, 1}, f16bytes(g.scales(rows, scale)))
		return b.Op("Multiply", "opset1", []string{"auto_broadcast", "numpy"}, "f16", ws, c, s)
	}
	axes := b.Const("i64", []int{1}, i64(-1))
	norm := func(v ovir.Value) ovir.Value {
		return b.Op("MVN", "opset6", []string{"eps", "1e-05", "normalize_variance", "true", "eps_mode", "inside_sqrt"}, "f16", v.Shape, v, axes)
	}
	linear := func(v ovir.Value, rows, cols int) ovir.Value {
		return b.Op("MatMul", "opset1", []string{"transpose_a", "false", "transpose_b", "true"}, "f16", []int{1, 1, rows}, v, weight(rows, cols))
	}
	bin2 := func(op string, p, q ovir.Value) ovir.Value {
		return b.Op(op, "opset1", []string{"auto_broadcast", "numpy"}, "f16", p.Shape, p, q)
	}
	reps := b.Const("i64", []int{3}, i64(1, 1, DModel/DKV))
	h := b.Parameter("x", "f16", shape...)
	for range layers {
		n := norm(h)
		q := linear(n, DModel, DModel)
		k := linear(n, DKV, DModel)
		v := linear(n, DKV, DModel)
		kv := b.Op("Tile", "opset1", nil, "f16", shape, bin2("Add", k, v), reps)
		h = bin2("Add", h, linear(bin2("Multiply", q, kv), DModel, DModel))
		n = norm(h)
		gate := b.Op("Swish", "opset4", nil, "f16", []int{1, 1, DFFN}, linear(n, DFFN, DModel))
		h = bin2("Add", h, linear(bin2("Multiply", gate, linear(n, DFFN, DModel)), DModel, DFFN))
	}
	b.Result("y", norm(h))
	if err != nil {
		return nil, nil, err
	}
	xmlData, bin = b.Bytes()
	return xmlData, bin, nil
}

// DirSize returns the bytes of all files under dir.
func DirSize(dir string) int64 {
	var n int64
	filepath.WalkDir(dir, func(_ string, d fs.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			if info, err := d.Info(); err == nil {
				n += info.Size()
			}
		}
		return nil
	})
	return n
}

// GiB converts bytes to GiB.
func GiB(n int64) float64 { return float64(n) / (1 << 30) }
