// Command bctest writes a tiny Core AI program with mlirbc's writer: two 1 x 1 convolutions of
// lut4 weights, each followed by relu (as maximum with a broadcast zero), (1, 64, 1, 8) fp16.
//
//	go run ./tools/coreai/bctest OUT.aimodel
package main

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu/internal/coreai/mlirbc"
)

func main() {
	out := os.Args[1]
	w := mlirbc.NewWriter("hwnpu", map[string][]byte{"aicode": {0x05}})
	A := func(s string) int { return w.Attr("aicode", s) }
	T := func(s string) int { return w.Type("aicode", s) }
	tensor := func(shape, elem string) int {
		if shape == "" {
			return T("!aicode.tensor_v1<!aicode." + elem + "_v1>")
		}
		return T(fmt.Sprintf("!aicode.tensor_v1<[%s] : !aicode.%s_v1>", shape, elem))
	}
	dense := func(splat bool, b []byte) string {
		return fmt.Sprintf("#aicode.serialization.dense_elements<[splat = %v] \"%s\">", splat, base64.StdEncoding.EncodeToString(b))
	}
	P := func(text string) mlirbc.Prop { return mlirbc.Prop{Attr: A(text)} }
	absent := mlirbc.Prop{Attr: -1, Optional: true}
	u32 := func(v ...uint32) []byte {
		b := make([]byte, 4*len(v))
		for i, x := range v {
			binary.LittleEndian.PutUint32(b[4*i:], x)
		}
		return b
	}
	xt := tensor("1, 64, 1, 8", "float16")
	g, args := w.NewBlock(xt)
	cst := func(typ int, val string) mlirbc.Value {
		return g.Op("aicode.primitive.constant_v1", []mlirbc.Prop{P(val)}, nil, []int{typ})[0]
	}
	one := cst(tensor("", "uint32"), dense(true, u32(1)))
	ones2 := cst(tensor("2", "uint32"), dense(true, u32(1)))
	axis := cst(tensor("", "sint16"), dense(true, []byte{0, 0}))
	zero := cst(tensor("", "float16"), dense(true, []byte{0, 0}))
	shape4 := cst(tensor("4", "sint32"), dense(true, u32(1)))
	dims := cst(tensor("2", "uint32"), dense(false, u32(64, 8)))
	bdims := cst(tensor("2", "sint32"), dense(false, u32(1, 3)))
	composite := len(os.Args) > 2 && os.Args[2] == "composite"
	var zb mlirbc.Value
	if !composite {
		z4 := g.Op("aicode.primitive.reshape_v2", nil, []mlirbc.Value{zero, shape4}, []int{tensor("1, 1, 1, 1", "float16")})[0]
		zb = g.Op("aicode.primitive.broadcast_in_dims_v1", nil, []mlirbc.Value{z4, dims, bdims}, []int{xt})[0]
	}
	// relu as coreai-core writes it: relu_v1 holding broadcasting_maximum_v1 holding the primitives
	relu := func(x mlirbc.Value) mlirbc.Value {
		inner, ia := w.NewBlock(xt, tensor("", "float16"))
		c := func(typ int, val string) mlirbc.Value {
			return inner.Op("aicode.primitive.constant_v1", []mlirbc.Prop{P(val)}, nil, []int{typ})[0]
		}
		s4 := c(tensor("4", "sint32"), dense(true, u32(1)))
		r := inner.Op("aicode.primitive.reshape_v2", nil, []mlirbc.Value{ia[1], s4}, []int{tensor("1, 1, 1, 1", "float16")})[0]
		d2 := c(tensor("2", "uint32"), dense(false, u32(64, 8)))
		b2 := c(tensor("2", "sint32"), dense(false, u32(1, 3)))
		bb := inner.Op("aicode.primitive.broadcast_in_dims_v1", nil, []mlirbc.Value{r, d2, b2}, []int{xt})[0]
		m := inner.Op("aicode.primitive.maximum_v1", nil, []mlirbc.Value{ia[0], bb}, []int{xt})[0]
		inner.Op("aicode.dataflow.yield_v1", nil, []mlirbc.Value{m}, nil)
		outer, oa := w.NewBlock(xt)
		z := outer.Op("aicode.primitive.constant_v1", []mlirbc.Prop{P(dense(true, []byte{0, 0}))}, nil, []int{tensor("", "float16")})[0]
		bm := outer.Op("aicode.broadcasting_maximum_v1", nil, []mlirbc.Value{oa[0], z}, []int{xt}, inner)[0]
		outer.Op("aicode.dataflow.yield_v1", nil, []mlirbc.Value{bm}, nil)
		return g.Op("aicode.relu_v1", nil, []mlirbc.Value{x}, []int{xt}, outer)[0]
	}
	rng := rand.New(rand.NewSource(1))
	x := args[0]
	for l := 0; l < 2; l++ {
		idx := make([]byte, 64*64/2)
		rng.Read(idx)
		lut := make([]byte, 64*16*2)
		for i := 0; i < 64*16; i++ {
			binary.LittleEndian.PutUint16(lut[2*i:], f16(float32(rng.NormFloat64()/8)))
		}
		ki, kl := fmt.Sprintf("hwnpu_w%d_idx", l), fmt.Sprintf("hwnpu_w%d_lut", l)
		w.Blob("aicode", ki, 16, idx)
		w.Blob("aicode", kl, 16, lut)
		wi := cst(tensor("64, 64, 1, 1", "uint4"), "#aicode.serialization.dense_resource<"+ki+">")
		wl := cst(tensor("64, 1, 1, 1, 16, 1", "float16"), "#aicode.serialization.dense_resource<"+kl+">")
		wd := g.Op("aicode.primitive.lut_to_dense_v1", nil, []mlirbc.Value{wi, wl, axis}, []int{tensor("64, 64, 1, 1", "float16")})[0]
		c := g.Op("aicode.primitive.conv2d_v1", nil, []mlirbc.Value{x, wd, ones2, ones2, one}, []int{xt})[0]
		if composite {
			x = relu(c)
		} else {
			x = g.Op("aicode.primitive.maximum_v1", nil, []mlirbc.Value{c, zb}, []int{xt})[0]
		}
	}
	g.Op("aicode.dataflow.output_v1", nil, []mlirbc.Value{x}, nil)
	name := func(s string) string {
		return fmt.Sprintf("#aicode.serialization.array<[#aicode.serialization.array<[#aicode.serialization.opaque<\"aicode.name\" = [#aicode.serialization.string<\"%s\">]>]>]>", s)
	}
	ft := "!aicode.tensor_v1<[1, 64, 1, 8] : !aicode.float16_v1>"
	zero1 := P("#aicode.serialization.integer<(1bit unsigned) 0>")
	gp := []mlirbc.Prop{P(name("x")), absent, zero1,
		P("#aicode.serialization.type<!aicode.function_v1<(operands " + ft + ") -> (results " + ft + ")>>"),
		zero1, P("#aicode.serialization.string<\"main\">"), zero1, P("#aicode.serialization.array<>"), P(name("y"))}
	asset, _ := w.NewBlock()
	asset.Op("aicode.dataflow.graph_v1", gp, nil, nil, g)
	top, _ := w.NewBlock()
	top.Op("aicode.serialization.asset", []mlirbc.Prop{absent, {Attr: A("#aicode.serialization.integer<(32bit unsigned) 20260204>"), Optional: true}}, nil, nil, asset)
	b := w.Bytes(top)
	if err := os.MkdirAll(out, 0o755); err != nil {
		log.Fatal(err)
	}
	sum := sha256.Sum256(b)
	must(os.WriteFile(filepath.Join(out, "main.mlirb"), b, 0o644))
	must(os.WriteFile(filepath.Join(out, "main.hash"), sum[:], 0o644))
	must(os.WriteFile(filepath.Join(out, "metadata.json"), []byte(`{"assetVersion":"2.0","producer":"hwnpu","creationDate":"20261112T000000Z"}`), 0o644))
	fmt.Printf("wrote %s (%d bytes)\n", out, len(b))
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func f16(f float32) uint16 {
	b := binary.LittleEndian.Uint32(binary.LittleEndian.AppendUint32(nil, mathFloat32bits(f)))
	sign := uint16(b>>16) & 0x8000
	exp := int((b>>23)&0xff) - 127 + 15
	mant := b & 0x7fffff
	switch {
	case exp <= 0:
		return sign
	case exp >= 31:
		return sign | 0x7c00
	}
	return sign | uint16(exp)<<10 | uint16(mant>>13)
}

func mathFloat32bits(f float32) uint32 { return math.Float32bits(f) }
