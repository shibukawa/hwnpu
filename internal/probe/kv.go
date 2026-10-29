package probe

import (
	"fmt"
	"math"
	"strings"

	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// KV cache probe models: one token through layers of the probe's size (lut4 weights) with real
// attention over a KV cache of a fixed number of tokens, the cache kept in one of two ways.
//
//   - "host": the cache is an input of every layer, kv_k_N and kv_v_N (1, kv_heads, cache, 128),
//     and the token's own key and value are outputs, kv_k_N_new and kv_v_N_new, for the host to
//     write in; the attention runs over the cache followed by the token, with the mask input
//     (1, 1, 1, cache+1). This is how internal/llm runs LFM3.
//   - "state": the cache is a Core ML state (MLState) per layer, k_N and v_N. The program writes
//     the token into it at the row the one-hot input onehot (1, 1, cache, 1) marks, as
//     cache + onehot * (k - cache), with elementwise operations only, reads it back (the Neural
//     Engine takes a written state only when it is read again, mlpackage.UpdateState) and attends
//     over it with the mask input (1, 1, 1, cache).
//
// The attention repeats the cache for every query head (repeat_kv as a tile). Reading it once per
// key and value head instead, with the query heads of a group as the rows of one matrix product,
// computes the same, but Core ML then puts the whole model on the CPU (host) or fails to compile
// it (state) (M3, macOS 27.0.1). The suffix "-mha" (host-mha, state-mha) gives every query head
// its own key and value head instead, so that nothing is repeated: the cache is 4 times larger,
// and the difference in time shows what the repeat costs.
//
// The output y is the hidden state after the layers.

// KVHeads and HeadDim are the probe's grouped key and value heads (DKV = KVHeads * HeadDim).
const (
	KVHeads = 4
	HeadDim = DKV / KVHeads
	QHeads  = DModel / HeadDim
)

// kvMode splits a mode into its base ("host" or "state") and its flags (mha, int8).
func kvMode(mode string) (base string, flags map[string]bool) {
	parts := strings.Split(mode, "-")
	flags = map[string]bool{}
	for _, f := range parts[1:] {
		flags[f] = true
	}
	return parts[0], flags
}

// kvHeads is the number of key and value heads of a probe model in the given mode.
func kvHeads(mode string) int {
	if _, f := kvMode(mode); f["mha"] {
		return QHeads
	}
	return KVHeads
}

// KVStates lists the state inputs of a "state" probe model, for Package.States.
func KVStates(mode string, layers, cache int) []mlpackage.Feature {
	var fs []mlpackage.Feature
	for i := 0; i < layers; i++ {
		for _, kv := range []string{"k", "v"} {
			fs = append(fs, mlpackage.Feature{Name: fmt.Sprintf("%s_%d", kv, i), DType: mlpackage.Float16, Shape: []int{1, kvHeads(mode), cache, HeadDim}})
		}
	}
	return fs
}

// WriteKVPackage writes a KV cache probe model with the given cache handling ("host" or "state",
// with "-mha" for a key and value head per query head, and "-int8" for a host cache held as int8
// with an fp16 scale per token and head, cast back to fp16 in the program).
func WriteKVPackage(dir, mode string, layers, cache int, seed uint64) error {
	nkv := kvHeads(mode)
	base, flags := kvMode(mode)
	if base != "host" && base != "state" || flags["int8"] && base != "host" {
		return fmt.Errorf("probe: KV cache mode %q is not host or state, optionally -mha, and -int8 for host", mode)
	}
	pkg, err := mlpackage.Create(dir)
	if err != nil {
		return err
	}
	pkg.Metadata["com.github.shibukawa.hwnpu.probe"] = fmt.Sprintf("kv %s %d layers cache %d lut4", mode, layers, cache)
	b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
	var g *weightGen
	var werr error
	linear := func(v *mlpackage.Var, rows, cols int) *mlpackage.Var {
		w, err := b.LUT4("w", []int{rows, cols}, []int{16, cols}, g.bytes(rows*cols/2), g.halves(rows/16*16))
		if err != nil && werr == nil {
			werr = err
		}
		return b.Op("linear", map[string]any{"x": v, "weight": w}, mlpackage.Float16, 1, 1, rows)
	}
	op := func(typ string, in map[string]any, shape ...int) *mlpackage.Var {
		return b.Op(typ, in, mlpackage.Float16, shape...)
	}
	bin := func(typ string, p, q *mlpackage.Var, shape ...int) *mlpackage.Var {
		return op(typ, map[string]any{"x": p, "y": q}, shape...)
	}
	shape := []int{1, 1, DModel}
	x := b.Input("x", mlpackage.Float16, shape...)
	ins := []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: shape}}
	input := func(name string, sh ...int) *mlpackage.Var {
		ins = append(ins, mlpackage.Feature{Name: name, DType: mlpackage.Float16, Shape: sh})
		return b.Input(name, mlpackage.Float16, sh...)
	}
	// cacheInput reads a cached tensor: fp16 as is, or int8 times its per-token scale.
	cacheInput := func(name string, sh ...int) *mlpackage.Var {
		if !flags["int8"] {
			return input(name, sh...)
		}
		ins = append(ins, mlpackage.Feature{Name: name, DType: mlpackage.Int8, Shape: sh})
		q := b.Input(name, mlpackage.Int8, sh...)
		scale := input(name+"_scale", sh[0], sh[1], sh[2], 1)
		f := b.Op("cast", map[string]any{"x": q, "dtype": b.Str("dtype", "fp16")}, mlpackage.Float16, sh...)
		return bin("mul", f, scale, sh...)
	}
	var outVars []*mlpackage.Var
	keys := cache + 1
	var mask, onehot *mlpackage.Var
	if base == "state" {
		keys = cache
		onehot = input("onehot", 1, 1, cache, 1)
		pkg.States = KVStates(mode, layers, cache)
	}
	mask = input("mask", 1, 1, 1, keys)
	gamma, err := b.F16("gamma", []int{DModel}, constHalves(DModel, 1))
	if err != nil {
		return err
	}
	beta, err := b.F16("beta", []int{DModel}, constHalves(DModel, 0))
	if err != nil {
		return err
	}
	norm := func(v *mlpackage.Var) *mlpackage.Var {
		return op("layer_norm", map[string]any{"x": v, "axes": b.Ints(-1), "gamma": gamma, "beta": beta, "epsilon": b.ScalarF16(1e-5)}, shape...)
	}
	heads := func(v *mlpackage.Var, n int) *mlpackage.Var { // (1, 1, n*128) -> (1, n, 1, 128)
		v = op("reshape", map[string]any{"x": v, "shape": b.Ints(1, 1, int32(n), HeadDim)}, 1, 1, n, HeadDim)
		return op("transpose", map[string]any{"x": v, "perm": b.Ints(0, 2, 1, 3)}, 1, n, 1, HeadDim)
	}
	expand := func(v *mlpackage.Var) *mlpackage.Var { // (1, 4, keys, 128) -> (1, 16, keys, 128)
		rep := QHeads / nkv
		if rep == 1 {
			return v
		}
		v = op("reshape", map[string]any{"x": v, "shape": b.Ints(1, int32(nkv), 1, int32(keys), HeadDim)}, 1, nkv, 1, keys, HeadDim)
		v = op("tile", map[string]any{"x": v, "reps": b.Ints(1, 1, int32(rep), 1, 1)}, 1, nkv, rep, keys, HeadDim)
		return op("reshape", map[string]any{"x": v, "shape": b.Ints(1, QHeads, int32(keys), HeadDim)}, 1, QHeads, keys, HeadDim)
	}
	h := x
	for i := 0; i < layers; i++ {
		g = newWeightGen(seed, i)
		n := norm(h)
		q := heads(linear(n, DModel, DModel), QHeads)
		k := heads(linear(n, nkv*HeadDim, DModel), nkv)
		v := heads(linear(n, nkv*HeadDim, DModel), nkv)
		cached := func(kv string, t *mlpackage.Var) *mlpackage.Var {
			if base == "host" {
				outVars = append(outVars, b.NamedOp("identity", fmt.Sprintf("kv_%s_%d_new", kv, i), map[string]any{"x": t}, mlpackage.Float16, t.Shape...))
				c := cacheInput(fmt.Sprintf("kv_%s_%d", kv, i), 1, nkv, cache, HeadDim)
				return op("concat", map[string]any{"values": []*mlpackage.Var{c, t}, "axis": b.ScalarI32(2), "interleave": b.ScalarBool(false)}, 1, nkv, keys, HeadDim)
			}
			s := b.State(fmt.Sprintf("%s_%d", kv, i), mlpackage.Float16, 1, nkv, cache, HeadDim)
			cur := op("read_state", map[string]any{"input": s}, 1, nkv, cache, HeadDim)
			d := bin("sub", t, cur, 1, nkv, cache, HeadDim)
			upd := bin("add", cur, bin("mul", onehot, d, 1, nkv, cache, HeadDim), 1, nkv, cache, HeadDim)
			return b.UpdateState(s, upd)
		}
		ka, va := cached("k", k), cached("v", v)
		scores := op("matmul", map[string]any{"x": q, "y": expand(ka), "transpose_x": b.ScalarBool(false), "transpose_y": b.ScalarBool(true)}, 1, QHeads, 1, keys)
		scores = bin("mul", scores, b.ScalarF16(float32(1/math.Sqrt(HeadDim))), 1, QHeads, 1, keys)
		scores = bin("add", scores, mask, 1, QHeads, 1, keys)
		probs := op("softmax", map[string]any{"x": scores, "axis": b.ScalarI32(-1)}, 1, QHeads, 1, keys)
		o := op("matmul", map[string]any{"x": probs, "y": expand(va), "transpose_x": b.ScalarBool(false), "transpose_y": b.ScalarBool(false)}, 1, QHeads, 1, HeadDim)
		o = op("transpose", map[string]any{"x": o, "perm": b.Ints(0, 2, 1, 3)}, 1, 1, QHeads, HeadDim)
		o = op("reshape", map[string]any{"x": o, "shape": b.Ints(1, 1, DModel)}, shape...)
		h = bin("add", h, linear(o, DModel, DModel), shape...)
		n = norm(h)
		gate := op("silu", map[string]any{"x": linear(n, DFFN, DModel)}, 1, 1, DFFN)
		h = bin("add", h, linear(bin("mul", gate, linear(n, DFFN, DModel), 1, 1, DFFN), DModel, DFFN), shape...)
		if werr != nil {
			return werr
		}
	}
	y := b.NamedOp("identity", "y", map[string]any{"x": norm(h)}, mlpackage.Float16, shape...)
	outs := []mlpackage.Feature{{Name: "y", DType: mlpackage.Float16, Shape: shape}}
	for _, v := range outVars {
		outs = append(outs, mlpackage.Feature{Name: v.Name, DType: mlpackage.Float16, Shape: v.Shape})
	}
	spec := int32(9)
	if flags["int8"] {
		spec = 10 // int8 multi array features need specification version 10 (macOS 26)
	}
	return pkg.Finish(b.Finish(append([]*mlpackage.Var{y}, outVars...)...), spec, ins, outs)
}

// KVCacheBytes is the size of the KV cache of a probe model: fp16, or int8 with an fp16 scale per
// token and head.
func KVCacheBytes(mode string, layers, cache int) int64 {
	perToken := int64(HeadDim * 2)
	if _, f := kvMode(mode); f["int8"] {
		perToken = HeadDim + 2
	}
	return int64(layers) * 2 * int64(kvHeads(mode)) * int64(cache) * perToken
}
