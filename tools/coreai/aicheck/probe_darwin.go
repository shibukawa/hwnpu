package main

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/coreai"
	"github.com/shibukawa/hwnpu/internal/backend/coreml"
	"github.com/shibukawa/hwnpu/internal/imagegen/milpart"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/types"
)

// probes are small MIL programs of the pieces the models are made of, for finding which keeps
// a lowered program off the Neural Engine.
var probes = map[string]func(g *milpart.Graph){
	"sdpa": func(g *milpart.Graph) {
		q := g.Input("q", 1, 8, 256, 64)
		k := g.Input("k", 1, 8, 256, 64)
		v := g.Input("v", 1, 8, 256, 64)
		g.Output(g.SDPAScale(q, k, v, 0.125, nil))
	},
	"sdpa_mask": func(g *milpart.Graph) {
		q := g.Input("q", 1, 8, 256, 64)
		k := g.Input("k", 1, 8, 256, 64)
		v := g.Input("v", 1, 8, 256, 64)
		m := g.CausalMask("m", 256, 0)
		g.Output(g.SDPAScale(q, k, v, 0.125, func() *mlpackage.Var { return m }))
	},
	"sdpa_grouped": func(g *milpart.Graph) {
		q := g.Input("q", 1, 8, 256, 64)
		k := g.Input("k", 1, 2, 256, 64)
		v := g.Input("v", 1, 2, 256, 64)
		m := g.CausalMask("m", 256, 0)
		g.Output(g.SDPAGrouped(q, k, v, 0.125, func() *mlpackage.Var { return m }))
	},
	"rope": func(g *milpart.Graph) {
		x := g.Input("x", 1, 256, 8, 64)
		cos, sin := g.RopeTables("t", milpart.InvFreq(64, 10000), 256, 64)
		g.Output(g.ApplyRope(x, cos, sin))
	},
	"rmsnorm": func(g *milpart.Graph) {
		x := g.Input("x", 1, 256, 512)
		w := make([]float32, 512)
		for i := range w {
			w[i] = 1
		}
		g.Output(g.RMSNormWith(g.MaxScaled(x, -1), "w", w, 1e-6, 1))
	},
	"layernorm": func(g *milpart.Graph) {
		x := g.Input("x", 1, 256, 512)
		g.Output(g.ScaledLayerNorm(x, 1e-6))
	},
	"silu": func(g *milpart.Graph) {
		x := g.Input("x", 1, 256, 512)
		g.Output(g.SiLU(x))
	},
	"gelu": func(g *milpart.Graph) {
		x := g.Input("x", 1, 256, 512)
		g.Output(g.Activation(x, "gelu"))
	},
	"linear": func(g *milpart.Graph) {
		x := g.Input("x", 1, 256, 512)
		w := make([]float32, 512*512)
		for i := range w {
			w[i] = float32(rand.NormFloat64() / 22)
		}
		g.Output(g.LinearConst(x, "w", w, 512, nil))
	},
	// indexing at run time, for mixtures of experts: 8 "experts" of 256 x 256
	"gnd_const": func(g *milpart.Graph) { // an expert's weight picked from a constant table
		w := constF16(g, "w", 8, 256, 256)
		i := g.InputI32("i", 2, 1)
		g.Output(g.Op("gather_nd", map[string]any{"x": w, "indices": i, "batch_dims": g.B.ScalarI32(0), "validate_indices": g.B.ScalarBool(false)}, 2, 256, 256))
	},
	"gnd_input": func(g *milpart.Graph) { // the same from an input table
		w := g.Input("w", 8, 256, 256)
		i := g.InputI32("i", 2, 1)
		g.Output(g.Op("gather_nd", map[string]any{"x": w, "indices": i, "batch_dims": g.B.ScalarI32(0), "validate_indices": g.B.ScalarBool(false)}, 2, 256, 256))
	},
	"gnd_mm": func(g *milpart.Graph) { // the picked weight times activations
		w := constF16(g, "w", 8, 256, 256)
		i := g.InputI32("i", 1, 1)
		a := g.Input("a", 1, 16, 256)
		e := g.Op("gather_nd", map[string]any{"x": w, "indices": i, "batch_dims": g.B.ScalarI32(0), "validate_indices": g.B.ScalarBool(false)}, 1, 256, 256)
		g.Output(g.Op("matmul", map[string]any{"x": a, "y": e, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(false)}, 1, 16, 256))
	},
	"gaa_input": func(g *milpart.Graph) { // gather_along_axis of an input
		x := g.Input("x", 1, 8, 256)
		i := g.InputI32("i", 1, 2, 256)
		g.Output(g.Op("gather_along_axis", map[string]any{"x": x, "indices": i, "axis": g.B.ScalarI32(1), "validate_indices": g.B.ScalarBool(false)}, 1, 2, 256))
	},
	"gaa_const": func(g *milpart.Graph) { // rows of a constant table (an embedding lookup)
		x := constF16(g, "t", 8, 256)
		i := g.InputI32("i", 2, 256)
		g.Output(g.Op("gather_along_axis", map[string]any{"x": x, "indices": i, "axis": g.B.ScalarI32(0), "validate_indices": g.B.ScalarBool(false)}, 2, 256))
	},
	"dslice_const": func(g *milpart.Graph) { // an expert's rows of a constant table at a run time offset
		w := constF16(g, "w", 8*256, 256)
		b := g.InputI32("b", 2)
		e := g.InputI32("e", 2)
		g.Output(g.Op("slice_by_index", map[string]any{"x": w, "begin": b, "end": e}, 256, 256))
	},
	"dslice_input": func(g *milpart.Graph) {
		w := g.Input("w", 8*256, 256)
		b := g.InputI32("b", 2)
		e := g.InputI32("e", 2)
		g.Output(g.Op("slice_by_index", map[string]any{"x": w, "begin": b, "end": e}, 256, 256))
	},
	"dslice_mm": func(g *milpart.Graph) {
		w := constF16(g, "w", 8*256, 256)
		b := g.InputI32("b", 2)
		e := g.InputI32("e", 2)
		a := g.Input("a", 16, 256)
		s := g.Op("slice_by_index", map[string]any{"x": w, "begin": b, "end": e}, 256, 256)
		g.Output(g.Op("matmul", map[string]any{"x": a, "y": s, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(false)}, 16, 256))
	},
	// the same with int16 indices: the Neural Engine takes fp16, f8E4M3, si8, ui8, si16, ui16 (no si32)
	"gnd16_const": func(g *milpart.Graph) {
		w := constF16(g, "w", 8, 256, 256)
		i := idx16(g, "i", 2, 1)
		g.Output(g.Op("gather_nd", map[string]any{"x": w, "indices": i, "batch_dims": g.B.ScalarI32(0), "validate_indices": g.B.ScalarBool(false)}, 2, 256, 256))
	},
	"gnd16_pts": func(g *milpart.Graph) { // elements by their 3 coordinates (the form the Neural Engine's gather asks for)
		w := constF16(g, "w", 8, 256, 256)
		i := idx16(g, "i", 1024, 3)
		g.Output(g.Op("gather_nd", map[string]any{"x": w, "indices": i, "batch_dims": g.B.ScalarI32(0), "validate_indices": g.B.ScalarBool(false)}, 1024))
	},
	"gnd16_pts_in": func(g *milpart.Graph) {
		w := g.Input("w", 8, 256, 256)
		i := idx16(g, "i", 1024, 3)
		g.Output(g.Op("gather_nd", map[string]any{"x": w, "indices": i, "batch_dims": g.B.ScalarI32(0), "validate_indices": g.B.ScalarBool(false)}, 1024))
	},
	"gnd16_pts_w1": func(g *milpart.Graph) { // the index tensor's width 1: (1024, 1, 3)
		w := constF16(g, "w", 8, 256, 256)
		i := idx16(g, "i", 1024, 1, 3)
		g.Output(g.Op("gather_nd", map[string]any{"x": w, "indices": i, "batch_dims": g.B.ScalarI32(0), "validate_indices": g.B.ScalarBool(false)}, 1024, 1))
	},
	"gnd16_pts_4d": func(g *milpart.Graph) { // (1, 1024, 1, 3)
		w := constF16(g, "w", 8, 256, 256)
		i := idx16(g, "i", 1, 1024, 1, 3)
		g.Output(g.Op("gather_nd", map[string]any{"x": w, "indices": i, "batch_dims": g.B.ScalarI32(0), "validate_indices": g.B.ScalarBool(false)}, 1, 1024, 1))
	},
	"gnd16_pts_hw": func(g *milpart.Graph) { // (32, 32, 1, 3)
		w := constF16(g, "w", 8, 256, 256)
		i := idx16(g, "i", 32, 32, 1, 3)
		g.Output(g.Op("gather_nd", map[string]any{"x": w, "indices": i, "batch_dims": g.B.ScalarI32(0), "validate_indices": g.B.ScalarBool(false)}, 32, 32, 1))
	},
	"gnd16_mm": func(g *milpart.Graph) {
		w := constF16(g, "w", 8, 256, 256)
		i := idx16(g, "i", 1, 1)
		a := g.Input("a", 1, 16, 256)
		e := g.Op("gather_nd", map[string]any{"x": w, "indices": i, "batch_dims": g.B.ScalarI32(0), "validate_indices": g.B.ScalarBool(false)}, 1, 256, 256)
		g.Output(g.Op("matmul", map[string]any{"x": a, "y": e, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(false)}, 1, 16, 256))
	},
	"gaa16_const": func(g *milpart.Graph) {
		x := constF16(g, "t", 8, 256)
		i := idx16(g, "i", 2, 256)
		g.Output(g.Op("gather_along_axis", map[string]any{"x": x, "indices": i, "axis": g.B.ScalarI32(0), "validate_indices": g.B.ScalarBool(false)}, 2, 256))
	},
	"gaa16_input": func(g *milpart.Graph) {
		x := g.Input("x", 1, 8, 256)
		i := idx16(g, "i", 1, 2, 256)
		g.Output(g.Op("gather_along_axis", map[string]any{"x": x, "indices": i, "axis": g.B.ScalarI32(1), "validate_indices": g.B.ScalarBool(false)}, 1, 2, 256))
	},
	"dslice16_const": func(g *milpart.Graph) {
		w := constF16(g, "w", 8*256, 256)
		b := idx16(g, "b", 2)
		e := idx16(g, "e", 2)
		g.Output(g.Op("slice_by_index", map[string]any{"x": w, "begin": b, "end": e}, 256, 256))
	},
	"dslice16_mm": func(g *milpart.Graph) {
		w := constF16(g, "w", 8*256, 256)
		b := idx16(g, "b", 2)
		e := idx16(g, "e", 2)
		a := g.Input("a", 16, 256)
		s := g.Op("slice_by_index", map[string]any{"x": w, "begin": b, "end": e}, 256, 256)
		g.Output(g.Op("matmul", map[string]any{"x": a, "y": s, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(false)}, 16, 256))
	},
	// weights passed at run time: the host picks the experts and binds their arrays
	"mm_dyn": func(g *milpart.Graph) {
		a := g.Input("a", 1, 16, 256)
		w := g.Input("w", 1, 256, 256)
		g.Output(g.Op("matmul", map[string]any{"x": a, "y": w, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(false)}, 1, 16, 256))
	},
	"mm_dyn_big": func(g *milpart.Graph) { // gpt-oss's width: a token's rows times an expert's 2880 x 2880
		a := g.Input("a", 1, 8, 2880)
		w := g.Input("w", 1, 2880, 2880)
		g.Output(g.Op("matmul", map[string]any{"x": a, "y": w, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(false)}, 1, 8, 2880))
	},
	"mm_dyn_i8": func(g *milpart.Graph) { // int8 weights at run time, cast and scaled on the device
		a := g.Input("a", 1, 16, 256)
		w := g.InputI8("w", 1, 256, 256)
		wf := g.B.Op("cast", map[string]any{"x": w, "dtype": g.B.Str("c", "fp16")}, mlpackage.Float16, 1, 256, 256)
		wf = g.Scale(wf, 0.01)
		g.Output(g.Op("matmul", map[string]any{"x": a, "y": wf, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(false)}, 1, 16, 256))
	},
	"mm_dyn_i8_big": func(g *milpart.Graph) { // int8 weights at run time at gpt-oss's width
		a := g.Input("a", 1, 8, 2880)
		w := g.InputI8("w", 1, 2880, 2880)
		wf := g.B.Op("cast", map[string]any{"x": w, "dtype": g.B.Str("c", "fp16")}, mlpackage.Float16, 1, 2880, 2880)
		wf = g.Scale(wf, 0.0005)
		g.Output(g.Op("matmul", map[string]any{"x": a, "y": wf, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(false)}, 1, 8, 2880))
	},
	"lin_const_big": func(g *milpart.Graph) { // the same product with constant fp16 weights, for comparison
		a := g.Input("a", 1, 8, 2880)
		g.Output(g.LinearConst(a, "w", randW(2880*2880, 2880), 2880, nil))
	},
	"lin_i8_big": int8Linear8(2880, 2880),
	"argmax": func(g *milpart.Graph) { // the router's choice on the device
		x := g.Input("x", 1, 32)
		i := g.B.Op("reduce_argmax", map[string]any{"x": x, "axis": g.B.ScalarI32(-1), "keep_dims": g.B.ScalarBool(true)}, mlpackage.Int32, 1, 1)
		g.Output(g.B.Op("cast", map[string]any{"x": i, "dtype": g.B.Str("c", "fp16")}, mlpackage.Float16, 1, 1))
	},
	"inputs16":       inputsN(16),
	"inputs20":       inputsN(20),
	"inputs22":       inputsN(22),
	"inputs24":       inputsN(24),
	"inputs26":       inputsN(26),
	"inputs28":       inputsN(28),
	"inputs30":       inputsN(30),
	"inputs32":       inputsN(32),
	"big6":           bigInputsN(6),
	"big8":           bigInputsN(8),
	"big9":           bigInputsN(9),
	"big10":          bigInputsN(10),
	"big11":          bigInputsN(11),
	"big12":          bigInputsN(12),
	"big14":          bigInputsN(14),
	"big16":          bigInputsN(16),
	"int8_row_big":   int8Linear(2048, 1024, 1024),
	"int8_row_small": int8Linear(64, 512, 512),
	"int8_blk32_big": int8Linear(2048, 1024, 32),
	"int8_row_mid":   int8Linear(512, 1024, 1024),
	"slice_concat": func(g *milpart.Graph) {
		x := g.Input("x", 1, 256, 512)
		a, b := g.Slice(x, -1, 0, 256), g.Slice(x, -1, 256, 512)
		g.Output(g.Concat(-1, b, a))
	},
	// Sana's VAE (design.md 16.189): the first stage at 1024 x 1024 falls off the Neural Engine
	// from its first EfficientViT block's q/k/v convolution (1024 -> 3072 at 32 x 32)
	"conv1x1_1024_3072_32": convProbe(1024, 3072, 32, 1, false),
	"conv1x1_1024_3072_16": convProbe(1024, 3072, 16, 1, false),
	"conv1x1_1024_1024_32": convProbe(1024, 1024, 32, 1, false),
	"conv1x1_1024_2048_32": convProbe(1024, 2048, 32, 1, false),
	"conv1x1_512_3072_32":  convProbe(512, 3072, 32, 1, false),
	"conv1x1_1024_3072_48": convProbe(1024, 3072, 48, 1, false),
	"conv3x3_32_1024_then_qkv_32": func(g *milpart.Graph) {
		z := g.Input("x", 1, 32, 32, 32)
		y := g.Conv2D(z, "a", randW(1024*32*9, 32*9), make([]float32, 1024), 1024, 3, 1)
		g.Output(g.Conv2D(y, "b", randW(3072*1024, 1024), nil, 3072, 1, 1))
	},
	"vae_in_qkv_32": vaeInProbe(32, true, true),
	"vae_in_32":     vaeInProbe(32, true, false),
	"vae_in_qkv_16": vaeInProbe(16, true, true),
	"tile_qkv_32":   vaeInProbe(32, false, true),
	"vae_in_onehot_qkv_32": func(g *milpart.Graph) {
		z := g.Scale(g.Input("x", 1, 32, 32, 32), 2)
		e := make([]float32, 1024*32)
		for c := 0; c < 1024; c++ {
			e[c*32+c/32] = 1
		}
		y := g.Add(g.Conv2D(z, "a", randW(1024*32*9, 32*9), make([]float32, 1024), 1024, 3, 1), g.Conv2D(z, "e", e, nil, 1024, 1, 1))
		g.Output(g.Conv2D(y, "b", randW(3072*1024, 1024), nil, 3072, 1, 1))
	},
	// MoGe-2's dense head on the M1 (design.md 16.190): the deconvolution 128 -> 64 at 200 x 200
	// (a 1 x 1 convolution to 256 channels, then pixel_shuffle) falls off the Neural Engine
	"ps_256_200":            psProbe(256, 200, 0, 1),
	"conv_ps_128_200":       psProbe(128, 200, 256, 1),
	"conv_ps_128_100":       psProbe(128, 100, 256, 1),
	"conv_ps_128_200_rows2": psProbe(128, 200, 256, 2),
	"conv_ps_128_200_rows4": psProbe(128, 200, 256, 4),
	"transpose": func(g *milpart.Graph) {
		x := g.Input("x", 1, 256, 8, 64)
		g.Output(g.Transpose(x, 0, 2, 1, 3))
	},
	// Qwen-Image's VAE on M1 (design.md 16.184): the Neural Engine's compiler fails ("Allocation
	// failure for L2-dep pair") on the norm over 1152 channels of 64 x 64, not on 768 x 64 x 64,
	// 1152 x 48 x 48 or 576 x 96 x 96, nor on the convolution alone
	"conv1152_64_neg":      conv3(1152, 64, "neg"),
	"norm1152_64":          conv3(1152, 64, "norm noconv"),
	"norm_conv1152_64_neg": conv3(1152, 64, "norm neg"),
	"norm_conv1152_48_neg": conv3(1152, 48, "norm neg"),
	"norm_conv768_64_neg":  conv3(768, 64, "norm neg"),
	"norm_conv576_96_neg":  conv3(576, 96, "norm neg"),
}

// conv3 is a 3 x 3 convolution of c channels over hw x hw, after SiLU (tail "pre ...") or after
// MaxScaled over the channels and SiLU ("norm ..."; "norm noconv" without the convolution), then
// the rest of tail: "" nothing, neg (times -1), max (MaxScaled over the channels) or silu.
func conv3(c, hw int, tail string) func(g *milpart.Graph) {
	return func(g *milpart.Graph) {
		x := g.Input("x", 1, c, hw, hw)
		switch {
		case strings.HasPrefix(tail, "pre"):
			x = g.SiLU(x)
		case strings.HasPrefix(tail, "norm"):
			x = g.SiLU(g.MaxScaled(x, 1))
		}
		tail = strings.TrimSpace(strings.TrimPrefix(strings.TrimPrefix(tail, "pre"), "norm"))
		if tail == "noconv" {
			g.Output(x)
			return
		}
		w := make([]float32, c*c*9)
		for i := range w {
			w[i] = float32(rand.NormFloat64() / 100)
		}
		bias := make([]float32, c)
		y := g.Conv2D(x, "w", w, bias, c, 3, 1)
		switch tail {
		case "neg":
			y = g.Scale(y, -1)
		case "max":
			y = g.MaxScaled(y, 1)
		case "silu":
			y = g.SiLU(y)
		}
		g.Output(y)
	}
}

// convProbe is a k x k convolution of cin to cout channels over an n x n input.
func convProbe(cin, cout, n, k int, bias bool) func(g *milpart.Graph) {
	return func(g *milpart.Graph) {
		x := g.Input("x", 1, cin, n, n)
		var b []float32
		if bias {
			b = make([]float32, cout)
		}
		g.Output(g.Conv2D(x, "w", randW(cout*cin*k*k, cin*k*k), b, cout, k, 1))
	}
}

// vaeInProbe is Sana's VAE conv_in over latents of n x n with (conv) and its shortcut of the 32
// channels each repeated 32 times (tile), then (qkv) the first EfficientViT block's q/k/v
// convolution, 1024 -> 3072.
func vaeInProbe(n int, conv, qkv bool) func(g *milpart.Graph) {
	return func(g *milpart.Graph) {
		z := g.Scale(g.Input("x", 1, 32, n, n), 2)
		t := g.Reshape(z, 1, 32, 1, n*n)
		t = g.Op("tile", map[string]any{"x": t, "reps": g.B.Ints(1, 1, 32, 1)}, 1, 32, 32, n*n)
		y := g.Reshape(t, 1, 1024, n, n)
		if conv {
			y = g.Add(g.Conv2D(z, "a", randW(1024*32*9, 32*9), make([]float32, 1024), 1024, 3, 1), y)
		}
		if qkv {
			y = g.Conv2D(y, "b", randW(3072*1024, 1024), nil, 3072, 1, 1)
		}
		g.Output(y)
	}
}

// psProbe is pixel_shuffle by 2 of (1, c, n, n), after a 1 x 1 convolution to conv channels when
// conv > 0, in rows bands of the rows concatenated back.
func psProbe(c, n, conv, rows int) func(g *milpart.Graph) {
	return func(g *milpart.Graph) {
		x := g.Input("x", 1, c, n, n)
		if conv > 0 {
			x = g.Conv2D(x, "w", randW(conv*c, c), nil, conv, 1, 1)
		}
		C := x.Shape[1]
		var parts []*mlpackage.Var
		for r := 0; r < rows; r++ {
			lo, hi := r*n/rows, (r+1)*n/rows
			band := x
			if rows > 1 {
				band = g.Slice(x, 2, lo, hi)
			}
			parts = append(parts, g.Op("pixel_shuffle", map[string]any{"x": band, "upscale_factor": g.B.ScalarI32(2)}, 1, C/4, 2*(hi-lo), 2*n))
		}
		if len(parts) == 1 {
			g.Output(parts[0])
			return
		}
		g.Output(g.Concat(2, parts...))
	}
}

// int8Linear8 is a linear of a constant int8 weight (out x in, a scale per row) on 8 rows.
func int8Linear8(out, in int) func(g *milpart.Graph) {
	return func(g *milpart.Graph) {
		x := g.Input("x", 1, 8, in)
		data := make([]byte, out*in)
		for i := range data {
			data[i] = byte(rand.Intn(255) - 127)
		}
		scales := make([]uint16, out)
		for i := range scales {
			scales[i] = mlpackage.Float16Bits(0.0005)
		}
		w, err := g.B.Int8Scaled("w", []int{out, in}, []int{1, in}, data, scales)
		if err != nil {
			panic(err)
		}
		g.Output(g.Op("linear", map[string]any{"x": x, "weight": w}, 1, 8, out))
	}
}

// idx16 is an fp16 input cast to int16 on the device: indices the Neural Engine takes (Core ML's
// model description has no int16 input).
func idx16(g *milpart.Graph, name string, shape ...int) *mlpackage.Var {
	x := g.Input(name, shape...)
	return g.B.Op("cast", map[string]any{"x": x, "dtype": g.B.Str("c", "int16")}, mlpackage.Int16, shape...)
}

// inputsN is n inputs of 64 x 64 summed: what a call costs by its number of inputs.
func inputsN(n int) func(g *milpart.Graph) {
	return func(g *milpart.Graph) {
		acc := g.Input("x0", 1, 64, 64)
		for i := 1; i < n; i++ {
			acc = g.Add(acc, g.Input(fmt.Sprintf("x%d", i), 1, 64, 64))
		}
		g.Output(acc)
	}
}

// bigInputsN is n inputs of 1 MiB (512 x 1024 fp16) summed.
func bigInputsN(n int) func(g *milpart.Graph) {
	return func(g *milpart.Graph) {
		acc := g.Input("x0", 1, 512, 1024)
		for i := 1; i < n; i++ {
			acc = g.Add(acc, g.Input(fmt.Sprintf("x%d", i), 1, 512, 1024))
		}
		g.Output(acc)
	}
}

// constF16 is a constant fp16 tensor of Gaussian values.
func constF16(g *milpart.Graph, name string, shape ...int) *mlpackage.Var {
	n := 1
	for _, d := range shape {
		n *= d
	}
	w, err := g.B.F16s(name, shape, randW(n, shape[len(shape)-1]))
	if err != nil {
		panic(err)
	}
	return w
}

// randW is n Gaussian weights for a fan-in of fan.
func randW(n, fan int) []float32 {
	w := make([]float32, n)
	s := 1 / math.Sqrt(float64(fan))
	for i := range w {
		w[i] = float32(rand.NormFloat64() * s)
	}
	return w
}

// int8Linear is a linear of an int8 weight (out x in) with a scale per block of colBlock columns
// of each row (hwnpu's int8 format: per row, colBlock = in).
func int8Linear(out, in, colBlock int) func(g *milpart.Graph) {
	return func(g *milpart.Graph) {
		x := g.Input("x", 1, 256, in)
		data := make([]byte, out*in)
		for i := range data {
			data[i] = byte(rand.Intn(255) - 127)
		}
		scales := make([]uint16, out*(in/colBlock))
		for i := range scales {
			scales[i] = mlpackage.Float16Bits(0.001)
		}
		w, err := g.B.Int8Scaled("w", []int{out, in}, []int{1, colBlock}, data, scales)
		if err != nil {
			panic(err)
		}
		g.Output(g.Op("linear", map[string]any{"x": x, "weight": w}, 1, 256, out))
	}
}

func probe(names []string, keep bool) error {
	ctx := context.Background()
	if err := llm.LockNPU(ctx); err != nil {
		return err
	}
	var dev backend.Device
	var err error
	if os.Getenv("COREML") != "" { // Core ML's compute plan: each operation's devices
		dev, err = coreml.Provider{}.Open(ctx, types.DeviceInfo{}, backend.Options{})
	} else {
		dev, err = coreai.Provider{}.Open(ctx, types.DeviceInfo{}, backend.Options{})
	}
	if err != nil {
		return err
	}
	dir, err := os.MkdirTemp("", "aicheck-probe-*")
	if err != nil {
		return err
	}
	if !keep {
		defer os.RemoveAll(dir)
	}
	if len(names) == 0 {
		for n := range probes {
			names = append(names, n)
		}
	}
	for _, n := range names {
		body, ok := probes[n]
		if !ok {
			return fmt.Errorf("no probe %s", n)
		}
		path := filepath.Join(dir, n+".mlpackage")
		part := &milpart.Part{Name: n, Model: "probe", Body: body}
		if _, err := llm.WritePart(path, part, nil, llm.Options{Backend: llm.BackendCoreML, Format: "fp16"}, nil); err != nil {
			return fmt.Errorf("%s: %w", n, err)
		}
		g, err := dev.Compile(ctx, types.GraphSource{Path: path})
		if err != nil {
			fmt.Printf("%-14s FAIL %v\n", n, err)
			continue
		}
		p := g.Placement()
		fmt.Printf("%-14s ane=%v verified=%v reason=%q ops=%v\n", n, p.OnNPU(), p.Verified, p.Reason, p.Ops)
		g.Close()
	}
	if keep {
		fmt.Println("packages in", dir)
	}
	return nil
}
