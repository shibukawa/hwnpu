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
	dev, err := coreai.Provider{}.Open(ctx, types.DeviceInfo{}, backend.Options{})
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
		fmt.Printf("%-14s ane=%v\n", n, g.Placement().OnNPU())
		g.Close()
	}
	if keep {
		fmt.Println("packages in", dir)
	}
	return nil
}
