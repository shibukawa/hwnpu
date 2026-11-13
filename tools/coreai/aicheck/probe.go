package main

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"

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
	"transpose": func(g *milpart.Graph) {
		x := g.Input("x", 1, 256, 8, 64)
		g.Output(g.Transpose(x, 0, 2, 1, 3))
	},
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
