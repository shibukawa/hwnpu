package main

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/coreai"
	"github.com/shibukawa/hwnpu/internal/backend/coreml"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/types"
)

// run loads each package with the Core AI backend (lowering it), runs it on random inputs
// (normal, times scale) and compares its outputs with the CPU interpreter's (milcpu, fp16
// rounding): the cosine and the relative RMS error, and whether it is all on the Neural Engine.
//
// The second time is the mean of calls more calls. An .aimodel runs as it is; -cpu then compares it
// with milcpu running ref, the package it was lowered from. With coreML the packages run on Core ML instead, to tell Core AI's errors from the program's.
func run(pkgs []string, cacheDir string, scale float64, cpu, coreML bool, calls int, ref string) error {
	ctx := context.Background()
	if os.Getenv("AICHECK_ENUMERATE") != "" { // as the commands do: every backend's devices first
		if _, err := hwnpu.Enumerate(ctx); err != nil {
			return err
		}
	}
	var prov backend.Provider = coreai.Provider{}
	if coreML {
		prov = coreml.Provider{}
	}
	dev, err := prov.Open(ctx, types.DeviceInfo{}, backend.Options{CacheDir: cacheDir})
	if err != nil {
		return err
	}
	for _, p := range pkgs {
		name := filepath.Base(p)
		t := time.Now()
		g, err := dev.Compile(ctx, types.GraphSource{Path: p})
		if err != nil {
			fmt.Printf("%-24s FAIL %v\n", name, err)
			continue
		}
		load := time.Since(t)
		rng := rand.New(rand.NewSource(1))
		req, _ := g.NewRequest()
		in := map[string]milcpu.Tensor{}
		var outs []types.ArgInfo
		var bufs []backend.Buffer
		for _, a := range g.Args() {
			b, err := dev.AllocBuffer(a.Bytes, types.BufferKind(0))
			if err != nil {
				return err
			}
			bufs = append(bufs, b)
			req.Bind(a.Index, b)
			if a.Kind == types.ArgInput {
				n := a.Bytes / 2
				f := make([]float32, n)
				for i := range f {
					f[i] = float32(rng.NormFloat64() * scale)
				}
				hwnpu.EncodeF16(b.Bytes(), f)
				hwnpu.DecodeF16(f, b.Bytes()) // the values the device sees
				in[a.Name] = milcpu.Tensor{Shape: a.Dims, Data: f}
			} else {
				outs = append(outs, a)
			}
		}
		t = time.Now()
		f, err := req.Submit()
		if err == nil {
			err = f.Wait(ctx)
		}
		call := time.Since(t)
		if err != nil {
			fmt.Printf("%-24s FAIL run: %v\n", name, err)
			g.Close()
			continue
		}
		t = time.Now()
		for range max(calls, 1) {
			f, _ = req.Submit()
			f.Wait(ctx)
		}
		call2 := time.Since(t) / time.Duration(max(calls, 1))
		on := g.Placement().OnNPU()
		line := fmt.Sprintf("%-24s load %6.2fs call %.2fms/%.2fms ane=%v cached=%v", name, load.Seconds(), float64(call.Microseconds())/1e3, float64(call2.Microseconds())/1e3, on, g.Cached())
		if cpu {
			src := p
			if ref != "" {
				src = ref // p is an .aimodel lowered from ref
			}
			prog, closer, err := milcpu.OpenPackage(src)
			if err != nil {
				return err
			}
			want, err := prog.Run(in, milcpu.Options{RoundF16: true})
			closer.Close()
			if err != nil {
				return err
			}
			for _, o := range outs {
				got := make([]float32, o.Bytes/2)
				hwnpu.DecodeF16(got, bufs[o.Index].Bytes())
				cos, rel, bad := compare(got, want[o.Name].Data)
				line += fmt.Sprintf(" | %s cos %.6f rel %.4f nonfinite %d", o.Name, cos, rel, bad)
			}
		}
		fmt.Println(line)
		g.Close()
		for _, b := range bufs {
			b.Close()
		}
	}
	return nil
}

func compare(got, want []float32) (cos, rel float64, bad int) {
	var dot, ng, nw, diff float64
	for i := range got {
		g, w := float64(got[i]), float64(want[i])
		if math.IsNaN(g) || math.IsInf(g, 0) {
			bad++
			continue
		}
		dot += g * w
		ng += g * g
		nw += w * w
		diff += (g - w) * (g - w)
	}
	return dot / math.Sqrt(ng*nw), math.Sqrt(diff / nw), bad
}
