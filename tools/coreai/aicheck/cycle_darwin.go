package main

import (
	"context"
	"fmt"
	"math/rand"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/coreai"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/types"
)

// cycle loads every package at once and times their calls two ways: each called rounds times in
// a row (as run's second time), and all of them in turn for rounds rounds (as a diffusion step
// runs its groups of blocks). The difference is what the Neural Engine pays when a program
// follows another (design.md 16.201).
func cycle(pkgs []string, rounds int) error {
	ctx := context.Background()
	if !coreai.OnCPU() {
		if err := llm.LockNPU(ctx); err != nil {
			return err
		}
	}
	dev, err := coreai.Provider{}.Open(ctx, types.DeviceInfo{}, backend.Options{})
	if err != nil {
		return err
	}
	type prog struct {
		name string
		g    backend.Graph
		req  backend.Request
	}
	var ps []prog
	defer func() {
		for _, p := range ps {
			p.g.Close()
		}
	}()
	rng := rand.New(rand.NewSource(1))
	for _, path := range pkgs {
		g, err := dev.Compile(ctx, types.GraphSource{Path: path})
		if err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		req, err := g.NewRequest()
		if err != nil {
			g.Close()
			return err
		}
		for _, a := range g.Args() {
			b, err := dev.AllocBuffer(a.Bytes, types.BufferKind(0))
			if err != nil {
				g.Close()
				return err
			}
			req.Bind(a.Index, b)
			if a.Kind == types.ArgInput {
				f := make([]float32, a.Bytes/2)
				for i := range f {
					f[i] = float32(rng.NormFloat64() * 0.5)
				}
				hwnpu.EncodeF16(b.Bytes(), f)
			}
		}
		ps = append(ps, prog{filepath.Base(path), g, req})
	}
	call := func(p prog) (time.Duration, error) {
		t := time.Now()
		f, err := p.req.Submit()
		if err == nil {
			err = f.Wait(ctx)
		}
		return time.Since(t), err
	}
	alone := make([]time.Duration, len(ps))
	for i, p := range ps {
		if _, err := call(p); err != nil { // the first call apart
			return fmt.Errorf("%s: %w", p.name, err)
		}
		for range rounds {
			d, err := call(p)
			if err != nil {
				return err
			}
			alone[i] += d
		}
	}
	turn := make([]time.Duration, len(ps))
	for r := 0; r <= rounds; r++ {
		for i, p := range ps {
			d, err := call(p)
			if err != nil {
				return err
			}
			if r > 0 {
				turn[i] += d
			}
		}
	}
	var sa, st time.Duration
	for i, p := range ps {
		a, t := alone[i]/time.Duration(rounds), turn[i]/time.Duration(rounds)
		sa, st = sa+a, st+t
		fmt.Printf("%-28s alone %8.2f ms  in turn %8.2f ms  (+%.2f)\n", p.name, ms(a), ms(t), ms(t-a))
	}
	fmt.Printf("%-28s alone %8.2f ms  in turn %8.2f ms  (+%.2f)\n", "all", ms(sa), ms(st), ms(st-sa))
	return nil
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1e3 }
