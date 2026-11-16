package main

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/coreai"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/types"
)

// timeGroups loads the programs and calls them k at a time, the k of a group side by side (as
// the 4 experts of a token's layer would be), for rounds rounds: the time a group takes.
func timeGroups(paths []string, k, rounds int, hold time.Duration) error {
	ctx := context.Background()
	if err := llm.LockNPU(ctx); err != nil {
		return err
	}
	dev, err := coreai.Provider{}.Open(ctx, types.DeviceInfo{}, backend.Options{})
	if err != nil {
		return err
	}
	var reqs []backend.Request
	rng := rand.New(rand.NewSource(1))
	t0 := time.Now()
	for i, path := range paths {
		if i%16 == 0 {
			fmt.Printf("loading %d (%.1f s)\n", i, time.Since(t0).Seconds())
		}
		g, err := dev.Compile(ctx, types.GraphSource{Path: path})
		if err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		defer g.Close()
		req, err := g.NewRequest()
		if err != nil {
			return err
		}
		for _, a := range g.Args() {
			b, err := dev.AllocBuffer(a.Bytes, types.BufferKind(0))
			if err != nil {
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
		reqs = append(reqs, req)
	}
	fmt.Printf("loaded %d programs in %.1f s\n", len(reqs), time.Since(t0).Seconds())
	if hold > 0 {
		for _, r := range reqs {
			if f, err := r.Submit(); err == nil {
				f.Wait(ctx)
			}
		}
		fmt.Println("holding")
		time.Sleep(hold)
		return nil
	}
	call := func(r backend.Request) error {
		f, err := r.Submit()
		if err == nil {
			err = f.Wait(ctx)
		}
		return err
	}
	for _, r := range reqs { // first calls apart
		if err := call(r); err != nil {
			return err
		}
	}
	for _, kk := range []int{1, k} {
		var total time.Duration
		groups := 0
		for range rounds {
			for i := 0; i+kk <= len(reqs); i += kk {
				t := time.Now()
				if kk == 1 {
					if err := call(reqs[i]); err != nil {
						return err
					}
				} else {
					var wg sync.WaitGroup
					errs := make([]error, kk)
					for j := 0; j < kk; j++ {
						wg.Add(1)
						go func(j int) { defer wg.Done(); errs[j] = call(reqs[i+j]) }(j)
					}
					wg.Wait()
					for _, e := range errs {
						if e != nil {
							return e
						}
					}
				}
				total += time.Since(t)
				groups++
			}
		}
		per := float64(total.Microseconds()) / 1e3 / float64(groups)
		fmt.Printf("%d at a time: %.3f ms a group, %.3f ms a program\n", kk, per, per/float64(kk))
	}
	return nil
}
