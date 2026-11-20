package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/coreai"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/types"
)

// runANE writes the program without its output layer as an .mlpackage, loads it with Core AI (on
// the Neural Engine), runs it on x (ones for any other input) and computes the final norm and the
// output layer on the host in float32 (a vocabulary of 248320 is past the Neural Engine's axis
// of 16384: whole, the program falls to the GPU).
func runANE(model llm.Model, w llm.Weights, opts llm.Options, x []float32, dir string) ([]float32, error) {
	ctx := context.Background()
	opts.NoHead = true
	if err := llm.LockNPU(ctx); err != nil {
		return nil, err
	}
	path := filepath.Join(dir, "model.mlpackage")
	if _, err := llm.WritePart(path, model, w, opts, nil); err != nil {
		return nil, err
	}
	dev, err := coreai.Provider{}.Open(ctx, types.DeviceInfo{}, backend.Options{})
	if err != nil {
		return nil, err
	}
	g, err := dev.Compile(ctx, types.GraphSource{Path: path})
	if err != nil {
		return nil, err
	}
	defer g.Close()
	if p := g.Placement(); !p.OnNPU() {
		fmt.Fprintf(os.Stderr, "not all on the Neural Engine: %v %s\n", p.Ops, p.Reason)
	}
	req, err := g.NewRequest()
	if err != nil {
		return nil, err
	}
	var out backend.Buffer
	for _, a := range g.Args() {
		b, err := dev.AllocBuffer(a.Bytes, types.BufferKind(0))
		if err != nil {
			return nil, err
		}
		req.Bind(a.Index, b)
		switch {
		case a.Kind != types.ArgInput:
			if a.Name == "y" || out == nil {
				out = b
			}
		case a.Name == "x":
			hwnpu.EncodeF16(b.Bytes(), x)
		default:
			ones := make([]float32, a.Bytes/2)
			for i := range ones {
				ones[i] = 1
			}
			hwnpu.EncodeF16(b.Bytes(), ones)
		}
	}
	f, err := req.Submit()
	if err == nil {
		err = f.Wait(ctx)
	}
	if err != nil {
		return nil, err
	}
	h := make([]float32, len(out.Bytes())/2)
	hwnpu.DecodeF16(h, out.Bytes())
	return hostHead(model, w, h)
}
