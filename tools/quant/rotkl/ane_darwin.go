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

// runANE runs the model's layers on the Neural Engine (Core AI) in parts of up to perPart layers,
// each an .mlpackage loaded and run on the hidden state the one before it gave (x for the first),
// ones for any other input, and computes the final norm and the output layer on the host in
// float32: whole, the program falls to the GPU (a vocabulary of 248320 is past the Neural
// Engine's axis of 16384, and 24 layers at once did not compile for it).
func runANE(model llm.Model, w llm.Weights, opts llm.Options, x []float32, dir string, perPart int) ([]float32, error) {
	ctx := context.Background()
	opts.NoHead = true
	if err := llm.LockNPU(ctx); err != nil {
		return nil, err
	}
	dev, err := coreai.Provider{}.Open(ctx, types.DeviceInfo{}, backend.Options{})
	if err != nil {
		return nil, err
	}
	h := x
	for start := 0; start < model.Layers(); start += perPart {
		o := opts
		o.Start, o.Layers = start, min(perPart, model.Layers()-start)
		if h, err = runPart(ctx, dev, model, w, o, h, filepath.Join(dir, fmt.Sprintf("part%d.mlpackage", start))); err != nil {
			return nil, fmt.Errorf("layers %d..%d: %w", o.Start, o.Start+o.Layers-1, err)
		}
	}
	return hostHead(model, w, h)
}

func runPart(ctx context.Context, dev backend.Device, model llm.Model, w llm.Weights, opts llm.Options, x []float32, path string) ([]float32, error) {
	if _, err := llm.WritePart(path, model, w, opts, nil); err != nil {
		return nil, err
	}
	defer os.RemoveAll(path)
	g, err := dev.Compile(ctx, types.GraphSource{Path: path})
	if err != nil {
		return nil, err
	}
	defer g.Close()
	if p := g.Placement(); !p.OnNPU() {
		fmt.Fprintf(os.Stderr, "%s: not all on the Neural Engine: %v %s\n", filepath.Base(path), p.Ops, p.Reason)
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
	return h, nil
}
