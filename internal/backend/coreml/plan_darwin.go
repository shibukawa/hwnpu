package coreml

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/ebitengine/purego/objc"

	"github.com/shibukawa/hwnpu/internal/types"
)

const planMethod = "coreml-compute-plan"

// planTimeout is how long the compute plan may take. A plan takes as long as loading the model;
// on an M1 whose Neural Engine another process is compiling for, Gemma 2's programs of 470 MB
// for Sana took over 2 minutes where they took 13 s alone (design.md 16.143).
const planTimeout = 10 * time.Minute

type planResult struct{ plan, err objc.ID }

// computePlacement loads an MLComputePlan (macOS 14.4+) for the compiled model and lists the
// ML Program operations whose preferred device is not the Neural Engine.
// With a function name only that function is walked; otherwise all of them.
//
// A plan that fails to load is asked for once more: on an M1 Core ML now and then answers
// "Failed to construct compute plan, internal failure" for a program whose plan the next try
// (or the next process) builds (a DiT group of Sana, design.md 16.143).
func computePlacement(ctx context.Context, compiledURL, config objc.ID, function string) (types.PlacementReport, error) {
	rep, err := computePlacementOnce(ctx, compiledURL, config, function)
	if err == nil && !rep.Verified && strings.Contains(rep.Reason, "load compute plan") {
		rep, err = computePlacementOnce(ctx, compiledURL, config, function)
	}
	return rep, err
}

func computePlacementOnce(ctx context.Context, compiledURL, config objc.ID, function string) (types.PlacementReport, error) {
	rep := types.PlacementReport{Method: planMethod}
	if class("MLComputePlan") == 0 {
		rep.Reason = "MLComputePlan is not available (requires macOS 14.4 or later)"
		return rep, nil
	}

	ch := make(chan planResult, 1)
	blk := objc.NewBlock(func(_ objc.Block, plan, e objc.ID) {
		ch <- planResult{retain(plan), retain(e)}
	})
	_ = withPool(func() error {
		send(class("MLComputePlan"), "loadContentsOfURL:configuration:completionHandler:", compiledURL, config, blk)
		return nil
	})
	defer blk.Release()

	var res planResult
	select {
	case res = <-ch:
	case <-ctx.Done():
		go func() { r := <-ch; _ = withPool(func() error { release(r.plan); release(r.err); return nil }) }()
		return rep, ctx.Err()
	case <-time.After(planTimeout):
		rep.Reason = fmt.Sprintf("MLComputePlan did not finish within %v", planTimeout)
		return rep, nil
	}

	err := withPool(func() error {
		defer release(res.plan)
		defer release(res.err)
		if res.plan == 0 {
			rep.Reason = nsError("load compute plan", res.err).Error()
			return nil
		}
		program := send(send(res.plan, "modelStructure"), "program")
		if program == 0 {
			rep.Reason = "the model is not an ML Program; Core ML cannot report per-operation placement"
			return nil
		}
		names, fns := dictItems(send(program, "functions"))
		for i, fn := range fns {
			if function == "" || names[i] == function {
				walkBlock(res.plan, send(fn, "block"), &rep)
			}
		}
		rep.Verified = true
		return nil
	})
	return rep, err
}

func walkBlock(plan, block objc.ID, rep *types.PlacementReport) {
	for _, op := range arrayItems(send(block, "operations")) {
		for _, inner := range arrayItems(send(op, "blocks")) {
			walkBlock(plan, inner, rep)
		}
		usage := send(plan, "computeDeviceUsageForMLProgramOperation:", op)
		if usage == 0 {
			continue // constants and other operations without a device
		}
		dev := send(usage, "preferredComputeDevice")
		if isKind(dev, "MLNeuralEngineComputeDevice") {
			rep.NPUOps++
			continue
		}
		device := "unknown"
		switch {
		case isKind(dev, "MLCPUComputeDevice"):
			device = "cpu"
		case isKind(dev, "MLGPUComputeDevice"):
			device = "gpu"
		}
		name := ""
		if outs := arrayItems(send(op, "outputs")); len(outs) > 0 {
			name = goString(send(outs[0], "name"))
		}
		rep.Ops = append(rep.Ops, types.OpPlacement{Name: name, Type: goString(send(op, "operatorName")), Device: device})
	}
}
