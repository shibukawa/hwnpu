package coreml

import (
	"context"
	"testing"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/types"
)

// Checks whether Core ML writes outputs straight into the bound buffers (outputBackings).
func TestOutputBackingZeroCopy(t *testing.T) {
	ctx := context.Background()
	infos, err := Provider{}.Enumerate(ctx)
	if err != nil || len(infos) == 0 {
		t.Skip("no Neural Engine")
	}
	dev, _ := Provider{}.Open(ctx, infos[0], backend.Options{})
	for _, m := range []string{"linear_relu_fp16", "linear_relu_fp32"} {
		g, err := dev.Compile(ctx, types.GraphSource{Path: "../../fixtures/coreml/" + m + ".mlpackage"})
		if err != nil {
			t.Fatal(err)
		}
		req, _ := g.NewRequest()
		for _, a := range g.Args() {
			b, _ := dev.AllocBuffer(a.Bytes, types.BufferDefault)
			defer b.Close()
			req.Bind(a.Index, b)
		}
		before := outputCopies.Load()
		f, err := req.Submit()
		if err != nil {
			t.Fatal(err)
		}
		if err := f.Wait(ctx); err != nil {
			t.Fatal(err)
		}
		t.Logf("%s: output copies %d (0 means zero copy)", m, outputCopies.Load()-before)
		req.Close()
		g.Close()
	}
}
