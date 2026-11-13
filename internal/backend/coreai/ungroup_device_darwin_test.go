package coreai_test

import (
	"context"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/coreai"
	"github.com/shibukawa/hwnpu/internal/imagegen/milpart"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/types"
)

// TestUngroupOnDevice loads, on Core AI (HWNPU_COREAI_DEVICE=1, M1), two exports' package of the
// same weights: Qwen-Image's VAE norm and convolution of 1152 channels at 32 x 32, which runs on
// the Neural Engine, and at 64 x 64, which the Neural Engine's compiler fails on (design.md
// 16.184). Lowered together, the first would be off the Neural Engine too: the group is taken
// apart and the first runs on it alone.
func TestUngroupOnDevice(t *testing.T) {
	if os.Getenv("HWNPU_COREAI_DEVICE") != "1" {
		t.Skip("HWNPU_COREAI_DEVICE=1 not set")
	}
	t.Setenv("HWNPU_CACHE", t.TempDir())
	ctx := context.Background()
	const c = 1152
	rng := rand.New(rand.NewPCG(1, 2))
	w := make([]float32, c*c*9)
	for i := range w {
		w[i] = float32(rng.NormFloat64() / 100)
	}
	bias := make([]float32, c)
	root := t.TempDir()
	write := func(hw int) string {
		dir := filepath.Join(root, "model-"+string(rune('a'+hw/32))+".coreml")
		os.MkdirAll(dir, 0o755)
		path := filepath.Join(dir, "vae.mlpackage")
		part := &milpart.Part{Name: "vae", Model: "probe", Body: func(g *milpart.Graph) {
			x := g.Input("x", 1, c, hw, hw)
			x = g.SiLU(g.MaxScaled(x, 1))
			g.Output(g.Scale(g.Conv2D(x, "w", w, bias, c, 3, 1), -1))
		}}
		if _, err := llm.WritePart(path, part, nil, llm.Options{Backend: llm.BackendCoreML, Format: "fp16"}, nil); err != nil {
			t.Fatal(err)
		}
		return path
	}
	small, large := write(32), write(64)
	dev, err := coreai.Provider{}.Open(ctx, types.DeviceInfo{}, backend.Options{CacheDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	for _, c := range []struct {
		path string
		on   bool
	}{{small, true}, {large, false}, {small, true}} {
		g, err := dev.Compile(ctx, types.GraphSource{Format: types.GraphFormatNative, Path: c.path})
		if err != nil {
			t.Fatal(err)
		}
		rep := g.Placement()
		g.Close()
		if rep.OnNPU() != c.on {
			t.Errorf("%s: on the Neural Engine %v (%+v), want %v", c.path, rep.OnNPU(), rep, c.on)
		}
	}
}
