package probe

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/coreml"
	"github.com/shibukawa/hwnpu/internal/types"
)

// Every KV cache probe model, the Core ML states, the int8 cache and the cache in two chunks
// included, is placed on the Neural Engine. A state written and then used without reading it
// back fails to compile for it (mlpackage.UpdateState).
func TestKVOnNeuralEngine(t *testing.T) {
	ctx := context.Background()
	infos, err := coreml.Provider{}.Enumerate(ctx)
	if err != nil || len(infos) == 0 {
		t.Skip("no Neural Engine")
	}
	dev, err := coreml.Provider{}.Open(ctx, infos[0], backend.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	for _, mode := range []string{"host", "state", "host-int8", "host-chunk128"} {
		dir := filepath.Join(t.TempDir(), fmt.Sprintf("kv_%s.mlpackage", mode))
		if err := WriteKVPackage(dir, mode, 2, 256, 1); err != nil {
			t.Fatal(err)
		}
		g, err := dev.Compile(ctx, types.GraphSource{Path: dir})
		if err != nil {
			t.Errorf("%s: %v", mode, err)
			continue
		}
		if p := g.Placement(); p.Verified && !p.OnNPU() {
			t.Errorf("%s: %d operations off the Neural Engine: %v", mode, len(p.Ops), p.Ops)
		}
		g.Close()
	}
}

// The chunk attention program of every form stays on the Neural Engine with one query row and
// with 64, over a chunk of the longest length a dimension takes.
func TestChunkAttnOnNeuralEngine(t *testing.T) {
	ctx := context.Background()
	infos, err := coreml.Provider{}.Enumerate(ctx)
	if err != nil || len(infos) == 0 {
		t.Skip("no Neural Engine")
	}
	dev, err := coreml.Provider{}.Open(ctx, infos[0], backend.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	for _, form := range AttnForms {
		for _, q := range []int{1, 64} {
			for _, c := range []int{1024, 16384} {
				dir := filepath.Join(t.TempDir(), fmt.Sprintf("attn_%s_%d_%d.mlpackage", form, q, c))
				if err := WriteChunkAttnPackage(dir, form, q, c); err != nil {
					t.Fatal(err)
				}
				g, err := dev.Compile(ctx, types.GraphSource{Path: dir})
				if err != nil {
					t.Errorf("%s Q=%d chunk %d: %v", form, q, c, err)
					continue
				}
				if p := g.Placement(); p.Verified && !p.OnNPU() {
					t.Errorf("%s Q=%d chunk %d: %d operations off the Neural Engine: %v", form, q, c, len(p.Ops), p.Ops)
				}
				g.Close()
			}
		}
	}
}
