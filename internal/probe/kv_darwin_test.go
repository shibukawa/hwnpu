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

// The chunk attention program of every form stays on the Neural Engine with 64 query rows, and
// with one over a chunk of the longest length a dimension takes. With one row over 1024 keys
// Core ML puts the whole program on the CPU (M3, macOS 27.0.1): too little work for the Neural
// Engine by its estimate. That case is logged, not required; npuprobe attn maps the boundary.
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
		for _, c := range []struct {
			q, chunk int
			want     bool
		}{{1, 1024, false}, {1, 4096, false}, {1, 16384, true}, {8, 1024, false}, {64, 1024, true}, {64, 16384, true}} {
			dir := filepath.Join(t.TempDir(), fmt.Sprintf("attn_%s_%d_%d.mlpackage", form, c.q, c.chunk))
			if err := WriteChunkAttnPackage(dir, form, c.q, c.chunk); err != nil {
				t.Fatal(err)
			}
			g, err := dev.Compile(ctx, types.GraphSource{Path: dir})
			if err != nil {
				t.Errorf("%s Q=%d chunk %d: %v", form, c.q, c.chunk, err)
				continue
			}
			p := g.Placement()
			switch {
			case !p.Verified:
				t.Logf("%s Q=%d chunk %d: placement not verified", form, c.q, c.chunk)
			case p.OnNPU():
				t.Logf("%s Q=%d chunk %d: on the Neural Engine", form, c.q, c.chunk)
			case c.want:
				t.Errorf("%s Q=%d chunk %d: %d operations off the Neural Engine: %v", form, c.q, c.chunk, len(p.Ops), p.Ops)
			default:
				t.Logf("%s Q=%d chunk %d: %d of %d operations off the Neural Engine", form, c.q, c.chunk, len(p.Ops), len(p.Ops)+p.NPUOps)
			}
			g.Close()
		}
	}
}
