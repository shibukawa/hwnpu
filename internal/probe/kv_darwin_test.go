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

// Every KV cache probe model, the Core ML states and the int8 cache included, is placed on the
// Neural Engine. A state written and then used without reading it back fails to compile for it
// (mlpackage.UpdateState).
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
	for _, mode := range []string{"host", "state", "host-int8"} {
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
