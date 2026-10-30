package chat

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/backend/coreml"
	"github.com/shibukawa/hwnpu/internal/llm/chain"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
)

// Times one call of each program of LFM3-1.7B converted by Prepare (1, 64 and the draft's
// tokens per call, cache 16320) on the Neural Engine: the cost of a call hardly depends on the
// tokens it takes, which is what speculative decoding relies on. Needs HWNPU_LFM3_DIR (the
// model's directory of hwnpu-chat) and prints the times.
func TestCallTimes(t *testing.T) {
	dir := os.Getenv("HWNPU_LFM3_DIR")
	if dir == "" {
		t.Skip("HWNPU_LFM3_DIR not set")
	}
	ctx := context.Background()
	if infos, err := (coreml.Provider{}).Enumerate(ctx); err != nil || len(infos) == 0 {
		t.Skip("no Neural Engine")
	}
	cache, _ := os.UserCacheDir()
	dev, err := hwnpu.Open(ctx, hwnpu.WithBackend("coreml"), hwnpu.WithPlacement(hwnpu.PlacementPrefer), hwnpu.WithCacheDir(filepath.Join(cache, "hwnpu")))
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	var first *chain.Chain
	var progs []decode.Program
	for _, name := range []string{"lfm3-c16320-q1.parts.json", "lfm3-c16320-q8.mlpackage", "lfm3-c16320-q64.mlpackage"} {
		path := filepath.Join(dir, name)
		if _, err := os.Stat(path); err != nil {
			t.Logf("%s: %v", name, err)
			continue
		}
		var c *chain.Chain
		if first == nil {
			c, err = chain.Open(ctx, dev, path)
			first = c
		} else {
			c, err = first.Share(ctx, dev, path)
		}
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		p := c.Program(ctx)
		progs = append(progs, p)
		for i := 0; i < 3; i++ {
			if err := p.Run(); err != nil {
				t.Fatal(err)
			}
		}
		const n = 20
		t0 := time.Now()
		for i := 0; i < n; i++ {
			if err := p.Run(); err != nil {
				t.Fatal(err)
			}
		}
		t.Logf("%s: %d parts, %.1f ms per call", name, len(c.Parts()), float64(time.Since(t0).Microseconds())/n/1000)
	}
	// alternating programs: does switching cost?
	for _, pair := range [][2]int{{0, 1}, {0, 2}, {1, 2}} {
		if pair[1] >= len(progs) {
			continue
		}
		const n = 10
		t0 := time.Now()
		for i := 0; i < n; i++ {
			for _, j := range pair {
				if err := progs[j].Run(); err != nil {
					t.Fatal(err)
				}
			}
		}
		t.Logf("alternating programs %d and %d: %.1f ms per pair of calls", pair[0], pair[1], float64(time.Since(t0).Microseconds())/n/1000)
	}
}
