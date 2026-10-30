package chat

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/backend/coreml"
	"github.com/shibukawa/hwnpu/internal/llm/chain"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
)

// Times one call of each of a model's programs on the Neural Engine, alone and alternating with
// the others: the cost of a call hardly depends on the tokens it takes (speculative decoding
// relies on that), and switching programs costs. Needs HWNPU_LFM3_DIR (the model's directory of
// hwnpu-chat); HWNPU_CALL_PKGS names the programs, groups that share one cache separated by ";"
// and programs by "," (default: the programs of 1, 8 and 64 tokens with the cache of 16320).
func TestCallTimes(t *testing.T) {
	dir := os.Getenv("HWNPU_LFM3_DIR")
	if dir == "" {
		t.Skip("HWNPU_LFM3_DIR not set")
	}
	ctx := context.Background()
	if infos, err := (coreml.Provider{}).Enumerate(ctx); err != nil || len(infos) == 0 {
		t.Skip("no Neural Engine")
	}
	groups := os.Getenv("HWNPU_CALL_PKGS")
	if groups == "" {
		groups = "lfm3-c16320-q1.parts.json,lfm3-c16320-q8.mlpackage,lfm3-c16320-q64.mlpackage"
	}
	cache, _ := os.UserCacheDir()
	dev, err := hwnpu.Open(ctx, hwnpu.WithBackend("coreml"), hwnpu.WithPlacement(hwnpu.PlacementPrefer), hwnpu.WithCacheDir(filepath.Join(cache, "hwnpu")))
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	var names []string
	var progs []decode.Program
	for _, group := range strings.Split(groups, ";") {
		var first *chain.Chain
		for _, name := range strings.Split(group, ",") {
			path := filepath.Join(dir, name)
			if _, err := os.Stat(path); err != nil {
				t.Logf("%s: %v", name, err)
				continue
			}
			t0 := time.Now()
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
			names, progs = append(names, name), append(progs, p)
			for i := 0; i < 3; i++ {
				if err := p.Run(); err != nil {
					t.Fatal(err)
				}
			}
			const n = 20
			t1 := time.Now()
			for i := 0; i < n; i++ {
				if err := p.Run(); err != nil {
					t.Fatal(err)
				}
			}
			t.Logf("%s: %d parts, loaded in %.1fs, %.1f ms per call", name, len(c.Parts()), t1.Sub(t0).Seconds(), float64(time.Since(t1).Microseconds())/n/1000)
		}
	}
	var ru syscall.Rusage
	syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	t.Logf("%d programs loaded, max RSS %.2f GiB", len(progs), float64(ru.Maxrss)/(1<<30))
	// alternating programs: does switching cost?
	for i := range progs {
		for j := i + 1; j < len(progs); j++ {
			const n = 10
			t0 := time.Now()
			for k := 0; k < n; k++ {
				if err := progs[i].Run(); err != nil {
					t.Fatal(err)
				}
				if err := progs[j].Run(); err != nil {
					t.Fatal(err)
				}
			}
			t.Logf("alternating %s and %s: %.1f ms per pair of calls", names[i], names[j], float64(time.Since(t0).Microseconds())/n/1000)
		}
	}
}
