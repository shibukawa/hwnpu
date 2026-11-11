package mil

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/backend/coreml"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
)

// TestGroupOnDevice writes one group of the DiT's image blocks (HWNPU_QWENIMAGE_ROOT, the
// checkpoint) in HWNPU_QWENIMAGE_FORMAT (lut4) at HWNPU_QWENIMAGE_SIZE (512) into
// HWNPU_QWENIMAGE_OUT, loads it on the Neural Engine twice and times its calls: the first load
// compiles it, the second shows whether the Neural Engine's cache keeps it.
func TestGroupOnDevice(t *testing.T) {
	root, out := os.Getenv("HWNPU_QWENIMAGE_ROOT"), os.Getenv("HWNPU_QWENIMAGE_OUT")
	if root == "" || out == "" {
		t.Skip("HWNPU_QWENIMAGE_ROOT and HWNPU_QWENIMAGE_OUT not set")
	}
	ctx := context.Background()
	if infos, err := (coreml.Provider{}).Enumerate(ctx); err != nil || len(infos) == 0 {
		t.Skip("no Neural Engine")
	}
	size, _ := strconv.Atoi(os.Getenv("HWNPU_QWENIMAGE_SIZE"))
	if size == 0 {
		size = 512
	}
	format := os.Getenv("HWNPU_QWENIMAGE_FORMAT")
	if format == "" {
		format = "lut4"
	}
	count, _ := strconv.Atoi(os.Getenv("HWNPU_QWENIMAGE_GROUP"))
	if count == 0 {
		count = 4
	}
	c, err := ReadDit(root, size, size)
	if err != nil {
		t.Fatal(err)
	}
	w, err := OpenWeights(root, "")
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	kind := os.Getenv("HWNPU_QWENIMAGE_PART") // blocks (default) or text
	qchunk := QChunk(c, 0)
	if q, err := strconv.Atoi(os.Getenv("HWNPU_QWENIMAGE_QCHUNK")); err == nil {
		qchunk = q
	}
	p := Blocks(c, 4, count, qchunk, false)
	if kind == "text" {
		p = TextBlocks(c, 4, count)
	}
	path := filepath.Join(out, p.Name+"-"+format+"-"+strconv.Itoa(size)+"-q"+strconv.Itoa(qchunk)+".mlpackage")
	if _, err := os.Stat(path); err != nil {
		t0 := time.Now()
		if _, err := llm.WritePart(path, p, w, llm.Options{Backend: llm.BackendCoreML, Format: format}, nil); err != nil {
			t.Fatal(err)
		}
		t.Logf("written in %.1f s", time.Since(t0).Seconds())
	}
	cache, _ := os.UserCacheDir()
	dev, err := hwnpu.Open(ctx, hwnpu.WithBackend("coreml"), hwnpu.WithPlacement(hwnpu.PlacementPrefer), hwnpu.WithCacheDir(filepath.Join(cache, "hwnpu")))
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	for round := 0; round < 2; round++ {
		t0 := time.Now()
		prog, err := runner.Open(ctx, dev, path)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("load %d: %.1f s, on the NPU %v", round, time.Since(t0).Seconds(), prog.OnNPU())
		in := map[string][]float32{}
		for name, shape := range prog.Inputs() {
			n := 1
			for _, d := range shape {
				n *= d
			}
			v := make([]float32, n)
			for i := range v {
				v[i] = float32((i*7919)%201-100) / 400
			}
			in[name] = v
		}
		for i := 0; i < 3; i++ {
			t1 := time.Now()
			if _, err := prog.Run(ctx, in); err != nil {
				t.Fatal(err)
			}
			t.Logf("  call %d: %.0f ms", i, float64(time.Since(t1).Microseconds())/1000)
		}
		prog.Close()
	}
}
