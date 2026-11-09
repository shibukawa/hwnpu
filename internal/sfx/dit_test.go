package sfx

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
)

func ditWeights(t *testing.T, dir string) *Weights {
	t.Helper()
	w, err := OpenWeights(nil, map[string][]string{"dit.": {filepath.Join(dir, "transformer", "diffusion_pytorch_model.safetensors")}})
	if err != nil {
		t.Fatal(err)
	}
	return w
}

// rows is x (c x n, channel major) as n rows of c.
func rows(x []float32, c, n int) []float32 {
	y := make([]float32, len(x))
	for i := range c {
		for j := range n {
			y[j*c+i] = x[i*n+j]
		}
	}
	return y
}

// On the Neural Engine (HWNPU_SFX_DEVICE; the DiT's format in HWNPU_SFX_DIT_FORMAT, fp16 by
// default; the programs kept in HWNPU_SFX_WRITE) the velocities of the reference's first steps
// (tools/ref/ref_moss_sfx.py run --dump, HWNPU_SFX_REF; its context in HWNPU_SFX_CONTEXT) are the
// reference's.
func TestVelocityOnDevice(t *testing.T) {
	dir, ref := refDir(t)
	ctxPath := os.Getenv("HWNPU_SFX_CONTEXT")
	if os.Getenv("HWNPU_SFX_DEVICE") == "" || ctxPath == "" {
		t.Skip("HWNPU_SFX_DEVICE and HWNPU_SFX_CONTEXT are not set")
	}
	format := os.Getenv("HWNPU_SFX_DIT_FORMAT")
	if format == "" {
		format = "fp16"
	}
	group := 5
	ctx := context.Background()
	dev, _, err := llm.OpenDevice(ctx, "", "")
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	w := ditWeights(t, dir)
	defer w.Close()
	out := os.Getenv("HWNPU_SFX_WRITE")
	if out == "" {
		out = t.TempDir()
	}
	open := func(p *Part, f string) Program {
		path := filepath.Join(out, fmt.Sprintf("%s-%s.mlpackage", p.Name, f))
		if _, err := os.Stat(path); err != nil {
			t0 := time.Now()
			if _, err := llm.WritePart(path, p, w, llm.Options{Backend: llm.BackendCoreML, Format: f}, nil); err != nil {
				t.Fatal(err)
			}
			t.Logf("%s: written in %s", filepath.Base(path), time.Since(t0).Round(time.Millisecond))
		}
		t0 := time.Now()
		prog, err := runner.Open(ctx, dev, path)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { prog.Close() })
		t.Logf("%s: loaded in %s, on the NPU: %v", filepath.Base(path), time.Since(t0).Round(time.Millisecond), prog.OnNPU())
		return prog
	}
	timing, err := LoadTiming(w, "dit.")
	if err != nil {
		t.Fatal(err)
	}
	d := &DiT{Timing: timing, Ctx: open(Context(), "fp16")}
	for first := 0; first < DitLayers; first += group {
		d.Groups = append(d.Groups, open(Blocks(first, group), format))
		d.Sizes = append(d.Sizes, group)
	}
	cond, err := d.Context(ctx, readF32(t, ctxPath))
	if err != nil {
		t.Fatal(err)
	}
	cond = append([]float32(nil), cond...)
	uncond, err := d.Context(ctx, make([]float32, TextRows*TextDim))
	if err != nil {
		t.Fatal(err)
	}
	uncond = append([]float32(nil), uncond...)
	sig := Sigmas(4, 5)
	for i := range 2 {
		x := rows(readF32(t, fmt.Sprintf("%s.step%d.x.bin", ref, i)), LatentDim, Frames)
		for _, c := range []struct {
			name string
			ctx  []float32
		}{{"pos", cond}, {"neg", uncond}} {
			t0 := time.Now()
			v, err := d.Velocity(ctx, x, sig[i]*1000, c.ctx)
			if err != nil {
				t.Fatal(err)
			}
			want := rows(readF32(t, fmt.Sprintf("%s.step%d.%s.bin", ref, i, c.name)), LatentDim, Frames)
			t.Logf("step %d %s: in %s, %.3g off the reference", i, c.name, time.Since(t0).Round(time.Millisecond), relRMS(v, want))
		}
	}
}

// TestGroupsOnDevice runs the context and each group of blocks from the reference's input to it
// (tools/ref/ref_moss_sfx.py hidden, HWNPU_SFX_HIDDEN; the programs in HWNPU_SFX_WRITE of
// TestVelocityOnDevice) against the reference's output.
func TestGroupsOnDevice(t *testing.T) {
	dir, _ := refDir(t)
	hid := os.Getenv("HWNPU_SFX_HIDDEN")
	out := os.Getenv("HWNPU_SFX_WRITE")
	if hid == "" || out == "" || os.Getenv("HWNPU_SFX_DEVICE") == "" {
		t.Skip("HWNPU_SFX_HIDDEN, HWNPU_SFX_WRITE and HWNPU_SFX_DEVICE are not set")
	}
	format := os.Getenv("HWNPU_SFX_DIT_FORMAT")
	if format == "" {
		format = "fp16"
	}
	ctx := context.Background()
	var d *hwnpu.Device
	if os.Getenv("HWNPU_SFX_CPU") == "" {
		dv, _, err := llm.OpenDevice(ctx, "", "")
		if err != nil {
			t.Fatal(err)
		}
		defer dv.Close()
		d = dv
	}
	w := ditWeights(t, dir)
	defer w.Close()
	timing, err := LoadTiming(w, "dit.")
	if err != nil {
		t.Fatal(err)
	}
	cp, err := runner.Open(ctx, d, filepath.Join(out, "dit_context-fp16.mlpackage"))
	if err != nil {
		t.Fatal(err)
	}
	defer cp.Close()
	c, err := cp.Run(ctx, map[string][]float32{"ctx": readF32(t, os.Getenv("HWNPU_SFX_CONTEXT"))})
	if err != nil {
		t.Fatal(err)
	}
	c = append([]float32(nil), c...)
	t.Logf("context: %.3g off the reference", relRMS(c, readF32(t, hid+".ctx.bin")))
	c = readF32(t, hid+".ctx.bin")
	temb, tmod := timing.Embed(1000)
	scale := func(v []float32, s float32) []float32 {
		out := make([]float32, len(v))
		for i := range v {
			out[i] = v[i] * s
		}
		return out
	}
	group := 5
	for first := 0; first < DitLayers; first += group {
		if only := os.Getenv("HWNPU_SFX_ONLY"); only != "" && only != fmt.Sprint(first) {
			continue
		}
		p, err := runner.Open(ctx, d, filepath.Join(out, fmt.Sprintf("dit_blocks_%d-%s.mlpackage", first, format)))
		if err != nil {
			t.Fatal(err)
		}
		var x []float32
		if first == 0 {
			x = rows(readF32(t, os.Getenv("HWNPU_SFX_X")), LatentDim, Frames)
		} else {
			x = scale(readF32(t, fmt.Sprintf("%s.h%d.bin", hid, first)), 1.0/DitScale)
		}
		y, err := p.Run(ctx, map[string][]float32{"x": x, "ctx": c, "mod": timing.Mod(first, group, temb, tmod)})
		p.Close()
		if err != nil {
			t.Fatal(err)
		}
		var want []float32
		if first+group == DitLayers {
			want = rows(readF32(t, hid+".v.bin"), LatentDim, Frames)
		} else {
			want = scale(readF32(t, fmt.Sprintf("%s.h%d.bin", hid, first+group)), 1.0/DitScale)
		}
		t.Logf("blocks %d to %d: %.3g off the reference", first, first+group-1, relRMS(y, want))
	}
}
