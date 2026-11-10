package sfx

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
)

// refDir is the model's directory (HWNPU_SFX_DIR) and the reference's prefix (HWNPU_SFX_REF,
// tools/ref/ref_moss_sfx.py's OUT).
func refDir(t *testing.T) (dir, ref string) {
	dir, ref = os.Getenv("HWNPU_SFX_DIR"), os.Getenv("HWNPU_SFX_REF")
	if dir == "" || ref == "" {
		t.Skip("HWNPU_SFX_DIR and HWNPU_SFX_REF are not set")
	}
	return dir, ref
}

func readF32(t *testing.T, path string) []float32 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v
}

// relRMS is |a - b| / |b|.
func relRMS(a, b []float32) float64 {
	var d, n float64
	for i := range b {
		e := float64(a[i]) - float64(b[i])
		d += e * e
		n += float64(b[i]) * float64(b[i])
	}
	return math.Sqrt(d / n)
}

func vaeWeights(t *testing.T, dir string) *Weights {
	w, err := OpenWeights([]string{filepath.Join(dir, "vae", "vae_128d_48k.pth")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return w
}

// writeStages writes the decoder's stages into a temporary directory and opens them on dev
// (nil: the CPU interpreter).
func openStages(t *testing.T, w *Weights, dev any) []Program {
	t.Helper()
	tmp := t.TempDir()
	var ps []Program
	for _, s := range DACStages() {
		path := filepath.Join(tmp, s.Part().Name+".mlpackage")
		if _, err := llm.WritePart(path, s.Part(), w, llm.Options{Backend: llm.BackendCoreML}, nil); err != nil {
			t.Fatal(err)
		}
		p, err := runner.Open(context.Background(), nil, path)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { p.Close() })
		ps = append(ps, p)
	}
	return ps
}

// The decoder's stages on the CPU interpreter give the reference's samples from the latents of
// its clip (the encoder's of 4 seconds of a WAV).
func TestDecodeLatentsCPU(t *testing.T) {
	dir, ref := refDir(t)
	w := vaeWeights(t, dir)
	defer w.Close()
	z := readF32(t, ref+".latents.bin")
	frames := len(z) / LatentDim
	want := readF32(t, ref+".samples.bin")
	ps := openStages(t, w, nil)
	got, err := DecodeLatents(context.Background(), ps, z, frames, frames, len(want))
	if err != nil {
		t.Fatal(err)
	}
	e := relRMS(got, want)
	t.Logf("%d frames: samples %.3g off the reference (relative RMS)", frames, e)
	if !(e <= 0.02) {
		t.Fatalf("samples %.3g off the reference", e)
	}
}

// On the Neural Engine (HWNPU_SFX_DEVICE=1) the decoder's stages are all on the NPU and give the
// reference's samples.
func TestDecodeLatentsOnDevice(t *testing.T) {
	dir, ref := refDir(t)
	if os.Getenv("HWNPU_SFX_DEVICE") == "" {
		t.Skip("HWNPU_SFX_DEVICE is not set")
	}
	ctx := context.Background()
	dev, _, err := llm.OpenDevice(ctx, "", "")
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	w := vaeWeights(t, dir)
	defer w.Close()
	tmp := t.TempDir()
	var ps []Program
	for _, s := range DACStages() {
		path := filepath.Join(tmp, s.Part().Name+".mlpackage")
		if _, err := llm.WritePart(path, s.Part(), w, llm.Options{Backend: llm.BackendCoreML}, nil); err != nil {
			t.Fatal(err)
		}
		t0 := time.Now()
		p, err := runner.Open(ctx, dev, path)
		if err != nil {
			t.Fatal(err)
		}
		defer p.Close()
		t.Logf("%s: loaded in %s, on the NPU: %v", s.Part().Name, time.Since(t0).Round(time.Millisecond), p.OnNPU())
		ps = append(ps, p)
	}
	z := readF32(t, ref+".latents.bin")
	frames := len(z) / LatentDim
	want := readF32(t, ref+".samples.bin")
	t0 := time.Now()
	got, err := DecodeLatents(ctx, ps, z, frames, frames, len(want))
	if err != nil {
		t.Fatal(err)
	}
	e := relRMS(got, want)
	t.Logf("%d frames in %s: samples %.3g off the reference (relative RMS)", frames, time.Since(t0).Round(time.Millisecond), e)
	if !(e <= 0.05) {
		t.Fatalf("samples %.3g off the reference", e)
	}
}

// TestWriteStages writes the decoder's stages into HWNPU_SFX_WRITE (for hwnpu-dev info -model).
func TestWriteStages(t *testing.T) {
	out := os.Getenv("HWNPU_SFX_WRITE")
	dir := os.Getenv("HWNPU_SFX_DIR")
	if out == "" || dir == "" {
		t.Skip("HWNPU_SFX_WRITE and HWNPU_SFX_DIR are not set")
	}
	w := vaeWeights(t, dir)
	defer w.Close()
	for _, s := range DACStages() {
		path := filepath.Join(out, s.Part().Name+".mlpackage")
		os.RemoveAll(path)
		if _, err := llm.WritePart(path, s.Part(), w, llm.Options{Backend: llm.BackendCoreML}, nil); err != nil {
			t.Fatal(err)
		}
	}
}

// TestStagesOnDevice runs each stage on the Neural Engine from the reference's input to it
// (HWNPU_SFX_STAGES: a directory of the stages, TestWriteStages'), against the reference's output.
func TestStagesOnDevice(t *testing.T) {
	_, ref := refDir(t)
	dir := os.Getenv("HWNPU_SFX_STAGES")
	if dir == "" {
		t.Skip("HWNPU_SFX_STAGES is not set")
	}
	ctx := context.Background()
	var dev *hwnpu.Device
	if os.Getenv("HWNPU_SFX_DEVICE") != "" {
		d, _, err := llm.OpenDevice(ctx, "", "")
		if err != nil {
			t.Fatal(err)
		}
		defer d.Close()
		dev = d
	}
	x := readF32(t, ref+".latents.bin")
	frames := len(x) / LatentDim
	clip := frames
	for i, s := range DACStages() {
		want := readF32(t, ref+fmt.Sprintf(".stage%d.bin", i))
		p, err := runner.Open(ctx, dev, filepath.Join(dir, s.Part().Name+".mlpackage"))
		if err != nil {
			t.Fatal(err)
		}
		n := clip * s.Rate
		got, err := runStage(ctx, p, s, x, clip, clip, n)
		p.Close()
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%s: %.3g off the reference", s.Part().Name, relRMS(got, want))
		x, clip = want, n
	}
}
