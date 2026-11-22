package sfx

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
)

// TestAMDDAC compares the decoder's stages on AMD's NPU (HWNPU_TEST_SFX_AMD_EXPORT: an export with dac-N/,
// tools/amd/iron_moss_dac.py) with the Core ML stages on the CPU interpreter (HWNPU_TEST_SFX_CPU_DAC_DIR:
// a directory of hwnpu-dac-N-rREV-fp16.mlpackage) over standard normal latents (or
// HWNPU_TEST_SFX_LATENTS: 1500 rows of 128 float32).
func TestAMDDAC(t *testing.T) {
	amdDir, cpuDir := os.Getenv("HWNPU_TEST_SFX_AMD_EXPORT"), os.Getenv("HWNPU_TEST_SFX_CPU_DAC_DIR")
	if amdDir == "" || cpuDir == "" {
		t.Skip("HWNPU_TEST_SFX_AMD_EXPORT and HWNPU_TEST_SFX_CPU_DAC_DIR are not set")
	}
	ctx := context.Background()
	x := Noise(7)
	if f := os.Getenv("HWNPU_TEST_SFX_LATENTS"); f != "" {
		x = readF32(t, f)
	}
	seconds := 5.0
	samples := int(seconds) * SampleRate

	dev, err := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
	if err != nil {
		t.Skip(err)
	}
	defer dev.Close()
	var dirs []string
	for i := range DACStages() {
		dirs = append(dirs, filepath.Join(amdDir, fmt.Sprintf("dac-%d", i)))
	}
	d, err := openAMDDAC(ctx, dev, dirs)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	t0 := time.Now()
	got, err := d.Decode(ctx, x, Frames, Frames, samples)
	if err != nil {
		t.Fatal(err)
	}
	took := time.Since(t0)
	t0 = time.Now()
	got, err = d.Decode(ctx, x, Frames, Frames, samples)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("AMD's NPU: %.0f s of sound in %s (then %s)", seconds, took.Round(time.Millisecond), time.Since(t0).Round(time.Millisecond))

	for _, s := range d.st {
		t.Logf("%s: %d calls, %s", s.name, s.calls, s.took.Round(time.Millisecond))
	}
	var stages []Program
	for _, st := range DACStages() {
		p, err := runner.Open(ctx, nil, filepath.Join(cpuDir, fmt.Sprintf("hwnpu-dac-%d-r%d-fp16.mlpackage", st.Index, DACRevision)))
		if err != nil {
			t.Fatal(err)
		}
		defer p.Close()
		stages = append(stages, p)
	}
	t0 = time.Now()
	want, err := DecodeLatents(ctx, stages, channels(x, Frames, LatentDim), Frames, Frames, samples)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("the CPU interpreter: in %s", time.Since(t0).Round(time.Millisecond))
	var e, n float64
	for i := range want {
		d := float64(got[i]) - float64(want[i])
		e += d * d
		n += float64(want[i]) * float64(want[i])
	}
	snr := 10 * math.Log10(n/e)
	t.Logf("relative RMS %.4g, SNR %.1f dB, cosine %.5f", relRMS(got, want), snr, cosine(got, want))
	if snr < 20 {
		t.Errorf("SNR %.1f dB", snr)
	}
	if out := os.Getenv("HWNPU_TEST_SFX_WAV_OUT"); out != "" {
		writeTestWAV(t, out+"-amd.wav", got)
		writeTestWAV(t, out+"-cpu.wav", want)
	}
}

func writeTestWAV(t *testing.T, path string, y []float32) {
	b := make([]byte, 44+2*len(y))
	copy(b, "RIFF")
	le := func(o int, v uint32, n int) {
		for k := range n {
			b[o+k] = byte(v >> (8 * k))
		}
	}
	le(4, uint32(36+2*len(y)), 4)
	copy(b[8:], "WAVEfmt ")
	le(16, 16, 4)
	le(20, 1, 2)
	le(22, 1, 2)
	le(24, SampleRate, 4)
	le(28, 2*SampleRate, 4)
	le(32, 2, 2)
	le(34, 16, 2)
	copy(b[36:], "data")
	le(40, uint32(2*len(y)), 4)
	for i, v := range y {
		s := int16(max(-32767, min(32767, math.Round(float64(v)*32767))))
		le(44+2*i, uint32(uint16(s)), 2)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
}
