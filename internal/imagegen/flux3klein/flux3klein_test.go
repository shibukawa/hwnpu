package flux3klein

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/imagegen/flux3klein/mil"
)

// Sigmas are FlowMatchEulerDiscreteScheduler's with a static shift of 6.93 over linspace(1, 1/16,
// 16), then 0: the reference's timesteps 1000, 990.47, ..., 316.01.
func TestSigmas(t *testing.T) {
	s := Sigmas()
	if len(s) != Steps+1 || s[0] != 1 || s[Steps] != 0 {
		t.Fatalf("sigmas %v", s)
	}
	for i, want := range map[int]float64{1: 0.9904717, 8: 0.8738966, 15: 0.3160055} {
		if math.Abs(s[i]-want) > 1e-6 {
			t.Errorf("sigma %d = %.7f, want %.7f", i, s[i], want)
		}
	}
}

// The bf16 rounding of the timestep the pipeline hands the embedders.
func TestRoundBF16(t *testing.T) {
	if r := roundBF16(0.9904717); r != 0.9921875 { // bf16's step is 1/256 below 1
		t.Errorf("bf16 of 0.9904717 = %v", r)
	}
	if r := roundBF16(roundBF16(0.9904717) * 1000); r != 992 { // 4 between 512 and 1024
		t.Errorf("bf16 of 992.1875 = %v", r)
	}
}

func readF32(t *testing.T, path string) []float32 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("%v", err)
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

// The host's modulation rows are diffusers' (Timesteps, TimestepEmbedding, Flux3Modulation,
// norm_out's linear in float32, the timestep rounded to bf16 as the pipeline hands it): with an
// export in HWNPU_TEST_FLUX3_EXPORT and the reference's rows in HWNPU_TEST_FLUX3_MODS_REF (a directory of mods_0.f32, the
// step 1 -> sigma 1, and mods_15.f32, the last), made from the checkpoint by diffusers' modules.
func TestModulation(t *testing.T) {
	dir, golden := os.Getenv("HWNPU_TEST_FLUX3_EXPORT"), os.Getenv("HWNPU_TEST_FLUX3_MODS_REF")
	if dir == "" || golden == "" {
		t.Skip("HWNPU_TEST_FLUX3_EXPORT and HWNPU_TEST_FLUX3_MODS_REF name an export and the reference's modulation rows")
	}
	te, err := readTimeFile(filepath.Join(dir, TimeFile), 3072, 256)
	if err != nil {
		t.Fatal(err)
	}
	s := Sigmas()
	for _, c := range []struct {
		file       string
		sigma, nxt float64
	}{{"mods_0.f32", s[0], s[1]}, {"mods_15.f32", s[15], s[16]}} {
		want := readF32(t, filepath.Join(golden, c.file))
		got := te.modulation(c.sigma, c.nxt)
		if len(got) != len(want) || len(got) != mil.ModRows*3072 {
			t.Fatalf("%s: %d rows' values, want %d", c.file, len(got), len(want))
		}
		var num, den float64
		for i := range want {
			d := float64(got[i] - want[i])
			num, den = num+d*d, den+float64(want[i])*float64(want[i])
		}
		if rel := math.Sqrt(num / den); rel > 1e-4 {
			t.Errorf("%s: the modulation rows are %.3g off diffusers' (relative RMS)", c.file, rel)
		} else {
			t.Logf("%s: relative RMS %.2g", c.file, rel)
		}
	}
}
