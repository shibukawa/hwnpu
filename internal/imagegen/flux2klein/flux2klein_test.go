package flux2klein

import (
	"math"
	"testing"
)

func testPipeline(size int) *Pipeline {
	p := &Pipeline{C: Config{Width: size, Height: size, TextRows: 512, Hidden: 3072, HeadDim: 128, Latent: 128, Text: 2560,
		Axes: []int{32, 32, 32, 32}, Theta: 2000, Doubles: 5, Singles: 20, DoubleGroup: 1, SingleGroup: 2}}
	p.Model.Steps = 4
	p.bnMean, p.bnStd = make([]float32, 128), make([]float32, 128)
	for i := range p.bnMean {
		p.bnMean[i], p.bnStd[i] = float32(i)/100-0.5, 1+float32(i%7)/10
	}
	return p
}

// The pipeline's sigmas of a 512 x 512 picture in 4 steps (tools/ref/ref_flux2klein.py prints them).
func TestSigmas(t *testing.T) {
	got := testPipeline(512).Sigmas(0)
	want := []float64{1, 0.95808536, 0.8839819, 0.7174966, 0}
	if len(got) != len(want) {
		t.Fatalf("%d sigmas, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-5 {
			t.Errorf("sigma %d: %.6f, want %.6f", i, got[i], want[i])
		}
	}
	// past 4300 rows the shift no longer depends on the steps
	if a, b := EmpiricalMu(8192, 4), EmpiricalMu(8192, 50); a != b {
		t.Errorf("mu of 8192 rows: %g at 4 steps, %g at 50", a, b)
	}
}

// Unpack un-normalizes and lays the 2 x 2 patches out: channel c of the 32, at (2 y + dy, 2 x + dx),
// is the row's value c*4 + dy*2 + dx (the pipeline's _unpatchify_latents).
func TestUnpack(t *testing.T) {
	p := testPipeline(64)
	c := p.C
	x := make([]float32, c.Tokens()*c.Latent)
	for i := range x {
		x[i] = float32(i%251) / 50
	}
	z := p.Unpack(x)
	H, W := 2*c.GH(), 2*c.GW()
	if len(z) != 32*H*W {
		t.Fatalf("%d values, want %d", len(z), 32*H*W)
	}
	for _, at := range [][5]int{{0, 0, 0, 0, 0}, {5, 1, 2, 1, 0}, {31, 3, 3, 1, 1}, {17, 2, 0, 0, 1}} {
		ch, py, px, dy, dx := at[0], at[1], at[2], at[3], at[4]
		k := ch*4 + dy*2 + dx
		want := x[(py*c.GW()+px)*c.Latent+k]*p.bnStd[k] + p.bnMean[k]
		if got := z[(ch*H+2*py+dy)*W+2*px+dx]; got != want {
			t.Errorf("channel %d at patch (%d, %d) + (%d, %d): %g, want %g", ch, py, px, dy, dx, got, want)
		}
	}
}

// The rows' positions: the picture's, each reference's on the first axis at 10, 20, ..., then the text's.
func TestPositions(t *testing.T) {
	p := testPipeline(64)
	N := p.C.Tokens()
	pos := p.positions(2, 1)
	if len(pos) != 3*N+p.C.TextRows {
		t.Fatalf("%d rows, want %d", len(pos), 3*N+p.C.TextRows)
	}
	for _, c := range []struct {
		row  int
		want [4]float64
	}{{0, [4]float64{0, 0, 0, 0}}, {N - 1, [4]float64{0, 3, 3, 0}}, {N + 5, [4]float64{10, 1, 1, 0}}, {2 * N, [4]float64{20, 0, 0, 0}},
		{3 * N, [4]float64{0, 0, 0, 0}}, {3*N + 7, [4]float64{0, 0, 0, 7}}} {
		if pos[c.row] != c.want {
			t.Errorf("row %d: %v, want %v", c.row, pos[c.row], c.want)
		}
	}
	// references of half the sides: a quarter of the rows each, their own grid
	half := p.positions(1, 2)
	if len(half) != N+N/4+p.C.TextRows {
		t.Fatalf("%d rows with a half-size reference, want %d", len(half), N+N/4+p.C.TextRows)
	}
	if got, want := half[N+3], ([4]float64{10, 1, 1, 0}); got != want {
		t.Errorf("row %d of a half-size reference: %v, want %v", N+3, got, want)
	}
	if suffix(0, 1) != "" || suffix(1, 1) != "_r1" || suffix(1, 2) != "_r1d2" {
		t.Errorf("suffixes %q %q %q", suffix(0, 1), suffix(1, 1), suffix(1, 2))
	}
	a := p.ropeAngles(pos[:1])
	for i := 0; i < p.C.HeadDim; i += 2 { // position 0: no turn
		if a[i] != 1 || a[i+1] != 0 {
			t.Fatalf("angle %d of position 0: cos %g sin %g", i/2, a[i], a[i+1])
		}
	}
}
