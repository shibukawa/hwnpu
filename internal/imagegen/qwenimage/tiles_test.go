package qwenimage

import (
	"context"
	"math/rand/v2"
	"testing"

	"github.com/shibukawa/hwnpu/internal/imagegen/qwenimage/mil"
)

// stencil is a program of the shape of a VAE stage: reach 3 x 3 sums (zero padded) of its input,
// mixed across the channels into outC, then twice larger (nearest) when up.
type stencil struct {
	c, h, w, outC, reach int
	up                   bool
}

func (s stencil) Inputs() map[string][]int    { return map[string][]int{"x": {1, s.c, s.h, s.w}} }
func (s stencil) Output() []int               { return nil }
func (s stencil) Metadata() map[string]string { return nil }
func (s stencil) OnNPU() bool                 { return true }
func (s stencil) Close() error                { return nil }

func (s stencil) Run(_ context.Context, in map[string][]float32) ([]float32, error) {
	x := append([]float32(nil), in["x"]...)
	h, w := len(x)/s.c/s.w, s.w
	for k := 0; k < s.reach; k++ {
		y := make([]float32, len(x))
		for ch := 0; ch < s.c; ch++ {
			for i := 0; i < h; i++ {
				for j := 0; j < w; j++ {
					var sum float32
					for di := -1; di <= 1; di++ {
						for dj := -1; dj <= 1; dj++ {
							if ii, jj := i+di, j+dj; ii >= 0 && ii < h && jj >= 0 && jj < w {
								sum += x[(ch*h+ii)*w+jj]
							}
						}
					}
					y[(ch*h+i)*w+j] = sum/9 + 0.25 // a bias: zero padding inside the input would show
				}
			}
		}
		x = y
	}
	sc := 1
	if s.up {
		sc = 2
	}
	out := make([]float32, s.outC*sc*h*sc*w)
	for o := 0; o < s.outC; o++ {
		for i := 0; i < sc*h; i++ {
			for j := 0; j < sc*w; j++ {
				var v float32
				for ch := 0; ch < s.c; ch++ {
					v += float32(o+ch+1) * x[(ch*h+i/sc)*w+j/sc]
				}
				out[(o*sc*h+i)*sc*w+j] = v
			}
		}
	}
	return out, nil
}

// TestRunTiled checks that a stage run on tiles (runTiled) gives what it gives on the whole input.
func TestRunTiled(t *testing.T) {
	ctx := context.Background()
	for _, tc := range []struct {
		h, w       int
		rows, cols mil.Tiling
		up         bool
	}{
		{64, 64, mil.Tiling{N: 2, Size: 64, Tile: 32, Halo: 7}, mil.Tiling{N: 2, Size: 64, Tile: 32, Halo: 7}, true},
		{128, 96, mil.Tiling{N: 4, Size: 128, Tile: 32, Halo: 14}, mil.Tiling{N: 3, Size: 96, Tile: 32, Halo: 14}, true},
		{50, 40, mil.Tiling{N: 3, Size: 50, Tile: 17, Halo: 14}, mil.Tiling{N: 1, Size: 40}, false},
	} {
		const c, outC, reach = 3, 2, 7
		x := make([]float32, c*tc.h*tc.w)
		for i := range x {
			x[i] = rand.Float32()*2 - 1
		}
		whole, _ := stencil{c, tc.h, tc.w, outC, reach, tc.up}.Run(ctx, map[string][]float32{"x": x})
		sc := 1
		if tc.up {
			sc = 2
		}
		tl := VaeTiling{C: c, H: tc.h, W: tc.w, OutC: outC, Scale: sc, Rows: tc.rows, Cols: tc.cols}
		got, err := runTiled(ctx, stencil{c, tc.rows.Window(), tc.cols.Window(), outC, reach, tc.up}, x, tl)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != len(whole) {
			t.Fatalf("%dx%d: %d values, want %d", tc.h, tc.w, len(got), len(whole))
		}
		for i := range got {
			if d := got[i] - whole[i]; d > 1e-4 || d < -1e-4 {
				t.Fatalf("%dx%d: value %d is %g, want %g", tc.h, tc.w, i, got[i], whole[i])
			}
		}
	}
}
