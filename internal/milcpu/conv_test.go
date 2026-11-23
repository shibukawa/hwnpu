package milcpu

import (
	"math/rand/v2"
	"testing"
)

// naiveConv2d is the definition: out[b][oc][y][x] = bias[oc] + the sum over the group's input
// channels and the taps of w times x at (y sh + ky dh - pt, x sw + kx dw - pl), zero outside.
func naiveConv2d(x, w, bias *value, sh, sw, dh, dw, pt, pb, pl, pr, groups int) []float32 {
	n, cin, ih, iw := x.shape[0], x.shape[1], x.shape[2], x.shape[3]
	cout, cg, kh, kw := w.shape[0], w.shape[1], w.shape[2], w.shape[3]
	oh := (ih+pt+pb-dh*(kh-1)-1)/sh + 1
	ow := (iw+pl+pr-dw*(kw-1)-1)/sw + 1
	out := make([]float32, n*cout*oh*ow)
	per := cout / groups
	for b := range n {
		for oc := range cout {
			g := oc / per
			for y := range oh {
				for xx := range ow {
					var s float64
					if bias != nil {
						s = float64(bias.f[oc])
					}
					for ic := range cg {
						for ky := range kh {
							for kx := range kw {
								sy, sx := y*sh+ky*dh-pt, xx*sw+kx*dw-pl
								if sy < 0 || sy >= ih || sx < 0 || sx >= iw {
									continue
								}
								s += float64(w.f[((oc*cg+ic)*kh+ky)*kw+kx]) * float64(x.f[((b*cin+g*cg+ic)*ih+sy)*iw+sx])
							}
						}
					}
					out[((b*cout+oc)*oh+y)*ow+xx] = float32(s)
				}
			}
		}
	}
	return out
}

// conv2d's paths (depthwise 3 x 3, direct, im2col on sgemm, the depthwise loop) against the
// definition, with the pool's buffers coming back dirty.
func TestConv2dPaths(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	rnd := func(shape ...int) *value {
		v := &value{shape: shape, f: make([]float32, numel(shape))}
		for i := range v.f {
			v.f[i] = rng.Float32()*2 - 1
		}
		return v
	}
	var bp pool
	type tc struct{ n, cin, ih, iw, cout, groups, kh, kw, sh, sw, dh, dw, pt, pb, pl, pr int }
	cases := []tc{
		{1, 3, 33, 41, 16, 1, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1}, // YuNet's first: direct, stride 2
		{1, 8, 17, 24, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1},  // depthwise 3 x 3
		{2, 6, 9, 13, 6, 6, 3, 3, 1, 1, 1, 1, 0, 2, 2, 0},   // depthwise 3 x 3, uneven pads
		{1, 4, 12, 15, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1},  // depthwise stride 2: direct
		{1, 4, 12, 15, 4, 4, 5, 5, 1, 1, 2, 2, 4, 4, 4, 4},  // dilated 5 x 5: direct
		{1, 4, 10, 11, 6, 2, 3, 2, 1, 3, 1, 1, 1, 0, 0, 1},  // grouped, mixed: direct
		{1, 16, 10, 12, 8, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1}, // 144 taps: im2col
		{1, 32, 8, 20, 16, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0}, // 1 x 1
		{1, 80, 6, 7, 80, 80, 9, 9, 1, 1, 1, 1, 4, 4, 4, 4}, // depthwise of 81 taps: the loop
		{1, 3, 7, 300, 5, 1, 1, 7, 1, 3, 1, 1, 0, 0, 3, 3},  // a 1 D shape: direct
	}
	for i, c := range cases {
		for _, withBias := range []bool{true, false} {
			for _, relu := range []bool{false, true} {
				x := rnd(c.n, c.cin, c.ih, c.iw)
				w := rnd(c.cout, c.cin/c.groups, c.kh, c.kw)
				var b *value
				if withBias {
					b = rnd(c.cout)
				}
				want := naiveConv2d(x, w, b, c.sh, c.sw, c.dh, c.dw, c.pt, c.pb, c.pl, c.pr, c.groups)
				if relu {
					for j := range want {
						want[j] = max(want[j], 0)
					}
				}
				for range 2 {
					got, err := conv2d(&bp, x, w, b, []int32{int32(c.sh), int32(c.sw)}, []int32{int32(c.dh), int32(c.dw)}, "custom",
						[]int32{int32(c.pt), int32(c.pb), int32(c.pl), int32(c.pr)}, c.groups, relu, false)
					if err != nil {
						t.Fatal(err)
					}
					if len(got.f) != len(want) {
						t.Fatalf("case %d: %d values, want %d", i, len(got.f), len(want))
					}
					for j := range want {
						if d := got.f[j] - want[j]; d > 1e-4 || d < -1e-4 {
							t.Fatalf("case %d bias %v relu %v [%d]: %v, want %v", i, withBias, relu, j, got.f[j], want[j])
						}
					}
					for j := range got.f {
						got.f[j] = 1e9 // dirty for the next run
					}
					bp.drop(got.f)
				}
			}
		}
	}
}
