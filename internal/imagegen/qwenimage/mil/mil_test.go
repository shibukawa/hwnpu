package mil

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"
)

// TestRopePermuted checks that blocks.ApplyRope's halves on a head permuted by headPerm, with
// RopeTables, give QwenImage21's complex rotation of the pairs (2i, 2i + 1), permuted the same way:
// so q . k is the reference's.
func TestRopePermuted(t *testing.T) {
	c := Dit{Width: 64, Height: 32, HeadDim: 128, Axes: [3]int{16, 56, 56}, Theta: 10000}
	pos := ImagePositions(c, 21)
	cos, sin := RopeTables(c, pos)
	D, half := c.HeadDim, c.HeadDim/2
	r := rand.New(rand.NewPCG(1, 2))
	perm := headPerm(D)
	for _, row := range []int{0, 3, len(pos) - 1} {
		x := make([]float64, D)
		for i := range x {
			x[i] = r.NormFloat64()
		}
		// the reference: pair i is x[2i] + i x[2i+1], times exp(i angle_i)
		var angles []float64
		for a, d := range c.Axes {
			for k := 0; k < d; k += 2 {
				angles = append(angles, pos[row][a]/math.Pow(c.Theta, float64(k)/float64(d)))
			}
		}
		want := make([]float64, D)
		for i, a := range angles {
			z := complex(x[2*i], x[2*i+1]) * cmplx.Exp(complex(0, a))
			want[2*i], want[2*i+1] = real(z), imag(z)
		}
		// ApplyRope on the permuted head: x cos + rotate_half(x) sin
		p := make([]float64, D)
		for i, src := range perm {
			p[i] = x[src]
		}
		got := make([]float64, D)
		for i := 0; i < D; i++ {
			var rot float64 // rotate_half: (-x2, x1)
			if i < half {
				rot = -p[i+half]
			} else {
				rot = p[i-half]
			}
			got[i] = p[i]*float64(cos[row*D+i]) + rot*float64(sin[row*D+i])
		}
		for i, src := range perm {
			if math.Abs(got[i]-want[src]) > 1e-5 {
				t.Fatalf("row %d, feature %d: %g, want %g", row, i, got[i], want[src])
			}
		}
	}
}

// TestImagePositions checks QwenImage21Rope's image positions: the frame at the text's length,
// the rows and columns centred on zero.
func TestImagePositions(t *testing.T) {
	c := Dit{Width: 3 * VAEScale, Height: 2 * VAEScale}
	pos := ImagePositions(c, 7)
	want := [][3]float64{{7, -1, -2}, {7, -1, -1}, {7, -1, 0}, {7, 0, -2}, {7, 0, -1}, {7, 0, 0}}
	if len(pos) != len(want) {
		t.Fatalf("%d positions, want %d", len(pos), len(want))
	}
	for i := range want {
		if pos[i] != want[i] {
			t.Errorf("position %d: %v, want %v", i, pos[i], want[i])
		}
	}
}

// TestDupSource checks DupUp3D's source channels against its definition: repeat_interleave of the
// channels, the (cout, ft, 2, 2) view, the last frame of the first chunk.
func TestDupSource(t *testing.T) {
	for _, tc := range []struct{ cin, cout, ft int }{{1152, 1152, 2}, {1152, 576, 2}, {576, 288, 1}, {8, 4, 2}} {
		src := DupSource(tc.cin, tc.cout, tc.ft)
		rep := tc.cout * tc.ft * 4 / tc.cin
		// channel k of the repeated tensor is input channel k / rep; the view puts (co, f, i, j)
		// at k = ((co ft + f) 2 + i) 2 + j
		for co := 0; co < tc.cout; co++ {
			for i := 0; i < 2; i++ {
				for j := 0; j < 2; j++ {
					k := ((co*tc.ft+tc.ft-1)*2+i)*2 + j
					if got := src[co*4+i*2+j]; got != k/rep {
						t.Fatalf("%v: (%d, %d, %d) takes %d, want %d", tc, co, i, j, got, k/rep)
					}
				}
			}
		}
	}
	// the decoder's: the first two blocks are nearest copies, the third every other channel
	if s := DupSource(1152, 1152, 2); s[4*5] != 5 || s[4*5+3] != 5 {
		t.Errorf("1152 -> 1152: channel 5 takes %v", s[20:24])
	}
	if s := DupSource(1152, 576, 2); s[4*5] != 11 {
		t.Errorf("1152 -> 576: channel 5 takes %d, want 11", s[20])
	}
}

// TestVaeStages checks that each stage's input is the one before's output, ending at the picture.
func TestVaeStages(t *testing.T) {
	v := Vae{Base: 144, Mult: []int{1, 2, 4, 8, 8}, Res: 2, TDown: []bool{false, true, true, true}, Z: 64, Out: 4}
	st := VaeStages(v, 512, 768)
	C, H, W := v.Z, 768/VAEScale, 512/VAEScale
	for i, s := range st {
		if s.C != C || s.H != H || s.W != W {
			t.Fatalf("stage %d takes (%d, %d, %d), the one before gives (%d, %d, %d)", i, s.C, s.H, s.W, C, H, W)
		}
		if s.Rows.N <= 1 && s.Rows.Window() != H || s.Cols.N <= 1 && s.Cols.Window() != W {
			t.Errorf("stage %d: a window of %d x %d for (%d, %d) untiled", i, s.Rows.Window(), s.Cols.Window(), H, W)
		}
		for _, op := range s.Ops {
			switch op.Kind {
			case "in", "res", "head":
				C = op.Out
			case "up":
				C, H, W = op.Out, 2*H, 2*W
			}
		}
	}
	if C != 4 || H != 768 || W != 512 {
		t.Errorf("the decoder gives (%d, %d, %d)", C, H, W)
	}
	// the up blocks' time factors: temperal_downsample reversed
	var ts []int
	for _, s := range st {
		for _, op := range s.Ops {
			if op.Kind == "up" {
				ts = append(ts, op.T)
			}
		}
	}
	if len(ts) != 4 || ts[0] != 2 || ts[3] != 1 {
		t.Errorf("time factors %v, want [2 2 2 1]", ts)
	}
}

// TestQChunk checks the image attention's chunks: all of 512 x 512's queries, 512 of 1024 x 1024's.
func TestQChunk(t *testing.T) {
	c := Dit{Heads: 32}
	for _, tc := range []struct{ size, want int }{{512, 1024}, {1024, 512}} {
		c.Width, c.Height = tc.size, tc.size
		if got := QChunk(c, 0); got != tc.want {
			t.Errorf("%d: %d queries a chunk, want %d", tc.size, got, tc.want)
		}
	}
}

// TestVaeTiles checks which stages are tiled (Qwen-Image-2.1's VAE: 1152 channels at 64 x 64 and
// more) and that the tiles cover the input once, each far enough from its window's inner edges.
func TestVaeTiles(t *testing.T) {
	// Qwen-Image-2.1's: up blocks of 1152 to 1152 channels (stages 1 and 2), then 1152 to 576
	v := Vae{Base: 144, Mult: []int{1, 2, 4, 8, 8}, Res: 2, TDown: []bool{false, true, true, true}, Z: 64, Out: 4}
	if s := VaeStages(v, 512, 512); len(s) != 7 || s[2].C != 1152 || s[2].H != 64 || s[3].C != 1152 || s[3].width() != 576 {
		t.Fatalf("stages %d, the third (%d, %d, %d)", len(s), s[2].C, s[2].H, s[2].W)
	}
	want := map[[2]int][]int{ // size, stage: tiles of an axis (1 not tiled)
		{512, 1}: {1}, {512, 2}: {2}, {512, 3}: {1},
		{1024, 1}: {2}, {1024, 2}: {4}, {1024, 3}: {1},
		{832, 1}: {2}, // 52 x 52: two of 26 and a halo of 7
		{832, 2}: {4}, // 104 x 104: four of 26 and a halo of 14
	}
	for k, n := range want {
		s := VaeStages(v, k[0], k[0])[k[1]]
		if s.Rows.N != n[0] && !(n[0] == 1 && s.Rows.N == 0) || s.Rows != s.Cols {
			t.Errorf("%d^2 stage %d: tiles %+v x %+v, want %d an axis", k[0], k[1], s.Rows, s.Cols, n[0])
		}
		if s.Rows.N > 1 && (s.Rows.Window() > wideWindow || s.width() <= wideChannels) {
			t.Errorf("%d^2 stage %d: a window of %d for %d channels", k[0], k[1], s.Rows.Window(), s.C)
		}
	}
	for _, tl := range []Tiling{{2, 64, 32, 7}, {4, 128, 32, 14}, {3, 50, 17, 14}, {2, 52, 26, 7}} {
		r := tl.Halo
		if tl.N > 2 {
			r = tl.Halo / 2
		}
		next := 0
		for i, w := range tl.Windows() {
			if w.Start < 0 || w.Start+tl.Window() > tl.Size || w.Start+w.Off != next || w.Len <= 0 {
				t.Fatalf("%+v: window %d %+v, the tile should start at %d", tl, i, w, next)
			}
			if w.Start > 0 && w.Off < r || w.Start+tl.Window() < tl.Size && tl.Window()-w.Off-w.Len < r {
				t.Errorf("%+v: window %d %+v: a tile nearer than %d to an inner edge", tl, i, w, r)
			}
			next += w.Len
		}
		if next != tl.Size {
			t.Errorf("%+v: the tiles cover %d", tl, next)
		}
	}
}
