package mil

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/safetensors"
)

// tiny is a FLUX.3 klein DiT of the checkpoint's layout, small: a 2 x 3 grid of patches (rows and
// columns differ, so that a swapped axis shows), 3 text rows, one mode block of each stream and two
// single blocks.
func tiny() Dit {
	return Dit{Width: 48, Height: 32, Hidden: 32, Heads: 2, HeadDim: 16, MLP: 48, Latent: 8, Context: 24, Modes: 1, Singles: 2,
		Axes: []int{4, 4, 4, 4}, Theta: 10000, Eps: 1e-6, TimeIn: 16, TextRows: 3}
}

// writeCheckpoint writes ws (name: (shape, values)) as root/transformer/diffusion_pytorch_model.safetensors.
func writeCheckpoint(t *testing.T, root string, names []string, shapes map[string][]int, ws map[string][]float32) {
	t.Helper()
	var outs []safetensors.Out
	for _, name := range names {
		var b bytes.Buffer
		binary.Write(&b, binary.LittleEndian, ws[name])
		outs = append(outs, safetensors.Out{Name: name, DType: "F32", Shape: shapes[name], Size: int64(b.Len()), Data: bytes.NewReader(b.Bytes())})
	}
	dir := filepath.Join(root, "transformer")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := safetensors.Write(filepath.Join(dir, "diffusion_pytorch_model.safetensors"), outs, nil); err != nil {
		t.Fatal(err)
	}
}

// tinyCheckpoint writes random weights of c's DiT (diffusers' names) and returns them by name.
func tinyCheckpoint(t *testing.T, c Dit, root string) map[string][]float32 {
	t.Helper()
	E, M, L := c.Hidden, c.MLP, c.Latent
	rng := rand.New(rand.NewSource(5))
	ws, shapes := map[string][]float32{}, map[string][]int{}
	var names []string
	add := func(name string, scale, offset float64, shape ...int) {
		n := 1
		for _, d := range shape {
			n *= d
		}
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(offset + rng.NormFloat64()*scale)
		}
		ws[name], shapes[name] = v, shape
		names = append(names, name)
	}
	s := 1 / math.Sqrt(float64(E))
	block := func(p string) {
		for _, n := range []string{"to_q", "to_k", "to_v", "to_out.0"} {
			add(p+"attn."+n+".weight", s, 0, E, E)
		}
		add(p+"attn.norm_q.weight", 0.3, 1, c.HeadDim)
		add(p+"attn.norm_k.weight", 0.3, 1, c.HeadDim)
		add(p+"ff.linear_in.weight", s, 0, 2*M, E)
		add(p+"ff.linear_out.weight", 1/math.Sqrt(float64(M)), 0, E, M)
	}
	add("x_embedders.image.weight", 1/math.Sqrt(float64(L)), 0, E, L)
	for i := 0; i < c.Modes; i++ {
		block(fmt.Sprintf("txt_mode_blocks.%d.", i))
		block(fmt.Sprintf("mode_blocks.image.%d.", i))
	}
	for i := 0; i < c.Singles; i++ {
		block(fmt.Sprintf("single_transformer_blocks.%d.", i))
	}
	add("proj_out.weight", s, 0, L, E)
	add("context_embedder.weight", 1/math.Sqrt(float64(c.Context)), 0, E, c.Context)
	writeCheckpoint(t, root, names, shapes, ws)
	return ws
}

// refDit is Flux3Transformer2DModel's blocks in float64, written from diffusers' (the rotary pairs
// (2i, 2i + 1) at each axis's frequencies, the weights as the checkpoint holds them): the velocity
// of the patches x (tokens x latent) under the text rows ctx (text rows x hidden, after
// context_embedder) and the modulation rows mods (ModRows x hidden).
func refDit(c Dit, w map[string][]float32, x, ctx, mods []float32) []float64 {
	E, H, D, M := c.Hidden, c.Heads, c.HeadDim, c.MLP
	N, T := c.Tokens(), c.TextRows
	f64 := func(v []float32) []float64 {
		out := make([]float64, len(v))
		for i, x := range v {
			out[i] = float64(x)
		}
		return out
	}
	lin := func(a []float64, rows int, wt []float32, out int) []float64 {
		in := len(wt) / out
		y := make([]float64, rows*out)
		for r := 0; r < rows; r++ {
			for o := 0; o < out; o++ {
				s := 0.0
				for i := 0; i < in; i++ {
					s += a[r*in+i] * float64(wt[o*in+i])
				}
				y[r*out+o] = s
			}
		}
		return y
	}
	mod := func(r int) []float64 { return f64(mods[r*E : (r+1)*E]) }
	modulate := func(a []float64, rows int, shift, scale []float64) []float64 {
		y := make([]float64, len(a))
		for r := 0; r < rows; r++ {
			row := a[r*E : (r+1)*E]
			mean, v := 0.0, 0.0
			for _, x := range row {
				mean += x
			}
			mean /= float64(E)
			for _, x := range row {
				v += (x - mean) * (x - mean)
			}
			inv := 1 / math.Sqrt(v/float64(E)+c.Eps)
			for i, x := range row {
				y[r*E+i] = (x-mean)*inv*(1+scale[i]) + shift[i]
			}
		}
		return y
	}
	// rope turns q or k (rows x heads x head_dim) at ids: pair (2i, 2i + 1) at axis a = i / (dim/2)'s
	// frequency theta^(-2k/dim), k = i % (dim/2)
	rope := func(a []float64, ids []Position) {
		for r, pos := range ids {
			for h := 0; h < H; h++ {
				i := 0
				for ax, dim := range c.Axes {
					for k := 0; k < dim/2; k++ {
						ang := pos[ax] * math.Pow(c.Theta, -float64(2*k)/float64(dim))
						cs, sn := math.Cos(ang), math.Sin(ang)
						o := (r*H+h)*D + 2*i
						x0, x1 := a[o], a[o+1]
						a[o], a[o+1] = x0*cs-x1*sn, x0*sn+x1*cs
						i++
					}
				}
			}
		}
	}
	rms := func(a []float64, rows int, wt []float32) {
		for r := 0; r < rows*H; r++ {
			v := a[r*D : (r+1)*D]
			s := 0.0
			for _, x := range v {
				s += x * x
			}
			inv := 1 / math.Sqrt(s/float64(D)+c.Eps)
			for i := range v {
				v[i] *= inv * float64(wt[i])
			}
		}
	}
	// parallel is a block's attention + feedforward of h (rows at ids, all attending to all)
	parallel := func(p string, h []float64, ids []Position) []float64 {
		rows := len(ids)
		q, k, v := lin(h, rows, w[p+"attn.to_q.weight"], E), lin(h, rows, w[p+"attn.to_k.weight"], E), lin(h, rows, w[p+"attn.to_v.weight"], E)
		rms(q, rows, w[p+"attn.norm_q.weight"])
		rms(k, rows, w[p+"attn.norm_k.weight"])
		rope(q, ids)
		rope(k, ids)
		o := make([]float64, rows*E)
		for hd := 0; hd < H; hd++ {
			for i := 0; i < rows; i++ {
				sc := make([]float64, rows)
				mx := math.Inf(-1)
				for j := 0; j < rows; j++ {
					s := 0.0
					for d := 0; d < D; d++ {
						s += q[(i*H+hd)*D+d] * k[(j*H+hd)*D+d]
					}
					sc[j] = s / math.Sqrt(float64(D))
					mx = math.Max(mx, sc[j])
				}
				sum := 0.0
				for j := range sc {
					sc[j] = math.Exp(sc[j] - mx)
					sum += sc[j]
				}
				for j := range sc {
					for d := 0; d < D; d++ {
						o[(i*H+hd)*D+d] += sc[j] / sum * v[(j*H+hd)*D+d]
					}
				}
			}
		}
		a := lin(o, rows, w[p+"attn.to_out.0.weight"], E)
		u := lin(h, rows, w[p+"ff.linear_in.weight"], 2*M)
		g := make([]float64, rows*M)
		for r := 0; r < rows; r++ {
			for i := 0; i < M; i++ {
				x1 := u[r*2*M+i]
				g[r*M+i] = x1 / (1 + math.Exp(-x1)) * u[r*2*M+M+i]
			}
		}
		f := lin(g, rows, w[p+"ff.linear_out.weight"], E)
		for i := range a {
			a[i] += f[i]
		}
		return a
	}
	// the position ids as the pipeline makes them, not through Dit's (so that a change there shows):
	// the text's (0, 0, 0, i), the image's (0, y, x, 0) with y and x halved (image_rope_scale)
	textIDs, imageIDs := make([]Position, T), make([]Position, N)
	for i := range textIDs {
		textIDs[i] = Position{0, 0, 0, float64(i)}
	}
	for r := range imageIDs {
		imageIDs[r] = Position{0, float64(r/c.GW()) * 0.5, float64(r%c.GW()) * 0.5, 0}
	}
	txt, img := f64(ctx), lin(f64(x), N, w["x_embedders.image.weight"], E)
	for b := 0; b < c.Modes; b++ {
		for _, s := range []struct {
			x   []float64
			p   string
			ids []Position
			r   int
		}{{txt, fmt.Sprintf("txt_mode_blocks.%d.", b), textIDs, ModTextEarly}, {img, fmt.Sprintf("mode_blocks.image.%d.", b), imageIDs, ModImageEarly}} {
			o := parallel(s.p, modulate(s.x, len(s.ids), mod(s.r), mod(s.r+1)), s.ids)
			g := mod(s.r + 2)
			for i := range s.x {
				s.x[i] += g[i%E] * o[i]
			}
		}
	}
	ids := append(append([]Position(nil), textIDs...), imageIDs...)
	for b := 0; b < c.Singles; b++ {
		p := fmt.Sprintf("single_transformer_blocks.%d.", b)
		h := append(modulate(txt, T, mod(ModTextSingle), mod(ModTextSingle+1)), modulate(img, N, mod(ModImageSingle), mod(ModImageSingle+1))...)
		o := parallel(p, h, ids)
		gt, gi := mod(ModTextSingle+2), mod(ModImageSingle+2)
		for i := range txt {
			txt[i] += gt[i%E] * o[i]
		}
		for i := range img {
			img[i] += gi[i%E] * o[T*E+i]
		}
	}
	h := modulate(img, N, mod(ModOut+1), mod(ModOut))
	return lin(h, N, w["proj_out.weight"], c.Latent)
}

// writeRun writes p as a Core ML package and runs it with the CPU interpreter.
func writeRun(t *testing.T, p *Part, w llm.Weights, in map[string]milcpu.Tensor) []float32 {
	t.Helper()
	path := filepath.Join(t.TempDir(), p.Name+".mlpackage")
	if _, err := llm.WritePart(path, p, w, llm.Options{Backend: llm.BackendCoreML, Format: "fp16"}, nil); err != nil {
		t.Fatal(err)
	}
	prog, closer, err := milcpu.OpenPackage(path)
	if err != nil {
		t.Fatal(err)
	}
	defer closer.Close()
	out, err := prog.Run(in, milcpu.Options{})
	if err != nil {
		t.Fatal(err)
	}
	return out["y"].Data
}

func relRMS(got []float32, want []float64) float64 {
	var num, den float64
	for i := range want {
		d := float64(got[i]) - want[i]
		num, den = num+d*d, den+want[i]*want[i]
	}
	return math.Sqrt(num / den)
}

// The DiT's programs (TextModes, ImageModes, then Singles ending with the output) give the velocity
// diffusers' blocks give: the checkpoint's weights through Weights (the rotary rows permuted, the
// feedforward's input in parts), the text's stream divided by TextStreamScale and the image's by
// DitScale.
func TestDit(t *testing.T) {
	c := tiny()
	root := t.TempDir()
	ws := tinyCheckpoint(t, c, root)
	w, err := OpenWeights(root, c)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	rng := rand.New(rand.NewSource(9))
	randn := func(n int, s float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64() * s)
		}
		return v
	}
	E, N, T := c.Hidden, c.Tokens(), c.TextRows
	x, ctx, mods := randn(N*c.Latent, 1), randn(T*E, 1), randn(ModRows*E, 0.3)
	want := refDit(c, ws, x, ctx, mods)
	cs := make([]float32, len(ctx))
	for i, v := range ctx {
		cs[i] = v / TextStreamScale
	}
	m := func(lo, hi int) milcpu.Tensor {
		return milcpu.Tensor{Shape: []int{1, hi - lo, E}, Data: mods[lo*E : hi*E]}
	}
	txt := writeRun(t, TextModes(c, 0, 1), w, map[string]milcpu.Tensor{"x": {Shape: []int{1, T, E}, Data: cs}, "mods": m(ModTextEarly, ModTextEarly+3)})
	img := writeRun(t, ImageModes(c, 0, 1), w, map[string]milcpu.Tensor{"x": {Shape: []int{1, N, c.Latent}, Data: x}, "mods": m(ModImageEarly, ModImageEarly+3)})
	if len(txt) != T*E || len(img) != N*E {
		t.Fatalf("the mode blocks give %d and %d values, want %d and %d", len(txt), len(img), T*E, N*E)
	}
	all := append(append([]float32(nil), txt...), img...)
	h := writeRun(t, Singles(c, 0, 1), w, map[string]milcpu.Tensor{"x": {Shape: []int{1, c.Rows(), E}, Data: all}, "mods": m(ModTextSingle, ModOut)})
	got := writeRun(t, Singles(c, 1, 1), w, map[string]milcpu.Tensor{"x": {Shape: []int{1, c.Rows(), E}, Data: h}, "mods": m(ModTextSingle, ModRows)})
	if len(got) != len(want) {
		t.Fatalf("%d values of velocity, want %d", len(got), len(want))
	}
	// fp16 weights and activations: about 1e-3 here; a rotary pair, an axis, a stream's scale or a
	// modulation row out of place is 0.1 and more
	if rel := relRMS(got, want); rel > 1e-2 {
		t.Errorf("the velocity is %.3g off diffusers' (relative RMS)", rel)
	} else {
		t.Logf("relative RMS %.2g", rel)
	}
}

// Ctx is context_embedder of the eight hidden states side by side, in two halves of columns, from
// TextScale to TextStreamScale.
func TestCtx(t *testing.T) {
	c := tiny()
	root := t.TempDir()
	ws := tinyCheckpoint(t, c, root)
	w, err := OpenWeights(root, c)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	rng := rand.New(rand.NewSource(4))
	C, half := c.Context, c.Context/2
	x := make([]float32, c.TextRows*C)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	x0, x1 := make([]float32, c.TextRows*half), make([]float32, c.TextRows*half)
	for r := 0; r < c.TextRows; r++ {
		for i := 0; i < half; i++ {
			x0[r*half+i] = x[r*C+i] / TextScale
			x1[r*half+i] = x[r*C+half+i] / TextScale
		}
	}
	got := writeRun(t, Ctx(c), w, map[string]milcpu.Tensor{"x0": {Shape: []int{1, c.TextRows, half}, Data: x0},
		"x1": {Shape: []int{1, c.TextRows, half}, Data: x1}})
	wt := ws["context_embedder.weight"]
	for r := 0; r < c.TextRows; r++ {
		for o := 0; o < c.Hidden; o++ {
			want := 0.0
			for i := 0; i < C; i++ {
				want += float64(x[r*C+i]) * float64(wt[o*C+i])
			}
			if d := math.Abs(float64(got[r*c.Hidden+o])*TextStreamScale - want); d > 1e-2*(1+math.Abs(want)) {
				t.Fatalf("row %d, %d: %g, want %g", r, o, float64(got[r*c.Hidden+o])*TextStreamScale, want)
			}
		}
	}
}

func TestTextGroups(t *testing.T) {
	if g := TextGroups(3); fmt.Sprint(g) != "[[0 3] [3 1] [4 3] [7 1] [8 3] [11 1] [12 3] [15 1] [16 3] [19 1] [20 3] [23 1] [24 3] [27 1] [28 3] [31 1]]" {
		t.Errorf("groups of 3: %v", g)
	}
	if g := TextGroups(4); len(g) != 8 || g[7] != [2]int{28, 4} {
		t.Errorf("groups of 4: %v", g)
	}
}

// The pads' position ids are 1 and the prompt's their index; the rotary tables turn frequency j at
// j and j + head_dim/2.
func TestTextPositions(t *testing.T) {
	if p := TextPositions(3, 6); fmt.Sprint(p) != "[0 1 2 1 1 1]" {
		t.Errorf("positions %v", p)
	}
	tc := Text{HeadDim: 8, Theta: 5e6}
	cos, sin := TextRope(tc, []int{0, 2})
	if cos[0] != 1 || sin[0] != 0 || cos[8] != float32(math.Cos(2)) || cos[8+4] != cos[8] || sin[8+1] != float32(math.Sin(2/math.Pow(5e6, 0.25))) {
		t.Errorf("rope tables cos %v sin %v", cos, sin)
	}
}
