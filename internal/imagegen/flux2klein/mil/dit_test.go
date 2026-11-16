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

// tiny is a FLUX.2 klein DiT of the checkpoint's layout, small: a 2 x 2 grid of patches, 3 text
// rows, one double block and two single blocks.
func tiny() Dit {
	return Dit{Width: 32, Height: 32, Hidden: 32, Heads: 2, HeadDim: 16, MLP: 48, Latent: 8, Context: 24, Doubles: 1, Singles: 2,
		Axes: []int{4, 4, 4, 4}, Theta: 2000, Eps: 1e-6, TimeIn: 16, TextRows: 3}
}

// tinyCheckpoint writes random weights of c's DiT as root/transformer/diffusion_pytorch_model.safetensors
// and returns them by name (without the "dit." prefix).
func tinyCheckpoint(t *testing.T, c Dit, root string) map[string][]float32 {
	t.Helper()
	E, M, L := c.Hidden, c.MLP, c.Latent
	rng := rand.New(rand.NewSource(5))
	ws := map[string][]float32{}
	var outs []safetensors.Out
	add := func(name string, scale float64, shape ...int) {
		n := 1
		for _, d := range shape {
			n *= d
		}
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64() * scale)
		}
		ws[name] = v
		var b bytes.Buffer
		binary.Write(&b, binary.LittleEndian, v)
		outs = append(outs, safetensors.Out{Name: name, DType: "F32", Shape: shape, Size: int64(4 * n), Data: bytes.NewReader(b.Bytes())})
	}
	s := 1 / math.Sqrt(float64(E))
	add("x_embedder.weight", 1/math.Sqrt(float64(L)), E, L)
	for i := 0; i < c.Doubles; i++ {
		p := fmt.Sprintf("transformer_blocks.%d.", i)
		for _, n := range []string{"to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"} {
			add(p+"attn."+n+".weight", s, E, E)
		}
		for _, n := range []string{"norm_q", "norm_k", "norm_added_q", "norm_added_k"} {
			add(p+"attn."+n+".weight", 0.5, c.HeadDim)
		}
		for _, f := range []string{"ff", "ff_context"} {
			add(p+f+".linear_in.weight", s, 2*M, E)
			add(p+f+".linear_out.weight", 1/math.Sqrt(float64(M)), E, M)
		}
	}
	for i := 0; i < c.Singles; i++ {
		p := fmt.Sprintf("single_transformer_blocks.%d.attn.", i)
		add(p+"to_qkv_mlp_proj.weight", s, 3*E+2*M, E)
		add(p+"to_out.weight", 1/math.Sqrt(float64(E+M)), E, E+M)
		add(p+"norm_q.weight", 0.5, c.HeadDim)
		add(p+"norm_k.weight", 0.5, c.HeadDim)
	}
	add("proj_out.weight", s, L, E)
	for i := range ws {
		if len(ws[i]) == c.HeadDim { // norms' weights about 1
			for j := range ws[i] {
				ws[i][j] += 1
			}
		}
	}
	// the norms were written before they were moved about 1: write the file from ws
	for i := range outs {
		var b bytes.Buffer
		binary.Write(&b, binary.LittleEndian, ws[outs[i].Name])
		outs[i].Data = bytes.NewReader(b.Bytes())
	}
	dir := filepath.Join(root, "transformer")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := safetensors.Write(filepath.Join(dir, "diffusion_pytorch_model.safetensors"), outs, nil); err != nil {
		t.Fatal(err)
	}
	return ws
}

// refDit is Flux2Transformer2DModel's blocks in float64, written from diffusers' (the rotary
// pairs (2i, 2i + 1), the weights as the checkpoint holds them): the velocity of the patches x
// (tokens x latent) under the text rows ctx (text rows x hidden, after context_embedder) and the
// modulation rows mods (ModRows x hidden).
func refDit(c Dit, w map[string][]float32, x, ctx, mods []float32) []float64 {
	E, H, D, M := c.Hidden, c.Heads, c.HeadDim, c.MLP
	N, T := c.Tokens(), c.TextRows
	R := N + T
	f64 := func(v []float32) []float64 {
		out := make([]float64, len(v))
		for i, x := range v {
			out[i] = float64(x)
		}
		return out
	}
	// lin is rows x (in) times w (out, in)^T
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
	// modulate: layer norm (eps, no affine) times 1 + scale plus shift, each row
	modulate := func(a []float64, rows int, shift, scale []float64) []float64 {
		y := make([]float64, len(a))
		for r := 0; r < rows; r++ {
			row := a[r*E : (r+1)*E]
			m, v := 0.0, 0.0
			for _, x := range row {
				m += x
			}
			m /= float64(E)
			for _, x := range row {
				v += (x - m) * (x - m)
			}
			v /= float64(E)
			for i, x := range row {
				y[r*E+i] = (x-m)/math.Sqrt(v+c.Eps)*(1+scale[i]) + shift[i]
			}
		}
		return y
	}
	// positions (index, row, column, text position) and the rotary embedding of FLUX's pairs
	pos := make([][4]float64, R)
	for r := 0; r < N; r++ {
		pos[r] = [4]float64{0, float64(r / c.GW()), float64(r % c.GW()), 0}
	}
	for l := 0; l < T; l++ {
		pos[N+l] = [4]float64{0, 0, 0, float64(l)}
	}
	var ang [][]float64 // rows x D/2
	for r := 0; r < R; r++ {
		var a []float64
		for ax, d := range c.Axes {
			for k := 0; k < d/2; k++ {
				a = append(a, pos[r][ax]*math.Pow(c.Theta, -float64(2*k)/float64(d)))
			}
		}
		ang = append(ang, a)
	}
	// head is the q or k rows (rows x E, rows from row lo) normed per head and turned
	head := func(a []float64, rows, lo int, norm []float32) []float64 {
		y := append([]float64(nil), a...)
		for r := 0; r < rows; r++ {
			for h := 0; h < H; h++ {
				v := y[r*E+h*D : r*E+(h+1)*D]
				ms := 0.0
				for _, x := range v {
					ms += x * x
				}
				ms = math.Sqrt(ms/float64(D) + c.Eps)
				for i := range v {
					v[i] = v[i] / ms * float64(norm[i])
				}
				for i := 0; i < D/2; i++ {
					cs, sn := math.Cos(ang[lo+r][i]), math.Sin(ang[lo+r][i])
					a0, a1 := v[2*i], v[2*i+1]
					v[2*i], v[2*i+1] = a0*cs-a1*sn, a0*sn+a1*cs
				}
			}
		}
		return y
	}
	attend := func(q, k, v []float64) []float64 {
		o := make([]float64, R*E)
		for h := 0; h < H; h++ {
			for i := 0; i < R; i++ {
				s := make([]float64, R)
				mx := math.Inf(-1)
				for j := 0; j < R; j++ {
					d := 0.0
					for t := 0; t < D; t++ {
						d += q[i*E+h*D+t] * k[j*E+h*D+t]
					}
					s[j] = d / math.Sqrt(float64(D))
					mx = math.Max(mx, s[j])
				}
				sum := 0.0
				for j := range s {
					s[j] = math.Exp(s[j] - mx)
					sum += s[j]
				}
				for j := range s {
					for t := 0; t < D; t++ {
						o[i*E+h*D+t] += s[j] / sum * v[j*E+h*D+t]
					}
				}
			}
		}
		return o
	}
	swiglu := func(a []float64, rows int, in, out []float32) []float64 {
		u := lin(a, rows, in, 2*M)
		g := make([]float64, rows*M)
		for r := 0; r < rows; r++ {
			for i := 0; i < M; i++ {
				x1, x2 := u[r*2*M+i], u[r*2*M+M+i]
				g[r*M+i] = x1 / (1 + math.Exp(-x1)) * x2
			}
		}
		return lin(g, rows, out, E)
	}
	img := lin(f64(x), N, w["x_embedder.weight"], E)
	txt := f64(ctx)
	for b := 0; b < c.Doubles; b++ {
		p := fmt.Sprintf("transformer_blocks.%d.", b)
		type st struct {
			x            []float64
			rows, lo, mr int
			q, k, v, o   string
			qn, kn, ff   string
		}
		ss := []*st{{img, N, 0, ModImage, "to_q", "to_k", "to_v", "to_out.0", "norm_q", "norm_k", "ff"},
			{txt, T, N, ModText, "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out", "norm_added_q", "norm_added_k", "ff_context"}}
		var q, k, v []float64
		for _, s := range ss {
			h := modulate(s.x, s.rows, mod(s.mr), mod(s.mr+1))
			q = append(q, head(lin(h, s.rows, w[p+"attn."+s.q+".weight"], E), s.rows, s.lo, w[p+"attn."+s.qn+".weight"])...)
			k = append(k, head(lin(h, s.rows, w[p+"attn."+s.k+".weight"], E), s.rows, s.lo, w[p+"attn."+s.kn+".weight"])...)
			v = append(v, lin(h, s.rows, w[p+"attn."+s.v+".weight"], E)...)
		}
		o := attend(q, k, v)
		for _, s := range ss {
			a := lin(o[s.lo*E:(s.lo+s.rows)*E], s.rows, w[p+"attn."+s.o+".weight"], E)
			g1, g2 := mod(s.mr+2), mod(s.mr+5)
			for i := range s.x {
				s.x[i] += g1[i%E] * a[i]
			}
			f := swiglu(modulate(s.x, s.rows, mod(s.mr+3), mod(s.mr+4)), s.rows, w[p+s.ff+".linear_in.weight"], w[p+s.ff+".linear_out.weight"])
			for i := range s.x {
				s.x[i] += g2[i%E] * f[i]
			}
		}
	}
	all := append(append([]float64(nil), img...), txt...)
	for b := 0; b < c.Singles; b++ {
		p := fmt.Sprintf("single_transformer_blocks.%d.attn.", b)
		h := modulate(all, R, mod(ModSingle), mod(ModSingle+1))
		u := lin(h, R, w[p+"to_qkv_mlp_proj.weight"], 3*E+2*M)
		col := func(lo, n int) []float64 {
			y := make([]float64, R*n)
			for r := 0; r < R; r++ {
				copy(y[r*n:(r+1)*n], u[r*(3*E+2*M)+lo:r*(3*E+2*M)+lo+n])
			}
			return y
		}
		q, k, v := head(col(0, E), R, 0, w[p+"norm_q.weight"]), head(col(E, E), R, 0, w[p+"norm_k.weight"]), col(2*E, E)
		o := attend(q, k, v)
		g1, g2 := col(3*E, M), col(3*E+M, M)
		cat := make([]float64, R*(E+M))
		for r := 0; r < R; r++ {
			copy(cat[r*(E+M):], o[r*E:(r+1)*E])
			for i := 0; i < M; i++ {
				x1 := g1[r*M+i]
				cat[r*(E+M)+E+i] = x1 / (1 + math.Exp(-x1)) * g2[r*M+i]
			}
		}
		y := lin(cat, R, w[p+"to_out.weight"], E)
		gate := mod(ModSingle + 2)
		for i := range all {
			all[i] += gate[i%E] * y[i]
		}
	}
	h := modulate(all[:N*E], N, mod(ModOut+1), mod(ModOut))
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

// The DiT's programs (Doubles, then Singles ending with the output) give the velocity diffusers'
// blocks give: the checkpoint's weights through Weights (the rotary rows permuted, the fused
// matrices in parts), the residual stream divided by DitScale.
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
	E, N, R := c.Hidden, c.Tokens(), c.Rows()
	x, ctx, mods := randn(N*c.Latent, 1), randn(c.TextRows*E, 1), randn(ModRows*E, 0.3)
	want := refDit(c, ws, x, ctx, mods)
	cs := make([]float32, len(ctx))
	for i, v := range ctx {
		cs[i] = v / DitScale
	}
	h := writeRun(t, Doubles(c, 0, 1), w, map[string]milcpu.Tensor{
		"x": {Shape: []int{1, N, c.Latent}, Data: x}, "ctx": {Shape: []int{1, c.TextRows, E}, Data: cs},
		"mods": {Shape: []int{1, ModSingle, E}, Data: mods[:ModSingle*E]}})
	if len(h) != R*E {
		t.Fatalf("the doubles give %d values, want %d", len(h), R*E)
	}
	got := writeRun(t, Singles(c, 0, 2), w, map[string]milcpu.Tensor{
		"x": {Shape: []int{1, R, E}, Data: h}, "mods": {Shape: []int{1, ModRows - ModSingle, E}, Data: mods[ModSingle*E:]}})
	if len(got) != len(want) {
		t.Fatalf("%d values of velocity, want %d", len(got), len(want))
	}
	var num, den float64
	for i := range want {
		d := float64(got[i]) - want[i]
		num, den = num+d*d, den+want[i]*want[i]
	}
	// fp16 weights and activations: about 1e-3 here; a rotary pair, a part or a modulation row
	// out of place is 0.1 and more
	if rel := math.Sqrt(num / den); rel > 1e-2 {
		t.Errorf("the velocity is %.3g off diffusers' (relative RMS)", rel)
	} else {
		t.Logf("relative RMS %.2g", rel)
	}
}

func TestTextGroups(t *testing.T) {
	g := TextGroups(4)
	want := [][2]int{{0, 4}, {4, 4}, {8, 1}, {9, 4}, {13, 4}, {17, 1}, {18, 4}, {22, 4}, {26, 1}}
	if fmt.Sprint(g) != fmt.Sprint(want) {
		t.Errorf("groups of 4: %v, want %v", g, want)
	}
	if g := TextGroups(3); len(g) != 9 || g[8] != [2]int{24, 3} {
		t.Errorf("groups of 3: %v", g)
	}
	if g := Groups(20, 6); fmt.Sprint(g) != "[[0 5] [5 5] [10 5] [15 5]]" {
		t.Errorf("20 by 6: %v", g)
	}
	if g := Groups(5, 2); fmt.Sprint(g) != "[[0 2] [2 2] [4 1]]" {
		t.Errorf("5 by 2: %v", g)
	}
}

// Ctx is context_embedder of the three hidden states side by side, from TextScale to DitScale.
func TestCtx(t *testing.T) {
	c := tiny()
	c.TextRows = 2
	root := t.TempDir()
	rng := rand.New(rand.NewSource(4))
	wt := make([]float32, c.Hidden*c.Context)
	for i := range wt {
		wt[i] = float32(rng.NormFloat64() * 0.2)
	}
	var b bytes.Buffer
	binary.Write(&b, binary.LittleEndian, wt)
	dir := filepath.Join(root, "transformer")
	os.MkdirAll(dir, 0o755)
	if err := safetensors.Write(filepath.Join(dir, "diffusion_pytorch_model.safetensors"), []safetensors.Out{{Name: "context_embedder.weight",
		DType: "F32", Shape: []int{c.Hidden, c.Context}, Size: int64(b.Len()), Data: bytes.NewReader(b.Bytes())}}, nil); err != nil {
		t.Fatal(err)
	}
	w, err := OpenWeights(root, c)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	x := make([]float32, c.TextRows*c.Context)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	xs := make([]float32, len(x))
	for i, v := range x {
		xs[i] = v / TextScale
	}
	got := writeRun(t, Ctx(c), w, map[string]milcpu.Tensor{"x": {Shape: []int{1, c.TextRows, c.Context}, Data: xs}})
	for r := 0; r < c.TextRows; r++ {
		for o := 0; o < c.Hidden; o++ {
			want := 0.0
			for i := 0; i < c.Context; i++ {
				want += float64(x[r*c.Context+i]) * float64(wt[o*c.Context+i])
			}
			if d := math.Abs(float64(got[r*c.Hidden+o])*DitScale - want); d > 1e-2*(1+math.Abs(want)) {
				t.Fatalf("row %d, %d: %g, want %g", r, o, float64(got[r*c.Hidden+o])*DitScale, want)
			}
		}
	}
}
