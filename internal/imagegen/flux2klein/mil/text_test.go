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

	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/safetensors"
)

// refQwen3 is Qwen3's decoder layers first..first+count-1 in float64 over x (TextRows x hidden),
// as transformers runs them with an attention mask of n tokens: the rows causal, no row attending
// to a pad's key (the pads' rows attend to the tokens').
func refQwen3(c Text, w map[string][]float32, x []float64, n, first, count int) []float64 {
	L, E, H, G, D := TextRows, c.Hidden, c.Heads, c.KVHeads, c.HeadDim
	rms := func(v []float64, wt []float32) {
		s := 0.0
		for _, a := range v {
			s += a * a
		}
		s = math.Sqrt(s/float64(len(v)) + c.Eps)
		for i := range v {
			v[i] = v[i] / s * float64(wt[i])
		}
	}
	lin := func(a []float64, wt []float32, out int) []float64 {
		in := len(wt) / out
		rows := len(a) / in
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
	rope := func(v []float64, pos int) { // HF's rotate_half: pairs (i, i + D/2)
		for i := 0; i < D/2; i++ {
			a := float64(pos) * math.Pow(c.Theta, -float64(2*i)/float64(D))
			cs, sn := math.Cos(a), math.Sin(a)
			x0, x1 := v[i], v[i+D/2]
			v[i], v[i+D/2] = x0*cs-x1*sn, x1*cs+x0*sn
		}
	}
	x = append([]float64(nil), x...)
	for l := first; l < first+count; l++ {
		p := fmt.Sprintf("model.layers.%d.", l)
		h := append([]float64(nil), x...)
		for r := 0; r < L; r++ {
			rms(h[r*E:(r+1)*E], w[p+"input_layernorm.weight"])
		}
		q, k, v := lin(h, w[p+"self_attn.q_proj.weight"], H*D), lin(h, w[p+"self_attn.k_proj.weight"], G*D), lin(h, w[p+"self_attn.v_proj.weight"], G*D)
		for r := 0; r < L; r++ {
			for hh := 0; hh < H; hh++ {
				rms(q[r*H*D+hh*D:r*H*D+(hh+1)*D], w[p+"self_attn.q_norm.weight"])
				rope(q[r*H*D+hh*D:r*H*D+(hh+1)*D], r)
			}
			for g := 0; g < G; g++ {
				rms(k[r*G*D+g*D:r*G*D+(g+1)*D], w[p+"self_attn.k_norm.weight"])
				rope(k[r*G*D+g*D:r*G*D+(g+1)*D], r)
			}
		}
		o := make([]float64, L*H*D)
		for hh := 0; hh < H; hh++ {
			g := hh / (H / G)
			for i := 0; i < L; i++ {
				var s []float64
				mx := math.Inf(-1)
				for j := 0; j < L; j++ {
					if j > i || j >= n {
						s = append(s, math.Inf(-1))
						continue
					}
					d := 0.0
					for t := 0; t < D; t++ {
						d += q[i*H*D+hh*D+t] * k[j*G*D+g*D+t]
					}
					s = append(s, d/math.Sqrt(float64(D)))
					mx = math.Max(mx, s[j])
				}
				sum := 0.0
				for j := range s {
					s[j] = math.Exp(s[j] - mx)
					sum += s[j]
				}
				for j := range s {
					for t := 0; t < D; t++ {
						o[i*H*D+hh*D+t] += s[j] / sum * v[j*G*D+g*D+t]
					}
				}
			}
		}
		a := lin(o, w[p+"self_attn.o_proj.weight"], E)
		for i := range x {
			x[i] += a[i]
		}
		h = append(h[:0], x...)
		for r := 0; r < L; r++ {
			rms(h[r*E:(r+1)*E], w[p+"post_attention_layernorm.weight"])
		}
		gt, up := lin(h, w[p+"mlp.gate_proj.weight"], c.MLP), lin(h, w[p+"mlp.up_proj.weight"], c.MLP)
		for i := range gt {
			gt[i] = gt[i] / (1 + math.Exp(-gt[i])) * up[i]
		}
		f := lin(gt, w[p+"mlp.down_proj.weight"], E)
		for i := range x {
			x[i] += f[i]
		}
	}
	return x
}

// A group of Qwen3's layers (TextGroup) gives the hidden states transformers gives, the pads'
// rows too (the DiT reads them), with the residual stream divided by TextScale.
func TestTextGroup(t *testing.T) {
	c := Text{Hidden: 16, Layers: 3, Heads: 4, KVHeads: 2, HeadDim: 8, MLP: 24, Theta: 1e6, Eps: 1e-6}
	rng := rand.New(rand.NewSource(3))
	ws := map[string][]float32{}
	var outs []safetensors.Out
	add := func(name string, mean, scale float64, shape ...int) {
		n := 1
		for _, d := range shape {
			n *= d
		}
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(mean + rng.NormFloat64()*scale)
		}
		ws[name] = v
		var b bytes.Buffer
		binary.Write(&b, binary.LittleEndian, v)
		outs = append(outs, safetensors.Out{Name: name, DType: "F32", Shape: shape, Size: int64(4 * n), Data: bytes.NewReader(b.Bytes())})
	}
	E, D := c.Hidden, c.HeadDim
	for l := 0; l < c.Layers; l++ {
		p := fmt.Sprintf("model.layers.%d.", l)
		add(p+"input_layernorm.weight", 1, 0.1, E)
		add(p+"post_attention_layernorm.weight", 1, 0.1, E)
		add(p+"self_attn.q_norm.weight", 1, 0.1, D)
		add(p+"self_attn.k_norm.weight", 1, 0.1, D)
		add(p+"self_attn.q_proj.weight", 0, 0.4, c.Heads*D, E)
		add(p+"self_attn.k_proj.weight", 0, 0.4, c.KVHeads*D, E)
		add(p+"self_attn.v_proj.weight", 0, 0.4, c.KVHeads*D, E)
		add(p+"self_attn.o_proj.weight", 0, 0.4, E, c.Heads*D)
		add(p+"mlp.gate_proj.weight", 0, 0.4, c.MLP, E)
		add(p+"mlp.up_proj.weight", 0, 0.4, c.MLP, E)
		add(p+"mlp.down_proj.weight", 0, 0.4, E, c.MLP)
	}
	root := t.TempDir()
	dir := filepath.Join(root, "text_encoder")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := safetensors.Write(filepath.Join(dir, "model.safetensors"), outs, nil); err != nil {
		t.Fatal(err)
	}
	w, err := OpenWeights(root, tiny())
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	const n = 9 // the prompt's tokens; the rest of the 512 rows are the pad's
	x := make([]float64, TextRows*E)
	pad := make([]float64, E)
	for i := range pad {
		pad[i] = rng.NormFloat64()
	}
	for r := 0; r < TextRows; r++ {
		for i := 0; i < E; i++ {
			if r < n {
				x[r*E+i] = rng.NormFloat64()
			} else {
				x[r*E+i] = pad[i]
			}
		}
	}
	want := refQwen3(c, ws, x, n, 1, 2)
	xs := make([]float32, len(x))
	for i, v := range x {
		xs[i] = float32(v / TextScale)
	}
	kb := make([]float32, TextRows)
	for i := n; i < TextRows; i++ {
		kb[i] = -1e4
	}
	got := writeRun(t, TextGroup(c, 1, 2), w, map[string]milcpu.Tensor{
		"x": {Shape: []int{1, TextRows, E}, Data: xs}, "kbias": {Shape: []int{1, 1, 1, TextRows}, Data: kb}})
	var num, den float64
	for i := range want {
		d := float64(got[i])*TextScale - want[i]
		num, den = num+d*d, den+want[i]*want[i]
	}
	if rel := math.Sqrt(num / den); rel > 1e-2 {
		t.Errorf("the hidden states are %.3g off transformers' (relative RMS)", rel)
	} else {
		t.Logf("relative RMS %.2g", rel)
	}
	// a pad's row: as far off as the tokens'
	var pn, pd float64
	for i := (TextRows - 1) * E; i < TextRows*E; i++ {
		d := float64(got[i])*TextScale - want[i]
		pn, pd = pn+d*d, pd+want[i]*want[i]
	}
	if rel := math.Sqrt(pn / pd); rel > 1e-2 {
		t.Errorf("the last pad's row is %.3g off", rel)
	}
}
