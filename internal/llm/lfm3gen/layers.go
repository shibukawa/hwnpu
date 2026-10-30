// Package lfm3gen converts the speech output of Liquid AI's LFM3-Audio and LFM3-Omni into Core ML
// programs: the depthformer (lfm3-depthformer), which turns the text model's final hidden state
// into the codes of one audio frame in three MaskGIT passes, and the decoder of the liquid codec
// (lfm3-codec), which turns codes into STFT frames. internal/speech runs them with the text model
// and the inverse STFT. The structures are llama.cpp's tools/mtmd/models/lfm3a-gen.cpp.
package lfm3gen

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// stack describes an LFM3 style stack of layers.
type stack struct {
	prefix         string // of the layers: PREFIXlayers.N.
	d              int
	layerTypes     []string // conv or full_attention
	heads, kvHeads int
	headDim        int
	ff             int
	convKernel     int
	eps            float64
	ropeTheta      float64
	sandwich       bool
	window         int // causal attention over the last window positions; 0: bidirectional
	qNorm, kNorm   string
}

type builder struct {
	*blocks.Graph
	st       stack
	seq      int
	cos, sin *mlpackage.Var
	mask     *mlpackage.Var
}

func (m *builder) layer(i int, h *mlpackage.Var) *mlpackage.Var {
	st := m.st
	p := fmt.Sprintf("%slayers.%d.", st.prefix, i)
	residual := h
	h = m.RMSNorm(h, p+"operator_norm.weight", st.d, st.eps, 1)
	if st.layerTypes[i] == "conv" {
		h = m.conv(p+"conv.", h)
	} else {
		h = m.attention(p+"self_attn.", h)
	}
	if st.sandwich {
		h = m.RMSNorm(h, p+"post_operator_norm.weight", st.d, st.eps, 1)
	}
	h = m.Op("add", map[string]any{"x": h, "y": residual}, h.Shape...)
	residual = h
	f := m.RMSNorm(h, p+"ffn_norm.weight", st.d, st.eps, 1)
	f = m.SwiGLU(f, p+"feed_forward.w1.weight", p+"feed_forward.w3.weight", p+"feed_forward.w2.weight", st.ff, st.d, 1)
	if st.sandwich {
		f = m.RMSNorm(f, p+"post_ffn_norm.weight", st.d, st.eps, 1)
	}
	return m.Op("add", map[string]any{"x": f, "y": residual}, f.Shape...)
}

// conv is the gated short convolution, causal from zeros, in taps.
func (m *builder) conv(p string, h *mlpackage.Var) *mlpackage.Var {
	d, s, k := m.st.d, m.seq, m.st.convKernel
	bcx := m.Linear(h, p+"in_proj.weight", "", 3*d)
	part := func(j int) *mlpackage.Var {
		return m.Op("slice_by_index", map[string]any{"x": bcx, "begin": m.B.Ints(0, 0, int32(j*d)), "end": m.B.Ints(1, int32(s), int32((j+1)*d)),
			"begin_mask": m.Bools(true, true, false), "end_mask": m.Bools(true, true, false)}, 1, s, d)
	}
	bg, cg, xg := part(0), part(1), part(2)
	bx := m.Op("mul", map[string]any{"x": bg, "y": xg}, 1, s, d)
	padded := m.Op("pad", map[string]any{"x": bx, "pad": m.B.Ints(int32(k-1), 0, 0, 0), "mode": m.Str("constant"), "constant_val": m.B.ScalarF16(0)}, 1, s+k-1, d)
	cw := m.Get(p+"conv.weight", d, 1, k)
	var y *mlpackage.Var
	for j := 0; j < k; j++ {
		tap := make([]float32, d)
		for c := range tap {
			tap[c] = cw[c*k+j]
		}
		wv := m.F16s(fmt.Sprintf("%sconv.weight.tap%d", p, j), []int{1, 1, d}, tap)
		seg := m.Op("slice_by_index", map[string]any{"x": padded, "begin": m.B.Ints(0, int32(j), 0), "end": m.B.Ints(1, int32(j+s), int32(d)),
			"begin_mask": m.Bools(true, false, true), "end_mask": m.Bools(true, false, true)}, 1, s, d)
		term := m.Op("mul", map[string]any{"x": seg, "y": wv}, 1, s, d)
		if y == nil {
			y = term
		} else {
			y = m.Op("add", map[string]any{"x": y, "y": term}, 1, s, d)
		}
	}
	y = m.Op("mul", map[string]any{"x": cg, "y": y}, 1, s, d)
	return m.Linear(y, p+"out_proj.weight", "", d)
}

// attention is LFM3's: grouped query attention with QK RMS norm, RoPE (rotate_half) over the
// whole head and a sigmoid gate per head; causal over the window, or bidirectional.
func (m *builder) attention(p string, h *mlpackage.Var) *mlpackage.Var {
	st := m.st
	s, nh, nkv, hd := m.seq, st.heads, st.kvHeads, st.headDim
	if m.cos == nil {
		inv := make([]float64, hd/2)
		for i := range inv {
			inv[i] = math.Pow(st.ropeTheta, -float64(2*i)/float64(hd))
		}
		m.cos, m.sin = m.RopeTables("gen", inv, s, hd)
	}
	heads := func(name string, k int, norm string) *mlpackage.Var {
		x := m.Linear(h, p+name+"_proj.weight", "", k*hd)
		x = m.Op("reshape", map[string]any{"x": x, "shape": m.B.Ints(1, int32(s), int32(k), int32(hd))}, 1, s, k, hd)
		if norm != "" {
			x = m.RMSNorm(x, p+norm+".weight", hd, st.eps, 1)
			x = m.ApplyRope(x, m.cos, m.sin)
		}
		return m.Op("transpose", map[string]any{"x": x, "perm": m.B.Ints(0, 2, 1, 3)}, 1, k, s, hd)
	}
	q, k, v := heads("q", nh, st.qNorm), heads("k", nkv, st.kNorm), heads("v", nkv, "")
	var mask func() *mlpackage.Var
	if st.window > 0 {
		if m.mask == nil {
			m.mask = m.CausalMask("mask_gen", s, st.window)
		}
		mask = func() *mlpackage.Var { return m.mask }
	}
	o := m.SDPA(q, k, v, mask)
	o = m.Op("transpose", map[string]any{"x": o, "perm": m.B.Ints(0, 2, 1, 3)}, 1, s, nh, hd)
	gate := m.Linear(h, p+"gate_proj.weight", "", nh)
	gate = m.Op("sigmoid", map[string]any{"x": gate}, 1, s, nh)
	gate = m.Op("reshape", map[string]any{"x": gate, "shape": m.B.Ints(1, int32(s), int32(nh), 1)}, 1, s, nh, 1)
	o = m.Op("mul", map[string]any{"x": o, "y": gate}, 1, s, nh, hd)
	o = m.Op("reshape", map[string]any{"x": o, "shape": m.B.Ints(1, int32(s), int32(nh*hd))}, 1, s, nh*hd)
	return m.Linear(o, p+"o_proj.weight", "", st.d)
}

// tensors lists the stack's tensors.
func (st *stack) tensors(t map[string][]int) {
	d, hd := st.d, st.headDim
	for i, typ := range st.layerTypes {
		p := fmt.Sprintf("%slayers.%d.", st.prefix, i)
		norms := []string{"operator_norm", "ffn_norm"}
		if st.sandwich {
			norms = append(norms, "post_operator_norm", "post_ffn_norm")
		}
		for _, n := range norms {
			t[p+n+".weight"] = []int{d}
		}
		t[p+"feed_forward.w1.weight"] = []int{st.ff, d}
		t[p+"feed_forward.w3.weight"] = []int{st.ff, d}
		t[p+"feed_forward.w2.weight"] = []int{d, st.ff}
		if typ == "conv" {
			t[p+"conv.in_proj.weight"] = []int{3 * d, d}
			t[p+"conv.conv.weight"] = []int{d, 1, st.convKernel}
			t[p+"conv.out_proj.weight"] = []int{d, d}
			continue
		}
		a := p + "self_attn."
		t[a+"q_proj.weight"] = []int{st.heads * hd, d}
		t[a+"k_proj.weight"] = []int{st.kvHeads * hd, d}
		t[a+"v_proj.weight"] = []int{st.kvHeads * hd, d}
		t[a+"o_proj.weight"] = []int{d, st.heads * hd}
		t[a+"gate_proj.weight"] = []int{st.heads, d}
		t[a+st.qNorm+".weight"] = []int{hd}
		t[a+st.kNorm+".weight"] = []int{hd}
	}
}
