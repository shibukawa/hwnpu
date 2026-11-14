package mil

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// TestWriteProbes writes parts of the DiT at 1024 x 1024 whole and with pieces taken out, as Core
// AI programs (Core ML packages with HWNPU_ANIMA_PROBE_COREML, for aicheck run -cpu), to time
// what each piece of a block costs with aicheck run -calls N and aicheck cycle (design.md
// 16.201): HWNPU_ANIMA_DIR is the Anima directory (diffusion_models/ ...), HWNPU_ANIMA_PROBE_OUT
// where they go, HWNPU_ANIMA_PROBE the comma separated keys of parts (default: all). v_ are one
// block (with self-attention it fails the Neural Engine's compiler alone: time groups), g_ and
// na are groups of blocks, a_ attention alone on its q, k and v.
func TestWriteProbes(t *testing.T) {
	out, root := os.Getenv("HWNPU_ANIMA_PROBE_OUT"), os.Getenv("HWNPU_ANIMA_DIR")
	if out == "" || root == "" {
		t.Skip("HWNPU_ANIMA_PROBE_OUT and HWNPU_ANIMA_DIR")
	}
	w, err := OpenWeights(root, "anima-turbo-v1.1.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	const W, H, i = 1024, 1024, 4
	gh, gw := Grid(W, H)
	n := gh * gw
	sc := 1 / math.Sqrt(float64(ditHeadDim))
	blockPart := func(name string, v variant) *Part {
		return part(name, func(g *graph) {
			x := g.Input("x", 1, n, DitHidden)
			tt := g.Input("t", 1, 1, TimeDim)
			ctx := g.Input("ctx", 1, TextRows, textHidden)
			temb := g.Slice(tt, -1, 0, 3*DitHidden)
			act := g.SiLU(g.Slice(tt, -1, 3*DitHidden, TimeDim))
			cos, sin := ropeTables(g, gh, gw)
			g.Output(ditBlockV(g, i, x, act, temb, ctx, cos, sin, v))
		})
	}
	var groupPartN func(name string, first, count int, v variant) *Part
	groupPart := func(name string, v variant) *Part { return groupPartN(name, 4, 4, v) }
	groupPartN = func(name string, first, count int, v variant) *Part {
		return part(name, func(g *graph) {
			x := g.Input("x", 1, n, DitHidden)
			tt := g.Input("t", 1, 1, TimeDim)
			rows := TextRows
			if v.rows > 0 {
				rows = v.rows
			}
			ctx := g.Input("ctx", 1, rows, textHidden)
			if v.rows > 0 {
				v.cbias = g.Input("cbias", 1, 1, 1, rows)
			}
			temb := g.Slice(tt, -1, 0, 3*DitHidden)
			act := g.SiLU(g.Slice(tt, -1, 3*DitHidden, TimeDim))
			cos, sin := ropeTables(g, gh, gw)
			for b := first; b < first+count; b++ {
				x = ditBlockV(g, b, x, act, temb, ctx, cos, sin, v)
			}
			g.Output(x)
		})
	}
	attnPart := func(name string, q int, scale float64, m int) *Part {
		return part(name, func(g *graph) {
			qq := g.Input("q", 1, ditHeads, n, ditHeadDim)
			k := g.Input("k", 1, ditHeads, m, ditHeadDim)
			v := g.Input("v", 1, ditHeads, m, ditHeadDim)
			g.Output(g.QChunkAttention(qq, k, v, scale, nil, q))
		})
	}
	parts := map[string]*Part{
		"full":       blockPart("v_full", variant{qchunk: 512}),
		"noattn":     blockPart("v_noattn", variant{qchunk: 512, noSelf: true, noCross: true}),
		"noself":     blockPart("v_noself", variant{qchunk: 512, noSelf: true}),
		"qkv":        blockPart("v_qkv", variant{qchunk: 512, fusedQKV: true}),
		"foldscale":  blockPart("v_foldscale", variant{qchunk: 512, foldScale: true}),
		"self512":    attnPart("a_self512", 512, sc, n),
		"self1024":   attnPart("a_self1024", 1024, sc, n),
		"self256":    attnPart("a_self256", 256, sc, n),
		"self512s1":  attnPart("a_self512s1", 512, 1, n),
		"cross512":   attnPart("a_cross512", 512, sc, TextRows),
		"cross4096":  attnPart("a_cross4096", 0, sc, TextRows),
		"mlp":        blockPart("v_mlp", variant{mlpOnly: true}),
		"hp":         blockPart("v_hp", variant{noSelf: true, noCross: true, keepQK: true}),
		"hp_abs":     blockPart("v_hp_abs", variant{noSelf: true, noCross: true, keepQK: true, absMax: true}),
		"hp_absmul":  blockPart("v_hp_absmul", variant{noSelf: true, noCross: true, keepQK: true, absMax: true, absMul: true}),
		"hp_absrope": blockPart("v_hp_absrope", variant{noSelf: true, noCross: true, keepQK: true, absMax: true, ropeSign: true}),
		"hp_norope":  blockPart("v_hp_norope", variant{noSelf: true, noCross: true, keepQK: true, noRope: true}),
		"hp_nonorm":  blockPart("v_hp_nonorm", variant{noSelf: true, noCross: true, keepQK: true, noHeadNorm: true}),
		"hp_nomax":   blockPart("v_hp_nomax", variant{noSelf: true, noCross: true, keepQK: true, noHeadMax: true}),
		"hp_bare":    blockPart("v_hp_bare", variant{noSelf: true, noCross: true, keepQK: true, noHeadNorm: true, noRope: true}),
		"hp_plainln": blockPart("v_hp_plainln", variant{noSelf: true, noCross: true, keepQK: true, plainLN: true}),
		"hp_nogelu":  blockPart("v_hp_nogelu", variant{noSelf: true, noCross: true, keepQK: true, gelu: "none"}),
		"hp_tanh":    blockPart("v_hp_tanh", variant{noSelf: true, noCross: true, keepQK: true, gelu: "tanh"}),
		"hp_sigmoid": blockPart("v_hp_sigmoid", variant{noSelf: true, noCross: true, keepQK: true, gelu: "sigmoid"}),
		"g_base":     groupPart("g_base", variant{qchunk: 512}),
		"g_fold":     groupPart("g_fold", variant{qchunk: 512, foldScale: true}),
		"g_foldabs":  groupPart("g_foldabs", variant{qchunk: 512, foldScale: true, absMax: true}),
		"g_prod":     Blocks(W, H, 4, 4, 512),
		"g7a":        groupPartN("g7a", 0, 7, variant{qchunk: 512, foldScale: true}),
		"g7b":        groupPartN("g7b", 7, 7, variant{qchunk: 512, foldScale: true}),
		"g4c":        groupPartN("g4c", 8, 4, variant{qchunk: 512, foldScale: true}),
		"na4a":       groupPartN("na4a", 0, 4, variant{noSelf: true, noCross: true}),
		"na4b":       groupPartN("na4b", 4, 4, variant{noSelf: true, noCross: true}),
		"na8a":       groupPartN("na8a", 0, 8, variant{noSelf: true, noCross: true}),
		"na8b":       groupPartN("na8b", 8, 8, variant{noSelf: true, noCross: true}),
		"g_foldx256": groupPart("g_foldx256", variant{qchunk: 512, foldScale: true, rows: 256}),
		"g_qkv":      groupPart("g_qkv", variant{qchunk: 512, fusedQKV: true}),
		"g_foldqkv":  groupPart("g_foldqkv", variant{qchunk: 512, foldScale: true, fusedQKV: true}),
		"cross128":   attnBias("a_cross128", 128),
		"cross256":   attnBias("a_cross256", 256),
		"crossb512":  attnBias("a_crossb512", 512),
		"keys":       attnKeys("a_keys", 512, n),
		// layouts of the self-attention's scores (attnLayout)
		"l_w":        attnLayout("l_w", "w", false, 512, false),
		"l_mm":       attnLayout("l_mm", "mm", false, 512, false),
		"l_phw":      attnLayout("l_phw", "w", true, 0, false),
		"l_phw5":     attnLayout("l_phw5", "w", true, 512, false),
		"l_phc":      attnLayout("l_phc", "c", true, 0, false),
		"l_phc5":     attnLayout("l_phc5", "c", true, 512, false),
		"l_phc1":     attnLayout("l_phc1", "c", true, 1024, false),
		"l_phcd":     attnLayout("l_phcd", "c", true, 0, true),
		"l_bc":       attnLayout("l_bc", "bc", false, 512, false),
		"l_bc0":      attnLayout("l_bc0", "bc", false, 0, false),
		"l_phq":      attnLayout("l_phq", "q", true, 0, false),
		"l_phq5":     attnLayout("l_phq5", "q", true, 512, false),
		"l_w128":     attnLayout("l_w128", "w", false, 128, false),
		"l_w64":      attnLayout("l_w64", "w", false, 64, false),
		"l_w32":      attnLayout("l_w32", "w", false, 32, false),
		"l_phq2":     attnLayout("l_phq2", "q", true, 256, false),
		"l_phq1":     attnLayout("l_phq1", "q", true, 128, false),
		"l_phw2":     attnLayout("l_phw2", "w", true, 256, false),
		"l_phw1":     attnLayout("l_phw1", "w", true, 128, false),
		"l_phc2":     attnLayout("l_phc2", "c", true, 256, false),
		"l_phq5d":    attnLayout("l_phq5d", "q", true, 512, true),
		"l_phq1d":    attnLayout("l_phq1d", "q", true, 128, true),
		"l_phq2d":    attnLayout("l_phq2d", "q", true, 256, true),
		"l_phqkd":    attnLayout("l_phqkd", "q", true, 1024, true),
		"l_phq0d":    attnLayout("l_phq0d", "q", true, 0, true),
		"l_phc5d":    attnLayout("l_phc5d", "c", true, 512, true),
		"l_cq5":      attnLayout("l_cq5", "q", false, 512, false),
		"l_cq5d":     attnLayout("l_cq5d", "q", false, 512, true),
		"l_cq2d":     attnLayout("l_cq2d", "q", false, 256, true),
		"l_cqkd":     attnLayout("l_cqkd", "q", false, 1024, true),
		"l_w5d":      attnLayout("l_w5d", "w", false, 512, true),
		"l_phqk":     attnLayout("l_phqk", "q", true, 1024, false),
		"l_phq2k":    attnLayout("l_phq2k", "q", true, 2048, false),
		"l_phq5m":    attnLayout("l_phq5m", "qm", true, 512, false),
		"l_phqkm":    attnLayout("l_phqkm", "qm", true, 1024, false),
		"x_w":        attnLayoutK("x_w", "w", false, 0, false, TextRows),
		"x_wd":       attnLayoutK("x_wd", "w", false, 0, true, TextRows),
		"x_phq0d":    attnLayoutK("x_phq0d", "q", true, 0, true, TextRows),
		"x_phqkd":    attnLayoutK("x_phqkd", "q", true, 1024, true, TextRows),
		"x_cq5d":     attnLayoutK("x_cq5d", "q", false, 512, true, TextRows),
		"x_cqkd":     attnLayoutK("x_cqkd", "q", false, 1024, true, TextRows),
		"x_cq0":      attnLayoutK("x_cq0", "q", false, 1024, false, TextRows),
		"defer":      attnVar("a_defer", 512, n, 0, false, false),
		"deferc":     attnVar("a_deferc", 512, n, 5.5, false, false),
		"nomax":      attnVar("a_nomax", 512, n, 0, true, false),
		"explicit":   attnVar("a_explicit", 512, n, 0, false, true),
		"defer1024":  attnVar("a_defer1024", 1024, n, 0, false, false),
		"defer256":   attnVar("a_defer256", 256, n, 0, false, false),
		"crossdefer": attnVar("a_crossdefer", 0, TextRows, 0, false, false),
	}
	want := os.Getenv("HWNPU_ANIMA_PROBE")
	for k, p := range parts {
		if want != "" && !strings.Contains(","+want+",", ","+k+",") {
			continue
		}
		// HWNPU_ANIMA_PROBE_FORMAT: the weights' format (default int8, the export's), in the name
		format, suffix := "int8", ""
		if f := os.Getenv("HWNPU_ANIMA_PROBE_FORMAT"); f != "" {
			format, suffix = f, "_"+f
		}
		path := filepath.Join(out, p.Name+suffix+probeExt())
		if _, err := llm.WritePart(path, p, w, llm.Options{Backend: probeBackend(), Format: format}, nil); err != nil {
			t.Fatalf("%s: %v", k, err)
		}
		t.Logf("wrote %s", path)
	}
}

// attnVar is attention over q pre-scaled, its softmax written out: exp(s - max - c), summed, and
// the sum divided after P V (explicit: before it, as ailower's softmax).
func attnVar(name string, chunk, m int, c float32, nomax, explicit bool) *Part {
	return part(name, func(g *graph) {
		gh, gw := Grid(1024, 1024)
		n := gh * gw
		qq := g.Input("q", 1, ditHeads, n, ditHeadDim)
		k := g.Input("k", 1, ditHeads, m, ditHeadDim)
		v := g.Input("v", 1, ditHeads, m, ditHeadDim)
		if chunk <= 0 {
			chunk = n
		}
		var outs []*mlpackage.Var
		for lo := 0; lo < n; lo += chunk {
			q := qq
			if chunk < n {
				q = g.Slice(qq, 2, lo, lo+chunk)
			}
			s := g.MatMul(q, k, false, true)
			var e *mlpackage.Var
			if nomax {
				e = g.Unary("exp", s)
			} else {
				mx := g.Reduce("max", s, -1)
				if c != 0 {
					mx = g.AddScalar(mx, c)
				}
				e = g.Unary("exp", g.Sub(s, mx))
			}
			l := g.Reduce("sum", e, -1)
			if explicit {
				outs = append(outs, g.MatMul(g.Div(e, l), v, false, false))
			} else {
				outs = append(outs, g.Div(g.MatMul(e, v, false, false), l))
			}
		}
		if len(outs) == 1 {
			g.Output(outs[0])
		} else {
			g.Output(g.Concat(2, outs...))
		}
	})
}

// attnKeys is attention with the scores keys first, (1, heads, keys, queries): the softmax's
// reductions over axis 2.
func attnKeys(name string, chunk, m int) *Part {
	return part(name, func(g *graph) {
		gh, gw := Grid(1024, 1024)
		n := gh * gw
		qq := g.Input("q", 1, ditHeads, n, ditHeadDim)
		k := g.Input("k", 1, ditHeads, m, ditHeadDim)
		v := g.Input("v", 1, ditHeads, m, ditHeadDim)
		var outs []*mlpackage.Var
		for lo := 0; lo < n; lo += chunk {
			q := g.Slice(qq, 2, lo, lo+chunk)
			s := g.MatMul(k, q, false, true) // (1, h, m, chunk)
			e := g.Unary("exp", g.Sub(s, g.Reduce("max", s, 2)))
			l := g.Reduce("sum", e, 2)                 // (1, h, 1, chunk)
			o := g.MatMul(v, g.Div(e, l), true, false) // (1, h, d, chunk)
			outs = append(outs, g.Transpose(o, 0, 1, 3, 2))
		}
		g.Output(g.Concat(2, outs...))
	})
}

// attnBias is the cross-attention over m keys with a bias (1, 1, 1, m) added to the scores, q
// pre-scaled, all queries at once.
func attnBias(name string, m int) *Part {
	return part(name, func(g *graph) {
		gh, gw := Grid(1024, 1024)
		n := gh * gw
		q := g.Input("q", 1, ditHeads, n, ditHeadDim)
		k := g.Input("k", 1, ditHeads, m, ditHeadDim)
		v := g.Input("v", 1, ditHeads, m, ditHeadDim)
		b := g.Input("b", 1, 1, 1, m)
		g.Output(g.QChunkAttention(q, k, v, 1, b, 0))
	})
}

// variant is what a probe's block changes: qchunk as Blocks'; noSelf and noCross take out the
// attention (its values, or its queries, pass on); fusedQKV the self-attention's projections as
// one linear; foldScale the scale in the queries' norm (as ditBlock now does); mlpOnly the MLP
// alone; rows the text's rows with the scores' bias cbias (the cross-attention on fewer keys).
type variant struct {
	qchunk              int
	noSelf, noCross     bool
	fusedQKV, foldScale bool
	mlpOnly             bool
	// keepQK, with noSelf, passes v + q + k on, so that the queries' and keys' projections,
	// norms and RoPE stay; noRope, noHeadNorm, noHeadMax (the norm without MaxScaled) and plainLN
	// (the block's LayerNorms without MaxScaled) take those out; gelu is the MLP's ("" exact,
	// tanh, sigmoid, none)
	keepQK, noRope, noHeadNorm, noHeadMax, plainLN bool
	// absMax writes the norms' division by the row's largest magnitude with abs (one operation
	// for neg and maximum)
	absMax bool
	// absMul, with absMax, multiplies by the reciprocal of the row's largest magnitude in place
	// of dividing by it; ropeSign takes RoPE's negation into the sine's table
	absMul, ropeSign bool
	gelu             string
	rows             int
	cbias            *mlpackage.Var
}

// ditBlockV is ditBlock before design.md 16.201 (the scale on the scores), changed by vr.
func ditBlockV(g *graph, i int, x, act, temb, ctx, cos, sin *mlpackage.Var, vr variant) *mlpackage.Var {
	p := fmt.Sprintf("dit.blocks.%d.", i)
	E, H, D := DitHidden, ditHeads, ditHeadDim
	sc := 1 / math.Sqrt(float64(D))
	norm := func(name string) (*mlpackage.Var, *mlpackage.Var) {
		shift, scale1, gate := modulation(g, p+"adaln_modulation_"+name, act, temb)
		ln := g.ScaledLayerNorm(x, 1e-6)
		if vr.absMax {
			ones, zeros := make([]float32, E), make([]float32, E)
			for j := range ones {
				ones[j] = 1
			}
			ln = g.Op("layer_norm", map[string]any{"x": absScaled(g, x, vr.absMul), "axes": g.B.Ints(-1), "gamma": g.F16s("pln_ones", []int{E}, ones),
				"beta": g.F16s("pln_zeros", []int{E}, zeros), "epsilon": g.B.ScalarF16(1e-6)}, x.Shape...)
		}
		if vr.plainLN {
			ones, zeros := make([]float32, E), make([]float32, E)
			for j := range ones {
				ones[j] = 1
			}
			ln = g.Op("layer_norm", map[string]any{"x": x, "axes": g.B.Ints(-1), "gamma": g.F16s("pln_ones", []int{E}, ones),
				"beta": g.F16s("pln_zeros", []int{E}, zeros), "epsilon": g.B.ScalarF16(1e-6)}, x.Shape...)
		}
		return g.Add(g.Mul(ln, scale1), shift), gate
	}
	act1 := func(f *mlpackage.Var) *mlpackage.Var {
		switch vr.gelu {
		case "none":
			return f
		case "tanh":
			return g.Op("gelu", map[string]any{"x": f, "mode": g.Str("TANH_APPROXIMATION")}, f.Shape...)
		case "sigmoid":
			return g.Op("gelu", map[string]any{"x": f, "mode": g.Str("SIGMOID_APPROXIMATION")}, f.Shape...)
		}
		return g.Activation(f, "gelu")
	}
	head := func(y *mlpackage.Var, normName string, rope bool, s float32) *mlpackage.Var {
		y = g.Reshape(y, 1, y.Shape[1], H, D)
		if normName != "" && !vr.noHeadNorm {
			wn := append([]float32(nil), g.Get(normName, D)...)
			for j := range wn {
				wn[j] *= s
			}
			in := y
			if vr.absMax {
				in = absScaled(g, y, vr.absMul)
			} else if !vr.noHeadMax {
				in = g.MaxScaled(y, -1)
			}
			y = g.RMSNormWith(in, normName, wn, 1e-6, 1)
		}
		if rope && vr.ropeSign {
			half := D / 2
			x1, x2 := g.Slice(y, -1, 0, half), g.Slice(y, -1, half, D)
			sv := make([]float32, y.Shape[1]*D)
			_, sn := RopeValues(Grid(1024, 1024))
			for t := 0; t < y.Shape[1]; t++ {
				for j := 0; j < half; j++ {
					sv[t*D+j], sv[t*D+half+j] = -sn[t*D+j], sn[t*D+half+j]
				}
			}
			ssin := g.F16s("dit_rope_ssin", []int{1, y.Shape[1], 1, D}, sv)
			y = g.Add(g.Mul(y, cos), g.Mul(g.Concat(-1, x2, x1), ssin))
		} else if rope && !vr.noRope {
			y = g.ApplyRope(y, cos, sin)
		}
		return g.Transpose(y, 0, 2, 1, 3)
	}
	if vr.mlpOnly {
		h, gate := norm("mlp")
		f := g.Activation(g.Linear(h, p+"mlp.layer1.weight", "", ditMLP), "gelu")
		f = g.Linear(f, p+"mlp.layer2.weight", "", E)
		return g.Add(x, g.Mul(f, gate))
	}
	attnScale := sc
	var qs float32 = 1
	if vr.foldScale {
		attnScale, qs = 1, float32(sc)
	}
	h, gate := norm("self_attn")
	a := p + "self_attn."
	var q, k, v *mlpackage.Var
	if vr.fusedQKV {
		wq, wk, wv := g.Get(a+"q_proj.weight", E, E), g.Get(a+"k_proj.weight", E, E), g.Get(a+"v_proj.weight", E, E)
		cat := append(append(append([]float32(nil), wq...), wk...), wv...)
		wt := g.Matrix(a+"qkv_proj.weight", cat, 3*E, E)
		qkv := g.Op("linear", map[string]any{"x": h, "weight": wt}, 1, h.Shape[1], 3*E)
		q = head(g.Slice(qkv, -1, 0, E), a+"q_norm.weight", true, qs)
		k = head(g.Slice(qkv, -1, E, 2*E), a+"k_norm.weight", true, 1)
		v = head(g.Slice(qkv, -1, 2*E, 3*E), "", false, 1)
	} else {
		q = head(g.Linear(h, a+"q_proj.weight", "", E), a+"q_norm.weight", true, qs)
		k = head(g.Linear(h, a+"k_proj.weight", "", E), a+"k_norm.weight", true, 1)
		v = head(g.Linear(h, a+"v_proj.weight", "", E), "", false, 1)
	}
	var o *mlpackage.Var
	if vr.noSelf && vr.keepQK {
		o = g.Add(g.Add(v, q), k)
	} else if vr.noSelf {
		o = v
	} else {
		o = g.QChunkAttention(q, k, v, attnScale, nil, vr.qchunk)
	}
	o = g.Linear(g.Merge(o), a+"output_proj.weight", "", E)
	x = g.Add(x, g.Mul(o, gate))
	h, gate = norm("cross_attn")
	a = p + "cross_attn."
	q = head(g.Linear(h, a+"q_proj.weight", "", E), a+"q_norm.weight", false, qs)
	k = head(g.Linear(ctx, a+"k_proj.weight", "", E), a+"k_norm.weight", false, 1)
	v = head(g.Linear(ctx, a+"v_proj.weight", "", E), "", false, 1)
	if vr.noCross {
		o = q
	} else if vr.cbias != nil {
		o = g.QChunkAttention(q, k, v, attnScale, vr.cbias, 0)
	} else {
		o = g.QChunkAttention(q, k, v, attnScale, nil, vr.qchunk)
	}
	o = g.Linear(g.Merge(o), a+"output_proj.weight", "", E)
	x = g.Add(x, g.Mul(o, gate))
	h, gate = norm("mlp")
	f := act1(g.Linear(h, p+"mlp.layer1.weight", "", ditMLP))
	f = g.Linear(f, p+"mlp.layer2.weight", "", E)
	return g.Add(x, g.Mul(f, gate))
}

func probeBackend() string {
	if os.Getenv("HWNPU_ANIMA_PROBE_COREML") != "" {
		return llm.BackendCoreML
	}
	return llm.BackendCoreAI
}

func probeExt() string {
	if os.Getenv("HWNPU_ANIMA_PROBE_COREML") != "" {
		return ".mlpackage"
	}
	return ".aimodel"
}

// absScaled is milpart's MaxScaled with abs for the magnitude.
func absScaled(g *graph, x *mlpackage.Var, mul bool) *mlpackage.Var {
	m := g.Reduce("max", g.Unary("abs", x), -1)
	m = g.Op("maximum", map[string]any{"x": m, "y": g.B.ScalarF16(1e-3)}, m.Shape...)
	if mul {
		return g.Mul(x, g.Op("real_div", map[string]any{"x": g.B.ScalarF16(1), "y": m}, m.Shape...))
	}
	return g.Op("real_div", map[string]any{"x": x, "y": m}, x.Shape...)
}

// attnLayout is the self-attention of q (pre-scaled), k and v (1, heads, n, d) with its scores
// laid out as layout says, the heads one at a time with perHead, chunk queries at a time (0:
// all), the sum divided after P V with deferred:
//
//	w   (1, heads, queries, keys), the softmax over the last axis (as SDPAScale)
//	mm  the two products alone, no softmax (what the products cost)
//	c   (1, keys, 1, queries), the softmax over the channels (one head)
//	bc  (heads, keys, 1, queries), the heads as the batch, the softmax over the channels
//	q   (1, queries, 1, keys), the softmax over the last axis (one head)
func attnLayout(name, layout string, perHead bool, chunk int, deferred bool) *Part {
	return attnLayoutK(name, layout, perHead, chunk, deferred, 0)
}

// attnLayoutK is attnLayout over keys keys (0: the patches; TextRows: the cross-attention).
func attnLayoutK(name, layout string, perHead bool, chunk int, deferred bool, keys int) *Part {
	return part(name, func(g *graph) {
		gh, gw := Grid(1024, 1024)
		n, H, D := gh*gw, ditHeads, ditHeadDim
		if keys <= 0 {
			keys = n
		}
		q := g.Input("q", 1, H, n, D)
		k := g.Input("k", 1, H, keys, D)
		v := g.Input("v", 1, H, keys, D)
		if chunk <= 0 {
			chunk = n
		}
		softmax := func(s *mlpackage.Var, axis int) (e, l *mlpackage.Var) {
			e = g.Unary("exp", g.Sub(s, g.Reduce("max", s, axis)))
			return e, g.Reduce("sum", e, axis)
		}
		// one is the attention of qq (1, h, m, D) over kk and vv (1, h, n, D): (1, h, m, D)
		one := func(qq, kk, vv *mlpackage.Var) *mlpackage.Var {
			h, m := qq.Shape[1], qq.Shape[2]
			switch layout {
			case "w":
				s := g.MatMul(qq, kk, false, true)
				e, l := softmax(s, -1)
				if deferred {
					return g.Div(g.MatMul(e, vv, false, false), l)
				}
				return g.MatMul(g.Div(e, l), vv, false, false)
			case "mm":
				return g.MatMul(g.MatMul(qq, kk, false, true), vv, false, false)
			case "c", "bc": // keys on the channels
				s := g.MatMul(kk, qq, false, true) // (1, h, n, m)
				s = g.Reshape(s, h, n, 1, m)
				e, l := softmax(s, 1)
				p := e
				if !deferred {
					p = g.Div(e, l)
				}
				o := g.MatMul(vv, g.Reshape(p, 1, h, n, m), true, false) // (1, h, D, m)
				if deferred {
					o = g.Div(o, g.Reshape(l, 1, h, 1, m))
				}
				return g.Transpose(o, 0, 1, 3, 2)
			case "qm": // as q, the sum's reciprocal multiplied in place of the division
				s := g.Reshape(g.MatMul(qq, kk, false, true), 1, h*m, 1, keys)
				e, l := softmax(s, -1)
				r := g.Op("real_div", map[string]any{"x": g.B.ScalarF16(1), "y": l}, l.Shape...)
				return g.MatMul(g.Reshape(g.Mul(e, r), 1, h, m, keys), vv, false, false)
			case "q": // queries on the channels, keys last
				// with every head (h > 1), the heads and the queries together on the channels
				s := g.Reshape(g.MatMul(qq, kk, false, true), 1, h*m, 1, keys)
				e, l := softmax(s, -1)
				if deferred {
					return g.Div(g.MatMul(g.Reshape(e, 1, h, m, keys), vv, false, false), g.Reshape(l, 1, h, m, 1))
				}
				return g.MatMul(g.Reshape(g.Div(e, l), 1, h, m, keys), vv, false, false)
			}
			panic(layout)
		}
		var heads []*mlpackage.Var
		hs := H
		if !perHead {
			hs = 1
		}
		for h := 0; h < hs; h++ {
			qh, kh, vh := q, k, v
			if perHead {
				qh, kh, vh = g.Slice(q, 1, h, h+1), g.Slice(k, 1, h, h+1), g.Slice(v, 1, h, h+1)
			}
			var outs []*mlpackage.Var
			for lo := 0; lo < n; lo += chunk {
				qc := qh
				if chunk < n {
					qc = g.Slice(qh, 2, lo, lo+chunk)
				}
				outs = append(outs, one(qc, kh, vh))
			}
			o := outs[0]
			if len(outs) > 1 {
				o = g.Concat(2, outs...)
			}
			heads = append(heads, o)
		}
		if len(heads) == 1 {
			g.Output(heads[0])
		} else {
			g.Output(g.Concat(1, heads...))
		}
	})
}
