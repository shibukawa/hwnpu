package milpart

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

// TestWriteAttentionProbes writes attentions alone (no weights: they compile in seconds) as Core
// AI programs, to time the forms of an image DiT's attention at a model's shape with aicheck run
// -calls N before changing its export (docs/design/image-apple.md). HWNPU_TEST_ATTN_PROBE_OUT is where
// they go, HWNPU_TEST_ATTN_PROBE_SHAPE "heads,queries,keys,head_dim" (Qwen-Image at 1024 x 1024:
// 32,4096,4608,128; Chroma: 24,4608,4608,128), HWNPU_TEST_ATTN_PROBE the comma separated forms
// (default: all), each FORM or FORM:CHUNK (queries a chunk; default 512):
//
//	old    QChunkAttention, the scale on the scores, the bias added
//	fold   QChunkAttention at scale 1 (the scale in q), the bias added
//	head   HeadAttention, the bias added to the scores
//	headn  HeadAttention without the bias
//	whole  HeadAttention without the bias, the keys in one tensor
//	split  HeadAttention with the last 512 keys another kind (a text's), the bias on them
//	piece  an attention over these keys alone (a part of an attention's keys, in a program of its
//	       own): its output, softmax over these keys, and each row's log-sum-exp (lse, (1, H,
//	       queries, 1)), which join puts together with the other parts'
//	pieces PARTS pieces and their join (joinSoftmax) in one program; kpieces the pieces of KeyParts
//	join   (joinmm: and a projection after, as the next program's start) the output of PARTS pieces (HWNPU_TEST_ATTN_PROBE_PARTS, default 2) of the queries
//	       and head_dim of SHAPE: each part's output weighted by exp(lse - the parts' largest)
func TestWriteAttentionProbes(t *testing.T) {
	out := os.Getenv("HWNPU_TEST_ATTN_PROBE_OUT")
	if out == "" {
		t.Skip("HWNPU_TEST_ATTN_PROBE_OUT")
	}
	H, n, m, D := 32, 4096, 4608, 128
	if s := os.Getenv("HWNPU_TEST_ATTN_PROBE_SHAPE"); s != "" {
		if _, err := fmt.Sscanf(s, "%d,%d,%d,%d", &H, &n, &m, &D); err != nil {
			t.Fatal(err)
		}
	}
	forms := "old,fold,head,headn,split"
	if s := os.Getenv("HWNPU_TEST_ATTN_PROBE"); s != "" {
		forms = s
	}
	for _, f := range strings.Split(forms, ",") {
		form, chunk := f, 512
		if i := strings.IndexByte(f, ':'); i >= 0 {
			form = f[:i]
			fmt.Sscan(f[i+1:], &chunk)
		}
		name := fmt.Sprintf("attn_%s_%d_h%d_q%d_k%d", form, chunk, H, n, m)
		parts := 2
		if s := os.Getenv("HWNPU_TEST_ATTN_PROBE_PARTS"); s != "" {
			fmt.Sscan(s, &parts)
		}
		if form == "join" || form == "joinmm" {
			name = fmt.Sprintf("attn_%s%d_h%d_q%d_d%d", form, parts, H, n, D)
		}
		p := &Part{Name: name, Model: "probe", Body: func(g *Graph) {
			if form == "join" || form == "joinmm" {
				var os_, ls []*mlpackage.Var
				for i := 0; i < parts; i++ {
					os_ = append(os_, g.Input(fmt.Sprintf("o%d", i), 1, H, n, D))
					ls = append(ls, g.Input(fmt.Sprintf("lse%d", i), 1, H, n, 1))
				}
				var x *mlpackage.Var
				switch os.Getenv("HWNPU_TEST_ATTN_PROBE_JOIN") {
				case "two":
					x = joinTwo(g, os_[0], os_[1], ls[0], ls[1])
				case "softmax":
					x = joinSoftmax(g, os_, ls)
				default:
					x = joinPieces(g, os_, ls)
				}
				if form == "joinmm" {
					// what the program after the attention does first: the heads side by side
					// (1, 1, queries, H x D) into a projection, its weight an input (as the output's)
					x = g.Reshape(g.Transpose(x, 0, 2, 1, 3), 1, 1, n, H*D)
					x = g.MatMul(x, g.Input("w", 1, 1, H*D, H*D), false, false)
				}
				g.Output(x)
				return
			}
			q := g.Input("q", 1, H, n, D)
			k := g.Input("k", 1, H, m, D)
			v := g.Input("v", 1, H, m, D)
			var bias *mlpackage.Var
			if form != "headn" && form != "whole" {
				bias = g.Input("kbias", 1, 1, 1, m)
			}
			switch form {
			case "old":
				g.Output(g.QChunkAttention(q, k, v, 1/math.Sqrt(float64(D)), bias, chunk))
			case "fold":
				g.Output(g.QChunkAttention(q, k, v, 1, bias, chunk))
			case "head", "headn":
				g.Output(g.HeadAttention(q, k, v, HeadOptions{Chunk: chunk, Bias: bias}))
			case "whole":
				g.Output(g.HeadAttention(q, k, v, HeadOptions{Chunk: chunk, Whole: true}))
			case "pieces", "kpieces":
				// PARTS pieces' attentions and their join in one program (kpieces: the pieces
				// KeyParts cuts)
				cuts := KeyParts(0, m)
				if form == "pieces" {
					cuts = nil
					for i := 0; i < parts; i++ {
						cuts = append(cuts, [2]int{i * m / parts, (i + 1) * m / parts})
					}
				}
				var os_, ls []*mlpackage.Var
				for _, c := range cuts {
					lo, hi := c[0], c[1]
					o, l := pieceAttention(g, q, g.Slice(k, 2, lo, hi), g.Slice(v, 2, lo, hi), chunk)
					os_, ls = append(os_, o), append(ls, l)
				}
				g.Output(joinSoftmax(g, os_, ls))
			case "piece":
				o, lse := pieceAttention(g, q, k, v, chunk)
				g.Output(g.Concat(-1, o, lse)) // (1, H, queries, D + 1): a part has one output
			case "split":
				g.Output(g.HeadAttention(q, k, v, HeadOptions{Chunk: chunk, Split: m - 512, Bias: g.Slice(bias, -1, m-512, m)}))
			default:
				g.Fail("form %q", form)
			}
		}}
		// HWNPU_TEST_ATTN_PROBE_BACKEND=coreml writes Core ML packages, for aicheck run -cpu (milcpu)
		backend, ext := llm.BackendCoreAI, ".aimodel"
		if os.Getenv("HWNPU_TEST_ATTN_PROBE_BACKEND") == "coreml" {
			backend, ext = llm.BackendCoreML, ".mlpackage"
		}
		path := filepath.Join(out, name+ext)
		if _, err := llm.WritePart(path, p, noWeights{}, llm.Options{Backend: backend}, nil); err != nil {
			t.Fatalf("%s: %v", f, err)
		}
		t.Logf("wrote %s", path)
	}
}

// pieceAttention is an attention over k's keys alone, a head at a time as HeadAttention's (scores
// (1, queries, 1, keys)), queries chunk at a time: its output (1, H, queries, D), softmax over these
// keys, and each row's log-sum-exp (1, H, queries, 1), for joinPieces.
func pieceAttention(g *Graph, q, k, v *mlpackage.Var, chunk int) (*mlpackage.Var, *mlpackage.Var) {
	heads, n, d := q.Shape[1], q.Shape[2], q.Shape[3]
	m := k.Shape[2]
	if chunk <= 0 || chunk > n {
		chunk = n
	}
	q = g.Scale(q, float32(1/math.Sqrt(float64(d))))
	var outs, lses []*mlpackage.Var
	for h := 0; h < heads; h++ {
		qh, kh, vh := g.Slice(q, 1, h, h+1), g.Slice(k, 1, h, h+1), g.Slice(v, 1, h, h+1)
		var rows, ls []*mlpackage.Var
		for lo := 0; lo < n; lo += chunk {
			hi := min(lo+chunk, n)
			c := hi - lo
			s := g.Reshape(g.MatMul(g.Slice(qh, 2, lo, hi), kh, false, true), 1, c, 1, m)
			mx := g.Reduce("max", s, -1)
			e := g.Unary("exp", g.Sub(s, mx))
			sum := g.Reduce("sum", e, -1)
			rows = append(rows, g.MatMul(g.Reshape(g.Div(e, sum), 1, 1, c, m), vh, false, false))
			ls = append(ls, g.Reshape(g.Add(mx, g.Unary("log", sum)), 1, 1, c, 1))
		}
		outs, lses = append(outs, g.Concat(2, rows...)), append(lses, g.Concat(2, ls...))
	}
	return g.Concat(1, outs...), g.Concat(1, lses...)
}

// joinPieces is the attention whose keys pieceAttention took in parts: each part's output o[i]
// weighted by exp(lse[i] - the largest lse), over the weights' sum.
func joinPieces(g *Graph, o, lse []*mlpackage.Var) *mlpackage.Var {
	top := lse[0]
	for _, l := range lse[1:] {
		top = g.Op("maximum", map[string]any{"x": top, "y": l}, top.Shape...)
	}
	var num, den *mlpackage.Var
	for i := range o {
		w := g.Unary("exp", g.Sub(lse[i], top))
		x := g.Mul(o[i], w)
		if num == nil {
			num, den = x, w
		} else {
			num, den = g.Add(num, x), g.Add(den, w)
		}
	}
	return g.Div(num, den)
}

// joinTwo is joinPieces of two parts without the largest lse taken off (a maximum, then a
// subtraction, which the Neural Engine's compiler fuses into an elementwise layer it cannot
// make: "NEFUSED_EW: KernelPreScales (1,-1)"): o0 + (o1 - o0) / (1 + exp(lse0 - lse1)), the second
// part's share of the weights. exp overflowing fp16 takes the share to 0, as it should.
func joinTwo(g *Graph, o0, o1, l0, l1 *mlpackage.Var) *mlpackage.Var {
	e := g.Unary("exp", g.Sub(l0, l1))
	w := g.Op("real_div", map[string]any{"x": g.B.ScalarF16(1), "y": g.Add(e, g.B.ScalarF16(1))}, e.Shape...)
	return g.Add(o0, g.Mul(g.Sub(o1, o0), w))
}

// joinSoftmax is joinPieces with the parts' weights a softmax over their lse side by side ((1, H,
// queries, parts), its maximum taken off as the attention's own softmax does, which the Neural
// Engine takes), and the output their weighted sum: only additions of like tensors, no
// subtraction of two of one shape.
func joinSoftmax(g *Graph, o, lse []*mlpackage.Var) *mlpackage.Var {
	l := g.Concat(-1, lse...)
	e := g.Unary("exp", g.Sub(l, g.Reduce("max", l, -1)))
	w := g.Div(e, g.Reduce("sum", e, -1))
	var out *mlpackage.Var
	for i := range o {
		x := g.Mul(o[i], g.Slice(w, -1, i, i+1))
		if out == nil {
			out = x
		} else {
			out = g.Add(out, x)
		}
	}
	return out
}
