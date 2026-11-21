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
// -calls N before changing its export (docs/design/image-apple.md). HWNPU_ATTN_PROBE_OUT is where
// they go, HWNPU_ATTN_PROBE_SHAPE "heads,queries,keys,head_dim" (Qwen-Image at 1024 x 1024:
// 32,4096,4608,128; Chroma: 24,4608,4608,128), HWNPU_ATTN_PROBE the comma separated forms
// (default: all), each FORM or FORM:CHUNK (queries a chunk; default 512):
//
//	old    QChunkAttention, the scale on the scores, the bias added
//	fold   QChunkAttention at scale 1 (the scale in q), the bias added
//	head   HeadAttention, the bias added to the scores
//	headn  HeadAttention without the bias
//	whole  HeadAttention without the bias, the keys in one tensor
//	split  HeadAttention with the last 512 keys another kind (a text's), the bias on them
func TestWriteAttentionProbes(t *testing.T) {
	out := os.Getenv("HWNPU_ATTN_PROBE_OUT")
	if out == "" {
		t.Skip("HWNPU_ATTN_PROBE_OUT")
	}
	H, n, m, D := 32, 4096, 4608, 128
	if s := os.Getenv("HWNPU_ATTN_PROBE_SHAPE"); s != "" {
		if _, err := fmt.Sscanf(s, "%d,%d,%d,%d", &H, &n, &m, &D); err != nil {
			t.Fatal(err)
		}
	}
	forms := "old,fold,head,headn,split"
	if s := os.Getenv("HWNPU_ATTN_PROBE"); s != "" {
		forms = s
	}
	for _, f := range strings.Split(forms, ",") {
		form, chunk := f, 512
		if i := strings.IndexByte(f, ':'); i >= 0 {
			form = f[:i]
			fmt.Sscan(f[i+1:], &chunk)
		}
		name := fmt.Sprintf("attn_%s_%d_h%d_q%d_k%d", form, chunk, H, n, m)
		p := &Part{Name: name, Model: "probe", Body: func(g *Graph) {
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
			case "split":
				g.Output(g.HeadAttention(q, k, v, HeadOptions{Chunk: chunk, Split: m - 512, Bias: g.Slice(bias, -1, m-512, m)}))
			default:
				g.Fail("form %q", form)
			}
		}}
		path := filepath.Join(out, name+".aimodel")
		if _, err := llm.WritePart(path, p, noWeights{}, llm.Options{Backend: llm.BackendCoreAI}, nil); err != nil {
			t.Fatalf("%s: %v", f, err)
		}
		t.Logf("wrote %s", path)
	}
}
