package experts

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/mxfp4"
)

// Host's experts of gpt-oss-20b's layer 0 (HWNPU_GPTOSS_CKPT: the checkpoint's directory) against
// the experts dequantized in float64 (transformers' convert_moe_packed_tensors and GptOssExperts).
func TestHostGptOss(t *testing.T) {
	dir := os.Getenv("HWNPU_GPTOSS_CKPT")
	if dir == "" {
		t.Skip("HWNPU_GPTOSS_CKPT: gpt-oss-20b's checkpoint")
	}
	sh := Shape{Hidden: 2880, Inter: 2880, Experts: 32, TopK: 4}
	rec := &HostRecord{Checkpoint: filepath.Join(dir, "model.safetensors.index.json"), Act: Act{Kind: ActGptOss, Alpha: 1.702, Limit: 7}, Layers: map[int]MXFP4{}}
	for _, l := range []int{0, 23} {
		p := fmt.Sprintf("model.layers.%d.mlp.experts.", l)
		rec.Layers[l] = MXFP4{p + "gate_up_proj_blocks", p + "gate_up_proj_scales", p + "gate_up_proj_bias", p + "down_proj_blocks", p + "down_proj_scales", p + "down_proj_bias"}
	}
	h, err := OpenHost(rec, sh)
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	r := rand.New(rand.NewPCG(3, 4))
	H, I := sh.Hidden, sh.Inter
	rows := make([][]float32, 3)
	for i := range rows {
		rows[i] = make([]float32, H)
		for j := range rows[i] {
			rows[i][j] = float32(r.NormFloat64())
		}
	}
	idx := [][]int{{1, 5, 9, 30}, {5, 2, 1, 7}, {31, 0, 9, 5}}
	w := [][]float32{{0.4, 0.3, 0.2, 0.1}, {0.25, 0.25, 0.25, 0.25}, {0.7, 0.1, 0.1, 0.1}}
	for _, layer := range []int{0, 23} {
		out := make([][]float32, len(rows))
		for i := range out {
			out[i] = make([]float32, H)
		}
		start := time.Now()
		if err := h.Run(layer, rows, idx, w, out); err != nil {
			t.Fatal(err)
		}
		t.Logf("layer %d: %d rows in %v", layer, len(rows), time.Since(start))
		hl := h.layers[layer]
		row := make([]float32, H)
		for i, x := range rows {
			want := make([]float64, H)
			for k, e := range idx[i] {
				act := make([]float32, I)
				for j := range I {
					var g, u float64
					for _, c := range []struct {
						o *float64
						r int
					}{{&g, 2 * j}, {&u, 2*j + 1}} {
						rr := e*2*I + c.r
						mxfp4.Dequantize(row, hl.gw[rr*H/2:(rr+1)*H/2], hl.ge[rr*H/32:(rr+1)*H/32])
						for m := range H {
							*c.o += float64(row[m]) * float64(x[m])
						}
						*c.o += float64(hl.gb[rr])
					}
					g = math.Min(g, 7)
					u = math.Max(math.Min(u, 7), -7)
					act[j] = float32(g / (1 + math.Exp(-1.702*g)) * (u + 1))
				}
				for c := range H {
					rr := e*H + c
					mxfp4.Dequantize(row[:I], hl.dw[rr*I/2:(rr+1)*I/2], hl.de[rr*I/32:(rr+1)*I/32])
					var y float64
					for m := range I {
						y += float64(row[m]) * float64(act[m])
					}
					want[c] += float64(w[i][k]) * (y + float64(hl.db[rr]))
				}
			}
			var dot, na, nb float64
			for c := range H {
				dot += want[c] * float64(out[i][c])
				na += want[c] * want[c]
				nb += float64(out[i][c]) * float64(out[i][c])
			}
			cos := dot / math.Sqrt(na*nb)
			t.Logf("layer %d row %d: cos %.6f, norms %.3f %.3f", layer, i, cos, math.Sqrt(na), math.Sqrt(nb))
			if cos < 0.999 {
				t.Errorf("layer %d row %d: cos %.6f", layer, i, cos)
			}
		}
	}
}
