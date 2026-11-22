package experts

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/mxfp4"
	"github.com/shibukawa/hwnpu/internal/safetensors"
)

// Host's experts of gpt-oss-20b's layer 0 (HWNPU_TEST_GPTOSS_CKPT_DIR: the checkpoint's directory) against
// the experts dequantized in float64 (transformers' convert_moe_packed_tensors and GptOssExperts).
func TestHostGptOss(t *testing.T) {
	dir := os.Getenv("HWNPU_TEST_GPTOSS_CKPT_DIR")
	if dir == "" {
		t.Skip("HWNPU_TEST_GPTOSS_CKPT_DIR: gpt-oss-20b's checkpoint")
	}
	sh := Shape{Hidden: 2880, Inter: 2880, Experts: 32, TopK: 4}
	rec := &HostRecord{Checkpoint: filepath.Join(dir, "model.safetensors.index.json"), Act: Act{Kind: ActGptOss, Alpha: 1.702, Limit: 7}, Layers: map[int]MXFP4{}}
	for _, l := range []int{0, 23} {
		p := fmt.Sprintf("model.layers.%d.mlp.experts.", l)
		rec.Layers[l] = MXFP4{GateUpBlocks: p + "gate_up_proj_blocks", GateUpScales: p + "gate_up_proj_scales", GateUpBias: p + "gate_up_proj_bias", DownBlocks: p + "down_proj_blocks", DownScales: p + "down_proj_scales", DownBias: p + "down_proj_bias"}
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

// writeHost writes a small checkpoint of MXFP4 experts (random blocks, scales around 2^-6) of L
// layers, with biases or without, and returns its record.
func writeHost(t *testing.T, sh Shape, L int, act Act, bias bool) *HostRecord {
	t.Helper()
	r := rand.New(rand.NewPCG(7, 8))
	E, H, I := sh.Experts, sh.Hidden, sh.Inter
	path := filepath.Join(t.TempDir(), "model.safetensors")
	rec := &HostRecord{Checkpoint: path, Act: act, Layers: map[int]MXFP4{}}
	var outs []safetensors.Out
	add := func(name, dtype string, shape []int, b []byte) {
		outs = append(outs, safetensors.Out{Name: name, DType: dtype, Shape: shape, Size: int64(len(b)), Data: bytes.NewReader(b)})
	}
	bytesOf := func(n int, scale bool) []byte {
		b := make([]byte, n)
		for i := range b {
			if scale {
				b[i] = byte(119 + r.IntN(4))
			} else {
				b[i] = byte(r.IntN(256))
			}
		}
		return b
	}
	floats := func(n int) []byte {
		b := make([]byte, 4*n)
		for i := range n {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(float32(r.NormFloat64())*0.1))
		}
		return b
	}
	for l := range L {
		p := fmt.Sprintf("layers.%d.", l)
		m := MXFP4{GateUpBlocks: p + "gu_blocks", GateUpScales: p + "gu_scales", DownBlocks: p + "dn_blocks", DownScales: p + "dn_scales"}
		add(m.GateUpBlocks, "U8", []int{E, 2 * I, H / 32, 16}, bytesOf(E*2*I*H/2, false))
		add(m.GateUpScales, "U8", []int{E, 2 * I, H / 32}, bytesOf(E*2*I*H/32, true))
		add(m.DownBlocks, "U8", []int{E, H, I / 32, 16}, bytesOf(E*H*I/2, false))
		add(m.DownScales, "U8", []int{E, H, I / 32}, bytesOf(E*H*I/32, true))
		if bias {
			m.GateUpBias, m.DownBias = p+"gu_bias", p+"dn_bias"
			add(m.GateUpBias, "F32", []int{E, 2 * I}, floats(E*2*I))
			add(m.DownBias, "F32", []int{E, H}, floats(E*H))
		}
		rec.Layers[l] = m
	}
	if err := safetensors.Write(path, outs, nil); err != nil {
		t.Fatal(err)
	}
	return rec
}

// A generated token's experts as one piece of work each (runRow) give bit for bit what the rows'
// form (three Runs over all the experts) gives for the one row, for gpt-oss's gated unit with
// biases and for the SwiGLU without, wired or read from the file as chosen.
func TestHostRowMatchesRows(t *testing.T) {
	sh := Shape{Hidden: 128, Inter: 64, Experts: 16, TopK: 4}
	for _, c := range []struct {
		name string
		act  Act
		bias bool
		wire string
	}{{"gptoss", Act{Kind: ActGptOss, Alpha: 1.702, Limit: 7}, true, "1"}, {"silu", Act{Kind: ActSiLU}, false, "1"}, {"silu not wired", Act{Kind: ActSiLU}, false, "0"}} {
		t.Run(c.name, func(t *testing.T) {
			rec := writeHost(t, sh, 2, c.act, c.bias)
			t.Setenv("HWNPU_MOE_WIRE", c.wire)
			run := func(debug string) [][]float32 {
				t.Setenv("HWNPU_DEV", debug)
				h, err := OpenHost(rec, sh)
				if err != nil {
					t.Fatal(err)
				}
				defer h.Close()
				r := rand.New(rand.NewPCG(1, 2))
				x := [][]float32{make([]float32, sh.Hidden)}
				for j := range x[0] {
					x[0][j] = float32(r.NormFloat64())
				}
				var outs [][]float32
				for l := range 2 {
					out := [][]float32{make([]float32, sh.Hidden)}
					if err := h.Run(l, x, [][]int{{9, 2, 15, 4}}, [][]float32{{0.4, 0.3, 0.2, 0.1}}, out); err != nil {
						t.Fatal(err)
					}
					outs = append(outs, out[0])
				}
				return outs
			}
			row, rows := run(""), run("moe.row=0")
			var nonzero bool
			for l := range row {
				for j := range row[l] {
					if row[l][j] != rows[l][j] {
						t.Fatalf("layer %d value %d: %g as a row, %g as rows", l, j, row[l][j], rows[l][j])
					}
					nonzero = nonzero || row[l][j] != 0
				}
			}
			if !nonzero {
				t.Fatal("every output is 0")
			}
		})
	}
}

// Host.Run on a small Host of random weights (no checkpoint) against the experts dequantized in
// float64, with one row (a generated token) and with enough rows on an expert for the kernels of
// 4 and 8 vectors.
func TestHostRunSmall(t *testing.T) {
	const L, E, K = 1, 4, 2
	h := fakeHost(L, E)
	h.act = Act{Kind: ActGptOss, Alpha: 1.702, Limit: 7}
	H, I := h.shape.Hidden, h.shape.Inter
	r := rand.New(rand.NewPCG(21, 22))
	hl := h.layers[0]
	for _, b := range [][]byte{hl.gw, hl.dw} {
		for i := range b {
			b[i] = byte(r.IntN(256))
		}
	}
	for _, b := range [][]byte{hl.ge, hl.de} {
		for i := range b {
			b[i] = byte(118 + r.IntN(6))
		}
	}
	hl.gb, hl.db = make([]float32, E*2*I), make([]float32, E*H)
	for _, b := range [][]float32{hl.gb, hl.db} {
		for i := range b {
			b[i] = float32(r.NormFloat64()) * 0.1
		}
	}
	for _, n := range []int{1, 3, 13, 21} {
		rows, idx, w, out := make([][]float32, n), make([][]int, n), make([][]float32, n), make([][]float32, n)
		for i := range rows {
			rows[i], out[i] = make([]float32, H), make([]float32, H)
			for j := range rows[i] {
				rows[i][j] = float32(r.NormFloat64())
			}
			idx[i], w[i] = r.Perm(E)[:K], []float32{0.7, 0.3}
		}
		if err := h.Run(0, rows, idx, w, out); err != nil {
			t.Fatal(err)
		}
		row := make([]float32, H)
		for i, x := range rows {
			want := make([]float64, H)
			for k, e := range idx[i] {
				act := make([]float32, I)
				for j := range I {
					var gu [2]float64
					for c := range gu {
						rr := e*2*I + 2*j + c
						mxfp4.Dequantize(row, hl.gw[rr*H/2:(rr+1)*H/2], hl.ge[rr*H/32:(rr+1)*H/32])
						for m := range H {
							gu[c] += float64(row[m]) * float64(x[m])
						}
						gu[c] += float64(hl.gb[rr])
					}
					g, u := math.Min(gu[0], 7), math.Max(math.Min(gu[1], 7), -7)
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
			if cos := dot / math.Sqrt(na*nb); cos < 0.999 || math.Abs(math.Sqrt(nb/na)-1) > 0.02 {
				t.Errorf("%d rows, row %d: cos %.6f, norms %.3f %.3f", n, i, cos, math.Sqrt(na), math.Sqrt(nb))
			}
		}
	}
}
