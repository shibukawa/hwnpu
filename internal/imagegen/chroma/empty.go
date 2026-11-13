package chroma

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu/internal/imagegen/chroma/mil"
)

// EmptyFile is the empty prompt's text rows in an export (text/empty.bin): the rows of </s> and
// of the first pad through T5 and context_embedder, float32, (2, hidden), not divided by
// mil.DitScale. Chroma1-HD's default negative prompt is empty: its conditioning is read, and T5
// runs for the prompt alone.
const EmptyFile = "text/empty.bin"

// emptyRows computes EmptyFile's rows in float32 on the host. With one token (</s>), each row's
// attention has one key: both rows' attention output is the value of </s>, and the position bias
// (one key) does not matter.
func emptyRows(w *mil.Weights, tc mil.T5, dc mil.Dit) ([]float32, error) {
	E := tc.Hidden
	x := make([][]float32, 2)
	for i, id := range []int{1, 0} { // </s>, <pad>
		r, err := w.Rows("t5.shared.weight", id, id+1)
		if err != nil {
			return nil, err
		}
		x[i] = append([]float32(nil), r...)
	}
	get := func(name string) ([]float32, error) {
		v, _, err := w.Get(name)
		return v, err
	}
	norm := func(v, wt []float32) []float32 {
		var ss float64
		for _, a := range v {
			ss += float64(a) * float64(a)
		}
		r := 1 / math.Sqrt(ss/float64(len(v))+tc.Eps)
		out := make([]float32, len(v))
		for i, a := range v {
			out[i] = float32(float64(a)*r) * wt[i]
		}
		return out
	}
	for l := 0; l < tc.Layers; l++ {
		p := fmt.Sprintf("t5.encoder.block.%d.layer.", l)
		ln0, err := get(p + "0.layer_norm.weight")
		if err != nil {
			return nil, err
		}
		wv, err := get(p + "0.SelfAttention.v.weight")
		if err != nil {
			return nil, err
		}
		wo, err := get(p + "0.SelfAttention.o.weight")
		if err != nil {
			return nil, err
		}
		o := matVec(wo, matVec(wv, norm(x[0], ln0), E), E)
		for i := range x {
			add(x[i], o)
		}
		ln1, err := get(p + "1.layer_norm.weight")
		if err != nil {
			return nil, err
		}
		f := p + "1.DenseReluDense."
		wi0, err := get(f + "wi_0.weight")
		if err != nil {
			return nil, err
		}
		wi1, err := get(f + "wi_1.weight")
		if err != nil {
			return nil, err
		}
		wout, err := get(f + "wo.weight")
		if err != nil {
			return nil, err
		}
		for i := range x {
			h := norm(x[i], ln1)
			g, u := matVec(wi0, h, tc.MLP), matVec(wi1, h, tc.MLP)
			for j := range g {
				a := float64(g[j])
				g[j] = float32(0.5*a*(1+math.Tanh(math.Sqrt(2/math.Pi)*(a+0.044715*a*a*a)))) * u[j]
			}
			add(x[i], matVec(wout, g, E))
		}
	}
	fn, err := get("t5.encoder.final_layer_norm.weight")
	if err != nil {
		return nil, err
	}
	ce, err := get("dit.context_embedder.weight")
	if err != nil {
		return nil, err
	}
	cb, err := get("dit.context_embedder.bias")
	if err != nil {
		return nil, err
	}
	out := make([]float32, 0, 2*dc.Hidden)
	for i := range x {
		y := matVec(ce, norm(x[i], fn), dc.Hidden)
		add(y, cb)
		out = append(out, y...)
	}
	return out, nil
}

// matVec is w (rows x len(v)) times v, summed in float64.
func matVec(w, v []float32, rows int) []float32 {
	n := len(v)
	out := make([]float32, rows)
	for r := 0; r < rows; r++ {
		var s float64
		for i, a := range w[r*n : (r+1)*n] {
			s += float64(a) * float64(v[i])
		}
		out[r] = float32(s)
	}
	return out
}

func add(x, y []float32) {
	for i := range x {
		x[i] += y[i]
	}
}

func writeEmpty(out string, rows []float32) error {
	b := make([]byte, 4*len(rows))
	for i, v := range rows {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
	}
	return os.WriteFile(filepath.Join(out, EmptyFile), b, 0o644)
}

func readEmpty(dir string, hidden int) ([]float32, error) {
	b, err := os.ReadFile(filepath.Join(dir, EmptyFile))
	if err != nil {
		return nil, err
	}
	if len(b) != 4*2*hidden {
		return nil, fmt.Errorf("chroma: %s is %d bytes, not 2 rows of %d", EmptyFile, len(b), hidden)
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out, nil
}
