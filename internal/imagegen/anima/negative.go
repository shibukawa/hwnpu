package anima

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu/internal/imagegen/anima/mil"
)

// EmptyCondFile is the empty prompt's conditioning in an export's text/: its one row (the
// adapter's hidden size) as float32, little endian. Classifier-free guidance takes it for an empty
// negative prompt.
//
// Diffusers runs an empty prompt as one Qwen3 token that is masked: its hidden state is zeroed
// before the adapter, and the T5 tokens are </s> alone. The adapter's cross-attention has no
// biases, so on a zero source its keys and values are zero and it adds nothing; its
// self-attention over one row is that row's value. What is left is a computation of one row on
// the weights alone, the same for every picture, which the export holds instead of the text
// graph running it (the graph runs Qwen3 on its tokens and has no way to mask them all).
const EmptyCondFile = "empty_cond.bin"

// t5EOS is the T5 tokenizer's </s>.
const t5EOS = 1

// EmptyCond computes the empty prompt's conditioning row from the checkpoint (mil.Weights'
// names), as AnimaTextConditioner does.
func EmptyCond(w *mil.Weights) ([]float32, error) {
	x, err := w.Rows("dit.llm_adapter.embed.weight", t5EOS, t5EOS+1)
	if err != nil {
		return nil, err
	}
	E := len(x)
	get := func(name string) ([]float32, error) {
		v, _, err := w.Get(name)
		return v, err
	}
	for i := 0; ; i++ {
		p := fmt.Sprintf("dit.llm_adapter.blocks.%d.", i)
		ns, err := get(p + "norm_self_attn.weight")
		if err != nil {
			if i == 0 {
				return nil, err
			}
			break // past the last block
		}
		var vw, ow, nm, m0, m0b, m2, m2b []float32
		for _, t := range []struct {
			dst  *[]float32
			name string
		}{{&vw, "self_attn.v_proj.weight"}, {&ow, "self_attn.o_proj.weight"}, {&nm, "norm_mlp.weight"},
			{&m0, "mlp.0.weight"}, {&m0b, "mlp.0.bias"}, {&m2, "mlp.2.weight"}, {&m2b, "mlp.2.bias"}} {
			if *t.dst, err = get(p + t.name); err != nil {
				return nil, err
			}
		}
		// self-attention over the one row: softmax of one score is 1, the output its value
		x = add(x, linear(ow, linear(vw, rmsNorm(x, ns, 1e-6), nil), nil))
		// the cross-attention adds nothing (above); the MLP
		h := linear(m0, rmsNorm(x, nm, 1e-6), m0b)
		for j, v := range h {
			h[j] = float32(0.5 * float64(v) * (1 + math.Erf(float64(v)/math.Sqrt2))) // nn.GELU()
		}
		x = add(x, linear(m2, h, m2b))
	}
	op, err := get("dit.llm_adapter.out_proj.weight")
	if err != nil {
		return nil, err
	}
	ob, err := get("dit.llm_adapter.out_proj.bias")
	if err != nil {
		return nil, err
	}
	n, err := get("dit.llm_adapter.norm.weight")
	if err != nil {
		return nil, err
	}
	out := rmsNorm(linear(op, x, ob), n, 1e-6)
	if len(out) != E {
		return nil, fmt.Errorf("anima: the adapter gives %d values a row, not %d", len(out), E)
	}
	return out, nil
}

// linear is w (out x len(x), row major) times x, plus b when not nil, in float64 sums.
func linear(w, x, b []float32) []float32 {
	n := len(x)
	out := make([]float32, len(w)/n)
	for i := range out {
		var s float64
		for j, r := range w[i*n : (i+1)*n] {
			s += float64(r) * float64(x[j])
		}
		if b != nil {
			s += float64(b[i])
		}
		out[i] = float32(s)
	}
	return out
}

func rmsNorm(x, w []float32, eps float64) []float32 {
	var s float64
	for _, v := range x {
		s += float64(v) * float64(v)
	}
	r := 1 / math.Sqrt(s/float64(len(x))+eps)
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = float32(float64(v) * r * float64(w[i]))
	}
	return out
}

func add(x, y []float32) []float32 {
	for i := range x {
		x[i] += y[i]
	}
	return x
}

// WriteEmptyCond writes the row into dir (an export's text/) as EmptyCondFile.
func WriteEmptyCond(dir string, row []float32) error {
	b := make([]byte, 4*len(row))
	for i, v := range row {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
	}
	return os.WriteFile(filepath.Join(dir, EmptyCondFile), b, 0o644)
}

// errNoEmptyCond is the error of an export made before EmptyCondFile.
var errNoEmptyCond = errors.New("anima: this export has no conditioning of an empty negative prompt (text/" + EmptyCondFile +
	"): give a negative prompt, or export it again")

// emptyCond is the conditioning of an empty negative prompt: EmptyCondFile's row, then zeros to
// Text.Rows rows (the adapter's padding to 512).
func (p *Pipeline) emptyCond() ([]float32, error) {
	b, err := os.ReadFile(filepath.Join(p.dir, "text", EmptyCondFile))
	if errors.Is(err, os.ErrNotExist) {
		return nil, errNoEmptyCond
	}
	if err != nil {
		return nil, err
	}
	E := p.Text.Hidden
	if len(b) != 4*E {
		return nil, fmt.Errorf("anima: %s holds %d bytes, not one row of %d float32", EmptyCondFile, len(b), E)
	}
	out := make([]float32, p.Text.Rows*E)
	for i := range E {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out, nil
}
