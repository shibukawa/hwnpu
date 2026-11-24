package flux3klein

import (
	"fmt"
	"io"
	"math"
	"strings"
	"time"

	"github.com/shibukawa/bonsaiai/internal/imagegen/flux3klein/mil"
	"github.com/shibukawa/bonsaiai/internal/palette"
	"github.com/shibukawa/bonsaiai/internal/safetensors"
)

// CodeIters are the passes of weighted Lloyd a table takes (tools/coreai/lutfit's).
const CodeIters = 20

// ditMatrices are the DiT's matrices the programs read in lutN (Weights' names, parts included):
// the text's and the image's mode blocks and the single blocks.
func ditMatrices(c mil.Dit) []string {
	var prefixes []string
	for i := 0; i < c.Modes; i++ {
		prefixes = append(prefixes, fmt.Sprintf("dit.txt_mode_blocks.%d.", i), fmt.Sprintf("dit.mode_blocks.image.%d.", i))
	}
	for i := 0; i < c.Singles; i++ {
		prefixes = append(prefixes, fmt.Sprintf("dit.single_transformer_blocks.%d.", i))
	}
	var out []string
	for _, p := range prefixes {
		for _, n := range []string{"attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight", "attn.to_out.0.weight",
			"ff.linear_in.weight#gate", "ff.linear_in.weight#up", "ff.linear_out.weight"} {
			out = append(out, p+n)
		}
	}
	return out
}

// statName is the name of a matrix's mean squares in a statistics file: its tensor's (a part's
// columns are the whole matrix's), + ".x2".
func statName(name string) string {
	base, _, _ := strings.Cut(name, "#")
	return base + ".x2"
}

// FitCodes writes into out the lut8 tables of the DiT's matrices (ditMatrices) fitted against the
// activations: weighted Lloyd from the uniform levels (palette.QuantizeWeightedMinMax), each
// weight's error counted by the mean square of its layer's input at its column, from stats (a
// safetensors file of NAME.x2, float32 by in-feature, for the "dit." names of Weights: tools/ref/
// flux3_quant.py's calibration). The matrices are read through w, as the programs read them (the
// rotary rows permuted, the feedforward's input in parts), so that their codes match there. A
// matrix stats leaves out is left to the format's own fitting.
func FitCodes(w *mil.Weights, c mil.Dit, stats, out string, log io.Writer) error {
	st, err := safetensors.Open(stats)
	if err != nil {
		return err
	}
	defer st.Close()
	cw, err := palette.NewCodesWriter(out)
	if err != nil {
		return err
	}
	t0 := time.Now()
	n, skipped := 0, 0
	var se, sr float64
	for _, name := range ditMatrices(c) {
		if _, ok := st.Lookup(statName(name)); !ok {
			skipped++
			continue
		}
		x2, err := st.Float32(statName(name))
		if err != nil {
			return err
		}
		v, shape, err := w.Get(name)
		if err != nil {
			return err
		}
		rows, cols := shape[0], shape[1]
		if len(x2) != cols || rows%16 != 0 {
			return fmt.Errorf("flux3klein: %s (%d x %d): %d mean squares", name, rows, cols, len(x2))
		}
		q := palette.QuantizeWeightedMinMax(v, rows, cols, 16, 0, 8, CodeIters, x2)
		if err := cw.Add(name, v, q); err != nil {
			return err
		}
		dq := q.Dequantize()
		for i, x := range v {
			d := float64(dq[i] - x)
			se += float64(x2[i%cols]) * d * d
			sr += float64(x2[i%cols]) * float64(x) * float64(x)
		}
		n++
	}
	if err := cw.Close(); err != nil {
		return err
	}
	if n == 0 {
		return fmt.Errorf("flux3klein: %s holds the statistics of none of the DiT's matrices", stats)
	}
	fmt.Fprintf(log, "lut8 tables of %d matrices fitted to the activations (%d left to the format's), weighted relative error %.4f, %s\n",
		n, skipped, math.Sqrt(se/sr), time.Since(t0).Round(time.Second))
	return nil
}
