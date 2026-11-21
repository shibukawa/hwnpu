// Command lutfit fits the lookup tables of a model's linear weights against its activations: each
// table of a block of rows (and columns) by weighted Lloyd, every weight's error counted by the
// mean square of the layer's input at its column (palette.QuantizeWeighted), and writes them as
// codes (palette.CodesWriter) that hwnpu-dev convert -codes and hwnpu-chat -codes read in place
// of the format's own fitting. The mean squares come from a calibration run in float32
// (tools/ref/ref_qwen2.py --stats): WEIGHT.x2 by in-feature for each weight they name; the weights
// without one are left to the format's fitting.
//
//	go run ./tools/coreai/lutfit -config DIR/config.json -weights DIR/model.safetensors -stats X2.safetensors -o CODES [-format lut4]
package main

import (
	"flag"
	"fmt"
	"log"
	"maps"
	"math"
	"slices"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/llm"
	_ "github.com/shibukawa/hwnpu/internal/llm/archs"
	"github.com/shibukawa/hwnpu/internal/palette"
	"github.com/shibukawa/hwnpu/internal/safetensors"
)

func main() {
	config := flag.String("config", "", "config.json of the model")
	weights := flag.String("weights", "", "the checkpoint (.safetensors, or the name it would have in one file)")
	stats := flag.String("stats", "", "the inputs' mean squares: WEIGHT.x2 float32 by in-feature")
	out := flag.String("o", "", "the codes directory to write")
	format := flag.String("format", "lut4", "lut4, lut6 or lut8")
	group := flag.Int("group", 16, "rows a table")
	colBlock := flag.Int("col-block", 0, "columns a table (0: whole rows; Core AI's Neural Engine reads only the first block's of a row, docs/design/llm-qwen.md)")
	iters := flag.Int("iters", 20, "passes of weighted Lloyd")
	flag.Parse()
	if *config == "" || *weights == "" || *stats == "" || *out == "" {
		log.Fatal("lutfit: -config, -weights, -stats and -o are needed")
	}
	bits := map[string]int{"lut4": 4, "lut6": 6, "lut8": 8}[*format]
	if bits == 0 {
		log.Fatalf("lutfit: format %q", *format)
	}
	a, m, err := llm.LoadConfig(*config, "")
	if err != nil {
		log.Fatal(err)
	}
	w, err := llm.OpenWeights(*weights, a)
	if err != nil {
		log.Fatal(err)
	}
	defer w.Close()
	st, err := safetensors.Open(*stats)
	if err != nil {
		log.Fatal(err)
	}
	defer st.Close()
	cw, err := palette.NewCodesWriter(*out)
	if err != nil {
		log.Fatal(err)
	}
	t0 := time.Now()
	n := 0
	var se, sr float64
	for _, name := range slices.Sorted(maps.Keys(m.Tensors())) {
		shape := m.Tensors()[name]
		if len(shape) != 2 || name == m.Embedding() {
			continue
		}
		x2, err := st.Float32(name + ".x2")
		if err != nil {
			if strings.Contains(err.Error(), "no tensor") {
				continue
			}
			log.Fatal(err)
		}
		rows, cols := shape[0], shape[1]
		if len(x2) != cols || rows%*group != 0 || (*colBlock > 0 && cols%*colBlock != 0) {
			log.Fatalf("lutfit: %s (%d x %d): %d mean squares", name, rows, cols, len(x2))
		}
		v, _, err := w.Get(name)
		if err != nil {
			log.Fatal(err)
		}
		q := palette.QuantizeWeighted(v, rows, cols, *group, *colBlock, bits, *iters, x2)
		if err := cw.Add(name, v, q); err != nil {
			log.Fatal(err)
		}
		// the weighted error over the matrix, as the fitting minimizes it
		dq := q.Dequantize()
		var e, r float64
		for i, x := range v {
			d := float64(dq[i] - x)
			e += float64(x2[i%cols]) * d * d
			r += float64(x2[i%cols]) * float64(x) * float64(x)
		}
		se, sr = se+e, sr+r
		n++
		if n%20 == 0 {
			fmt.Printf("%d weights, %.0f s, %s: weighted relative error %.4f\n", n, time.Since(t0).Seconds(), name, math.Sqrt(e/r))
		}
	}
	if err := cw.Close(); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%d weights in %s, weighted relative error %.4f, %.0f s\n", n, *format, math.Sqrt(se/sr), time.Since(t0).Seconds())
}
