package flux2klein

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/shibukawa/hwnpu/internal/imagegen/flux2klein/mil"
	"github.com/shibukawa/hwnpu/internal/safetensors"
)

// TimeFile is an export's file of the timestep's embedding and the shared modulations' weights
// (bf16, as the checkpoint holds them): the host computes them in float32, 17 rows a step that
// every block takes (they drifted on the Neural Engine in Sana: new-model-checklist 3.3), 170 M
// multiply-adds.
const TimeFile = "time.safetensors"

// timeTensors are the time embedding's weights in the checkpoint ("dit."), in timeEmbed's order;
// their biases, when the checkpoint holds them, follow under the same names with .bias.
var timeTensors = []string{
	"dit.time_guidance_embed.timestep_embedder.linear_1.weight",
	"dit.time_guidance_embed.timestep_embedder.linear_2.weight",
	"dit.double_stream_modulation_img.linear.weight",
	"dit.double_stream_modulation_txt.linear.weight",
	"dit.single_stream_modulation.linear.weight",
	"dit.norm_out.linear.weight",
}

func biasOf(name string) string { return name[:len(name)-len("weight")] + "bias" }

// timeEmbed is the timestep's embedding and the modulations on the host: the weights and their
// biases (nil: none).
type timeEmbed struct {
	E    int
	w, b [][]float32
}

// writeTimeFile writes the time embedding's weights (and biases) to path as bf16.
func writeTimeFile(path string, w *mil.Weights) error {
	var outs []safetensors.Out
	for _, n := range timeTensors {
		for _, name := range []string{n, biasOf(n)} {
			if name != n && !w.Has(name) {
				continue
			}
			v, shape, err := w.Get(name)
			if err != nil {
				return err
			}
			outs = append(outs, safetensors.Out{Name: name, DType: "BF16", Shape: shape, Size: int64(2 * len(v)), Data: bfReader(v)})
		}
	}
	return safetensors.Write(path, outs, nil)
}

func readTimeFile(path string, E, timeIn int) (*timeEmbed, error) {
	f, err := safetensors.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	e := &timeEmbed{E: E}
	outs := []int{E, E, 6 * E, 6 * E, 3 * E, 2 * E}
	for i, n := range timeTensors {
		v, err := f.Float32(n)
		if err != nil {
			return nil, err
		}
		in := E
		if i == 0 {
			in = timeIn
		}
		if len(v) != outs[i]*in {
			return nil, fmt.Errorf("flux2klein: %s: %s holds %d values for (%d, %d)", path, n, len(v), outs[i], in)
		}
		var b []float32
		if _, ok := f.Lookup(biasOf(n)); ok {
			if b, err = f.Float32(biasOf(n)); err != nil {
				return nil, err
			}
		}
		e.w, e.b = append(e.w, v), append(e.b, b)
	}
	return e, nil
}

// linear is w (out, len(x)) times x plus b (nil: none), the rows in parallel, in float64.
func linear(w, b, x []float32) []float32 {
	in := len(x)
	out := make([]float32, len(w)/in)
	var wg sync.WaitGroup
	n := runtime.GOMAXPROCS(0)
	per := (len(out) + n - 1) / n
	for lo := 0; lo < len(out); lo += per {
		hi := min(lo+per, len(out))
		wg.Add(1)
		go func() {
			defer wg.Done()
			for o := lo; o < hi; o++ {
				row := w[o*in : (o+1)*in]
				s := 0.0
				if b != nil {
					s = float64(b[o])
				}
				for i, v := range x {
					s += float64(row[i]) * float64(v)
				}
				out[o] = float32(s)
			}
		}()
	}
	wg.Wait()
	return out
}

func silu(x []float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = float32(float64(v) / (1 + math.Exp(-float64(v))))
	}
	return out
}

// modulation is the modulation rows at sigma (mil.ModRows x hidden, in mil.ModImage's order): the
// timestep's embedding (linear_2 of the SiLU of linear_1 of the sinusoid of 1000 sigma), and each
// modulation's linear of its SiLU.
func (e *timeEmbed) modulation(sigma float64) []float32 {
	temb := linear(e.w[1], e.b[1], silu(linear(e.w[0], e.b[0], sinusoid(1000*sigma, len(e.w[0])/e.E))))
	a := silu(temb)
	var out []float32
	for i := 2; i < len(e.w); i++ {
		out = append(out, linear(e.w[i], e.b[i], a)...)
	}
	return out
}

// bfReader is xs as bf16 bytes.
func bfReader(xs []float32) *bytes.Reader {
	b := make([]byte, 2*len(xs))
	for i, x := range xs {
		binary.LittleEndian.PutUint16(b[2*i:], bf16(x))
	}
	return bytes.NewReader(b)
}
