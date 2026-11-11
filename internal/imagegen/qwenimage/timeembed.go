package qwenimage

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/shibukawa/hwnpu/internal/imagegen/qwenimage/mil"
	"github.com/shibukawa/hwnpu/internal/safetensors"
)

// TimeFile is a Core ML export's file of the timestep's embedding and the shared modulation's
// weights (bf16, as the checkpoint holds them): the host computes them in float32. They are one row
// a step that every block takes, which drifted on the Neural Engine in Sana (new-model-checklist
// 3.3); here 100 M multiply-adds a step.
const TimeFile = "time.safetensors"

// timeTensors are the time embedding's weights in the checkpoint ("dit."), in timeEmbed's order:
// none has a bias.
var timeTensors = []string{
	"dit.time_text_embed.timestep_embedder.linear_1.weight",
	"dit.time_text_embed.timestep_embedder.linear_2.weight",
	"dit.modulation.1.weight",
	"dit.norm_out.linear.weight",
}

// timeEmbed is the timestep's embedding and the modulations on the host.
type timeEmbed struct {
	E                 int
	t1, t2, mod, norm []float32
}

// writeTimeFile writes the time embedding's weights to path as bf16.
func writeTimeFile(path string, w *mil.Weights) error {
	var outs []safetensors.Out
	for _, n := range timeTensors {
		v, shape, err := w.Get(n)
		if err != nil {
			return err
		}
		outs = append(outs, safetensors.Out{Name: n, DType: "BF16", Shape: shape, Size: int64(2 * len(v)), Data: bfReader(v)})
	}
	return safetensors.Write(path, outs, nil)
}

func readTimeFile(path string, E int) (*timeEmbed, error) {
	f, err := safetensors.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var vs [][]float32
	for _, n := range timeTensors {
		v, err := f.Float32(n)
		if err != nil {
			return nil, err
		}
		vs = append(vs, v)
	}
	e := &timeEmbed{E: E, t1: vs[0], t2: vs[1], mod: vs[2], norm: vs[3]}
	if len(e.t1) != E*256 || len(e.t2) != E*E || len(e.mod) != 4*E*E || len(e.norm) != E*E {
		return nil, fmt.Errorf("qwenimage: %s holds weights of other shapes than a hidden width of %d", path, E)
	}
	return e, nil
}

// matVec is w (out, len(x)) times x, the rows in parallel, in float64.
func matVec(w, x []float32) []float32 {
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

// modulation is the blocks' modulation rows at sigma (Blocks' and TextBlocks' mod: 1 + scale1,
// tanh gate1 / DitScale, 1 + scale2, tanh gate2 / DitScale) and norm_out's 1 + scale.
func (e *timeEmbed) modulation(sigma float64) (mod, scale []float32) {
	E := e.E
	temb := matVec(e.t2, silu(matVec(e.t1, timestepSinusoid(1000*sigma, 256))))
	a := silu(temb)
	m := matVec(e.mod, a)
	for i := 0; i < E; i++ {
		m[i]++
		m[E+i] = float32(math.Tanh(float64(m[E+i])) / mil.DitScale)
		m[2*E+i]++
		m[3*E+i] = float32(math.Tanh(float64(m[3*E+i])) / mil.DitScale)
	}
	s := matVec(e.norm, a)
	for i := range s {
		s[i]++
	}
	return m, s
}

// bfReader is xs as bf16 bytes.
func bfReader(xs []float32) *bytes.Reader {
	b := make([]byte, 2*len(xs))
	for i, x := range xs {
		binary.LittleEndian.PutUint16(b[2*i:], bf16(x))
	}
	return bytes.NewReader(b)
}
