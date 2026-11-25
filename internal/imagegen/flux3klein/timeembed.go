package flux3klein

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/shibukawa/bonsaiai/internal/imagegen/flux3klein/mil"
	"github.com/shibukawa/bonsaiai/internal/safetensors"
)

// TimeFile is an export's file of the timesteps' embeddings and the modulations' weights (bf16, as
// the checkpoint holds them): the host computes them in float32, 14 rows a step that every block of
// a stream takes (they drifted on the Neural Engine in Sana: new-model-checklist 3.3).
const TimeFile = "time.safetensors"

// timeTensors are the weights of the host's part, in timeEmbed's order (none has a bias).
var timeTensors = []string{
	"dit.timestep_embedder.linear_1.weight",
	"dit.timestep_embedder.linear_2.weight",
	"dit.target_timestep_embedder.linear_1.weight",
	"dit.target_timestep_embedder.linear_2.weight",
	"dit.early_stream_modulations.txt.linear.weight",
	"dit.early_stream_modulations.image.linear.weight",
	"dit.single_stream_modulations.txt.linear.weight",
	"dit.single_stream_modulations.image.linear.weight",
	"dit.norm_out.linear.weight",
}

// timeEmbed is the timesteps' embeddings and the modulations on the host. The weights are the
// file's bf16 values, mapped where the system maps files and decoded as they are multiplied: 305
// MB that were 610 MB of float32 for each picture.
type timeEmbed struct {
	E, In int
	w     [][]byte // bf16 rows, of f
	f     *safetensors.File
}

// close lets the file go.
func (e *timeEmbed) close() {
	e.w = nil
	e.f.Close()
}

func writeTimeFile(path string, w *mil.Weights) error {
	var outs []safetensors.Out
	for _, n := range timeTensors {
		if w.Has(n[:len(n)-len("weight")] + "bias") {
			return fmt.Errorf("flux3klein: %s has a bias, which the host's modulation leaves out", n)
		}
		v, shape, err := w.Get(n)
		if err != nil {
			return err
		}
		outs = append(outs, safetensors.Out{Name: n, DType: "BF16", Shape: shape, Size: int64(2 * len(v)), Data: bfReader(v)})
	}
	return safetensors.Write(path, outs, nil)
}

func readTimeFile(path string, E, timeIn int) (*timeEmbed, error) {
	f, err := safetensors.Open(path)
	if err != nil {
		return nil, err
	}
	e := &timeEmbed{E: E, In: timeIn, f: f}
	outs := []int{E, E, E, E, 3 * E, 3 * E, 3 * E, 3 * E, 2 * E}
	for i, n := range timeTensors {
		t, ok := f.Lookup(n)
		if !ok || t.DType != "BF16" {
			e.close()
			return nil, fmt.Errorf("flux3klein: %s: no bf16 tensor %s", path, n)
		}
		v, err := f.Map(n)
		if err != nil {
			e.close()
			return nil, err
		}
		in := E
		if i == 0 || i == 2 {
			in = timeIn
		}
		if len(v) != 2*outs[i]*in {
			e.close()
			return nil, fmt.Errorf("flux3klein: %s: %s holds %d values for (%d, %d)", path, n, len(v)/2, outs[i], in)
		}
		e.w = append(e.w, v)
	}
	return e, nil
}

// embed is a timestep embedder's output (the time embedder's when target is false) at t, which the
// pipeline hands it in bf16 and multiplies by 1000 in bf16 (timestep.to(dtype) * 1000): the
// sinusoid, linear_1, SiLU, linear_2.
func (e *timeEmbed) embed(t float64, target bool) []float32 {
	i := 0
	if target {
		i = 2
	}
	ts := roundBF16(roundBF16(float32(t)) * 1000)
	return linear(e.w[i+1], silu(linear(e.w[i], sinusoid(float64(ts), e.In))))
}

// modulation is the modulation rows of a step from sigma to next (mil.ModRows x hidden, in
// mil.ModTextEarly's order): the image's embedding is the time embedder's at sigma plus the target
// embedder's at next; the text's (and a reference's) the time embedder's at 0 plus the target's at
// next (Flux3Transformer2DModel.forward). Each modulation is a linear of its embedding's SiLU:
// shift, scale and gate rows, and norm_out's scale and shift.
func (e *timeEmbed) modulation(sigma, next float64) []float32 {
	tgt := e.embed(next, true)
	img, zero := e.embed(sigma, false), e.embed(0, false)
	for i := range img {
		img[i] += tgt[i]
		zero[i] += tgt[i]
	}
	ai, az := silu(img), silu(zero)
	var out []float32
	for i, a := range [][]float32{az, ai, az, ai, ai} {
		out = append(out, linear(e.w[4+i], a)...)
	}
	return out
}

// linear is w (out, len(x)) of bf16 values times x, the rows in parallel, in float64.
func linear(w []byte, x []float32) []float32 {
	in := len(x)
	out := make([]float32, len(w)/2/in)
	var wg sync.WaitGroup
	n := runtime.GOMAXPROCS(0)
	per := (len(out) + n - 1) / n
	for lo := 0; lo < len(out); lo += per {
		hi := min(lo+per, len(out))
		wg.Add(1)
		go func() {
			defer wg.Done()
			for o := lo; o < hi; o++ {
				row := w[2*o*in : 2*(o+1)*in]
				s := 0.0
				for i, v := range x {
					s += float64(math.Float32frombits(uint32(binary.LittleEndian.Uint16(row[2*i:]))<<16)) * float64(v)
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

// sinusoid is diffusers' get_timestep_embedding(t, dim, flip_sin_to_cos=True,
// downscale_freq_shift=0): the cosines, then the sines.
func sinusoid(t float64, dim int) []float32 {
	half := dim / 2
	out := make([]float32, dim)
	for i := 0; i < half; i++ {
		a := t * math.Exp(-math.Log(10000)*float64(i)/float64(half))
		out[i] = float32(math.Cos(a))
		out[half+i] = float32(math.Sin(a))
	}
	return out
}

// roundBF16 is x rounded to bf16 (to nearest, ties to even).
func roundBF16(x float32) float32 {
	return math.Float32frombits(uint32(bf16(x)) << 16)
}

func bf16(x float32) uint16 {
	u := math.Float32bits(x)
	return uint16((u + 0x7fff + (u>>16)&1) >> 16)
}

func bfReader(xs []float32) *bytes.Reader {
	b := make([]byte, 2*len(xs))
	for i, x := range xs {
		binary.LittleEndian.PutUint16(b[2*i:], bf16(x))
	}
	return bytes.NewReader(b)
}

func fromBF16s(b []byte) []float32 {
	out := make([]float32, len(b)/2)
	for i := range out {
		out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[2*i:])) << 16)
	}
	return out
}
