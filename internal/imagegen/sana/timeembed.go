package sana

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/imagegen/sana/mil"
	"github.com/shibukawa/hwnpu/internal/safetensors"
)

// TimeFile is the export's file of the time embedding's weights (float32): the host computes it.
// On the Neural Engine it came out up to 1 % off (the CPU interpreter in fp16: 0.05 %), and as
// the six modulations of every block, one DiT call's output went from cosine 0.9999 to 0.983
// with diffusers' fp32 (M1, design.md 16.145). It is one row: about 11 M multiply-adds a step.
const TimeFile = "time.safetensors"

// timeEmbed is the time embedding of the DiT on the host.
type timeEmbed struct {
	E                int
	t1, t1b, t2, t2b []float32 // the timestep's TimestepEmbedding
	lin, linb        []float32 // the modulations (6 E, E)
}

// timeTensors are the names of the time embedding's weights in the checkpoint ("dit."), in the
// order timeEmbed holds them.
func timeTensors() []string {
	p := "dit.time_embed.emb.timestep_embedder."
	names := []string{}
	for _, n := range []string{"linear_1.weight", "linear_1.bias", "linear_2.weight", "linear_2.bias"} {
		names = append(names, p+n)
	}
	return append(names, "dit.time_embed.linear.weight", "dit.time_embed.linear.bias")
}

// writeTimeFile writes the time embedding's weights to path as float32 under their names.
func writeTimeFile(path string, w *mil.Weights) error {
	var outs []safetensors.Out
	for _, n := range timeTensors() {
		v, shape, err := w.Get(n)
		if err != nil {
			return err
		}
		var b bytes.Buffer
		if err := binary.Write(&b, binary.LittleEndian, v); err != nil {
			return err
		}
		outs = append(outs, safetensors.Out{Name: n, DType: "F32", Shape: shape, Size: int64(b.Len()), Data: &b})
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
	for _, n := range timeTensors() {
		v, err := f.Float32(n)
		if err != nil {
			return nil, err
		}
		vs = append(vs, v)
	}
	e := &timeEmbed{E: E, t1: vs[0], t1b: vs[1], t2: vs[2], t2b: vs[3]}
	e.lin, e.linb = vs[len(vs)-2], vs[len(vs)-1]
	if len(e.t1) != E*256 || len(e.lin) != 6*E*E {
		return nil, fmt.Errorf("sana: %s holds weights of %d and %d for a hidden width of %d", path, len(e.t1), len(e.lin), E)
	}
	return e, nil
}

// linear is w (out, len(x)) times x plus b, in float64.
func linear(w, b, x []float32) []float32 {
	in := len(x)
	out := make([]float32, len(b))
	for o := range out {
		s := float64(b[o])
		row := w[o*in : (o+1)*in]
		for i, v := range x {
			s += float64(row[i]) * float64(v)
		}
		out[o] = float32(s)
	}
	return out
}

func silu(x []float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = float32(float64(v) / (1 + math.Exp(-float64(v))))
	}
	return out
}

// embed is the six modulations (6 E) and the embedded timestep (E) of the sinusoid tproj.
func (e *timeEmbed) embed(tproj []float32) (t6, emb []float32) {
	emb = linear(e.t2, e.t2b, silu(linear(e.t1, e.t1b, tproj)))
	return linear(e.lin, e.linb, silu(emb)), emb
}
