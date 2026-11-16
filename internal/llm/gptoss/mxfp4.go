package gptoss

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
)

// fp4 is e2m1's value of each 4 bit code (transformers' FP4_VALUES).
var fp4 = [16]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6}

// decodeMXFP4 turns MXFP4 rows into float32, as transformers' convert_moe_packed_tensors does:
// blocks holds a row's groups of 32 values as 16 bytes each, the low nibble first, and scales
// one byte a group, the power of two 2^(scale-127) its values are multiplied by. The result is
// rows x (groups*32), row major.
func decodeMXFP4(blocks, scales []byte, rows, groups int) []float32 {
	out := make([]float32, rows*groups*32)
	for r := 0; r < rows; r++ {
		for g := 0; g < groups; g++ {
			s := float32(math.Ldexp(1, int(scales[r*groups+g])-127))
			blk := blocks[(r*groups+g)*16 : (r*groups+g+1)*16]
			o := out[(r*groups+g)*32:]
			for j, b := range blk {
				o[2*j] = fp4[b&15] * s
				o[2*j+1] = fp4[b>>4] * s
			}
		}
	}
	return out
}

// expertWeights is one expert's weights, out x in as nn.Linear's: gate and up (intermediate,
// hidden) and down (hidden, intermediate), with the gate's and the up's biases.
type expertWeights struct {
	gate, up, down   []float32
	gateBias, upBias []float32
}

// experts reads a layer's experts from the checkpoint: its MXFP4 blocks and scales
// (mlp.experts.gate_up_proj_blocks, ...), an expert at a time, or transformers' dense parameters
// (mlp.experts.gate_up_proj (experts, hidden, 2 intermediate) and down_proj (experts,
// intermediate, hidden), in by out) of a checkpoint without quantization. The gate and the up are
// interleaved, the gate on the even rows of gate_up_proj (GptOssExperts: gate_up[..., ::2]).
type experts struct {
	cfg *Config
	w   llm.Weights

	// a dense checkpoint's layer, read once for its experts
	layer          int
	gateUp, downPr []float32
}

func newExperts(cfg *Config, w llm.Weights) *experts { return &experts{cfg: cfg, w: w, layer: -1} }

// get returns expert e of layer i; a failure stops g's build.
func (x *experts) get(g *blocks.Graph, i, e int) expertWeights {
	c := x.cfg
	d, f := c.HiddenSize, c.IntermediateSize
	p := layerPrefix(i) + "mlp.experts."
	var gu, down []float32 // (2f, d) and (d, f), out by in
	if c.mxfp4() {
		gu = x.mxfp4(g, p+"gate_up_proj", e, 2*f, d)
		down = x.mxfp4(g, p+"down_proj", e, d, f)
	} else {
		if x.layer != i {
			x.gateUp, x.downPr, x.layer = g.Get(p+"gate_up_proj", c.Experts, d, 2*f), g.Get(p+"down_proj", c.Experts, f, d), i
		}
		gu = make([]float32, 2*f*d)
		src := x.gateUp[e*d*2*f : (e+1)*d*2*f] // (d, 2f)
		for in := 0; in < d; in++ {
			for out := 0; out < 2*f; out++ {
				gu[out*d+in] = src[in*2*f+out]
			}
		}
		down = make([]float32, d*f)
		src = x.downPr[e*f*d : (e+1)*f*d] // (f, d)
		for in := 0; in < f; in++ {
			for out := 0; out < d; out++ {
				down[out*f+in] = src[in*d+out]
			}
		}
	}
	gb := g.Get(p+"gate_up_proj_bias", c.Experts, 2*f)[e*2*f : (e+1)*2*f]
	ew := expertWeights{gate: make([]float32, f*d), up: make([]float32, f*d), down: down,
		gateBias: make([]float32, f), upBias: make([]float32, f)}
	for j := 0; j < f; j++ {
		copy(ew.gate[j*d:(j+1)*d], gu[2*j*d:(2*j+1)*d])
		copy(ew.up[j*d:(j+1)*d], gu[(2*j+1)*d:(2*j+2)*d])
		ew.gateBias[j], ew.upBias[j] = gb[2*j], gb[2*j+1]
	}
	return ew
}

// downBias is layer i's down_proj_bias, (experts, hidden).
func (x *experts) downBias(g *blocks.Graph, i int) []float32 {
	c := x.cfg
	return g.Get(layerPrefix(i)+"mlp.experts.down_proj_bias", c.Experts, c.HiddenSize)
}

// mxfp4 decodes expert e of the MXFP4 tensor prefix (prefix_blocks (experts, rows, cols/32, 16),
// prefix_scales (experts, rows, cols/32)) into rows x cols.
func (x *experts) mxfp4(g *blocks.Graph, prefix string, e, rows, cols int) []float32 {
	raw, ok := x.w.(llm.RawWeights)
	if !ok {
		g.Fail("%s: the checkpoint gives no raw bytes for MXFP4", prefix)
	}
	groups := cols / 32
	read := func(name string, n int64, want []int) []byte {
		b, dtype, shape, err := raw.RawRange(name, int64(e)*n, int64(e+1)*n)
		if err != nil {
			g.Fail("%v", err)
		}
		if dtype != "U8" || fmt.Sprint(shape) != fmt.Sprint(want) {
			g.Fail("%s: %s %v, expected U8 %v", name, dtype, shape, want)
		}
		return b
	}
	n := x.cfg.Experts
	blocks := read(prefix+"_blocks", int64(rows*groups*16), []int{n, rows, groups, 16})
	scales := read(prefix+"_scales", int64(rows*groups), []int{n, rows, groups})
	return decodeMXFP4(blocks, scales, rows, groups)
}
