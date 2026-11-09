package pitch

import (
	"fmt"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Arch names SwiftF0's program in its metadata.
const Arch = "swiftf0"

// Revision counts the changes to the program that call for converting it again.
const Revision = 3 // 3: the dilated convolutions as stacked shifts (shifted)

// Program is SwiftF0's CNN for llm.WritePart: inputs x (1, 3, 128, frames), the features of
// Features, and mask (1, 1, 1, frames), 1 for the clip's frames and 0 past them; output y (1, 96,
// frames), the 95 pitch logits of each frame and its voicing logit. The graph pads every
// convolution with zeros at the clip's ends; the program masks each convolution's input, so
// that the frames past the clip are those zeros and a program of a fixed length gives a shorter
// clip's values exactly.
type Program struct {
	Frames int
}

// The llm.Model methods the program does not use.
func (p *Program) Hidden() int               { return 0 }
func (p *Program) Layers() int               { return 0 }
func (p *Program) Output() (int, bool)       { return 0, false }
func (p *Program) Embedding() string         { return "" }
func (p *Program) Tensors() map[string][]int { return nil }

type graph struct {
	*blocks.Graph
	frames int
	mask   *mlpackage.Var
}

// Build writes the program.
func (p *Program) Build(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins []mlpackage.Feature, out mlpackage.Feature, err error) {
	if opts.Format == "" {
		opts.Format = "fp16"
	}
	if opts.Group == 0 {
		opts.Group = 16
	}
	T := p.Frames
	if T <= 0 {
		return nil, out, fmt.Errorf("pitch: a program of %d frames", T)
	}
	g := &graph{Graph: &blocks.Graph{Name: Arch, B: b, W: w, Opts: opts}, frames: T}
	var y *mlpackage.Var
	err = blocks.Run(func() {
		x := b.Input("x", mlpackage.Float16, 1, 3, Bands, T)
		g.mask = b.Input("mask", mlpackage.Float16, 1, 1, 1, T)
		ins = []mlpackage.Feature{
			{Name: "x", DType: mlpackage.Float16, Shape: x.Shape},
			{Name: "mask", DType: mlpackage.Float16, Shape: g.mask.Shape},
		}
		x = g.masked(x)
		h := g.relu(g.conv(x, stem))
		for _, blk := range body {
			r := g.conv(g.relu(g.conv(h, blk[0])), blk[1])
			h = g.relu(g.Op("add", map[string]any{"x": h, "y": r}, h.Shape...))
		}
		// the head reads the lowest headBands bands
		h = g.Op("slice_by_index", map[string]any{"x": h, "begin": b.Ints(0, 0, 0, 0), "end": b.Ints(1, 12, int32(headBands), int32(T))}, 1, 12, headBands, T)
		h1 := g.relu(g.conv(h, head1)) // (1, 4, 95, T)
		h2 := g.conv(h1, head2)
		peaks := g.Op("reduce_max", map[string]any{"x": h2, "axes": b.Ints(2), "keep_dims": b.ScalarBool(true)}, 1, 4, 1, T)
		mean := g.Op("reduce_mean", map[string]any{"x": x, "axes": b.Ints(1, 2), "keep_dims": b.ScalarBool(true)}, 1, 1, 1, T)
		v := g.Op("concat", map[string]any{"values": []*mlpackage.Var{peaks, mean}, "axis": b.ScalarI32(1), "interleave": b.ScalarBool(false)}, 1, 5, 1, T)
		v = g.conv(g.relu(g.conv(g.masked(v), voice1)), voice2) // (1, 1, 1, T)
		logits := g.conv(h1, pitchConv)                         // (1, 1, 95, T)
		logits = g.Op("reshape", map[string]any{"x": logits, "shape": b.Ints(1, Bins, int32(T))}, 1, Bins, T)
		v = g.Op("reshape", map[string]any{"x": v, "shape": b.Ints(1, 1, int32(T))}, 1, 1, T)
		y = b.NamedOp("concat", "y", map[string]any{"values": []*mlpackage.Var{logits, v}, "axis": b.ScalarI32(1), "interleave": b.ScalarBool(false)}, mlpackage.Float16, 1, Bins+1, T)
	})
	if err != nil {
		return nil, out, err
	}
	return ins, mlpackage.Feature{Name: "y", DType: mlpackage.Float16, Shape: y.Shape}, nil
}

// masked zeroes the frames past the clip.
func (g *graph) masked(x *mlpackage.Var) *mlpackage.Var {
	return g.Op("mul", map[string]any{"x": x, "y": g.mask}, x.Shape...)
}

// relu is ReLU, then the mask: every ReLU's output is a convolution's input.
func (g *graph) relu(x *mlpackage.Var) *mlpackage.Var {
	return g.masked(g.Op("relu", map[string]any{"x": x}, x.Shape...))
}

// conv is one of the graph's convolutions over (bands, frames), with its weight and bias.
func (g *graph) conv(x *mlpackage.Var, c conv) *mlpackage.Var {
	w, shape, err := g.W.Get(c.weight)
	if err != nil {
		g.Fail("%v", err)
	}
	bias, bshape, err := g.W.Get(c.bias)
	if err != nil {
		g.Fail("%v", err)
	}
	if len(shape) == 3 { // a 1 D convolution over the frames: (out, in, k) as (out, in, 1, k)
		shape = []int{shape[0], shape[1], 1, shape[2]}
	}
	if len(shape) != 4 || shape[1] != x.Shape[1] || len(bshape) != 1 || bshape[0] != shape[0] {
		g.Fail("%s %v and bias %v for an input %v", c.weight, shape, bshape, x.Shape)
	}
	if c.dil[0] > maxDilation && shape[3] == 1 && c.pad[0] == c.pad[1] && c.pad[0] == c.dil[0]*(shape[2]-1)/2 {
		return g.shifted(x, c, w, shape, bias)
	}
	h := (x.Shape[2]+c.pad[0]+c.pad[1]-c.dil[0]*(shape[2]-1)-1)/1 + 1
	wd := (x.Shape[3]+c.pad[2]+c.pad[3]-c.dil[1]*(shape[3]-1)-1)/1 + 1
	return g.Op("conv", map[string]any{
		"x": x, "weight": g.F16s(c.weight, shape, w), "bias": g.F16s(c.bias, []int{shape[0]}, bias),
		"strides": g.B.Ints(1, 1), "pad_type": g.Str("custom"),
		"pad":       g.B.Ints(int32(c.pad[0]), int32(c.pad[1]), int32(c.pad[2]), int32(c.pad[3])),
		"dilations": g.B.Ints(int32(c.dil[0]), int32(c.dil[1])), "groups": g.B.ScalarI32(1),
	}, 1, shape[0], h, wd)
}

// maxDilation is the largest dilation over the bands that the Neural Engine runs a convolution
// with: on an M1, Core ML puts those of 17, 27, 40 and 48 on the CPU, and runs that of 2.
const maxDilation = 2

// shifted is a convolution of kernel (k, 1) dilated by d over the bands, padded to keep the
// bands, as one 1x1 convolution: the input shifted by (j - (k-1)/2) * d bands for each tap j
// (zeros shifted in, as the padding), the k copies stacked on the channels in tap order, and the
// weight's taps laid out to match (out, k * in, 1, 1).
func (g *graph) shifted(x *mlpackage.Var, c conv, w []float32, shape []int, bias []float32) *mlpackage.Var {
	co, ci, k := shape[0], shape[1], shape[2]
	d, H, T := c.dil[0], x.Shape[2], x.Shape[3]
	p := d * (k - 1) / 2
	padded := g.Op("pad", map[string]any{"x": x, "pad": g.B.Ints(0, 0, 0, 0, int32(p), int32(p), 0, 0), "mode": g.Str("constant"), "constant_val": g.B.ScalarF16(0)}, 1, ci, H+2*p, T)
	var taps []*mlpackage.Var
	for j := range k {
		taps = append(taps, g.Op("slice_by_index", map[string]any{"x": padded, "begin": g.B.Ints(0, 0, int32(j*d), 0), "end": g.B.Ints(1, int32(ci), int32(j*d+H), int32(T))}, 1, ci, H, T))
	}
	stacked := g.Op("concat", map[string]any{"values": taps, "axis": g.B.ScalarI32(1), "interleave": g.B.ScalarBool(false)}, 1, k*ci, H, T)
	wt := make([]float32, co*k*ci)
	for o := range co {
		for i := range ci {
			for j := range k {
				wt[(o*k+j)*ci+i] = w[(o*ci+i)*k+j]
			}
		}
	}
	return g.Op("conv", map[string]any{
		"x": stacked, "weight": g.F16s(c.weight+"_taps", []int{co, k * ci, 1, 1}, wt), "bias": g.F16s(c.bias, []int{co}, bias),
		"strides": g.B.Ints(1, 1), "pad_type": g.Str("custom"), "pad": g.B.Ints(0, 0, 0, 0), "dilations": g.B.Ints(1, 1), "groups": g.B.ScalarI32(1),
	}, 1, co, H, T)
}

// Weights serves the model's initializers to the program builder (llm.Weights).
func (m *Model) Weights() llm.Weights { return weights{m} }

type weights struct{ m *Model }

func (w weights) Get(name string) ([]float32, []int, error) {
	t, ok := w.m.t[name]
	if !ok {
		return nil, nil, fmt.Errorf("pitch: no initializer %s", name)
	}
	out := make([]float32, len(t.vals))
	for i, v := range t.vals {
		out[i] = float32(v)
	}
	return out, t.shape, nil
}

func (w weights) Rows(name string, begin, end int) ([]float32, error) {
	return nil, fmt.Errorf("pitch: no rows of %s", name)
}

func (w weights) Source() string { return w.m.source }
func (w weights) Close() error   { return nil }
