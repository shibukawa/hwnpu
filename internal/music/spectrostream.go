// Package music generates music with Magenta RealTime 2 (google/magenta-realtime-2: code
// Apache-2.0, weights CC-BY 4.0): a style from text (MusicCoCa), a depthformer language model
// that writes SpectroStream's audio tokens 25 frames a second, and SpectroStream's decoder that
// turns them into 48 kHz stereo.
//
// The structures follow the MLX implementation of magenta/magenta-realtime (magenta_rt/mlx) and
// the sequence_layers it builds on; the weights are the Flax checkpoints of the Hugging Face
// repository.
package music

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/safetensors"
)

// SpectroStream's constants (stft_spectrostream_40ms_generic_48khz_stereo_config).
const (
	SampleRate    = 48000
	FrameRate     = 25
	Codebooks     = 12   // the RVQ levels the language model writes
	CodebookSize  = 1024 // codes a level
	EmbeddingSize = 256
	stftHop       = 480
	stftLength    = 960
	stftBins      = 480 // the DC bin and up, the Nyquist one left out
	timeStride    = 4   // STFT frames a frame
	lookahead     = 4   // STFT frames the decoder looks ahead (one frame)
	// SamplesPerFrame are the samples (each stereo) of a frame.
	SamplesPerFrame = timeStride * stftHop
)

// decoderScales are what the decoder's blocks divide their values by (dgraph.scale): their
// largest values on the reference's clip are about 480, 9e3, 1.1e5, 2.8e5, 8.3e5, 1.3e7 and
// 1.4e7 from decoder_0 on, and with these the largest of all is 1.8e4, 3.7 times under fp16's
// limit (a block's sum is clipped there besides, clipSum). The scales are as small as that
// allows: the Neural Engine flushes fp16 subnormals (under 6.1e-5) to zero, and with 8 times
// larger scales the biases and the ELU's negative side divided by them fell there (samples 3.5%
// off the reference instead of 0.57%, as milcpu's RoundF16 with FlushSubnormals shows).
// DecoderOutScale is the output layer's: y is the STFT divided by it.
var decoderScales = map[string]float32{
	"decoder_0": 1, "decoder_1": 1, "decoder_2": 8, "decoder_3": 16, "decoder_4": 64, "decoder_5": 1024, "decoder_6": 1024,
}

const DecoderOutScale = 64

// decoderLevels are the decoder's blocks after decoder_0, in order: time and frequency strides,
// the transpose convolution's kernel and the output channels; the last ones run on the two
// halves of the channels (left and right) with the same weights.
var decoderLevels = []struct {
	name       string
	st, sf     int
	kt, kf     int
	in, out    int
	shortcut   bool // a 1x1 convolution on the shortcut (channels change)
	transposed string
}{
	{"decoder_1", 2, 2, 4, 4, 512, 256, true, "conv2dtranspose_4x4"},
	{"decoder_2", 1, 2, 3, 4, 256, 256, false, "conv2dtranspose_3x4"},
	{"decoder_3", 1, 2, 3, 4, 256, 256, false, "conv2dtranspose_3x4"},
	{"decoder_4", 1, 3, 3, 6, 256, 128, true, "conv2dtranspose_3x6"},
	{"decoder_5", 1, 2, 3, 4, 128, 128, false, "conv2dtranspose_3x4"},
	{"decoder_6", 1, 2, 3, 4, 128, 64, true, "conv2dtranspose_3x4"},
}

// Weights reads the Flax checkpoints of SpectroStream and the language model (llm.Weights).
type Weights struct {
	files  []*safetensors.File
	prefix map[*safetensors.File]string
}

// OpenWeights opens safetensors files; names are looked up in each in turn with its prefix
// ("params/" or "params/soundstream/") taken off.
func OpenWeights(paths map[string]string) (*Weights, error) {
	w := &Weights{prefix: map[*safetensors.File]string{}}
	for path, prefix := range paths {
		f, err := safetensors.Open(path)
		if err != nil {
			w.Close()
			return nil, err
		}
		w.files = append(w.files, f)
		w.prefix[f] = prefix
	}
	return w, nil
}

func (w *Weights) Get(name string) ([]float32, []int, error) {
	for _, f := range w.files {
		full := w.prefix[f] + name
		if t, ok := f.Lookup(full); ok {
			v, err := f.Float32(full)
			return v, t.Shape, err
		}
	}
	return nil, nil, fmt.Errorf("music: no tensor %s", name)
}

func (w *Weights) Rows(name string, begin, end int) ([]float32, error) {
	for _, f := range w.files {
		full := w.prefix[f] + name
		if t, ok := f.Lookup(full); ok {
			cols := t.Shape[len(t.Shape)-1] // rows of the last axis
			return f.Float32Range(full, begin*cols, end*cols)
		}
	}
	return nil, fmt.Errorf("music: no tensor %s", name)
}

func (w *Weights) Source() string { return "google/magenta-realtime-2" }

func (w *Weights) Close() error {
	for _, f := range w.files {
		f.Close()
	}
	return nil
}

// Decoder is SpectroStream's decoder as a program over Frames frames: inputs x (1, 256, frames,
// 1), the sums of the codes' quantizer embeddings channels first, and the masks (ssMasks), 1 for
// the frames of the clip and 0 for those before it; output y (2, 2, 4 * frames, 480), the STFT
// (the real and the imaginary parts of the left channel, then of the right) before the
// lookahead. The inputs come laid out as the convolutions read them: the M2's Core ML put the
// reshape, the transpose and the upsampling that made them on the CPU. The convolutions
// are causal in time and pad with zeros before the clip, which the program gets by zeroing every
// convolution's input before it (mask), so that a window of frames gives the clip's values.
type Decoder struct {
	Frames int
}

// The llm.Model methods the program does not use.
func (p *Decoder) Hidden() int               { return 0 }
func (p *Decoder) Layers() int               { return 0 }
func (p *Decoder) Output() (int, bool)       { return 0, false }
func (p *Decoder) Embedding() string         { return "" }
func (p *Decoder) Tensors() map[string][]int { return nil }

type dgraph struct {
	*blocks.Graph
	masks map[int]*mlpackage.Var // by time resolution (1, 2, 4 STFT frames a frame)
	// scale is what the values of the block being built are divided by (a power of 2): the
	// decoder's activations grow from hundreds to 1.4e7 through its blocks, past fp16's 65504,
	// and within a block they span a few hundred times their RMS. Each block computes on its
	// values divided by its scale (decoderScales), exactly: convolutions are linear (the bias
	// divided too), the ELU is computed for the scale (elu), and the change between blocks is a
	// multiplication by a power of 2. The output is y divided by the last scale.
	scale float32
}

func (g *dgraph) op(typ string, inputs map[string]any, shape ...int) *mlpackage.Var {
	return g.B.Op(typ, inputs, mlpackage.Float16, shape...)
}

func (g *dgraph) consts(name string, shape []int, vals []float32) *mlpackage.Var {
	return g.F16s(name, shape, vals)
}

// bias is a convolution's bias for the values divided by the scale.
func (g *dgraph) bias(name string, vals []float32) *mlpackage.Var {
	if g.scale == 1 {
		return g.consts(name, []int{len(vals)}, vals)
	}
	b := make([]float32, len(vals))
	for i, v := range vals {
		b[i] = v / g.scale
	}
	return g.consts(name, []int{len(b)}, b)
}

func (g *dgraph) scalar(v float32) *mlpackage.Var { return g.B.ScalarF16(v) }

func (g *dgraph) in(name string, shape ...int) *mlpackage.Var {
	return g.B.Input(name, mlpackage.Float16, shape...)
}

func (g *dgraph) out(x *mlpackage.Var) *mlpackage.Var {
	return g.B.NamedOp("identity", "y", map[string]any{"x": x}, mlpackage.Float16, x.Shape...)
}

// rescale is x, divided by the scale so far, as divided by s; s becomes the scale.
func (g *dgraph) rescale(x *mlpackage.Var, s float32) *mlpackage.Var {
	if s == g.scale {
		return x
	}
	x = g.op("mul", map[string]any{"x": x, "y": g.scalar(g.scale / s)}, x.Shape...)
	g.scale = s
	return x
}

const dec = "decoder/"

// Build writes the program.
func (p *Decoder) Build(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins []mlpackage.Feature, out mlpackage.Feature, err error) {
	if opts.Format == "" {
		opts.Format = "fp16"
	}
	if opts.Group == 0 {
		opts.Group = 16
	}
	T := p.Frames
	g := &dgraph{Graph: &blocks.Graph{Name: "spectrostream", B: b, W: w, Opts: opts}, masks: map[int]*mlpackage.Var{}, scale: 1}
	var y *mlpackage.Var
	err = blocks.Run(func() {
		h := g.in("x", 1, EmbeddingSize, T, 1)
		ins = append([]mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: h.Shape}}, g.maskInputs(T)...)
		// input_layer: conv1x1_first(x) + conv1x1_b2(elu(conv1x1_b1(x)))
		first := g.conv(h, dec+"input_layer/conv1x1_first/conv", 1, 0)
		sc := g.conv(g.elu(g.conv(h, dec+"input_layer/shortcut_layer/conv1x1_b1/conv", 1, 0)), dec+"input_layer/shortcut_layer/conv1x1_b2/conv", 1, 0)
		h = g.add(first, sc) // (1, 2560, T, 1): channel f*512 + c
		h = g.op("reshape", map[string]any{"x": h, "shape": b.Ints(1, 5, 512, int32(T))}, 1, 5, 512, T)
		h = g.op("transpose", map[string]any{"x": h, "perm": b.Ints(0, 2, 3, 1)}, 1, 512, T, 5)
		// input_layers_residual_unit
		r := g.preConv(h, dec+"input_layers_residual_unit/conv2d_3x3_a/conv", 1)
		r = g.preConv(r, dec+"input_layers_residual_unit/conv2d_3x3/conv", 1)
		h = g.add(h, r)
		// decoder_0: time stride 2, to 1024 channels
		h = g.rescale(h, decoderScales["decoder_0"])
		r = g.preTransposed(h, dec+"decoder_0/conv2dtranspose_4x3/conv", 2, 1, 1)
		r = g.preConv(r, dec+"decoder_0/conv2d_3x3/conv", 2)
		s := g.conv(g.masked(h, 1), dec+"decoder_0/shortcut_layer/conv1x1/conv", 1, 0)
		s = g.upsample(s, 2, 1)
		h = g.clipSum(g.add(r, s)) // (1, 1024, 2T, 5)
		// the two halves of the channels as a batch of two, with the same weights
		sh := h.Shape
		h = g.op("reshape", map[string]any{"x": h, "shape": b.Ints(2, 512, int32(sh[2]), int32(sh[3]))}, 2, 512, sh[2], sh[3])
		res := 2
		for _, lv := range decoderLevels {
			p := dec + lv.name + "/"
			h = g.rescale(h, decoderScales[lv.name])
			r := g.preTransposed(h, p+lv.transposed+"/conv", lv.st, lv.sf, res)
			nres := res * lv.st
			r = g.preConv(r, p+"conv2d_3x3/conv", nres)
			s := h
			if lv.shortcut {
				s = g.conv(g.masked(s, res), p+"shortcut_layer/conv1x1/conv", res, 0)
			}
			if lv.st != 1 || lv.sf != 1 {
				s = g.upsample(s, lv.st, lv.sf)
			}
			h = g.clipSum(g.add(r, s))
			res = nres
		}
		// output_layer: elu, a causal 7x7 convolution to the real and imaginary parts
		h = g.rescale(h, DecoderOutScale)
		h = g.convPadded(g.masked(g.elu(h), res), dec+"output_layer/base_conv_last/conv", 6, 0, 3, 3) // (2, 2, 4T, 480)
		y = g.out(h)
	})
	if err != nil {
		return nil, out, err
	}
	return ins, mlpackage.Feature{Name: "y", DType: mlpackage.Float16, Shape: y.Shape}, nil
}

// maskInputs are the masks of the frames at 1, 2 and 4 time steps a frame: mask (1, 1, T, 1),
// mask2 (1, 1, 2T, 1) and mask4 (1, 1, 4T, 1), as ssMasks makes them.
func (g *dgraph) maskInputs(T int) []mlpackage.Feature {
	var fs []mlpackage.Feature
	for _, r := range []int{1, 2, 4} {
		name := "mask"
		if r > 1 {
			name = fmt.Sprintf("mask%d", r)
		}
		g.masks[r] = g.in(name, 1, 1, r*T, 1)
		fs = append(fs, mlpackage.Feature{Name: name, DType: mlpackage.Float16, Shape: g.masks[r].Shape})
	}
	return fs
}

// ssMasks are the decoder's and the encoder's mask inputs for a mask of frames, into in.
func ssMasks(in map[string][]float32, mask []float32) {
	in["mask"] = mask
	for _, r := range []int{2, 4} {
		m := make([]float32, r*len(mask))
		for i, v := range mask {
			for k := range r {
				m[r*i+k] = v
			}
		}
		in[fmt.Sprintf("mask%d", r)] = m
	}
}

func (g *dgraph) add(x, y *mlpackage.Var) *mlpackage.Var {
	return g.op("add", map[string]any{"x": x, "y": y}, x.Shape...)
}

// clipSum is a block's sum saturated within fp16 (a louder clip than the scales were made for
// is distorted rather than turned to inf and NaN).
func (g *dgraph) clipSum(x *mlpackage.Var) *mlpackage.Var {
	if g.scale == 1 {
		return x
	}
	return g.op("clip", map[string]any{"x": x, "alpha": g.scalar(-60000), "beta": g.scalar(60000)}, x.Shape...)
}

// elu is ELU for values divided by the scale s: elu(s u) / s = max(u, 0) + (exp(min(s u, 0)) -
// 1) / s, s u clipped at -30 (where exp is 0 in fp16) so that it does not overflow.
func (g *dgraph) elu(x *mlpackage.Var) *mlpackage.Var {
	if g.scale == 1 {
		return g.op("elu", map[string]any{"x": x, "alpha": g.scalar(1)}, x.Shape...)
	}
	sh := x.Shape
	pos := g.op("relu", map[string]any{"x": x}, sh...)
	neg := g.op("clip", map[string]any{"x": x, "alpha": g.scalar(-30 / g.scale), "beta": g.scalar(0)}, sh...)
	neg = g.op("mul", map[string]any{"x": neg, "y": g.scalar(g.scale)}, sh...)
	neg = g.op("exp", map[string]any{"x": neg}, sh...)
	neg = g.op("sub", map[string]any{"x": neg, "y": g.scalar(1)}, sh...)
	neg = g.op("mul", map[string]any{"x": neg, "y": g.scalar(1 / g.scale)}, sh...)
	return g.op("add", map[string]any{"x": pos, "y": neg}, sh...)
}

// masked zeroes the time steps before the clip, at a resolution of res steps a frame.
func (g *dgraph) masked(x *mlpackage.Var, res int) *mlpackage.Var {
	return g.op("mul", map[string]any{"x": x, "y": g.masks[res]}, x.Shape...)
}

func (g *dgraph) upsample(x *mlpackage.Var, st, sf int) *mlpackage.Var {
	s := x.Shape
	return g.op("upsample_nearest_neighbor", map[string]any{"x": x, "scale_factor_height": g.B.ScalarI32(int32(st)), "scale_factor_width": g.B.ScalarI32(int32(sf))}, s[0], s[1], s[2]*st, s[3]*sf)
}

// preConv is elu, then a causal 3x3 convolution (time padded 2 before, frequency 1 each side).
func (g *dgraph) preConv(x *mlpackage.Var, name string, res int) *mlpackage.Var {
	return g.convPadded(g.masked(g.elu(x), res), name, 2, 0, 1, 1)
}

// conv is a 1x1 convolution (k 1) of a Flax kernel (1, 1, in, out).
func (g *dgraph) conv(x *mlpackage.Var, name string, res, _ int) *mlpackage.Var {
	return g.convPadded(x, name, 0, 0, 0, 0)
}

// convPadded is a convolution of a Flax kernel (kt, kf, in, out) and its bias, the input padded
// with zeros (time before and after, frequency before and after).
func (g *dgraph) convPadded(x *mlpackage.Var, name string, tb, ta, fb, fa int) *mlpackage.Var {
	k, shape, err := g.W.Get(name + "/kernel")
	if err != nil {
		g.Fail("%v", err)
	}
	bias, _, err := g.W.Get(name + "/bias")
	if err != nil {
		g.Fail("%v", err)
	}
	kt, kf, ci, co := shape[0], shape[1], shape[2], shape[3]
	if x.Shape[1] != ci {
		g.Fail("%s takes %d channels, not %d", name, ci, x.Shape[1])
	}
	wt := make([]float32, co*ci*kt*kf)
	for a := range kt {
		for e := range kf {
			for c := range ci {
				for o := range co {
					wt[((o*ci+c)*kt+a)*kf+e] = k[((a*kf+e)*ci+c)*co+o]
				}
			}
		}
	}
	s := x.Shape
	return g.op("conv", map[string]any{
		"x": x, "weight": g.consts(name+"_w", []int{co, ci, kt, kf}, wt), "bias": g.bias(name+"_b", bias),
		"strides": g.B.Ints(1, 1), "pad_type": g.Str("custom"), "pad": g.B.Ints(int32(tb), int32(ta), int32(fb), int32(fa)),
		"dilations": g.B.Ints(1, 1), "groups": g.B.ScalarI32(1),
	}, s[0], co, s[2]+tb+ta-kt+1, s[3]+fb+fa-kf+1)
}

// preTransposed is elu, then a transposed convolution of a Flax ConvTranspose kernel (kt, kf, in,
// out), which is the true transposed convolution of the kernel reversed in both axes, with
// strides (st, sf), trimmed as sequence_layers trims it: causal in time (the last kt - st steps
// off), "same" in frequency.
func (g *dgraph) preTransposed(x *mlpackage.Var, name string, st, sf, res int) *mlpackage.Var {
	x = g.masked(g.elu(x), res)
	k, shape, err := g.W.Get(name + "/kernel")
	if err != nil {
		g.Fail("%v", err)
	}
	bias, _, err := g.W.Get(name + "/bias")
	if err != nil {
		g.Fail("%v", err)
	}
	kt, kf, ci, co := shape[0], shape[1], shape[2], shape[3]
	if x.Shape[1] != ci {
		g.Fail("%s takes %d channels, not %d", name, ci, x.Shape[1])
	}
	wt := make([]float32, ci*co*kt*kf)
	for a := range kt {
		for e := range kf {
			for c := range ci {
				for o := range co {
					wt[((c*co+o)*kt+a)*kf+e] = k[(((kt-1-a)*kf+(kf-1-e))*ci+c)*co+o]
				}
			}
		}
	}
	tt := max(0, kt-st)
	ft := max(0, kf-sf)
	fl := ft / 2
	s := x.Shape
	oh, ow := (s[2]-1)*st+kt-tt, (s[3]-1)*sf+kf-ft
	return g.op("conv_transpose", map[string]any{
		"x": x, "weight": g.consts(name+"_w", []int{ci, co, kt, kf}, wt), "bias": g.bias(name+"_b", bias),
		"strides": g.B.Ints(int32(st), int32(sf)), "pad_type": g.Str("custom"), "pad": g.B.Ints(0, int32(tt), int32(fl), int32(ft-fl)),
		"dilations": g.B.Ints(1, 1), "groups": g.B.ScalarI32(1),
	}, s[0], co, oh, ow)
}

// Embed sums the quantizer embeddings of a frame's codes (each 0 to 1023) into x's row.
func Embed(w llm.Weights, codes []int, row []float32) error {
	clear(row)
	for level, c := range codes {
		v, err := w.Rows("quantizer/embedding", level*CodebookSize+c, level*CodebookSize+c+1)
		if err != nil {
			return err
		}
		for i := range row {
			row[i] += v[i]
		}
	}
	return nil
}

// InverseSTFT turns the decoder's STFT frames into stereo samples as sequence_layers'
// InverseSTFT with causal padding does: each frame's 481 bins (the Nyquist one 0) to 960
// samples, windowed by the Hann window divided by the overlap of its squares, overlapped and
// added every 480 samples, each frame giving the 480 samples that are complete with it. spec is
// (4, frames, 480) as the decoder lays it out (real and imaginary parts of the left channel,
// then of the right); prev is the 480 samples of each channel the last frame left (nil at the
// start). It returns the samples interleaved (left, right) and the tail for the next call.
func InverseSTFT(spec []float32, frames int, prev [2][]float64) ([]float32, [2][]float64) {
	win := synthesisWindow()
	cosT, sinT := dftTables()
	out := make([]float32, 2*frames*stftHop)
	var next [2][]float64
	buf := make([]float64, stftLength)
	for ch := range 2 {
		tail := prev[ch]
		if tail == nil {
			tail = make([]float64, stftHop)
		}
		re, im := spec[(2*ch)*frames*stftBins:], spec[(2*ch+1)*frames*stftBins:]
		for t := range frames {
			// irfft of bins 0..480 (480 the Nyquist bin, 0)
			for n := range stftLength {
				s := float64(re[t*stftBins])
				for k := 1; k < stftBins; k++ {
					idx := (k * n) % stftLength
					s += 2 * (float64(re[t*stftBins+k])*cosT[idx] - float64(im[t*stftBins+k])*sinT[idx])
				}
				buf[n] = s / stftLength * win[n]
			}
			for i := range stftHop {
				out[2*(t*stftHop+i)+ch] = float32(tail[i] + buf[i])
			}
			tail = append([]float64(nil), buf[stftHop:]...)
		}
		next[ch] = tail
	}
	return out, next
}

func dftTables() (cosT, sinT []float64) {
	cosT, sinT = make([]float64, stftLength), make([]float64, stftLength)
	for i := range stftLength {
		a := 2 * math.Pi * float64(i) / stftLength
		cosT[i], sinT[i] = math.Cos(a), math.Sin(a)
	}
	return cosT, sinT
}

// synthesisWindow is inverse_stft_window_fn(480, hann_window) of length 960: the periodic Hann
// window divided by the sum of the squares of the windows that overlap at each sample.
func synthesisWindow() []float64 {
	w := make([]float64, stftLength)
	for i := range w {
		w[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/stftLength)
	}
	out := make([]float64, stftLength)
	for i := range w {
		var den float64
		for j := i % stftHop; j < stftLength; j += stftHop {
			den += w[j] * w[j]
		}
		out[i] = w[i] / den
	}
	return out
}
