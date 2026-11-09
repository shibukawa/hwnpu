package music

import (
	"math"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// encoderLevels are the encoder's residual units encoder_0 to encoder_5, which run on the two
// halves of the STFT's channels (left, right) with the same weights: time and frequency strides,
// the resampling convolution's kernel and name, the channels, and whether a 1x1 convolution
// changes the shortcut's channels.
var encoderLevels = []struct {
	name     string
	st, sf   int
	kt, kf   int
	fb, fa   int // the resampling convolution's frequency padding
	in, out  int
	shortcut bool
	conv     string
}{
	{"encoder_0", 1, 2, 3, 4, 1, 1, 32, 64, true, "conv2d_3x4_a"},
	{"encoder_1", 1, 2, 3, 4, 1, 1, 64, 64, false, "conv2d_3x4_a"},
	{"encoder_2", 1, 3, 3, 6, 1, 2, 64, 128, true, "conv2d_3x6_a"},
	{"encoder_3", 1, 2, 3, 4, 1, 1, 128, 128, false, "conv2d_3x4_a"},
	{"encoder_4", 1, 2, 3, 4, 1, 1, 128, 128, false, "conv2d_3x4_a"},
	{"encoder_5", 2, 2, 4, 4, 1, 1, 128, 256, true, "conv2d_4x4_a"},
}

// Encoder is SpectroStream's encoder as a program over Frames frames (4 STFT frames each): inputs
// x (2, 2, 4 * frames, 480), the STFT (the real and imaginary parts of the left channel, then of
// the right), and the masks (ssMasks), 1 for the frames of the clip and 0 for those around it;
// output y (1, 256, frames, 1), the embeddings the quantizer reads, channels first. Every convolution pads with
// zeros around the clip (causal in time, the strided ones one step after too), which the program
// gets by zeroing each convolution's input outside it, so that a window of frames gives the
// clip's values. The delays the reference puts for streaming do nothing over a whole clip.
type Encoder struct {
	Frames int
}

func (p *Encoder) Hidden() int               { return 0 }
func (p *Encoder) Layers() int               { return 0 }
func (p *Encoder) Output() (int, bool)       { return 0, false }
func (p *Encoder) Embedding() string         { return "" }
func (p *Encoder) Tensors() map[string][]int { return nil }

const enc = "encoder/"

func (p *Encoder) BuildCached(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins, outs []mlpackage.Feature, err error) {
	if opts.Format == "" {
		opts.Format = "fp16"
	}
	if opts.Group == 0 {
		opts.Group = 16
	}
	T := p.Frames
	g := &dgraph{Graph: &blocks.Graph{Name: "spectrostream_encoder", B: b, W: w, Opts: opts}, masks: map[int]*mlpackage.Var{}, scale: 1}
	var y *mlpackage.Var
	err = blocks.Run(func() {
		// the halves of the channels as a batch of two: (2, 2, 4T, 480)
		h := g.in("x", 2, 2, 4*T, stftBins)
		ins = append([]mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: h.Shape}}, g.maskInputs(T)...)
		res := 4 // time steps a frame
		h = g.convPadded(g.masked(h, res), enc+"base_conv_first/conv", 6, 0, 3, 3)
		for _, lv := range encoderLevels {
			p := enc + lv.name + "/"
			r := g.preConv(h, p+"conv2d_3x3/conv", res)
			r = g.strided(g.masked(g.elu(r), res), p+lv.conv+"/conv", lv.st, lv.sf, lv.fb, lv.fa)
			s := g.avgPool(g.masked(h, res), lv.st, lv.sf)
			nres := res / lv.st
			if lv.shortcut {
				s = g.conv(s, p+"shortcut_layer/conv1x1/conv", nres, 0)
			}
			h = g.add(r, s)
			res = nres
		}
		// the halves joined: (1, 512, 2T, 5), channel half * 256 + c
		sh := h.Shape
		h = g.op("reshape", map[string]any{"x": h, "shape": b.Ints(1, int32(2*sh[1]), int32(sh[2]), int32(sh[3]))}, 1, 2*sh[1], sh[2], sh[3])
		// encoder_6: time stride 2, to 256 channels
		r := g.preConv(h, enc+"encoder_6/conv2d_3x3/conv", res)
		r = g.strided(g.masked(g.elu(r), res), enc+"encoder_6/conv2d_4x3_a/conv", 2, 1, 1, 1)
		s := g.conv(g.avgPool(g.masked(h, res), 2, 1), enc+"encoder_6/shortcut_layer/conv1x1/conv", 1, 0)
		h = g.add(r, s)
		res = 1
		// bottleneck
		r = g.preConv(h, enc+"bottleneck/conv2d_3x3/conv", res)
		r = g.preConv(r, enc+"bottleneck/conv2d_3x3_a/conv", res)
		h = g.add(h, r) // (1, 256, T, 5)
		// flattened to (frame, f * 256 + c) as channels: (1, 1280, T, 1)
		h = g.op("transpose", map[string]any{"x": h, "perm": b.Ints(0, 3, 1, 2)}, 1, 5, EmbeddingSize, T)
		h = g.op("reshape", map[string]any{"x": h, "shape": b.Ints(1, 5*EmbeddingSize, int32(T), 1)}, 1, 5*EmbeddingSize, T, 1)
		last := g.conv(h, enc+"output_convs/conv1x1_last/conv", 1, 0)
		sc := g.conv(g.elu(g.conv(g.elu(h), enc+"output_convs/shortcut_layer/conv1x1_b1/conv", 1, 0)), enc+"output_convs/shortcut_layer/conv1x1_b2/conv", 1, 0)
		h = g.add(last, sc) // (1, 256, T, 1)
		y = g.out(h)
	})
	if err != nil {
		return nil, nil, err
	}
	return ins, []mlpackage.Feature{{Name: "y", DType: mlpackage.Float16, Shape: y.Shape}}, nil
}

// strided is a convolution of strides (st, sf), padded as sequence_layers' semicausal time
// padding does (kt - st before, st - 1 after) and in frequency by (fb, fa).
func (g *dgraph) strided(x *mlpackage.Var, name string, st, sf, fb, fa int) *mlpackage.Var {
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
	tb := max(kt-st, 0)
	ta := kt - 1 - tb
	s := x.Shape
	oh := (s[2]+tb+ta-kt)/st + 1
	ow := (s[3]+fb+fa-kf)/sf + 1
	return g.op("conv", map[string]any{
		"x": x, "weight": g.consts(name+"_w", []int{co, ci, kt, kf}, wt), "bias": g.bias(name+"_b", bias),
		"strides": g.B.Ints(int32(st), int32(sf)), "pad_type": g.Str("custom"), "pad": g.B.Ints(int32(tb), int32(ta), int32(fb), int32(fa)),
		"dilations": g.B.Ints(1, 1), "groups": g.B.ScalarI32(1),
	}, s[0], co, oh, ow)
}

// avgPool is the mean over windows (st, sf) of strides (st, sf) (the shortcut's pooling; time
// padded after, where the masked input is zero).
func (g *dgraph) avgPool(x *mlpackage.Var, st, sf int) *mlpackage.Var {
	s := x.Shape
	if sf > 1 {
		x = g.op("reshape", map[string]any{"x": x, "shape": g.B.Ints(int32(s[0]), int32(s[1]), int32(s[2]), int32(s[3]/sf), int32(sf))}, s[0], s[1], s[2], s[3]/sf, sf)
		x = g.op("reduce_mean", map[string]any{"x": x, "axes": g.B.Ints(4), "keep_dims": g.B.ScalarBool(false)}, s[0], s[1], s[2], s[3]/sf)
		s = x.Shape
	}
	if st > 1 {
		x = g.op("reshape", map[string]any{"x": x, "shape": g.B.Ints(int32(s[0]), int32(s[1]), int32(s[2]/st), int32(st), int32(s[3]))}, s[0], s[1], s[2]/st, st, s[3])
		x = g.op("reduce_mean", map[string]any{"x": x, "axes": g.B.Ints(3), "keep_dims": g.B.ScalarBool(false)}, s[0], s[1], s[2]/st, s[3])
	}
	return x
}

// STFT is SpectroStream's STFT of stereo samples (interleaved) as sequence_layers' STFT with
// reverse causal padding: a frame every 480 samples of the 960 from it on (zeros after the
// clip), the periodic Hann window, the 481 bins of the real DFT less the Nyquist one. It returns
// (4, frames, 480): the real and imaginary parts of the left channel, then of the right, frames
// the samples / 480 rounded up.
func STFT(samples []float32) ([]float32, int) {
	n := len(samples) / 2
	frames := (n + stftHop - 1) / stftHop
	out := make([]float32, 4*frames*stftBins)
	win := make([]float64, stftLength)
	for i := range win {
		win[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/stftLength)
	}
	cosT, sinT := dftTables()
	buf := make([]float64, stftLength)
	for ch := range 2 {
		re, im := out[(2*ch)*frames*stftBins:], out[(2*ch+1)*frames*stftBins:]
		for t := range frames {
			for i := range stftLength {
				j := t*stftHop + i
				buf[i] = 0
				if j < n {
					buf[i] = float64(samples[2*j+ch]) * win[i]
				}
			}
			for k := range stftBins {
				var sr, si float64
				for i, v := range buf {
					idx := (k * i) % stftLength
					sr += v * cosT[idx]
					si -= v * sinT[idx]
				}
				re[t*stftBins+k], im[t*stftBins+k] = float32(sr), float32(si)
			}
		}
	}
	return out, frames
}

// Quantize is the residual vector quantizer's codes of an embedding (levels of them, each 0 to
// 1023): at each level the nearest of its embeddings, subtracted before the next.
func Quantize(w llm.Weights, emb []float32, levels int) ([]int, error) {
	r := make([]float64, len(emb))
	for i, v := range emb {
		r[i] = float64(v)
	}
	codes := make([]int, levels)
	for l := range levels {
		tab, err := w.Rows("quantizer/embedding", l*CodebookSize, (l+1)*CodebookSize)
		if err != nil {
			return nil, err
		}
		best, bd := 0, math.Inf(1)
		for c := range CodebookSize {
			var d float64
			for i, v := range tab[c*EmbeddingSize : (c+1)*EmbeddingSize] {
				e := r[i] - float64(v)
				d += e * e
			}
			if d < bd {
				best, bd = c, d
			}
		}
		codes[l] = best
		for i, v := range tab[best*EmbeddingSize : (best+1)*EmbeddingSize] {
			r[i] -= float64(v)
		}
	}
	return codes, nil
}
