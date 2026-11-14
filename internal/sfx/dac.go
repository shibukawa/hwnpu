package sfx

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// The DAC VAE's decoder (vae/vae_128d_48k.pth: decoder_dim 2048, decoder_rates 8 5 4 3 2,
// latent_dim 128, continuous): latents at 50 a second to 48 kHz mono.
const (
	LatentDim     = 128
	LatentRate    = 50
	SampleRate    = 48000
	HopLength     = SampleRate / LatentRate // 960
	decoderDim    = 2048
	residualReach = 3 + 9 + 27 // a DecoderBlock's residual units' reach at its output rate (k 7, dilations 1, 3, 9)
)

var decoderRates = []int{8, 5, 4, 3, 2}

// DACStage is a stage of the decoder as a program: In frames at its input rate in, Rate times as
// many out. Stage 0 is post_quant_conv, the first convolution and the first DecoderBlock;
// stages 1 to 3 a DecoderBlock each; stage 4 the last DecoderBlock and the output layer (snake,
// convolution, tanh).
type DACStage struct {
	Index    int
	In       int // the frames a call takes, halos included
	Channels int // the input's channels
	Out      int // its output's channels
	Rate     int // output frames an input frame
	// Halo is the input frames a call reads before and after the frames it gives: more than
	// the stage's receptive field.
	Halo int
}

// DACStages are the stages, each over about as many frames as its output fits in the Neural
// Engine's 16384 of a dimension (stage 0 over about 1024: Core ML put its transposed convolution on
// the CPU over 1536), its output a multiple of 9 frames for phaseConv.
func DACStages() []DACStage {
	ins := []int{1026, 3195, 3996, 4998, 7992}
	var st []DACStage
	ch := decoderDim
	for i, r := range decoderRates {
		out := ch / 2
		in := ch
		if i == 0 {
			in = LatentDim
		}
		halo := (residualReach+r-1)/r + 2
		if i == 0 {
			halo += 3 // the first convolution's
		}
		s := DACStage{Index: i, In: ins[i], Channels: in, Out: out, Rate: r, Halo: halo}
		if i == len(decoderRates)-1 {
			s.Out = 1
		}
		st = append(st, s)
		ch = out
	}
	return st
}

// Part is the stage's program: inputs x (1, Channels, 1, In), m (1, 1, 1, In) and mo (1, 1, 1,
// In * Rate), 1 for the frames of the clip and 0 outside it, and mg (1, 1, 1, 1), Magic (the decoder's convolutions pad with
// zeros, which the program gets by zeroing every convolution's input outside the clip); output y
// (1, Out, 1, In * Rate), the frame k of x giving frames k * Rate to (k + 1) * Rate - 1.
func (s DACStage) Part() *Part {
	return &Part{Name: fmt.Sprintf("dac_%d", s.Index), build: func(g *graph) {
		x := g.input("x", 1, s.Channels, 1, s.In)
		m := g.input("m", 1, 1, 1, s.In)
		mo := g.input("mo", 1, 1, 1, s.In*s.Rate)
		d := &dac{graph: g, prefix: "state_dict.", mg: g.input("mg", 1, 1, 1, 1)}
		i := s.Index
		if i == 0 {
			x = d.conv(g.mul(x, m), "post_quant_conv", 1, 0, false)
			x = d.conv(g.mul(x, m), "decoder.model.0", 1, 3, true)
		}
		x = d.decoderBlock(x, m, mo, fmt.Sprintf("decoder.model.%d.block.", i+1), s.Rate)
		if i == len(decoderRates)-1 {
			x = g.mul(d.snake(x, "decoder.model.6"), mo)
			x = g.unary("tanh", d.conv(x, "decoder.model.7", 1, 3, true))
		}
		g.output(x)
	}}
}

type dac struct {
	*graph
	prefix string
	mg     *mlpackage.Var // the input mg, Magic
}

// weightNorm is a weight-normed convolution's weight: weight_g * weight_v / |weight_v|, the norm
// over all but the first axis (out for Conv1d, in for ConvTranspose1d).
func (d *dac) weightNorm(name string) ([]float32, []int) {
	v, shape, err := d.W.Get(d.prefix + name + ".weight_v")
	if err != nil {
		d.Fail("%v", err)
	}
	gv, _, err := d.W.Get(d.prefix + name + ".weight_g")
	if err != nil {
		d.Fail("%v", err)
	}
	n := len(v) / shape[0]
	w := make([]float32, len(v))
	for o := range shape[0] {
		var ss float64
		for _, x := range v[o*n : (o+1)*n] {
			ss += float64(x) * float64(x)
		}
		f := float64(gv[o]) / math.Sqrt(ss)
		for j := range n {
			w[o*n+j] = float32(float64(v[o*n+j]) * f)
		}
	}
	return w, shape
}

// conv is a Conv1d (weight-normed when wn) over x (1, C, 1, T), padded with zeros by pad on
// each side, of the given dilation.
func (d *dac) conv(x *mlpackage.Var, name string, dil, pad int, wn bool) *mlpackage.Var {
	var w []float32
	var shape []int
	if wn {
		w, shape = d.weightNorm(name)
	} else {
		var err error
		w, shape, err = d.W.Get(d.prefix + name + ".weight")
		if err != nil {
			d.Fail("%v", err)
		}
	}
	bias := d.Get(d.prefix + name + ".bias")
	co, ci, k := shape[0], shape[1], shape[2]
	if x.Shape[1] != ci {
		d.Fail("%s takes %d channels, not %d", name, ci, x.Shape[1])
	}
	if dil > 1 && d.Opts.Backend != llm.BackendIntel {
		return d.phaseConv(x, name, w, bias, ci, co, k, dil, pad)
	}
	return d.conv1d(x, name, w, bias, co, k, dil, pad, pad)
}

// conv1d is a convolution of weight w (co, ci, k) and bias over x (1, ci, 1, T) along T,
// padded with zeros by padL and padR, of the given dilation. Except for Intel's NPU it runs
// along the height, x laid out as (1, ci, T, 1): Core AI's Neural Engine miscomputes a
// convolution along the width of a kernel over 1 and 512 or more channels whose input another
// operation computes (design.md 16.197), and runs it right along the height.
func (d *dac) conv1d(x *mlpackage.Var, name string, w, bias []float32, co, k, dil, padL, padR int) *mlpackage.Var {
	ci, t := x.Shape[1], x.Shape[3]
	out := t + padL + padR - dil*(k-1)
	if d.Opts.Backend == llm.BackendIntel {
		return d.op("conv", map[string]any{
			"x": x, "weight": d.F16s(name+"_w", []int{co, ci, 1, k}, w), "bias": d.F16s(name+"_b", []int{co}, bias),
			"strides": d.B.Ints(1, 1), "pad_type": d.Str("custom"), "pad": d.B.Ints(0, 0, int32(padL), int32(padR)),
			"dilations": d.B.Ints(1, int32(dil)), "groups": d.B.ScalarI32(1),
		}, 1, co, 1, out)
	}
	y := d.op("conv", map[string]any{
		"x": d.reshape(x, 1, ci, t, 1), "weight": d.F16s(name+"_w", []int{co, ci, k, 1}, w), "bias": d.F16s(name+"_b", []int{co}, bias),
		"strides": d.B.Ints(1, 1), "pad_type": d.Str("custom"), "pad": d.B.Ints(int32(padL), int32(padR), 0, 0),
		"dilations": d.B.Ints(int32(dil), 1), "groups": d.B.ScalarI32(1),
	}, 1, co, out, 1)
	return d.reshape(y, 1, co, 1, out)
}

// phaseConv is conv's dilated convolution as an undilated one: x (T frames, a multiple of dil)
// is laid out as (1, ci, T / dil, dil) (frame row dil + phase), convolved along the rows by a
// kernel of k by 1, padded by pad / dil rows, and laid back; the output's frame row dil + phase
// reads the input's frames (row + j) dil + phase. Core AI's Neural Engine miscomputes some
// dilated convolutions whose input another operation computes (dilation 3 over 128 channels at
// 8000 and 15000 frames but not 7680 or 16000) and an explicit pad before one (design.md 16.197),
// and runs this right.
func (d *dac) phaseConv(x *mlpackage.Var, name string, w, bias []float32, ci, co, k, dil, pad int) *mlpackage.Var {
	t := x.Shape[3]
	if t%dil != 0 || pad%dil != 0 || 2*pad != dil*(k-1) {
		d.Fail("%s: %d frames, padding %d for dilation %d (a multiple of it, the same frames out)", name, t, pad, dil)
	}
	rows, pr := t/dil, int32(pad/dil)
	z := d.op("conv", map[string]any{
		"x": d.reshape(x, 1, ci, rows, dil), "weight": d.F16s(name+"_w", []int{co, ci, k, 1}, w), "bias": d.F16s(name+"_b", []int{co}, bias),
		"strides": d.B.Ints(1, 1), "pad_type": d.Str("custom"), "pad": d.B.Ints(pr, pr, 0, 0),
		"dilations": d.B.Ints(1, 1), "groups": d.B.ScalarI32(1),
	}, 1, co, rows, dil)
	return d.reshape(z, 1, co, 1, t)
}

// convTranspose is the DecoderBlock's WNConvTranspose1d (kernel 2 stride, padding ceil(stride
// / 2), output_padding stride % 2): In frames to In * stride. Core ML puts a conv_transpose of
// stride 5 (or 7) on the CPU, and those of strides 8, 4 and 3 gave 22 %, 45 % and 4 % off the
// reference on the M1's Neural Engine (stride 2 0.3 %), so all but stride 2 are polyphase.
func (d *dac) convTranspose(x *mlpackage.Var, name string, stride int) *mlpackage.Var {
	w, shape := d.weightNorm(name)
	bias := d.Get(d.prefix + name + ".bias")
	ci, co, k := shape[0], shape[1], shape[2]
	if x.Shape[1] != ci || k != 2*stride {
		d.Fail("%s: %v for %d channels, stride %d", name, shape, x.Shape[1], stride)
	}
	p := (stride + 1) / 2
	op := stride % 2
	t := x.Shape[3]
	if stride != 2 {
		return d.polyphase(x, name, w, bias, ci, co, stride, p)
	}
	return d.op("conv_transpose", map[string]any{
		"x": x, "weight": d.F16s(name+"_w", []int{ci, co, 1, k}, w), "bias": d.F16s(name+"_b", []int{co}, bias),
		"strides": d.B.Ints(1, int32(stride)), "pad_type": d.Str("custom"), "pad": d.B.Ints(0, 0, int32(p), int32(p-op)),
		"dilations": d.B.Ints(1, 1), "groups": d.B.ScalarI32(1),
	}, 1, co, 1, t*stride)
}

// polyphase is the transposed convolution of weight w (ci, co, 2 s) and stride s as a convolution
// and an interleave: the full output (before the padding p is cut off) at q s + phase is x[q]
// w[phase] + x[q - 1] w[s + phase], a kernel-2 convolution of s co channels (phase major) over
// q = 0..T; its phases interleaved into time, the output is frames p to p + T s of it.
func (d *dac) polyphase(x *mlpackage.Var, name string, w, bias []float32, ci, co, s, p int) *mlpackage.Var {
	t := x.Shape[3]
	k := 2 * s
	pw := make([]float32, s*co*ci*2)
	pb := make([]float32, s*co)
	for ph := range s {
		for o := range co {
			row := (ph*co + o) * ci * 2
			for c := range ci {
				pw[row+c*2] = w[(c*co+o)*k+s+ph] // tap 0: x[q - 1]
				pw[row+c*2+1] = w[(c*co+o)*k+ph] // tap 1: x[q]
			}
			pb[ph*co+o] = bias[o]
		}
	}
	z := d.conv1d(x, name+"_p", pw, pb, s*co, 2, 1, 1, 1)
	z = d.reshape(z, s, co, t+1)
	z = d.transpose(z, 1, 2, 0)
	z = d.reshape(z, 1, co, 1, (t+1)*s)
	return d.slice(z, 3, p, p+t*s)
}

// snakeSmall is the alpha under which snake takes sin(alpha x)^2 / alpha by its series: 1 /
// alpha would be over fp16's range (the first DecoderBlock's snake has alphas down to 8e-6) and
// alpha x under it.
const snakeSmall = 1e-2

// Magic is the value of every stage's input mg: added to a value of fp16 under 512 and taken off
// again, it rounds the value to an integer (fp16's spacing from 1024 to 2048 is 1). It comes as an
// input, as a constant Core ML folds the addition and the subtraction away.
const Magic = 1536

// sinTerms are sin(pi / 2 w) = w (b0 + b1 w^2 + ... + b4 w^8) for |w| <= 1, Taylor's terms (the
// next, 3.6e-6, is under fp16's precision and its smallest normal number).
var sinTerms = func() []float32 {
	b := make([]float32, 5)
	f := math.Pi / 2
	for k := range b {
		v := math.Pow(f, float64(2*k+1))
		for i := 2; i <= 2*k+1; i++ {
			v /= float64(i)
		}
		if k%2 == 1 {
			v = -v
		}
		b[k] = float32(v)
	}
	return b
}()

// snake is Snake1d: x + sin(alpha x)^2 / (alpha + 1e-9), alpha a channel's. The Neural Engine has
// no sin (Core ML puts it on the CPU), so y = alpha x is reduced to r = y - n pi (n = round(y / pi),
// mg's rounding; pi in two parts, 3.125 n exact for |n| < 32) and sin(r) is its series in r / (pi
// / 2); sin^2 has the period pi. For the channels of an alpha under snakeSmall, sin(y)^2 / alpha =
// alpha x^2 (1 - y^2 / 3 + 2 y^4 / 45) (the next term, y^6 / 315, under 1e-5 of it while |y| <
// 0.4), alpha x^2 computed as (x sqrt|alpha|)^2 with alpha's sign.
func (d *dac) snake(x *mlpackage.Var, name string) *mlpackage.Var {
	c := x.Shape[1]
	alpha := d.Get(d.prefix+name+".alpha", 1, c, 1)
	big, inv := make([]float32, c), make([]float32, c)
	small, root, sign := make([]float32, c), make([]float32, c), make([]float32, c)
	anySmall := false
	for i, a := range alpha {
		if math.Abs(float64(a)) >= snakeSmall {
			big[i], inv[i] = a, float32(1/(float64(a)+1e-9))
			continue
		}
		anySmall = true
		small[i] = a
		root[i] = float32(math.Sqrt(math.Abs(float64(a))))
		sign[i] = 1
		if a < 0 {
			sign[i] = -1
		}
	}
	vec := func(suffix string, v []float32) *mlpackage.Var {
		return d.F16s(name+suffix, []int{1, c, 1, 1}, v)
	}
	y := d.clip(d.mul(x, vec("_alpha", big)), -1500, 1500)
	var sn *mlpackage.Var
	if d.Opts.Backend != llm.BackendIntel {
		// Core AI runs sin on the Neural Engine, and folds (v + mg) - mg into v even with mg an
		// input, so the rounding below does not round there; the CPU interpreter, which runs the
		// Core ML export, computes in fp32 and does not round it either (design.md 16.197)
		sn = d.unary("sin", y)
	} else {
		n := d.sub(d.add(d.scale(y, float32(1/math.Pi)), d.mg), d.mg)
		r := d.sub(d.sub(y, d.scale(n, 3.125)), d.scale(n, float32(math.Pi-3.125)))
		w := d.scale(r, float32(2/math.Pi))
		u := d.mul(w, w)
		b := sinTerms
		p := d.addScalar(d.scale(u, b[4]), b[3])
		for k := 2; k >= 0; k-- {
			p = d.addScalar(d.mul(p, u), b[k])
		}
		sn = d.mul(w, p)
	}
	out := d.add(x, d.mul(d.mul(sn, sn), vec("_inv_alpha", inv)))
	if !anySmall {
		return out
	}
	rt := d.mul(x, vec("_root_alpha", root))
	ax2 := d.mul(d.mul(rt, rt), vec("_sign_alpha", sign))
	v := d.mul(x, vec("_small_alpha", small))
	v2 := d.mul(v, v)
	// 1 - v2 / 3 + 2 v2^2 / 45 = 1 + v2 (2 v2 / 45 - 1 / 3)
	q := d.addScalar(d.mul(v2, d.addScalar(d.scale(v2, 2.0/45), -1.0/3)), 1)
	return d.add(out, d.mul(ax2, q))
}

// residualUnit is x + conv1x1(snake(conv7 dilated(snake(x)))).
func (d *dac) residualUnit(x, mo *mlpackage.Var, p string, dil int) *mlpackage.Var {
	y := d.mul(d.snake(x, p+"block.0"), mo)
	y = d.conv(y, p+"block.1", dil, 3*dil, true)
	y = d.conv(d.snake(y, p+"block.2"), p+"block.3", 1, 0, true)
	return d.add(x, y)
}

// decoderBlock is snake, the transposed convolution, and the residual units of dilations 1, 3, 9.
func (d *dac) decoderBlock(x, m, mo *mlpackage.Var, p string, stride int) *mlpackage.Var {
	x = d.mul(d.snake(x, p+"0"), m)
	x = d.convTranspose(x, p+"1", stride)
	for i, dil := range []int{1, 3, 9} {
		x = d.residualUnit(x, mo, fmt.Sprintf("%s%d.", p, i+2), dil)
	}
	return x
}
