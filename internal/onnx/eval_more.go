package onnx

import (
	"fmt"
	"math"
	"slices"
)

// evalIf runs the branch of an If its condition (in[0]) picks, the outer values it reads given
// as the If's other inputs (Node.Graphs).
func evalIf(n *Node, in []*Tensor) ([]*Tensor, error) {
	if len(n.Graphs) != 2 || len(in) == 0 || in[0] == nil || len(in[0].F) == 0 {
		return nil, fmt.Errorf("an If without its branches or condition")
	}
	b := n.Graphs[1]
	if in[0].F[0] != 0 {
		b = n.Graphs[0]
	}
	given := map[string]*Tensor{}
	for i, name := range n.In[1:] {
		if in[i+1] != nil {
			given[name] = in[i+1]
		}
	}
	var names []string
	for _, o := range b.Outputs {
		names = append(names, o.Name)
	}
	out, err := b.Eval(given, names)
	if err != nil {
		return nil, err
	}
	res := make([]*Tensor, len(names))
	for i, nm := range names {
		res[i] = out[nm]
	}
	return res, nil
}

// scatterND is ONNX's ScatterND: data with the slices at indices (..., k) replaced by updates
// (or combined with them by reduction).
func scatterND(data, idx, upd *Tensor, reduction string) (*Tensor, error) {
	out := &Tensor{Shape: slices.Clone(data.Shape), F: slices.Clone(data.F), Type: data.Type}
	k := idx.Shape[len(idx.Shape)-1]
	if k > len(data.Shape) {
		return nil, fmt.Errorf("ScatterND: indices of %d axes into %v", k, data.Shape)
	}
	st := strides(data.Shape)
	slice := Len(data.Shape[k:])
	n := Len(idx.Shape[:len(idx.Shape)-1])
	if len(upd.F) != n*slice {
		return nil, fmt.Errorf("ScatterND: updates %v for %d slices of %d", upd.Shape, n, slice)
	}
	for i := range n {
		at := 0
		for j := range k {
			v := int(idx.F[i*k+j])
			if v < 0 {
				v += data.Shape[j]
			}
			if v < 0 || v >= data.Shape[j] {
				return nil, fmt.Errorf("ScatterND: index %d out of %d", v, data.Shape[j])
			}
			at += v * st[j]
		}
		for e := range slice {
			u := upd.F[i*slice+e]
			switch reduction {
			case "add":
				out.F[at+e] += u
			case "mul":
				out.F[at+e] *= u
			case "max":
				out.F[at+e] = math.Max(out.F[at+e], u)
			case "min":
				out.F[at+e] = math.Min(out.F[at+e], u)
			default:
				out.F[at+e] = u
			}
		}
	}
	return out, nil
}

// convTranspose is a 2-D ConvTranspose: x (N, C, H, W), w (C, M/group, kh, kw).
func convTranspose(n *Node, x, w, b *Tensor) (*Tensor, error) {
	if len(x.Shape) != 4 || len(w.Shape) != 4 {
		return nil, fmt.Errorf("ConvTranspose of %v and %v: only 2-D", x.Shape, w.Shape)
	}
	N, C, H, W := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	group := int(n.Int("group", 1))
	mg, kh, kw := w.Shape[1], w.Shape[2], w.Shape[3]
	M := mg * group
	two := func(name string, def int) [2]int {
		v := n.IntList(name)
		if len(v) >= 2 {
			return [2]int{v[0], v[1]}
		}
		return [2]int{def, def}
	}
	s, d, op := two("strides", 1), two("dilations", 1), two("output_padding", 0)
	pads := n.IntList("pads")
	if len(pads) < 4 {
		pads = []int{0, 0, 0, 0}
	}
	if n.Str("auto_pad", "NOTSET") != "NOTSET" || len(n.IntList("output_shape")) > 0 {
		return nil, fmt.Errorf("ConvTranspose: auto_pad and output_shape are not evaluated")
	}
	Ho := s[0]*(H-1) + op[0] + (kh-1)*d[0] + 1 - pads[0] - pads[2]
	Wo := s[1]*(W-1) + op[1] + (kw-1)*d[1] + 1 - pads[1] - pads[3]
	if group == 1 && s == [2]int{kh, kw} && d == [2]int{1, 1} && op == [2]int{0, 0} && slices.Equal(pads[:4], []int{0, 0, 0, 0}) {
		return convTransposeTiles(x, w, b)
	}
	out := &Tensor{Shape: []int{N, M, Ho, Wo}, F: make([]float64, N*M*Ho*Wo), Type: x.Type}
	cg := C / group
	for nn := range N {
		for c := range C {
			g := c / cg
			xp := x.F[(nn*C+c)*H*W:][:H*W]
			for m := range mg {
				oc := g*mg + m
				op := out.F[(nn*M+oc)*Ho*Wo:][:Ho*Wo]
				wk := w.F[(c*mg+m)*kh*kw:][:kh*kw]
				for y := range H {
					for i := range W {
						v := xp[y*W+i]
						if v == 0 {
							continue
						}
						for ky := range kh {
							oy := y*s[0] + ky*d[0] - pads[0]
							if oy < 0 || oy >= Ho {
								continue
							}
							for kx := range kw {
								ox := i*s[1] + kx*d[1] - pads[1]
								if ox < 0 || ox >= Wo {
									continue
								}
								op[oy*Wo+ox] += v * wk[ky*kw+kx]
							}
						}
					}
				}
			}
		}
	}
	if b != nil {
		for nn := range N {
			for m := range M {
				o := out.F[(nn*M+m)*Ho*Wo:][:Ho*Wo]
				for j := range o {
					o[j] += b.F[m]
				}
			}
		}
	}
	return out, nil
}

// convTransposeTiles is a ConvTranspose whose stride is its kernel (no overlap, no pads): each
// input pixel's kernel-sized tile of every output channel, one matrix product.
func convTransposeTiles(x, w, b *Tensor) (*Tensor, error) {
	N, C, H, W := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	M, kh, kw := w.Shape[1], w.Shape[2], w.Shape[3]
	Ho, Wo := H*kh, W*kw
	out := &Tensor{Shape: []int{N, M, Ho, Wo}, F: make([]float64, N*M*Ho*Wo), Type: x.Type}
	K := M * kh * kw
	for nn := range N {
		// p[(y W + x) K + (m kh + ky) kw + kx] = sum over c of x[c, y, x] w[c, m, ky, kx]
		xt := transpose(&Tensor{Shape: []int{C, H * W}, F: x.F[nn*C*H*W:][:C*H*W], Type: x.Type}, []int{1, 0})
		p, err := matmul(xt, &Tensor{Shape: []int{C, K}, F: w.F, Type: w.Type})
		if err != nil {
			return nil, err
		}
		o := out.F[nn*M*Ho*Wo:]
		for y := range H {
			for i := range W {
				row := p.F[(y*W+i)*K:][:K]
				for m := range M {
					bias := 0.0
					if b != nil {
						bias = b.F[m]
					}
					for ky := range kh {
						dst := o[(m*Ho+y*kh+ky)*Wo+i*kw:][:kw]
						src := row[(m*kh+ky)*kw:][:kw]
						for kx, v := range src {
							dst[kx] = v + bias
						}
					}
				}
			}
		}
	}
	return out, nil
}

// instanceNorm normalizes each (N, C) plane of x over its other axes.
func instanceNorm(x, scale, bias *Tensor, eps float64) *Tensor {
	N, C := x.Shape[0], x.Shape[1]
	sp := Len(x.Shape[2:])
	out := &Tensor{Shape: slices.Clone(x.Shape), F: make([]float64, len(x.F)), Type: x.Type}
	for nn := range N {
		for c := range C {
			p := x.F[(nn*C+c)*sp:][:sp]
			var m, v float64
			for _, a := range p {
				m += a
			}
			m /= float64(sp)
			for _, a := range p {
				v += (a - m) * (a - m)
			}
			v /= float64(sp)
			r := 1 / math.Sqrt(v+eps)
			o := out.F[(nn*C+c)*sp:][:sp]
			for j, a := range p {
				o[j] = (a-m)*r*scale.F[c] + bias.F[c]
			}
		}
	}
	return out
}

// layerNorm normalizes x over its axes from axis on, times scale plus bias (broadcast).
func layerNorm(x, scale, bias *Tensor, ax int, eps float64) (*Tensor, error) {
	a := axis(ax, len(x.Shape))
	inner := Len(x.Shape[a:])
	if len(scale.F) != inner && len(scale.F) != 1 || bias != nil && len(bias.F) != inner && len(bias.F) != 1 {
		return nil, fmt.Errorf("LayerNormalization of %v by %v", x.Shape, scale.Shape)
	}
	out := &Tensor{Shape: slices.Clone(x.Shape), F: make([]float64, len(x.F)), Type: x.Type}
	at := func(t *Tensor, j int) float64 {
		if len(t.F) == 1 {
			return t.F[0]
		}
		return t.F[j]
	}
	for o := range Len(x.Shape[:a]) {
		p := x.F[o*inner:][:inner]
		var m, v float64
		for _, e := range p {
			m += e
		}
		m /= float64(inner)
		for _, e := range p {
			v += (e - m) * (e - m)
		}
		v /= float64(inner)
		r := 1 / math.Sqrt(v+eps)
		q := out.F[o*inner:][:inner]
		for j, e := range p {
			y := (e - m) * r * at(scale, j)
			if bias != nil {
				y += at(bias, j)
			}
			q[j] = y
		}
	}
	return out, nil
}

// resize is ONNX's Resize of the last two axes (the others unscaled): nearest or linear, by
// scales or sizes, with the common coordinate transformations.
func resize(n *Node, x, scales, sizes *Tensor) (*Tensor, error) {
	r := len(x.Shape)
	if r < 2 {
		return nil, fmt.Errorf("Resize of %v", x.Shape)
	}
	out := slices.Clone(x.Shape)
	sc := make([]float64, r)
	for i := range sc {
		sc[i] = 1
	}
	switch {
	case sizes != nil && len(sizes.F) == r:
		for i := range r {
			out[i] = int(sizes.F[i])
			sc[i] = float64(out[i]) / float64(x.Shape[i])
		}
	case scales != nil && len(scales.F) == r:
		for i := range r {
			sc[i] = scales.F[i]
			out[i] = int(math.Floor(float64(x.Shape[i]) * sc[i]))
		}
	default:
		return nil, fmt.Errorf("Resize without scales or sizes")
	}
	for i := range r - 2 {
		if out[i] != x.Shape[i] {
			return nil, fmt.Errorf("Resize of %v to %v: only the last two axes", x.Shape, out)
		}
	}
	mode := n.Str("mode", "nearest")
	ct := n.Str("coordinate_transformation_mode", "half_pixel")
	nm := n.Str("nearest_mode", "round_prefer_floor")
	ca := n.Float("cubic_coeff_a", -0.75)
	if mode != "nearest" && mode != "linear" && mode != "cubic" {
		return nil, fmt.Errorf("Resize: mode %s is not evaluated", mode)
	}
	if n.Int("exclude_outside", 0) != 0 || n.Int("antialias", 0) != 0 {
		return nil, fmt.Errorf("Resize: exclude_outside and antialias are not evaluated")
	}
	src := func(o, nOut, nIn int, s float64) float64 {
		f := float64(o)
		switch ct {
		case "asymmetric":
			return f / s
		case "align_corners":
			if nOut == 1 {
				return 0
			}
			return f * float64(nIn-1) / float64(nOut-1)
		case "pytorch_half_pixel":
			if nOut == 1 {
				return 0
			}
			return (f+0.5)/s - 0.5
		default: // half_pixel
			return (f+0.5)/s - 0.5
		}
	}
	// an output's taps: input indices (clamped to the edges) and their weights
	type tap struct {
		i []int
		w []float64
	}
	cubic := func(t float64) [4]float64 {
		w := [4]float64{
			((ca*(t+1)-5*ca)*(t+1)+8*ca)*(t+1) - 4*ca,
			((ca+2)*t-(ca+3))*t*t + 1,
			((ca+2)*(1-t)-(ca+3))*(1-t)*(1-t) + 1,
		}
		w[3] = 1 - w[0] - w[1] - w[2]
		return w
	}
	taps := func(nOut, nIn int, s float64) []tap {
		t := make([]tap, nOut)
		clamp := func(i int) int { return min(max(i, 0), nIn-1) }
		for o := range nOut {
			f := src(o, nOut, nIn, s)
			switch mode {
			case "nearest":
				var i int
				switch nm {
				case "floor":
					i = int(math.Floor(f))
				case "ceil":
					i = int(math.Ceil(f))
				case "round_prefer_ceil":
					i = int(math.Floor(f + 0.5))
				default:
					i = int(math.Ceil(f - 0.5))
				}
				t[o] = tap{[]int{clamp(i)}, []float64{1}}
			case "linear":
				f = min(max(f, 0), float64(nIn-1))
				i0 := int(math.Floor(f))
				t[o] = tap{[]int{i0, clamp(i0 + 1)}, []float64{1 - (f - float64(i0)), f - float64(i0)}}
			default: // cubic: four taps around floor(f)
				i0 := int(math.Floor(f))
				w := cubic(f - float64(i0))
				t[o] = tap{[]int{clamp(i0 - 1), clamp(i0), clamp(i0 + 1), clamp(i0 + 2)}, w[:]}
			}
		}
		return t
	}
	H, W, Ho, Wo := x.Shape[r-2], x.Shape[r-1], out[r-2], out[r-1]
	ty, tx := taps(Ho, H, sc[r-2]), taps(Wo, W, sc[r-1])
	res := &Tensor{Shape: out, F: make([]float64, Len(out)), Type: x.Type}
	for p := range Len(x.Shape[:r-2]) {
		in := x.F[p*H*W:][:H*W]
		o := res.F[p*Ho*Wo:][:Ho*Wo]
		for y, a := range ty {
			for i, b := range tx {
				var v float64
				for ka, iy := range a.i {
					row := in[iy*W:][:W]
					var h float64
					for kb, ix := range b.i {
						h += row[ix] * b.w[kb]
					}
					v += h * a.w[ka]
				}
				o[y*Wo+i] = v
			}
		}
	}
	return res, nil
}
