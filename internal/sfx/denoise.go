package sfx

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/shibukawa/hwnpu/internal/llm"
)

// HostTensors are the DiT's weights the host computes with: the time embedding, the time
// projection and the scale_shift_tables (the blocks' and the head's).
var HostTensors = func() []string {
	names := []string{
		"condition_embedder.time_embedder.linear_1.weight", "condition_embedder.time_embedder.linear_1.bias",
		"condition_embedder.time_embedder.linear_2.weight", "condition_embedder.time_embedder.linear_2.bias",
		"condition_embedder.time_proj.weight", "condition_embedder.time_proj.bias",
		"scale_shift_table",
	}
	for i := range DitLayers {
		names = append(names, fmt.Sprintf("blocks.%d.scale_shift_table", i))
	}
	return names
}()

// Timing is the host's view of the DiT: the time embedding and the modulation of a step.
type Timing struct {
	w1, b1, w2, b2, wp, bp []float32
	head                   []float32   // (2, 1536)
	blocks                 [][]float32 // (6, 1536) each
}

// LoadTiming reads HostTensors (named without the "dit." prefix the programs' weights have).
func LoadTiming(w llm.Weights, prefix string) (*Timing, error) {
	get := func(name string) ([]float32, error) {
		v, _, err := w.Get(prefix + name)
		return v, err
	}
	t := &Timing{}
	var err error
	for _, x := range []struct {
		dst  *[]float32
		name string
	}{
		{&t.w1, "condition_embedder.time_embedder.linear_1.weight"}, {&t.b1, "condition_embedder.time_embedder.linear_1.bias"},
		{&t.w2, "condition_embedder.time_embedder.linear_2.weight"}, {&t.b2, "condition_embedder.time_embedder.linear_2.bias"},
		{&t.wp, "condition_embedder.time_proj.weight"}, {&t.bp, "condition_embedder.time_proj.bias"},
		{&t.head, "scale_shift_table"},
	} {
		if *x.dst, err = get(x.name); err != nil {
			return nil, err
		}
	}
	for i := range DitLayers {
		v, err := get(fmt.Sprintf("blocks.%d.scale_shift_table", i))
		if err != nil {
			return nil, err
		}
		t.blocks = append(t.blocks, v)
	}
	return t, nil
}

func linearHost(w, b, x []float32, out int) []float32 {
	in := len(x)
	y := make([]float32, out)
	for o := range out {
		s := float64(b[o])
		row := w[o*in : (o+1)*in]
		for i, v := range x {
			s += float64(row[i]) * float64(v)
		}
		y[o] = float32(s)
	}
	return y
}

func siluHost(x []float32) []float32 {
	y := make([]float32, len(x))
	for i, v := range x {
		y[i] = float32(float64(v) / (1 + math.Exp(-float64(v))))
	}
	return y
}

// Embed is the time embedding t (1536) and the time projection (6 x 1536) of timestep (sigma
// times 1000): sinusoidal_embedding_1d (cos first), time_embedding, time_projection.
func (t *Timing) Embed(timestep float64) (temb, tmod []float32) {
	half := FreqDim / 2
	e := make([]float32, FreqDim)
	for i := range half {
		a := timestep * math.Pow(10000, -float64(i)/float64(half))
		e[i], e[half+i] = float32(math.Cos(a)), float32(math.Sin(a))
	}
	temb = linearHost(t.w2, t.b2, siluHost(linearHost(t.w1, t.b1, e, DitDim)), DitDim)
	tmod = linearHost(t.wp, t.bp, siluHost(temb), 6*DitDim)
	return temb, tmod
}

// Mod is a group's input mod (ModRows x 1536) for the step's embedding.
func (t *Timing) Mod(first, count int, temb, tmod []float32) []float32 {
	D := DitDim
	out := make([]float32, ModRows(first, count)*D)
	for j := range count {
		sst := t.blocks[first+j]
		for k := range 6 {
			row := out[(6*j+k)*D : (6*j+k+1)*D]
			for i := range D {
				v := sst[k*D+i] + tmod[k*D+i]
				switch k {
				case 1, 4:
					v++
				case 2, 5:
					v /= DitScale
				}
				row[i] = v
			}
		}
	}
	if first+count == DitLayers {
		r := 6 * count
		for i := range D {
			out[r*D+i] = t.head[i] + temb[i]
			out[(r+1)*D+i] = t.head[D+i] + temb[i] + 1
		}
	}
	return out
}

// Sigmas are the flow-match scheduler's (shift, sigma_min 0, extra_one_step): steps values from
// 1 down, linear in the unshifted sigma, shifted.
func Sigmas(steps int, shift float64) []float64 {
	s := make([]float64, steps)
	for i := range s {
		u := 1 - float64(i)/float64(steps)
		s[i] = shift * u / (1 + (shift-1)*u)
	}
	return s
}

// DiT runs the DiT's programs.
type DiT struct {
	Timing *Timing
	Ctx    Program   // Context
	Groups []Program // Blocks, in order
	Sizes  []int     // the blocks of each
}

// Context is the context of a text encoder's output (TextRows x 2048): TextRows x 1536.
func (d *DiT) Context(ctx context.Context, text []float32) ([]float32, error) {
	return d.Ctx.Run(ctx, map[string][]float32{"ctx": text})
}

// Velocity is the DiT's velocity (Frames x 128, rows) at the latents x (Frames x 128, rows) and
// the timestep, for the context.
func (d *DiT) Velocity(ctx context.Context, x []float32, timestep float64, c []float32) ([]float32, error) {
	temb, tmod := d.Timing.Embed(timestep)
	first := 0
	h := x
	for i, p := range d.Groups {
		n := d.Sizes[i]
		y, err := p.Run(ctx, map[string][]float32{"x": h, "ctx": c, "mod": d.Timing.Mod(first, n, temb, tmod)})
		if err != nil {
			return nil, fmt.Errorf("sfx: DiT blocks %d: %w", first, err)
		}
		h = y
		first += n
	}
	return h, nil
}

// Denoise runs the flow-match Euler steps from the noise x (Frames x 128, rows) with
// classifier-free guidance: the velocity is uncond + cfg (cond - uncond), the unconditional
// context that of the empty prompt (all zeros). step is called after each step.
func (d *DiT) Denoise(ctx context.Context, x, cond, uncond []float32, sigmas []float64, cfg float64, step func(i int, took time.Duration)) ([]float32, error) {
	x = append([]float32(nil), x...)
	for i, s := range sigmas {
		t0 := time.Now()
		vp, err := d.Velocity(ctx, x, s*1000, cond)
		if err != nil {
			return nil, err
		}
		vp = append([]float32(nil), vp...)
		vn, err := d.Velocity(ctx, x, s*1000, uncond)
		if err != nil {
			return nil, err
		}
		next := 0.0
		if i+1 < len(sigmas) {
			next = sigmas[i+1]
		}
		dt := float32(next - s)
		for j := range x {
			v := vn[j] + float32(cfg)*(vp[j]-vn[j])
			x[j] += v * dt
		}
		if step != nil {
			step(i, time.Since(t0))
		}
	}
	return x, nil
}
