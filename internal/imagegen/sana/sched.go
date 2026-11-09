package sana

import (
	"math"
)

// SCMTimesteps are Sana-Sprint's TrigFlow times of steps SCM steps (diffusers' SCMScheduler):
// max, 1.3, 0 for two steps (the pipeline's intermediate time), else steps+1 evenly from max
// to 0.
func SCMTimesteps(steps int, max float64) []float64 {
	if steps == 2 {
		return []float64{max, 1.3, 0}
	}
	t := make([]float64, steps+1)
	for i := range t {
		t[i] = max - max*float64(i)/float64(steps)
	}
	return t
}

// SCMInput is the factor the latents x take into the DiT at TrigFlow time s: (x / sigma_data)
// times sqrt(st^2 + (1 - st)^2), and st, the time the DiT is told.
func SCMInput(s float64) (scale, st float64) {
	st = math.Sin(s) / (math.Cos(s) + math.Sin(s))
	return math.Sqrt(st*st + (1-st)*(1-st)), st
}

// DPM is diffusers' DPMSolverMultistepScheduler as Sana's repositories set it: DPM-Solver++ of
// order 2 (midpoint) on flow sigmas shifted by Shift, the final sigma 0 and the last step of
// the first order.
type DPM struct {
	Sigmas    []float64 // steps+1, the last 0
	Timesteps []float64 // steps, what the DiT is told (int(sigma * 1000))
	prev      []float32 // the previous step's x0 prediction
	order     int       // lower_order_nums
}

// NewDPM sets steps steps: sigmas 1 - linspace(1, 1/train, steps+1), shifted and reversed,
// without the last.
func NewDPM(steps int, shift float64, train int) *DPM {
	d := &DPM{}
	raw := make([]float64, steps+1)
	for i := range raw {
		a := 1 - float64(i)*(1-1/float64(train))/float64(steps)
		s := 1 - a
		raw[i] = shift * s / (1 + (shift-1)*s)
	}
	for i := steps; i >= 1; i-- {
		// float32 as numpy's astype in set_timesteps; the timestep from the float64 sigma
		d.Sigmas = append(d.Sigmas, float64(float32(raw[i])))
		d.Timesteps = append(d.Timesteps, math.Trunc(raw[i]*float64(train)))
	}
	d.Sigmas = append(d.Sigmas, 0)
	return d
}

func lambda(sigma float64) float64 { return math.Log(1-sigma) - math.Log(sigma) }

// Step takes the flow prediction v of step i at the latents x and returns the next latents
// (in place).
func (d *DPM) Step(i int, v, x []float32) []float32 {
	n := len(d.Timesteps)
	ss, st := d.Sigmas[i], d.Sigmas[i+1]
	x0 := make([]float32, len(x))
	for j := range x {
		x0[j] = x[j] - float32(ss)*v[j]
	}
	lowerFinal := i == n-1 // final_sigmas_type "zero"
	lowerSecond := i == n-2 && n < 15
	_ = lowerSecond // order 2 at most: the second order is the highest anyway
	if st == 0 {
		copy(x, x0)
	} else if d.order < 1 || lowerFinal {
		h := lambda(st) - lambda(ss)
		a, b := float32(st/ss), float32((1-st)*(math.Exp(-h)-1))
		for j := range x {
			x[j] = a*x[j] - b*x0[j]
		}
	} else {
		s1 := d.Sigmas[i-1]
		h, h0 := lambda(st)-lambda(ss), lambda(ss)-lambda(s1)
		r0 := h0 / h
		a, b := float32(st/ss), float32((1-st)*(math.Exp(-h)-1))
		for j := range x {
			d0 := x0[j]
			d1 := float32(1/r0) * (x0[j] - d.prev[j])
			x[j] = a*x[j] - b*d0 - 0.5*b*d1
		}
	}
	d.prev = x0
	if d.order < 2 {
		d.order++
	}
	return x
}
