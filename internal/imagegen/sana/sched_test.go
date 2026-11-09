package sana

import (
	"math"
	"testing"
)

// TestDPM checks NewDPM and Step against diffusers' DPMSolverMultistepScheduler with
// Efficient-Large-Model/Sana_600M_1024px_diffusers' scheduler_config.json (diffusers 0.42.1).
func TestDPM(t *testing.T) {
	d := NewDPM(20, 3, 1000)
	ts := []float64{999, 982, 963, 944, 922, 899, 874, 847, 817, 785, 749, 710, 666, 617, 562, 499, 428, 345, 249, 136}
	sig := []float64{0.9996665, 0.9824195, 0.9639411, 0.9440944, 0.9227217, 0.8996398, 0.8746352, 0.8474572, 0.8178097, 0.78534, 0.7496248, 0.7101521, 0.6662961, 0.6172836, 0.5621483, 0.4996665, 0.4282652, 0.3458875, 0.2497916, 0.1362397, 0}
	for i := range ts {
		if d.Timesteps[i] != ts[i] {
			t.Errorf("timestep %d: %v, want %v", i, d.Timesteps[i], ts[i])
		}
	}
	for i := range sig {
		if math.Abs(d.Sigmas[i]-sig[i]) > 1e-6 {
			t.Errorf("sigma %d: %v, want %v", i, d.Sigmas[i], sig[i])
		}
	}
	d = NewDPM(5, 3, 1000)
	want := [][]float64{
		{0.9692221, -0.419208, 0.209604, 1.869194},
		{0.8933488, -0.1957732, 0.09788661, 1.521038},
		{0.7248967, 0.3237682, -0.1618841, 0.7232113},
		{0.377209, 1.418149, -0.7090747, -0.9468505},
		{-0.2813434, 3.498741, -1.749371, -4.118289},
	}
	x := []float32{1, -0.5, 0.25, 2}
	base := []float32{0.3, -1, 0.5, 1.5}
	for i := range want {
		v := make([]float32, len(x))
		for j := range v {
			v[j] = base[j]*float32(i+1) + x[j]*0.1
		}
		x = d.Step(i, v, x)
		for j := range x {
			if math.Abs(float64(x[j])-want[i][j]) > 1e-5*math.Max(1, math.Abs(want[i][j])) {
				t.Errorf("step %d: x %v, want %v", i, x, want[i])
				break
			}
		}
	}
}

func TestSCM(t *testing.T) {
	ts := SCMTimesteps(2, 1.5708)
	if ts[0] != 1.5708 || ts[1] != 1.3 || ts[2] != 0 {
		t.Errorf("SCMTimesteps(2) = %v", ts)
	}
	// diffusers' first step: st = sin(1.5708)/(cos+sin) = 1.0 (ref_sana.py's step 0)
	_, st := SCMInput(1.5708)
	if math.Abs(st-1) > 1e-5 {
		t.Errorf("st = %v", st)
	}
	_, st = SCMInput(1.3)
	if math.Abs(st-0.7827) > 1e-4 {
		t.Errorf("st(1.3) = %v, want 0.7827", st)
	}
}
