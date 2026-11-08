package lfm3audio

import (
	"math"
	"math/rand/v2"
	"testing"
)

// refFeatures is Features as first written (one thread, a division per term): Features must
// give the same values to the bit.
func refFeatures(samples []float32, mels int) ([]float32, int) {
	n := len(samples)
	padded := nFFT
	if n > nFFT {
		padded = n + (hop-(n-nFFT)%hop)%hop
	}
	frames := (padded-nFFT)/hop + 1
	window := make([]float64, nFFT)
	for i := range window {
		window[i] = 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/nFFT))
	}
	filters := MelFilters(mels, nFFT, SampleRate)
	bins := nFFT/2 + 1
	cos, sin := make([]float64, nFFT), make([]float64, nFFT)
	for i := range cos {
		a := 2 * math.Pi * float64(i) / nFFT
		cos[i], sin[i] = math.Cos(a), math.Sin(a)
	}
	out := make([]float32, frames*mels)
	in := make([]float64, nFFT)
	power := make([]float64, bins)
	for f := 0; f < frames; f++ {
		for j := range in {
			in[j] = 0
			if k := f*hop + j; k < n {
				in[j] = window[j] * float64(samples[k])
			}
		}
		for k := 0; k < bins; k++ {
			var re, im float64
			for j, v := range in {
				idx := (k * j) % nFFT
				re += v * cos[idx]
				im -= v * sin[idx]
			}
			power[k] = re*re + im*im
		}
		for m := 0; m < mels; m++ {
			var sum float64
			for k, p := range power {
				sum += p * filters[m*bins+k]
			}
			out[f*mels+m] = float32((math.Log10(math.Max(sum, 1e-12)) + 4) / 4)
		}
	}
	return out, frames
}

func TestFeaturesReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	for _, n := range []int{0, 100, 400, 401, 16000*3 + 77} {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.Float64()*2 - 1)
		}
		want, wf := refFeatures(s, 128)
		got, gf := Features(s, 128)
		if gf != wf || len(got) != len(want) {
			t.Fatalf("%d samples: %d frames, want %d", n, gf, wf)
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%d samples: value %d is %g, want %g", n, i, got[i], want[i])
			}
		}
	}
}

func BenchmarkFeatures(b *testing.B) {
	s := make([]float32, 30*SampleRate)
	for i := range s {
		s[i] = float32(i%97) * 0.01
	}
	for range b.N {
		Features(s, 128)
	}
}
