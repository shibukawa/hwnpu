package music

import (
	"math"
	"sync"
)

// resampy's kaiser_best filter (resampy/data/kaiser_best.npz): the right half of a sinc of 50
// zero crossings, 8192 values a crossing, rolled off at 0.9173 of the Nyquist frequency and
// tapered by a Kaiser window of beta 12.98 (the table's values to 2e-12).
const (
	kbZeros     = 50
	kbPrecision = 8192
	kbRolloff   = 0.9173473712608761
	kbBeta      = 12.984585250325175
)

var kaiserBest = sync.OnceValue(func() []float64 {
	n := kbZeros * kbPrecision
	w := make([]float64, n+1)
	i0b := besselI0(kbBeta)
	for k := range w {
		x := float64(k) / kbPrecision // in zero crossings
		s := 1.0
		if x != 0 {
			s = math.Sin(math.Pi*kbRolloff*x) / (math.Pi * kbRolloff * x)
		}
		r := float64(k) / float64(n)
		w[k] = kbRolloff * s * besselI0(kbBeta*math.Sqrt(math.Max(0, 1-r*r))) / i0b
	}
	return w
})

func besselI0(x float64) float64 {
	sum, term := 1.0, 1.0
	for k := 1; k < 500; k++ {
		term *= (x / 2) * (x / 2) / float64(k*k)
		sum += term
		if term < sum*1e-17 {
			break
		}
	}
	return sum
}

// Resample is resampy.resample(x, from, to) with its default filter (kaiser_best), as
// magenta-rt's Waveform.resample: MusicCoCa's audio side reads 16 kHz made so, and its log mel
// spectrogram up to 8 kHz is sensitive to the filter near the Nyquist frequency (a Hann windowed
// sinc moved an embedding by 40%).
func Resample(x []float32, from, to int) []float32 {
	if from == to {
		return x
	}
	ratio := float64(to) / float64(from)
	win := kaiserBest()
	scale := math.Min(1, ratio)
	gain := 1.0
	if ratio < 1 {
		gain = ratio
	}
	n := int(float64(len(x)) * ratio)
	out := make([]float32, n)
	step := int(scale * kbPrecision)
	nwin := len(win)
	at := func(i int, eta float64) float64 {
		d := 0.0
		if i+1 < nwin {
			d = win[i+1] - win[i]
		}
		return gain * (win[i] + eta*d)
	}
	for t := range out {
		reg := float64(t) / ratio
		m := int(reg)
		frac := scale * (reg - float64(m))
		idx := frac * kbPrecision
		off := int(idx)
		eta := idx - float64(off)
		var y float64
		for i := range min(m+1, (nwin-off)/step) {
			y += at(off+i*step, eta) * float64(x[m-i])
		}
		frac = scale - frac
		idx = frac * kbPrecision
		off = int(idx)
		eta = idx - float64(off)
		for k := range min(len(x)-m-1, (nwin-off)/step) {
			y += at(off+k*step, eta) * float64(x[m+k+1])
		}
		out[t] = float32(y)
	}
	return out
}
