package whisper

import (
	"math"
	"runtime"
	"sync"

	"github.com/shibukawa/hwnpu/internal/llm/lfm3audio"
)

// The feature extractor of Whisper (transformers' WhisperFeatureExtractor).
const (
	// SampleRate is the rate the encoder hears.
	SampleRate = 16000
	// ClipSamples is 30 seconds: every clip is padded with zeros, or cut, to it.
	ClipSamples = 30 * SampleRate
	nFFT        = 400
	hop         = 160
)

// Features returns the log mel frames of the first 30 seconds of samples (at SampleRate, in
// [-1, 1]), mel bins wide, time major: ClipSamples / hop of them, as WhisperFeatureExtractor
// computes them for one clip. The clip is padded with zeros to 30 seconds and by n_fft/2
// reflected samples on each side (a centered STFT), framed by 400 samples every 160 under a
// periodic Hann window, its power spectrum taken through the Slaney mel filters (0 to 8 kHz, area
// normalized), log10 with a floor of 1e-10, clamped below at the largest value less 8, then
// (v + 4) / 4. The STFT's last frame is dropped, as the extractor does.
func Features(samples []float32, mels int) []float32 {
	n := min(len(samples), ClipSamples)
	clip := make([]float32, ClipSamples)
	copy(clip, samples[:n])
	return logMel(clip, mels)
}

// FeaturesLong is Features over the whole of samples, however long: len(samples) / hop frames,
// the clamp below taken from the largest value of them all, as the extractor computes a long
// clip's features (truncation off) for transformers' long form generation, which then reads
// windows of 30 seconds of them.
func FeaturesLong(samples []float32, mels int) []float32 { return logMel(samples, mels) }

// logMel is the frames of samples, len(samples) / hop of them.
func logMel(samples []float32, mels int) []float32 {
	n := len(samples)
	x := make([]float64, n+nFFT)
	for i, v := range samples {
		x[nFFT/2+i] = float64(v)
	}
	for i := 1; i <= nFFT/2 && i < n; i++ { // reflect, without repeating the edge
		x[nFFT/2-i] = x[nFFT/2+i]
		x[nFFT/2+n-1+i] = x[nFFT/2+n-1-i]
	}
	frames := n / hop
	window := make([]float64, nFFT)
	for i := range window {
		window[i] = 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/nFFT))
	}
	filters := lfm3audio.MelFilters(mels, nFFT, SampleRate)
	bins := nFFT/2 + 1
	cos, sin := make([]float64, nFFT), make([]float64, nFFT)
	for i := range cos {
		a := 2 * math.Pi * float64(i) / nFFT
		cos[i], sin[i] = math.Cos(a), math.Sin(a)
	}
	logs := make([]float64, frames*mels)
	var wg sync.WaitGroup
	workers := runtime.GOMAXPROCS(0)
	for wk := 0; wk < workers; wk++ {
		wg.Add(1)
		go func(wk int) {
			defer wg.Done()
			in := make([]float64, nFFT)
			power := make([]float64, bins)
			for f := wk; f < frames; f += workers {
				for j := range in {
					in[j] = window[j] * x[f*hop+j]
				}
				lfm3audio.PowerSpectrum(power, in, cos, sin)
				for m := 0; m < mels; m++ {
					var s float64
					for k, p := range power {
						s += filters[m*bins+k] * p
					}
					logs[f*mels+m] = math.Log10(math.Max(s, 1e-10))
				}
			}
		}(wk)
	}
	wg.Wait()
	top := math.Inf(-1)
	for _, v := range logs {
		top = math.Max(top, v)
	}
	out := make([]float32, len(logs))
	for i, v := range logs {
		out[i] = float32((math.Max(v, top-8) + 4) / 4)
	}
	return out
}
