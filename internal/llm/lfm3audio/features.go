package lfm3audio

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

// The feature extractor of LFM3-Audio and LFM3-Omni, as llama.cpp's converter checks it and
// mtmd_audio_preprocessor_lfm3a computes it.
const (
	SampleRate = 16000
	nFFT       = 400
	hop        = 160
)

// Features returns the log mel frames of a clip at SampleRate (samples in [-1, 1]), mel bins
// wide, time major, and their number: the clip padded with zeros on the right so that no sample
// is dropped, frames of 400 samples every 160 under a periodic Hann window, the power spectrum,
// a Slaney mel filter bank (0 to 8 kHz, area normalized), log10 with a floor of 1e-12, then
// (v + 4) / 4.
func Features(samples []float32, mels int) ([]float32, int) {
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

// MelFilters is the Slaney mel filter bank (mels, n_fft/2+1) from 0 Hz to half the sample rate,
// each filter normalized to unit area (librosa's default, llama.cpp's fill_mel_filterbank_matrix).
func MelFilters(mels, nfft, rate int) []float64 {
	const minLogHz, linSlope = 1000.0, 3.0 / 200
	minLogMel := minLogHz * linSlope
	logStep := math.Log(6.4) / 27
	toMel := func(f float64) float64 {
		if f < minLogHz {
			return f * linSlope
		}
		return minLogMel + math.Log(f/minLogHz)/logStep
	}
	toHz := func(m float64) float64 {
		if m < minLogMel {
			return m / linSlope
		}
		return minLogHz * math.Exp((m-minLogMel)*logStep)
	}
	lo, hi := toMel(0), toMel(float64(rate)/2)
	hz := make([]float64, mels+2)
	for i := range hz {
		hz[i] = toHz(lo + (hi-lo)*float64(i)/float64(mels+1))
	}
	bins := nfft/2 + 1
	out := make([]float64, mels*bins)
	for m := 0; m < mels; m++ {
		l, c, r := hz[m], hz[m+1], hz[m+2]
		norm := 2 / math.Max(1e-30, r-l)
		for k := 0; k < bins; k++ {
			f := float64(k) * float64(rate) / float64(nfft)
			var w float64
			switch {
			case f >= l && f <= c:
				w = (f - l) / math.Max(1e-30, c-l)
			case f > c && f <= r:
				w = (r - f) / math.Max(1e-30, r-c)
			}
			out[m*bins+k] = w * norm
		}
	}
	return out
}

// ReadWAV decodes a WAV file (PCM of 8, 16, 24 or 32 bits, or 32 bit float; channels are
// averaged) and resamples it to SampleRate.
func ReadWAV(data []byte) ([]float32, error) {
	if len(data) < 12 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, errors.New("audio: not a WAV file")
	}
	var format, channels, bits int
	var rate int
	var pcm []byte
	for p := 12; p+8 <= len(data); {
		id, size := string(data[p:p+4]), int(binary.LittleEndian.Uint32(data[p+4:]))
		body := data[p+8 : min(p+8+size, len(data))]
		switch id {
		case "fmt ":
			if len(body) < 16 {
				return nil, errors.New("audio: short fmt chunk")
			}
			format, channels = int(binary.LittleEndian.Uint16(body)), int(binary.LittleEndian.Uint16(body[2:]))
			rate, bits = int(binary.LittleEndian.Uint32(body[4:])), int(binary.LittleEndian.Uint16(body[14:]))
			if format == 0xFFFE && len(body) >= 26 { // WAVE_FORMAT_EXTENSIBLE: the sub format's first two bytes
				format = int(binary.LittleEndian.Uint16(body[24:]))
			}
		case "data":
			pcm = body
		}
		p += 8 + size + size%2
	}
	if channels == 0 || rate == 0 || pcm == nil {
		return nil, errors.New("audio: WAV without fmt or data")
	}
	width := bits / 8
	if !(format == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32) || format == 3 && bits == 32) {
		return nil, fmt.Errorf("audio: WAV format %d with %d bits is not supported", format, bits)
	}
	n := len(pcm) / (width * channels)
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		var sum float64
		for c := 0; c < channels; c++ {
			b := pcm[(i*channels+c)*width:]
			var v float64
			switch {
			case format == 3:
				v = float64(math.Float32frombits(binary.LittleEndian.Uint32(b)))
			case bits == 8:
				v = (float64(b[0]) - 128) / 128
			case bits == 16:
				v = float64(int16(binary.LittleEndian.Uint16(b))) / (1 << 15)
			case bits == 24:
				v = float64(int32(uint32(b[0])<<8|uint32(b[1])<<16|uint32(b[2])<<24)>>8) / (1 << 23)
			default:
				v = float64(int32(binary.LittleEndian.Uint32(b))) / (1 << 31)
			}
			sum += v
		}
		out[i] = float32(sum / float64(channels))
	}
	return Resample(out, rate, SampleRate), nil
}

// Resample converts samples from one rate to another with a Hann windowed sinc low pass at the
// lower of the two Nyquist frequencies (32 zero crossings). llama.cpp decodes with miniaudio,
// whose resampler differs; a clip that is not at SampleRate is only approximately the same.
func Resample(x []float32, from, to int) []float32 {
	if from == to || len(x) == 0 {
		return x
	}
	ratio := float64(to) / float64(from)
	cutoff := math.Min(1, ratio) // of the input's Nyquist frequency
	const zeros = 32
	half := zeros / cutoff // the filter's half width, in input samples
	n := int(math.Ceil(float64(len(x)) * ratio))
	out := make([]float32, n)
	for i := range out {
		center := float64(i) / ratio
		lo, hi := int(math.Ceil(center-half)), int(math.Floor(center+half))
		var acc, wsum float64
		for j := max(lo, 0); j <= min(hi, len(x)-1); j++ {
			t := (float64(j) - center) * cutoff
			sinc := 1.0
			if t != 0 {
				sinc = math.Sin(math.Pi*t) / (math.Pi * t)
			}
			w := sinc * (0.5 + 0.5*math.Cos(math.Pi*t/zeros))
			acc += w * float64(x[j])
			wsum += w
		}
		if wsum != 0 {
			out[i] = float32(acc / wsum)
		}
	}
	return out
}
