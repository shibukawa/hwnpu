package speech

import (
	"bytes"
	"encoding/binary"
	"io"
	"math"
)

// ISTFT turns STFT frames into samples as llama.cpp's mtmd_audio_streaming_istft does: each
// frame's inverse DFT (the negative frequencies the conjugates of the positive ones) under a
// periodic Hann window, overlapped and added every hop samples and divided by the windows'
// summed squares; the first (n_fft-hop)/2 samples are padding and dropped, and Flush gives the
// rest after the last frame, of which the last (n_fft-hop)/2 are dropped too.
type ISTFT struct {
	nfft, hop  int
	window     []float64
	cos, sin   []float64
	overlap    []float64
	wsum       []float64
	toRemove   int
	re, im, td []float64
}

// NewISTFT returns an empty ISTFT.
func NewISTFT(nfft, hop int) *ISTFT {
	t := &ISTFT{nfft: nfft, hop: hop, window: make([]float64, nfft), cos: make([]float64, nfft), sin: make([]float64, nfft),
		overlap: make([]float64, nfft), wsum: make([]float64, nfft), toRemove: (nfft - hop) / 2,
		re: make([]float64, nfft), im: make([]float64, nfft), td: make([]float64, nfft)}
	for i := range t.window {
		t.window[i] = 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(nfft)))
		a := 2 * math.Pi * float64(i) / float64(nfft)
		t.cos[i], t.sin[i] = math.Cos(a), math.Sin(a)
	}
	return t
}

// Frame adds one frame given as log magnitudes then phases of the nfft/2+1 bins (the codec's
// output) and returns the samples that are complete.
func (t *ISTFT) Frame(polar []float32) []float32 {
	bins := t.nfft/2 + 1
	for j := 0; j < bins; j++ {
		mag := math.Exp(float64(polar[j]))
		ph := float64(polar[bins+j])
		t.re[j], t.im[j] = mag*math.Cos(ph), mag*math.Sin(ph)
	}
	for j := 1; j < bins-1; j++ {
		t.re[t.nfft-j], t.im[t.nfft-j] = t.re[j], -t.im[j]
	}
	for n := 0; n < t.nfft; n++ {
		var acc float64
		for k := 0; k < t.nfft; k++ {
			idx := (k * n) % t.nfft
			acc += t.re[k]*t.cos[idx] - t.im[k]*t.sin[idx]
		}
		t.td[n] = acc / float64(t.nfft)
	}
	for j := 0; j < t.nfft; j++ {
		t.wsum[j] += t.window[j] * t.window[j]
		t.overlap[j] += t.td[j] * t.window[j]
	}
	out := t.take(t.hop)
	n := min(t.toRemove, len(out))
	t.toRemove -= n
	return out[n:]
}

// take returns the first n samples, normalized, and shifts the buffers.
func (t *ISTFT) take(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		v := t.overlap[i]
		if t.wsum[i] > 1e-8 {
			v /= t.wsum[i]
		}
		out[i] = float32(v)
	}
	copy(t.overlap, t.overlap[n:])
	copy(t.wsum, t.wsum[n:])
	for i := t.nfft - n; i < t.nfft; i++ {
		t.overlap[i], t.wsum[i] = 0, 0
	}
	return out
}

// Flush returns the samples after the last frame, without the trailing padding.
func (t *ISTFT) Flush() []float32 {
	var out []float32
	for rest := t.nfft - t.hop; rest > 0; {
		n := min(rest, t.hop)
		out = append(out, t.take(n)...)
		rest -= n
	}
	trim := (t.nfft - t.hop) / 2
	return out[:max(0, len(out)-trim)]
}

// WriteWAV writes samples in [-1, 1] as a 16 bit mono WAV file.
func WriteWAV(w io.Writer, pcm []float32, rate int) error {
	var b bytes.Buffer
	b.WriteString("RIFF")
	binary.Write(&b, binary.LittleEndian, uint32(36+2*len(pcm)))
	b.WriteString("WAVEfmt ")
	for _, v := range []any{uint32(16), uint16(1), uint16(1), uint32(rate), uint32(2 * rate), uint16(2), uint16(16)} {
		binary.Write(&b, binary.LittleEndian, v)
	}
	b.WriteString("data")
	binary.Write(&b, binary.LittleEndian, uint32(2*len(pcm)))
	for _, v := range pcm {
		binary.Write(&b, binary.LittleEndian, int16(math.Round(math.Max(-1, math.Min(1, float64(v)))*32767)))
	}
	_, err := w.Write(b.Bytes())
	return err
}
