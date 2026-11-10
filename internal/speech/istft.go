package speech

import (
	"bytes"
	"encoding/binary"
	"io"
	"math"
	"runtime"
	"sync"
	"sync/atomic"
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
	t.inverse(polar, t.re, t.im, t.td)
	return t.add(t.td)
}

// Frames adds the frames of stft one after another, as Frame each, with their inverse DFTs
// taken in parallel, and returns the samples that are complete.
func (t *ISTFT) Frames(stft []float32) []float32 {
	w := 2 * (t.nfft/2 + 1)
	frames := len(stft) / w
	td := make([]float64, frames*t.nfft)
	var next atomic.Int64
	var wg sync.WaitGroup
	for range min(runtime.GOMAXPROCS(0), frames) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			re, im := make([]float64, t.nfft), make([]float64, t.nfft)
			for f := int(next.Add(1) - 1); f < frames; f = int(next.Add(1) - 1) {
				t.inverse(stft[f*w:(f+1)*w], re, im, td[f*t.nfft:(f+1)*t.nfft])
			}
		}()
	}
	wg.Wait()
	var out []float32
	for f := range frames {
		out = append(out, t.add(td[f*t.nfft:(f+1)*t.nfft])...)
	}
	return out
}

// inverse writes the inverse DFT of the frame polar to td, with re and im as scratch.
func (t *ISTFT) inverse(polar []float32, re, im, td []float64) {
	nfft := t.nfft
	bins := nfft/2 + 1
	for j := 0; j < bins; j++ {
		mag := math.Exp(float64(polar[j]))
		ph := float64(polar[bins+j])
		re[j], im[j] = mag*math.Cos(ph), mag*math.Sin(ph)
	}
	for j := 1; j < bins-1; j++ {
		re[nfft-j], im[nfft-j] = re[j], -im[j]
	}
	re, im = re[:nfft], im[:nfft]
	cs, sn := t.cos[:nfft], t.sin[:nfft]
	// Four outputs at a time, each summed over k in order as one would alone (the result is the
	// same to the bit); the table index (k*n) mod nfft steps by n without a division.
	n := 0
	for ; n+4 <= nfft; n += 4 {
		var a0, a1, a2, a3 float64
		var i0, i1, i2, i3 int
		for k := range re {
			r, m := re[k], im[k]
			a0 += r*cs[i0] - m*sn[i0]
			a1 += r*cs[i1] - m*sn[i1]
			a2 += r*cs[i2] - m*sn[i2]
			a3 += r*cs[i3] - m*sn[i3]
			if i0 += n; i0 >= nfft {
				i0 -= nfft
			}
			if i1 += n + 1; i1 >= nfft {
				i1 -= nfft
			}
			if i2 += n + 2; i2 >= nfft {
				i2 -= nfft
			}
			if i3 += n + 3; i3 >= nfft {
				i3 -= nfft
			}
		}
		td[n], td[n+1], td[n+2], td[n+3] = a0/float64(nfft), a1/float64(nfft), a2/float64(nfft), a3/float64(nfft)
	}
	for ; n < nfft; n++ {
		var acc float64
		idx := 0
		for k := range re {
			acc += re[k]*cs[idx] - im[k]*sn[idx]
			if idx += n; idx >= nfft {
				idx -= nfft
			}
		}
		td[n] = acc / float64(nfft)
	}
}

// add overlaps and adds the inverse DFT td of a frame and returns the samples that are complete.
func (t *ISTFT) add(td []float64) []float32 {
	for j := 0; j < t.nfft; j++ {
		t.wsum[j] += t.window[j] * t.window[j]
		t.overlap[j] += td[j] * t.window[j]
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
func WriteWAV(w io.Writer, pcm []float32, rate int) error { return WriteWAVChannels(w, pcm, rate, 1) }

// WriteWAVChannels writes interleaved samples in [-1, 1] of channels as a 16 bit WAV file.
func WriteWAVChannels(w io.Writer, pcm []float32, rate, channels int) error {
	var b bytes.Buffer
	b.WriteString("RIFF")
	binary.Write(&b, binary.LittleEndian, uint32(36+2*len(pcm)))
	b.WriteString("WAVEfmt ")
	for _, v := range []any{uint32(16), uint16(1), uint16(channels), uint32(rate), uint32(2 * channels * rate), uint16(2 * channels), uint16(16)} {
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
