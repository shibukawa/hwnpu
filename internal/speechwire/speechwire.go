// Package speechwire is what the speech engines (internal/speech, internal/llm/lfm3audio,
// internal/llm/whisper) and their client (ai/speech) share without the engine: the rate a
// transcriber reads, and WAV files read and written as those packages do (docs/design/hwnpud.md,
// "Clients without the engine").
//
// ReadWAVAt and WriteWAV are copies of lfm3audio.ReadWAVAt and speech.WriteWAV, whose packages
// are in the models' code (model-recipes.yaml) and are left as they are; ai/speech/engine's tests
// check that they give the same.
package speechwire

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"runtime"
	"sync"
	"sync/atomic"
)

// TranscribeRate is the samples a second a transcriber (Whisper) reads: whisper.SampleRate,
// which ai/speech/engine checks.
const TranscribeRate = 16000

// ReadWAVAt decodes a WAV file (PCM of 8, 16, 24 or 32 bits, or 32 bit float; channels are
// averaged) and resamples it to the rate given.
func ReadWAVAt(data []byte, to int) ([]float32, error) {
	chans, rate, err := decodeWAV(data)
	if err != nil {
		return nil, err
	}
	out := make([]float32, len(chans[0]))
	for i := range out {
		var sum float64
		for _, c := range chans {
			sum += float64(c[i])
		}
		out[i] = float32(sum / float64(len(chans)))
	}
	return Resample(out, rate, to), nil
}

func decodeWAV(data []byte) ([][]float32, int, error) {
	if len(data) < 12 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, 0, errors.New("audio: not a WAV file")
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
				return nil, 0, errors.New("audio: short fmt chunk")
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
		return nil, 0, errors.New("audio: WAV without fmt or data")
	}
	width := bits / 8
	if !(format == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32) || format == 3 && bits == 32) {
		return nil, 0, fmt.Errorf("audio: WAV format %d with %d bits is not supported", format, bits)
	}
	n := len(pcm) / (width * channels)
	out := make([][]float32, channels)
	for c := range out {
		out[c] = make([]float32, n)
	}
	for i := 0; i < n; i++ {
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
			out[c][i] = float32(v)
		}
	}
	return out, rate, nil
}

// Resample converts samples from one rate to another with a Hann windowed sinc low pass at the
// lower of the two Nyquist frequencies (32 zero crossings). llama.cpp decodes with miniaudio,
// whose resampler differs; a clip that is not at its rate is only approximately the same.
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
	// each output on its own: in parts of 4096 on every thread (a song is minutes of sines)
	const part = 4096
	var next atomic.Int64
	var wg sync.WaitGroup
	for range min(runtime.GOMAXPROCS(0), (n+part-1)/part) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for p := int(next.Add(1) - 1); p*part < n; p = int(next.Add(1) - 1) {
				for i := p * part; i < min(n, (p+1)*part); i++ {
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
			}
		}()
	}
	wg.Wait()
	return out
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
