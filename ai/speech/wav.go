package speech

import (
	"io"

	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/internal/speechwire"
)

// ReadWAV decodes a WAV file (PCM of 8, 16, 24 or 32 bits, or 32 bit float) into mono samples in
// [-1, 1] at rate samples a second: its channels are averaged and it is resampled. A
// Transcriber and a VAD read TranscribeRate.
func ReadWAV(data []byte, rate int) ([]float32, error) {
	return speechwire.ReadWAVAt(data, rate)
}

// WriteWAV writes mono samples in [-1, 1] as a 16 bit WAV file.
func WriteWAV(w io.Writer, pcm []float32, rate int) error {
	return speechwire.WriteWAV(w, pcm, rate)
}

// Chunks is audio (mono, at rate) as a stream of chunks of n samples, the last one final and
// shorter when the audio does not fill it: a recording as a microphone's frames would come
// (with flow.Live, at the pace they would come too).
func Chunks(pcm []float32, rate, n int) flow.Stream[[]float32] {
	return flow.Rechunk(flow.Of(pcm, flow.Samples(len(pcm), rate)), rate, n)
}
