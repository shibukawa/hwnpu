// Package capture gives the samples their live inputs as streams of the flow package: the
// microphone (and, with the samples of pictures, the camera) through hwcamera
// (github.com/shibukawa/hwcamera), which reaches each operating system's own API without cgo.
// A recording read from a file is the same kind of stream (speech.Chunks, flow.Live), so a
// sample runs on either.
package capture

import (
	"cmp"
	"context"
	"fmt"
	"time"

	"github.com/shibukawa/hwcamera/audio"

	"github.com/shibukawa/hwnpu/ai/flow"
)

// MicOptions are how a microphone is read. The zero value is the system's default input at
// 16 kHz.
type MicOptions struct {
	// Device is an input of Mics; "" is the system's default input.
	Device string
	// Rate is the samples a second (0: 16000, what a transcriber and a VAD read).
	Rate int
	// Buffer is how much sound waits for a reader that is busy (0: a minute). A microphone does
	// not wait: what a stage after it does not read in time is lost. The stream is read in a
	// goroutine of its own and kept, so that a transcriber and a translator that take seconds
	// for an utterance lose nothing of the next one.
	Buffer time.Duration
	// Lost (nil for none) is called when sound was lost all the same: the stages after the
	// microphone are further behind than Buffer.
	Lost func()
}

// Mic is the stream of a microphone's sound: mono, in [-1, 1], in the buffers the device gives
// (20 ms), timed by the samples read. The microphone is opened at the first read and closed
// when the stream ends. The stream ends, without an error, when ctx is cancelled, after the
// sound that was waiting: what follows it then works off what was heard (give those stages a
// context of their own).
func Mic(ctx context.Context, o MicOptions) flow.Stream[[]float32] {
	rate := cmp.Or(o.Rate, 16000)
	raw := func(yield func(flow.Chunk[[]float32], error) bool) {
		mic, err := audio.Open(ctx, o.Device, audio.WithSampleRate(rate), audio.WithChannels(1), audio.WithFormat(audio.F32))
		if err != nil {
			if ctx.Err() == nil {
				yield(flow.Chunk[[]float32]{}, fmt.Errorf("the microphone: %w", err))
			}
			return
		}
		defer mic.Close()
		pos := 0
		for {
			b, err := mic.Read(ctx)
			if err != nil {
				if ctx.Err() == nil {
					yield(flow.Chunk[[]float32]{}, fmt.Errorf("the microphone: %w", err))
				}
				return
			}
			if b.Discontinuity && o.Lost != nil {
				o.Lost()
			}
			pcm := b.F32() // the buffer is the caller's: not copied
			c := flow.Chunk[[]float32]{Data: pcm, Start: flow.Samples(pos, rate)}
			pos += len(pcm)
			c.Len = flow.Samples(pos, rate) - c.Start
			if !yield(c, nil) {
				return
			}
		}
	}
	// read ahead of the reader by up to Buffer; the reading ends with ctx, and what was read by
	// then is still passed on
	n := int(cmp.Or(o.Buffer, time.Minute) / period)
	return flow.Async(context.WithoutCancel(ctx), raw, n)
}

// period is the length of the buffers a microphone gives (hwcamera's default).
const period = 20 * time.Millisecond

// Input is a microphone, or on Linux the monitor of an output (what the speakers play).
type Input struct {
	ID, Name string
	Default  bool
	Monitor  bool
}

// Mics lists the audio inputs, for a sample's --device.
func Mics(ctx context.Context) ([]Input, error) {
	ds, err := audio.Devices(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]Input, len(ds))
	for i, d := range ds {
		out[i] = Input{ID: d.ID, Name: d.Name, Default: d.Default, Monitor: d.Monitor}
	}
	return out, nil
}
