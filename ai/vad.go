package ai

import (
	"cmp"
	"context"
	"math"
	"time"
)

// VADOptions tune VAD, a detector of speech by loudness. The zero value of a field takes its
// default.
type VADOptions struct {
	// Rate is the samples a second of the audio (default 16000, Transcriber's).
	Rate int
	// Frame is how much audio each decision looks at (default 20 ms).
	Frame time.Duration
	// Margin is how far above the background noise, in dB, a frame must be to be speech
	// (default 12). The background is followed as it changes: it falls at once to a quieter
	// frame and rises slowly through the frames that are not speech.
	Margin float64
	// MinLevel is the least level, in dBFS, that is ever speech (default -50), so that a quiet
	// room's noise is not speech however far above its floor.
	MinLevel float64
	// Hangover is the silence that ends an utterance (default 600 ms).
	Hangover time.Duration
	// MinSpeech is the shortest utterance (default 250 ms): a shorter burst (a click, a cough)
	// is left out.
	MinSpeech time.Duration
	// Padding is the audio kept before an utterance's first speech and after its last (default
	// 200 ms), so that soft beginnings and endings of words are not cut off.
	Padding time.Duration
	// MaxSpeech ends an utterance that goes on this long (default 25 s), so that a transcriber
	// of 30 second windows reads it in one.
	MaxSpeech time.Duration
}

func (o VADOptions) withDefaults() VADOptions {
	o.Rate = cmp.Or(o.Rate, TranscribeRate)
	o.Frame = cmp.Or(o.Frame, 20*time.Millisecond)
	o.Margin = cmp.Or(o.Margin, 12)
	o.MinLevel = cmp.Or(o.MinLevel, -50)
	o.Hangover = cmp.Or(o.Hangover, 600*time.Millisecond)
	o.MinSpeech = cmp.Or(o.MinSpeech, 250*time.Millisecond)
	o.Padding = cmp.Or(o.Padding, 200*time.Millisecond)
	o.MaxSpeech = cmp.Or(o.MaxSpeech, 25*time.Second)
	return o
}

// VAD finds speech in audio by its loudness against the background. It needs no model and runs
// on the CPU; Silero VAD on the NPU is to replace it (docs/app-api-tasks.md F2).
type VAD struct{ o VADOptions }

// NewVAD is a VAD with options o.
func NewVAD(o VADOptions) *VAD { return &VAD{o.withDefaults()} }

// Options are the VAD's options with the defaults filled in.
func (v *VAD) Options() VADOptions { return v.o }

// level is a frame's level in dBFS.
func level(frame []float32) float64 {
	var sum float64
	for _, x := range frame {
		sum += float64(x) * float64(x)
	}
	return 10 * math.Log10(sum/float64(len(frame))+1e-12)
}

// Detect passes on the speech of a stream of audio (mono, at the VAD's rate, in [-1, 1]) and
// leaves out the rest: each utterance as chunks as it is heard, the last one final. Its first
// chunk comes once the speech has lasted MinSpeech, with the Padding before it; its final chunk
// once Hangover of silence has followed (with Padding of that silence), at MaxSpeech, or at the
// end of the stream. The chunks' times are the input's, counted in samples, so that an
// utterance can be found in the recording.
func (v *VAD) Detect(ctx context.Context, s Stream[[]float32]) Stream[[]float32] {
	o := v.o
	dur := func(d time.Duration) int { return int(int64(d) * int64(o.Rate) / int64(time.Second)) }
	fn, hang, minN, padN, maxN := max(1, dur(o.Frame)), dur(o.Hangover), dur(o.MinSpeech), dur(o.Padding), dur(o.MaxSpeech)
	return func(yield func(Chunk[[]float32], error) bool) {
		const (
			silence = iota
			onset   // speech for less than MinSpeech: not passed on yet
			speech
		)
		var (
			state             = silence
			buf               []float32 // input not yet a whole frame
			pos               int64     // the sample of buf[0]
			floor             = math.NaN()
			pre               []float32 // the silence before, up to padN samples
			held              []float32 // onset: the utterance so far, pre included
			heldAt            int64     // the sample of held[0]
			pending           []float32 // speech: silence after the last speech, not passed on yet
			sent              int64     // speech: the sample after the last one passed on
			uttAt             int64     // speech: the utterance's first sample
			speechN, silenceN int
		)
		emit := func(data []float32, at int64, final bool) bool {
			c := Chunk[[]float32]{Data: data, Start: samples(at, o.Rate), Final: final}
			c.Len = samples(at+int64(len(data)), o.Rate) - c.Start
			sent = at + int64(len(data))
			return yield(c, nil)
		}
		keepPre := func(tail []float32) {
			pre = append(pre, tail...)
			if len(pre) > padN {
				pre = append([]float32(nil), pre[len(pre)-padN:]...)
			}
		}
		// end passes on the end of an utterance: Padding of the silence after it, final; the
		// rest of the silence is the next one's padding.
		end := func() bool {
			n := min(len(pending), padN)
			ok := emit(append([]float32(nil), pending[:n]...), sent, true)
			pre = nil
			keepPre(pending[n:])
			pending, state = nil, silence
			return ok
		}
		step := func(frame []float32, at int64) bool {
			lv := level(frame)
			if math.IsNaN(floor) {
				floor = lv
			}
			loud := lv > max(floor+o.Margin, o.MinLevel)
			switch {
			case lv < floor:
				floor += (lv - floor) * 0.3
			case !loud:
				floor += (lv - floor) * 0.01
			}
			switch state {
			case silence:
				if !loud {
					keepPre(frame)
					return true
				}
				held = append(append([]float32(nil), pre...), frame...)
				heldAt = at - int64(len(pre))
				pre, state, speechN, silenceN = nil, onset, len(frame), 0
			case onset:
				held = append(held, frame...)
				if loud {
					speechN += len(frame)
					silenceN = 0
				} else {
					silenceN += len(frame)
				}
				switch {
				case silenceN >= hang:
					state = silence
					keepPre(held)
					held = nil
				case speechN >= minN:
					state, uttAt, pending, silenceN = speech, heldAt, nil, 0
					h := held
					held = nil
					return emit(h, heldAt, false)
				}
			case speech:
				if !loud {
					pending = append(pending, frame...)
					silenceN += len(frame)
					if silenceN >= hang {
						return end()
					}
					return true
				}
				silenceN = 0
				data := append(pending, frame...)
				pending = nil
				if at+int64(len(frame))-uttAt >= int64(maxN) {
					state = silence
					return emit(data, sent, true)
				}
				return emit(data, sent, false)
			}
			return true
		}
		for c, err := range s {
			if err != nil {
				yield(Chunk[[]float32]{}, err)
				return
			}
			if err := ctx.Err(); err != nil {
				yield(Chunk[[]float32]{}, err)
				return
			}
			buf = append(buf, c.Data...)
			for len(buf) >= fn {
				frame := make([]float32, fn)
				copy(frame, buf)
				buf = buf[fn:]
				if !step(frame, pos) {
					return
				}
				pos += int64(fn)
			}
		}
		if state == speech {
			pending = append(pending, buf...)
			end()
		}
	}
}

// Segments are the utterances a batch of audio holds: the start and the end of each, with
// padding, as Detect finds them.
func (v *VAD) Segments(pcm []float32) []Span {
	var out []Span
	open := false
	for c := range v.Detect(context.Background(), Of(pcm, samples(int64(len(pcm)), v.o.Rate))) {
		if !open {
			out = append(out, Span{Start: c.Start})
			open = true
		}
		out[len(out)-1].End = c.Start + c.Len
		if c.Final {
			open = false
		}
	}
	return out
}

// Span is a stretch of a stream's time.
type Span struct{ Start, End time.Duration }
