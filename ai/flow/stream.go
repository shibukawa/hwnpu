package flow

import (
	"iter"
	"time"
)

// Chunk is a piece of a stream: its data, where it is on the stream's clock, and whether it
// ends a segment (an utterance, a sentence, a picture that is finished).
type Chunk[T any] struct {
	Data T
	// Start is where the chunk begins, from the start of the stream; Len is how long it lasts.
	// Streams of audio count them in samples (Rechunk), not by the wall clock, so that a late
	// microphone or a jittery network does not move them.
	Start, Len time.Duration
	// Final marks the last chunk of a segment. In a stream of Text, the chunks before a final
	// one are provisional versions of the segment, each replacing the one before it (captions
	// while someone speaks); the final one is the settled text.
	Final bool
}

// End is where the chunk ends: Start + Len.
func (c Chunk[T]) End() time.Duration { return c.Start + c.Len }

// Stream is a sequence of chunks, read with range; reading stops at a break, at an error, or
// when the context the stream was made with is cancelled. After an error a stream yields
// nothing more.
type Stream[T any] = iter.Seq2[Chunk[T], error]

// Of is the stream of one final chunk: a value that is whole, given to a stage that reads a
// stream.
func Of[T any](data T, length time.Duration) Stream[T] {
	return func(yield func(Chunk[T], error) bool) {
		yield(Chunk[T]{Data: data, Len: length, Final: true}, nil)
	}
}

// From is the stream of the values, each a final chunk, one after another with no times.
func From[T any](values ...T) Stream[T] {
	return func(yield func(Chunk[T], error) bool) {
		for _, v := range values {
			if !yield(Chunk[T]{Data: v, Final: true}, nil) {
				return
			}
		}
	}
}

// Collect reads a whole stream: the chunks up to its end, or up to its error.
func Collect[T any](s Stream[T]) ([]Chunk[T], error) {
	var out []Chunk[T]
	for c, err := range s {
		if err != nil {
			return out, err
		}
		out = append(out, c)
	}
	return out, nil
}

// Last reads a stream to its end and returns its last chunk's data: the result of a stage once
// it has finished all of it, for a stage that starts only then (a picture generated, then its
// depth). ok is false for a stream without chunks.
func Last[T any](s Stream[T]) (data T, ok bool, err error) {
	for c, err := range s {
		if err != nil {
			return data, ok, err
		}
		data, ok = c.Data, true
	}
	return data, ok, nil
}

// Finals keeps the final chunks of a stream: the settled texts of a transcript.
func Finals[T any](s Stream[T]) Stream[T] {
	return func(yield func(Chunk[T], error) bool) {
		for c, err := range s {
			if err != nil {
				yield(c, err)
				return
			}
			if c.Final && !yield(c, nil) {
				return
			}
		}
	}
}

// Map makes a stage of a function: each chunk's data goes through fn and keeps its times and
// its Final. An error of fn ends the stream.
func Map[T, U any](s Stream[T], fn func(Chunk[T]) (U, error)) Stream[U] {
	return func(yield func(Chunk[U], error) bool) {
		for c, err := range s {
			if err != nil {
				yield(Chunk[U]{}, err)
				return
			}
			u, err := fn(c)
			if err != nil {
				yield(Chunk[U]{}, err)
				return
			}
			if !yield(Chunk[U]{Data: u, Start: c.Start, Len: c.Len, Final: c.Final}, nil) {
				return
			}
		}
	}
}

// samples is how long n samples at rate last.
func samples(n int64, rate int) time.Duration {
	return time.Duration(n * int64(time.Second) / int64(rate))
}

// Samples is how long n samples at rate samples a second last: the time of a stream of audio.
func Samples(n int, rate int) time.Duration { return samples(int64(n), rate) }

// Rechunk cuts a stream of audio at rate samples a second into chunks of n samples, whatever
// sizes come in, as a program of a fixed shape takes them. A final chunk coming in ends the
// segment: what is left of it goes out as a shorter final chunk. The chunks' times are counted
// in samples from the start of the stream.
func Rechunk(s Stream[[]float32], rate, n int) Stream[[]float32] {
	return func(yield func(Chunk[[]float32], error) bool) {
		var buf []float32
		var pos int64 // samples sent
		send := func(data []float32, final bool) bool {
			c := Chunk[[]float32]{Data: data, Start: samples(pos, rate), Final: final}
			pos += int64(len(data))
			c.Len = samples(pos, rate) - c.Start
			return yield(c, nil)
		}
		for c, err := range s {
			if err != nil {
				yield(Chunk[[]float32]{}, err)
				return
			}
			buf = append(buf, c.Data...)
			for len(buf) >= n {
				last := c.Final && len(buf) == n
				if !send(buf[:n:n], last) {
					return
				}
				buf = buf[n:]
				if last {
					break
				}
			}
			if c.Final && len(buf) > 0 {
				if !send(buf, true) {
					return
				}
				buf = nil
			}
		}
		if len(buf) > 0 {
			send(buf, true)
		}
	}
}
