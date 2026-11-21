package flow

import (
	"context"
	"errors"
	"time"
)

// ErrEmpty is the error of a stream that ended without a chunk where one was needed.
var ErrEmpty = errors.New("flow: the stream ended without a chunk")

// Metronome is a steady beat on the wall clock, for the stages that must keep time: a
// generator of music that owes the player a chunk every two seconds, a renderer that draws a
// frame every sixtieth of one. Its stream (Ticks) gives one tick a beat; the stage reads a tick,
// makes what is due at that beat, and reads the next.
type Metronome struct {
	// Period is the time from a beat to the next (BPM gives it for music).
	Period time.Duration
	// Lead is how long before its beat a tick is given: the time the stage has to make what is
	// due at the beat. The first tick is given at once and its beat is Lead later.
	Lead time.Duration
	// Skip drops the beats that were over before a slow reader came back and gives the one
	// whose period it is now, for a stage that shows the present (a renderer). Without it every
	// beat is given, late ones at once, for a stage whose output must have no holes (audio);
	// Tick.Late then says how far behind the stage is.
	Skip bool
}

// Tick is a beat of a Metronome.
type Tick struct {
	// N is the beat's number, from 0.
	N int64
	// At is the beat's time on the wall clock.
	At time.Time
	// Late is how long after its time (Lead before At) the tick was given because the reader
	// was still busy with the one before: 0 for a stage that keeps up.
	Late time.Duration
	// Skipped is how many beats were dropped just before this one (Metronome.Skip).
	Skipped int64
}

// BPM is the period of a beat at bpm beats a minute.
func BPM(bpm float64) time.Duration {
	return time.Duration(float64(time.Minute) / bpm)
}

// sleepUntil waits for t, or for the end of ctx.
func sleepUntil(ctx context.Context, t time.Time) error {
	d := time.Until(t)
	if d <= 0 {
		return ctx.Err()
	}
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Ticks is the stream of the metronome's beats, each a final chunk whose Start is the beat's
// time from beat 0 and whose Len is the period. It starts at its first read and ends when ctx
// is cancelled (with ctx's error) or the reader breaks.
func (m Metronome) Ticks(ctx context.Context) Stream[Tick] {
	return func(yield func(Chunk[Tick], error) bool) {
		if m.Period <= 0 {
			yield(Chunk[Tick]{}, errors.New("flow: a metronome needs a period above 0"))
			return
		}
		origin := time.Now().Add(m.Lead) // beat 0
		for n := int64(0); ; n++ {
			due := origin.Add(time.Duration(n)*m.Period - m.Lead)
			var skipped int64
			if m.Skip {
				if behind := time.Since(due); behind >= m.Period {
					skipped = int64(behind / m.Period)
					n += skipped
					due = origin.Add(time.Duration(n)*m.Period - m.Lead)
				}
			}
			if err := sleepUntil(ctx, due); err != nil {
				yield(Chunk[Tick]{}, err)
				return
			}
			t := Tick{N: n, At: due.Add(m.Lead), Late: max(0, time.Since(due)), Skipped: skipped}
			if !yield(Chunk[Tick]{Data: t, Start: time.Duration(n) * m.Period, Len: m.Period, Final: true}, nil) {
				return
			}
		}
	}
}

// Pace passes the chunks of a stream on when they are due on the wall clock, for a sink that
// takes them in real time (a player): the first chunk at once, and each later one when its
// Start, counted from the first one's, has come, less ahead (the audio a player keeps buffered
// in front of what it plays). A chunk that comes late passes at once: nothing is dropped. With
// Async before it, Pace keeps a stage that is faster than real time from running further ahead
// than Async's chunks.
func Pace[T any](ctx context.Context, s Stream[T], ahead time.Duration) Stream[T] {
	return pace(ctx, s, func(c Chunk[T]) time.Duration { return c.Start - ahead })
}

// Live passes the chunks of a recorded stream on as a live source would give them: each when
// its end, counted from the first one's start, has come (a microphone gives a chunk once its
// last sample is heard). It makes a file stand in for a microphone or a camera, with the same
// timing of everything after it. Unlike those it waits for a reader that is late and loses
// nothing: a live source's own stream has to be read as it comes (Async after it for sound that
// must all be heard, Latest for frames that go stale).
func Live[T any](ctx context.Context, s Stream[T]) Stream[T] {
	return pace(ctx, s, func(c Chunk[T]) time.Duration { return c.Start + c.Len })
}

// pace passes each chunk on when at(c), counted from the first chunk's Start, has come since
// the first read.
func pace[T any](ctx context.Context, s Stream[T], at func(Chunk[T]) time.Duration) Stream[T] {
	return func(yield func(Chunk[T], error) bool) {
		var origin time.Time // the wall clock at the first chunk's Start
		first := true
		for c, err := range s {
			if err != nil {
				yield(c, err)
				return
			}
			if first {
				origin, first = time.Now().Add(-c.Start), false
			}
			if err := sleepUntil(ctx, origin.Add(at(c))); err != nil {
				yield(Chunk[T]{}, err)
				return
			}
			if !yield(c, nil) {
				return
			}
		}
	}
}
