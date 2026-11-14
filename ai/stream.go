package ai

import (
	"context"
	"iter"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

// Chunk is a piece of a stream: its data, where it is on the stream's clock, and whether it
// ends a segment (an utterance, a sentence).
type Chunk[T any] struct {
	Data T
	// Start is where the chunk begins, from the start of the stream; Len is how long it lasts.
	// Streams of audio count them in samples (Rechunk, VAD.Detect), not by the wall clock, so
	// that a late microphone or a jittery network does not move them.
	Start, Len time.Duration
	// Final marks the last chunk of a segment. In a stream of Text, the chunks before a final
	// one are provisional versions of the segment, each replacing the one before it (captions
	// while someone speaks); the final one is the settled text.
	Final bool
}

// Stream is a sequence of chunks, read with range; reading stops at a break, at an error, or
// when the context the stream was made with is cancelled. A batch call is a stream of one
// final chunk (Of).
type Stream[T any] = iter.Seq2[Chunk[T], error]

// Text is a chunk of text: a transcript, a translation, a sentence to speak.
type Text struct {
	Text string
	// Lang is the language's code ("ja", "en"), "" when not known.
	Lang string
}

// Of is the stream of one final chunk.
func Of[T any](data T, length time.Duration) Stream[T] {
	return func(yield func(Chunk[T], error) bool) {
		yield(Chunk[T]{Data: data, Len: length, Final: true}, nil)
	}
}

// Collect reads a whole stream.
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

// samples is how long n samples at rate last.
func samples(n int64, rate int) time.Duration {
	return time.Duration(n * int64(time.Second) / int64(rate))
}

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

// sentenceEnds are the characters that end a sentence: Japanese and Chinese ones end it where
// they are, Western ones before a space or the end of the text.
const (
	sentenceEndsCJK  = "。！？．!?"
	sentenceEndsWest = ".!?"
)

// SplitSentences cuts a text into its sentences, each with what follows it up to the next one
// (the spaces after a period stay with the sentence before).
func SplitSentences(text string) []string {
	var out []string
	start := 0
	for i := 0; i < len(text); {
		r, size := utf8.DecodeRuneInString(text[i:])
		i += size
		end := false
		switch {
		case r < utf8.RuneSelf && strings.ContainsRune(sentenceEndsWest, r):
			end = i == len(text) || text[i] == ' ' || text[i] == '\n' || text[i] == '"' || text[i] == ')'
		case strings.ContainsRune(sentenceEndsCJK, r) || r == '\n':
			end = true
		}
		if !end {
			continue
		}
		// closing quotes and brackets, and the spaces after them, stay with the sentence
		for i < len(text) {
			r, size := utf8.DecodeRuneInString(text[i:])
			if !strings.ContainsRune("」』）)\"' \n", r) {
				break
			}
			i += size
		}
		if s := text[start:i]; strings.TrimSpace(s) != "" {
			out = append(out, s)
		}
		start = i
	}
	if s := text[start:]; strings.TrimSpace(s) != "" {
		out = append(out, s)
	}
	return out
}

// Sentences cuts each final chunk of a stream of text into its sentences, each a final chunk of
// its own, so that a reader of sentences (Speaker.SpeakStream) starts on the first one sooner.
// The chunks before a final one, provisional, are left out. The sentences share their chunk's
// times.
func Sentences(s Stream[Text]) Stream[Text] {
	return func(yield func(Chunk[Text], error) bool) {
		for c, err := range s {
			if err != nil {
				yield(c, err)
				return
			}
			if !c.Final {
				continue
			}
			for _, t := range SplitSentences(c.Data.Text) {
				out := c
				out.Data.Text = strings.TrimSpace(t)
				if !yield(out, nil) {
					return
				}
			}
		}
	}
}

// Tee splits a stream into n streams that each read all of its chunks, to be read in other
// goroutines (captions and a translation of the same transcript). The stream is read once, as
// fast as the fastest reader; the chunks a slower reader has not read yet wait for it in memory.
// A reader that stops early stops receiving; the stream stops when all have stopped.
func Tee[T any](s Stream[T], n int) []Stream[T] {
	type item struct {
		c   Chunk[T]
		err error
	}
	var (
		mu      sync.Mutex
		next    func() (Chunk[T], error, bool)
		stop    func()
		bufs    = make([][]item, n)
		open    = make([]bool, n)
		left    = n
		started bool
		ended   bool
	)
	for i := range open {
		open[i] = true
	}
	// pull reads the next chunk for reader i: from its buffer, else from the stream, buffered
	// for the other readers. ok is false at the end.
	pull := func(i int) (item, bool) {
		mu.Lock()
		defer mu.Unlock()
		if len(bufs[i]) > 0 {
			it := bufs[i][0]
			bufs[i] = bufs[i][1:]
			return it, true
		}
		if ended {
			return item{}, false
		}
		if !started {
			next, stop = iter.Pull2(s)
			started = true
		}
		c, err, ok := next()
		if !ok {
			ended = true
			stop()
			return item{}, false
		}
		it := item{c, err}
		for j := range bufs {
			if j != i && open[j] {
				bufs[j] = append(bufs[j], it)
			}
		}
		return it, true
	}
	done := func(i int) {
		mu.Lock()
		defer mu.Unlock()
		if !open[i] {
			return
		}
		open[i], bufs[i] = false, nil
		if left--; left == 0 && started && !ended {
			ended = true
			stop()
		}
	}
	out := make([]Stream[T], n)
	for i := range out {
		out[i] = func(yield func(Chunk[T], error) bool) {
			defer done(i)
			for {
				it, ok := pull(i)
				if !ok || !yield(it.c, it.err) || it.err != nil {
					return
				}
			}
		}
	}
	return out
}

// Async reads a stream in a goroutine of its own, up to n chunks ahead of the reader, so that
// the stages of a pipeline overlap: the speaker speaks the first sentence while the translator
// translates the second. The goroutine ends before the returned stream does, whether the reader
// reads to the end, breaks, or ctx is cancelled; s must stop when ctx is cancelled.
func Async[T any](ctx context.Context, s Stream[T], n int) Stream[T] {
	return func(yield func(Chunk[T], error) bool) {
		parent := ctx
		ctx, cancel := context.WithCancel(ctx)
		type item struct {
			c   Chunk[T]
			err error
		}
		ch := make(chan item, max(n, 0))
		done := make(chan struct{})
		go func() {
			defer close(done)
			defer close(ch)
			for c, err := range s {
				select {
				case ch <- item{c, err}:
				case <-ctx.Done():
					return
				}
				if err != nil {
					return
				}
			}
		}()
		defer func() {
			cancel()
			for range ch {
			}
			<-done
		}()
		for {
			select {
			case it, ok := <-ch:
				if !ok {
					// the source may have stopped for the cancel of ctx without saying so
					if err := parent.Err(); err != nil {
						yield(Chunk[T]{}, err)
					}
					return
				}
				if !yield(it.c, it.err) || it.err != nil {
					return
				}
			case <-ctx.Done():
				yield(Chunk[T]{}, ctx.Err())
				return
			}
		}
	}
}
