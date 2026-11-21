package flow

import (
	"context"
	"iter"
	"sync"
)

// item is a chunk or the error that ends its stream.
type item[T any] struct {
	c   Chunk[T]
	err error
}

// Async reads a stream in a goroutine of its own, up to n chunks ahead of the reader, so that
// the stages of a pipeline overlap: the speaker speaks the first sentence while the translator
// translates the second. No chunk is dropped: a stage that is ahead by n chunks waits. The
// goroutine ends before the returned stream does, whether the reader reads to the end, breaks,
// or ctx is cancelled; s must stop when ctx is cancelled.
func Async[T any](ctx context.Context, s Stream[T], n int) Stream[T] {
	return func(yield func(Chunk[T], error) bool) {
		parent := ctx
		ctx, cancel := context.WithCancel(ctx)
		ch := make(chan item[T], max(n, 0))
		done := make(chan struct{})
		go func() {
			defer close(done)
			defer close(ch)
			for c, err := range s {
				select {
				case ch <- item[T]{c, err}:
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

// Latest reads a stream in a goroutine of its own and gives its reader the newest chunk only:
// the chunks that came while the reader was busy are dropped, all but the last. It is for
// streams whose chunks each stand alone and go stale (a camera's frames into a detector that
// takes longer than a frame): the reader always works on what is now, and the source never
// waits for it. The last chunk of the stream and its error are never dropped. The goroutine ends
// before the returned stream does; s must stop when ctx is cancelled.
func Latest[T any](ctx context.Context, s Stream[T]) Stream[T] {
	return func(yield func(Chunk[T], error) bool) {
		parent := ctx
		ctx, cancel := context.WithCancel(ctx)
		var (
			mu    sync.Mutex
			cur   item[T]
			has   bool
			ended bool
		)
		wake := make(chan struct{}, 1)
		signal := func() {
			select {
			case wake <- struct{}{}:
			default:
			}
		}
		done := make(chan struct{})
		go func() {
			defer close(done)
			defer func() {
				mu.Lock()
				ended = true
				mu.Unlock()
				signal()
			}()
			for c, err := range s {
				mu.Lock()
				cur, has = item[T]{c, err}, true
				mu.Unlock()
				signal()
				if err != nil || ctx.Err() != nil {
					return
				}
			}
		}()
		defer func() {
			cancel()
			<-done
		}()
		for {
			select {
			case <-wake:
			case <-ctx.Done():
				yield(Chunk[T]{}, ctx.Err())
				return
			}
			mu.Lock()
			it, ok, end := cur, has, ended
			has = false
			mu.Unlock()
			if ok && (!yield(it.c, it.err) || it.err != nil) {
				return
			}
			if end {
				if err := parent.Err(); err != nil {
					yield(Chunk[T]{}, err)
				}
				return
			}
		}
	}
}

// Held is the newest chunk of a stream that is read in the background (Hold).
type Held[T any] struct {
	mu     sync.Mutex
	cur    Chunk[T]
	has    bool
	err    error
	first  chan struct{} // closed at the first chunk, or at the end without one
	done   chan struct{} // closed at the end of the stream
	cancel context.CancelFunc
}

// Hold reads a stream in a goroutine of its own and keeps its newest chunk, for a stage that
// runs on its own beat and takes whatever is newest at each one (a renderer drawing 60 times a
// second with the head's position a detector finds 15 times a second). Stop ends the reading;
// s must stop when ctx is cancelled.
func Hold[T any](ctx context.Context, s Stream[T]) *Held[T] {
	ctx, cancel := context.WithCancel(ctx)
	h := &Held[T]{first: make(chan struct{}), done: make(chan struct{}), cancel: cancel}
	go func() {
		defer close(h.done)
		seen := false
		defer func() {
			if !seen {
				close(h.first)
			}
		}()
		for c, err := range s {
			h.mu.Lock()
			if err != nil {
				h.err = err
			} else {
				h.cur, h.has = c, true
			}
			h.mu.Unlock()
			if err == nil && !seen {
				seen = true
				close(h.first)
			}
			if err != nil || ctx.Err() != nil {
				return
			}
		}
	}()
	return h
}

// Latest is the newest chunk read so far; ok is false before the first one.
func (h *Held[T]) Latest() (c Chunk[T], ok bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.cur, h.has
}

// Wait waits for the first chunk and returns the newest one. It fails when the stream ends
// without a chunk (with the stream's error, if it had one) or ctx is cancelled first.
func (h *Held[T]) Wait(ctx context.Context) (Chunk[T], error) {
	select {
	case <-h.first:
	case <-ctx.Done():
		return Chunk[T]{}, ctx.Err()
	}
	if c, ok := h.Latest(); ok {
		return c, nil
	}
	if err := h.Err(); err != nil {
		return Chunk[T]{}, err
	}
	return Chunk[T]{}, ErrEmpty
}

// Done is closed when the stream has ended.
func (h *Held[T]) Done() <-chan struct{} { return h.done }

// Err is the error the stream ended with, nil while it runs or when it ended without one.
func (h *Held[T]) Err() error {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.err
}

// Stop ends the reading and waits for its goroutine. The newest chunk stays readable.
func (h *Held[T]) Stop() {
	h.cancel()
	<-h.done
}

// Tee splits a stream into n streams that each read all of its chunks, to be read in other
// goroutines (captions and a translation of the same transcript). The stream is read once, as
// fast as the fastest reader; the chunks a slower reader has not read yet wait for it in memory.
// A reader that stops early stops receiving; the stream stops when all have stopped.
func Tee[T any](s Stream[T], n int) []Stream[T] {
	var (
		mu      sync.Mutex
		next    func() (Chunk[T], error, bool)
		stop    func()
		bufs    = make([][]item[T], n)
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
	pull := func(i int) (item[T], bool) {
		mu.Lock()
		defer mu.Unlock()
		if len(bufs[i]) > 0 {
			it := bufs[i][0]
			bufs[i] = bufs[i][1:]
			return it, true
		}
		if ended {
			return item[T]{}, false
		}
		if !started {
			next, stop = iter.Pull2(s)
			started = true
		}
		c, err, ok := next()
		if !ok {
			ended = true
			stop()
			return item[T]{}, false
		}
		it := item[T]{c, err}
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
