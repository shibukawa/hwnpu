package flow

import (
	"context"
	"errors"
	"runtime"
	"slices"
	"sync"
	"testing"
	"testing/synctest"
	"time"
)

// from is a stream of audio in chunks of the given sizes, the values counting up, the last
// chunk final when final.
func from(sizes []int, final bool) Stream[[]float32] {
	return func(yield func(Chunk[[]float32], error) bool) {
		v := float32(0)
		for i, n := range sizes {
			data := make([]float32, n)
			for j := range data {
				data[j] = v
				v++
			}
			if !yield(Chunk[[]float32]{Data: data, Final: final && i == len(sizes)-1}, nil) {
				return
			}
		}
	}
}

// endless counts up until ctx is cancelled or the reader stops.
func endless(ctx context.Context) Stream[int] {
	return func(yield func(Chunk[int], error) bool) {
		for i := 0; ; i++ {
			if ctx.Err() != nil || !yield(Chunk[int]{Data: i}, nil) {
				return
			}
		}
	}
}

// every gives n chunks, one each period on the wall clock, timed on the stream's clock too.
func every(ctx context.Context, n int, period time.Duration) Stream[int] {
	return func(yield func(Chunk[int], error) bool) {
		for i := range n {
			if sleepUntil(ctx, time.Now().Add(period)) != nil {
				return
			}
			if !yield(Chunk[int]{Data: i, Start: time.Duration(i) * period, Len: period, Final: true}, nil) {
				return
			}
		}
	}
}

func TestRechunk(t *testing.T) {
	const rate = 1000
	got, err := Collect(Rechunk(from([]int{3, 7, 1, 13, 5, 9, 12}, false), rate, 8))
	if err != nil {
		t.Fatal(err)
	}
	var sizes []int
	var all []float32
	for i, c := range got {
		sizes = append(sizes, len(c.Data))
		if want := time.Duration(len(all)) * time.Millisecond; c.Start != want {
			t.Errorf("chunk %d starts at %v, want %v", i, c.Start, want)
		}
		if want := time.Duration(len(c.Data)) * time.Millisecond; c.Len != want {
			t.Errorf("chunk %d lasts %v, want %v", i, c.Len, want)
		}
		all = append(all, c.Data...)
	}
	if want := []int{8, 8, 8, 8, 8, 8, 2}; !slices.Equal(sizes, want) {
		t.Errorf("sizes %v, want %v", sizes, want)
	}
	for i, v := range all {
		if v != float32(i) {
			t.Fatalf("sample %d is %v: samples lost or reordered", i, v)
		}
	}
	if !got[len(got)-1].Final || got[0].Final {
		t.Errorf("only the last chunk is final at the end of the stream")
	}
}

func TestRechunkFinalEndsSegment(t *testing.T) {
	// two segments of 10 and 16 samples: each ends with a final chunk of its rest
	s := func(yield func(Chunk[[]float32], error) bool) {
		_ = yield(Chunk[[]float32]{Data: make([]float32, 10), Final: true}, nil) &&
			yield(Chunk[[]float32]{Data: make([]float32, 16), Final: true}, nil)
	}
	got, err := Collect(Rechunk(s, 1000, 8))
	if err != nil {
		t.Fatal(err)
	}
	var shape []int
	for _, c := range got {
		n := len(c.Data)
		if c.Final {
			n = -n
		}
		shape = append(shape, n)
	}
	if want := []int{8, -2, 8, -8}; !slices.Equal(shape, want) {
		t.Errorf("chunks %v (negative: final), want %v", shape, want)
	}
	if got[3].Start != 18*time.Millisecond {
		t.Errorf("last chunk starts at %v, want 18ms", got[3].Start)
	}
}

func TestSplitSentences(t *testing.T) {
	for _, c := range []struct {
		in   string
		want []string
	}{
		{"こんにちは。元気ですか？はい", []string{"こんにちは。", "元気ですか？", "はい"}},
		{"「そうです。」と言った。", []string{"「そうです。」", "と言った。"}},
		{"Hello world. How are you? Fine", []string{"Hello world. ", "How are you? ", "Fine"}},
		{"Version 1.5 is out.", []string{"Version 1.5 is out."}},
		{"  ", nil},
	} {
		if got := SplitSentences(c.in); !slices.Equal(got, c.want) {
			t.Errorf("SplitSentences(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestSentencesAndFinals(t *testing.T) {
	in := func(yield func(Chunk[Text], error) bool) {
		_ = yield(Chunk[Text]{Data: Text{Text: "こんに"}}, nil) &&
			yield(Chunk[Text]{Data: Text{Text: "こんにちは。元気？", Lang: "ja"}, Start: time.Second, Final: true}, nil)
	}
	got, err := Collect(Sentences(in))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].Data.Text != "こんにちは。" || got[1].Data.Text != "元気？" || !got[1].Final || got[1].Data.Lang != "ja" || got[1].Start != time.Second {
		t.Errorf("Sentences = %+v", got)
	}
	f, _ := Collect(Finals(Stream[Text](in)))
	if len(f) != 1 || !f[0].Final {
		t.Errorf("Finals = %+v", f)
	}
}

func TestOfFromLastMap(t *testing.T) {
	got, _ := Collect(Of("x", time.Second))
	if len(got) != 1 || !got[0].Final || got[0].Data != "x" || got[0].Len != time.Second {
		t.Errorf("Of = %+v", got)
	}
	doubled := Map(From(1, 2, 3), func(c Chunk[int]) (int, error) { return c.Data * 2, nil })
	if v, ok, err := Last(doubled); v != 6 || !ok || err != nil {
		t.Errorf("Last(Map(From(1, 2, 3))) = %v %v %v, want 6", v, ok, err)
	}
	if _, ok, err := Last(From[int]()); ok || err != nil {
		t.Errorf("Last of nothing = %v %v", ok, err)
	}
	boom := errors.New("boom")
	failing := Map(From(1, 2, 3), func(c Chunk[int]) (int, error) {
		if c.Data == 2 {
			return 0, boom
		}
		return c.Data, nil
	})
	if got, err := Collect(failing); !errors.Is(err, boom) || len(got) != 1 {
		t.Errorf("Map with an error = %+v %v, want one chunk and boom", got, err)
	}
}

func TestTee(t *testing.T) {
	src := from([]int{1, 1, 1, 1, 1, 1, 1, 1}, true)
	outs := Tee(src, 3)
	var wg sync.WaitGroup
	counts := make([]int, 3)
	for i, s := range outs {
		wg.Go(func() {
			for range s {
				counts[i]++
				if i == 2 && counts[i] == 3 {
					break // a reader that stops early does not stop the others
				}
			}
		})
	}
	wg.Wait()
	if want := []int{8, 8, 3}; !slices.Equal(counts, want) {
		t.Errorf("readers read %v chunks, want %v", counts, want)
	}
}

// settle waits for the goroutines to come back to n, at most a second.
func settle(n int) int {
	for i := 0; i < 100 && runtime.NumGoroutine() > n; i++ {
		time.Sleep(10 * time.Millisecond)
	}
	return runtime.NumGoroutine()
}

func TestAsyncLeavesNoGoroutine(t *testing.T) {
	before := runtime.NumGoroutine()
	ctx := context.Background()
	n := 0
	for c := range Async(ctx, endless(ctx), 4) {
		if c.Data != n {
			t.Fatalf("chunk %d is %d", n, c.Data)
		}
		if n++; n == 100 {
			break
		}
	}
	if after := settle(before); after > before {
		t.Errorf("%d goroutines after a break, %d before", after, before)
	}

	cctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var err error
	n = 0
	for _, e := range Async(cctx, endless(cctx), 4) {
		if e != nil {
			err = e
			break
		}
		if n++; n == 10 {
			cancel()
		}
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("after cancel: %v, want context.Canceled", err)
	}
	if after := settle(before); after > before {
		t.Errorf("%d goroutines after a cancel, %d before", after, before)
	}

	got, err := Collect(Async(ctx, Stream[[]float32](from([]int{2, 3, 4}, true)), 1))
	if err != nil || len(got) != 3 || !got[2].Final {
		t.Errorf("Async to the end: %v %+v", err, got)
	}
}

// TestAsyncOverlaps: two stages of 100 ms a chunk take 100 ms a chunk together behind Async,
// not 200 ms.
func TestAsyncOverlaps(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx := t.Context()
		start := time.Now()
		n := 0
		for _, err := range Async(ctx, every(ctx, 5, 100*time.Millisecond), 1) {
			if err != nil {
				t.Fatal(err)
			}
			time.Sleep(100 * time.Millisecond) // the second stage's work
			n++
		}
		if el := time.Since(start); n != 5 || el != 600*time.Millisecond {
			t.Errorf("%d chunks in %v, want 5 in 600ms (the stages overlap)", n, el)
		}
	})
}

// TestLatestDropsStale: a reader three times slower than its source reads every third chunk,
// always the newest, and the last one.
func TestLatestDropsStale(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx := t.Context()
		var got []int
		for c, err := range Latest(ctx, every(ctx, 10, 10*time.Millisecond)) {
			if err != nil {
				t.Fatal(err)
			}
			got = append(got, c.Data)
			time.Sleep(33 * time.Millisecond) // busy until 43, 76 and 109 ms: never as a chunk comes
		}
		if want := []int{0, 3, 6, 9}; !slices.Equal(got, want) {
			t.Errorf("read %v, want %v", got, want)
		}
	})
}

func TestLatestErrorAndCancel(t *testing.T) {
	boom := errors.New("boom")
	failing := func(yield func(Chunk[int], error) bool) {
		_ = yield(Chunk[int]{Data: 1}, nil) && yield(Chunk[int]{}, boom)
	}
	var err error
	for _, e := range Latest(context.Background(), failing) {
		err = e
	}
	if !errors.Is(err, boom) {
		t.Errorf("the stream's error is %v, want boom", err)
	}

	before := runtime.NumGoroutine()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	n := 0
	for _, e := range Latest(ctx, endless(ctx)) {
		if e != nil {
			err = e
			break
		}
		if n++; n == 5 {
			cancel()
		}
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("after cancel: %v, want context.Canceled", err)
	}
	if after := settle(before); after > before {
		t.Errorf("%d goroutines after a cancel, %d before", after, before)
	}
}

func TestHold(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx := t.Context()
		h := Hold(ctx, every(ctx, 4, 100*time.Millisecond))
		if _, ok := h.Latest(); ok {
			t.Error("a chunk before the stream gave one")
		}
		if c, err := h.Wait(ctx); err != nil || c.Data != 0 {
			t.Errorf("Wait = %+v %v, want chunk 0", c, err)
		}
		time.Sleep(250 * time.Millisecond) // at 350 ms: chunks 0, 1, 2 have come
		if c, _ := h.Latest(); c.Data != 2 {
			t.Errorf("at 350 ms the newest is %d, want 2", c.Data)
		}
		<-h.Done()
		if c, _ := h.Latest(); c.Data != 3 || h.Err() != nil {
			t.Errorf("at the end the newest is %d (%v), want 3", c.Data, h.Err())
		}
		h.Stop()
	})
	if _, err := Hold(context.Background(), From[int]()).Wait(context.Background()); !errors.Is(err, ErrEmpty) {
		t.Errorf("Wait on a stream without chunks: %v, want ErrEmpty", err)
	}
}

func TestMetronome(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		start := time.Now()
		m := Metronome{Period: 500 * time.Millisecond, Lead: 200 * time.Millisecond}
		var given, lates []time.Duration
		for c, err := range m.Ticks(t.Context()) {
			if err != nil {
				t.Fatal(err)
			}
			tick := c.Data
			given = append(given, time.Since(start))
			lates = append(lates, tick.Late)
			if want := start.Add(m.Lead + time.Duration(tick.N)*m.Period); !tick.At.Equal(want) {
				t.Errorf("beat %d is at %v, want %v", tick.N, tick.At.Sub(start), want.Sub(start))
			}
			if c.Start != time.Duration(tick.N)*m.Period || c.Len != m.Period {
				t.Errorf("beat %d: chunk at %v for %v", tick.N, c.Start, c.Len)
			}
			if tick.N == 1 {
				time.Sleep(700 * time.Millisecond) // a slow beat: the next tick comes late, none is lost
			}
			if tick.N == 4 {
				break
			}
		}
		ms := time.Millisecond
		if want := []time.Duration{0, 500 * ms, 1200 * ms, 1500 * ms, 2000 * ms}; !slices.Equal(given, want) {
			t.Errorf("ticks given at %v, want %v", given, want)
		}
		if want := []time.Duration{0, 0, 200 * ms, 0, 0}; !slices.Equal(lates, want) {
			t.Errorf("ticks late by %v, want %v", lates, want)
		}
	})
}

func TestMetronomeSkip(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		m := Metronome{Period: 100 * time.Millisecond, Skip: true}
		var beats, skipped []int64
		for c, err := range m.Ticks(t.Context()) {
			if err != nil {
				t.Fatal(err)
			}
			beats, skipped = append(beats, c.Data.N), append(skipped, c.Data.Skipped)
			if c.Data.N == 1 {
				time.Sleep(350 * time.Millisecond) // until 450 ms: beats 2 and 3 are over, beat 4 is now
			}
			if len(beats) == 4 {
				break
			}
		}
		if want := []int64{0, 1, 4, 5}; !slices.Equal(beats, want) {
			t.Errorf("beats %v, want %v", beats, want)
		}
		if want := []int64{0, 0, 2, 0}; !slices.Equal(skipped, want) {
			t.Errorf("skipped %v, want %v", skipped, want)
		}
	})
	if _, err := Collect(Metronome{}.Ticks(context.Background())); err == nil {
		t.Error("a metronome without a period gave no error")
	}
	if p := BPM(120); p != 500*time.Millisecond {
		t.Errorf("BPM(120) = %v, want 500ms", p)
	}
}

// recorded is a recording of n chunks of length each, all there at once.
func recorded(n int, length time.Duration) Stream[int] {
	return func(yield func(Chunk[int], error) bool) {
		for i := range n {
			if !yield(Chunk[int]{Data: i, Start: time.Duration(i) * length, Len: length}, nil) {
				return
			}
		}
	}
}

func TestPaceAndLive(t *testing.T) {
	ms := time.Millisecond
	synctest.Test(t, func(t *testing.T) {
		ctx := t.Context()
		times := func(s Stream[int]) []time.Duration {
			start := time.Now()
			var at []time.Duration
			for _, err := range s {
				if err != nil {
					t.Fatal(err)
				}
				at = append(at, time.Since(start))
			}
			return at
		}
		if got, want := times(Pace(ctx, recorded(4, 100*ms), 0)), []time.Duration{0, 100 * ms, 200 * ms, 300 * ms}; !slices.Equal(got, want) {
			t.Errorf("Pace passes at %v, want %v", got, want)
		}
		// a player's buffer of 150 ms: the chunks that start within it pass at once
		if got, want := times(Pace(ctx, recorded(4, 100*ms), 150*ms)), []time.Duration{0, 0, 50 * ms, 150 * ms}; !slices.Equal(got, want) {
			t.Errorf("Pace with 150 ms ahead passes at %v, want %v", got, want)
		}
		if got, want := times(Live(ctx, recorded(3, 100*ms))), []time.Duration{100 * ms, 200 * ms, 300 * ms}; !slices.Equal(got, want) {
			t.Errorf("Live passes at %v, want %v", got, want)
		}
	})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Collect(Pace(ctx, recorded(2, time.Hour), 0)); !errors.Is(err, context.Canceled) {
		t.Errorf("Pace after cancel: %v, want context.Canceled", err)
	}
}
