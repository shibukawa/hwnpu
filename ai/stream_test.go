package ai

import (
	"context"
	"errors"
	"runtime"
	"slices"
	"sync"
	"testing"
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

func TestTee(t *testing.T) {
	src := from([]int{1, 1, 1, 1, 1, 1, 1, 1}, true)
	outs := Tee(src, 3)
	var wg sync.WaitGroup
	counts := make([]int, 3)
	for i, s := range outs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range s {
				counts[i]++
				if i == 2 && counts[i] == 3 {
					break // a reader that stops early does not stop the others
				}
			}
		}()
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
	// an endless source that stops on ctx
	endless := func(ctx context.Context) Stream[int] {
		return func(yield func(Chunk[int], error) bool) {
			for i := 0; ; i++ {
				if ctx.Err() != nil || !yield(Chunk[int]{Data: i}, nil) {
					return
				}
			}
		}
	}
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

func TestOf(t *testing.T) {
	got, _ := Collect(Of("x", time.Second))
	if len(got) != 1 || !got[0].Final || got[0].Data != "x" || got[0].Len != time.Second {
		t.Errorf("Of = %+v", got)
	}
}
