package airpc

import (
	"context"
	"errors"
	"iter"
	"testing"
)

// encoding is a value whose state stays in the server (a reference: RegisterRef), as
// vision.PictureEncoding's does.
type encoding struct {
	state  string
	N      int
	remote *Handle
}

// stats are filled in as a stream goes, as sound.MusicStats are.
type stats struct{ Items, Sum int }

func init() {
	RegisterRef(func(e *encoding, h *Handle) { e.remote = h }, func(e *encoding) *Handle { return e.remote })
}

func (s *service) Encode(ctx context.Context, text string) (*encoding, error) {
	return &encoding{state: "state of " + text, N: len(text)}, nil
}

func (s *service) Decode(e *encoding) (string, error) {
	if e.state == "" {
		return "", errors.New("no state")
	}
	return e.state, nil
}

func (s *service) Sum(ctx context.Context, scale int, xs ...int) int {
	n := 0
	for _, x := range xs {
		n += x * scale
	}
	return n
}

func (s *service) Gen(ctx context.Context, n int) (iter.Seq[int], *stats) {
	st := &stats{}
	return func(yield func(int) bool) {
		for i := range n {
			st.Items++
			st.Sum += i
			if !yield(i) {
				return
			}
		}
	}, st
}

func TestRefs(t *testing.T) {
	h, _ := pair(t)
	ctx := context.Background()
	e, err := Call1[*encoding](h, ctx, "Encode", "a picture")
	if err != nil || e.N != 9 || e.remote == nil || e.state != "" {
		t.Fatalf("a reference: %+v %v", e, err)
	}
	if s, err := Call1[string](h, ctx, "Decode", e); err != nil || s != "state of a picture" {
		t.Errorf("the reference given back: %q %v", s, err)
	}
	if _, err := Call1[string](h, ctx, "Decode", &encoding{N: 1}); err == nil {
		t.Error("a value of the client's own has the server's state")
	}
	e.remote.Release()
	if _, err := Call1[string](h, ctx, "Decode", e); err == nil {
		t.Error("a released reference still answers")
	}
}

func TestVariadic(t *testing.T) {
	h, _ := pair(t)
	if n := Value[int](h, "Sum", 2, 1, 2, 3); n != 12 {
		t.Errorf("one by one: %d", n)
	}
	if n := Value[int](h, "Sum", 3, []int{1, 1}); n != 6 {
		t.Errorf("as a slice: %d", n)
	}
	if n := Value[int](h, "Sum", 3); n != 0 {
		t.Errorf("of none: %d", n)
	}
}

func TestResultsAfterStreams(t *testing.T) {
	h, _ := pair(t)
	ctx := context.Background()
	seq, st, err := Call2[iter.Seq[int], *stats](h, ctx, "Gen", 5)
	if err != nil || st == nil {
		t.Fatal(err)
	}
	if st.Items != 0 {
		t.Errorf("stats before the stream: %+v", st)
	}
	for range seq {
	}
	if st.Items != 5 || st.Sum != 10 {
		t.Errorf("stats after the stream: %+v", st)
	}
	// a loop that stops early waits for the producer's end, and its stats (the producer runs
	// ahead of the reader: a few more items than were read)
	seq, st, _ = Call2[iter.Seq[int], *stats](h, ctx, "Gen", 100000)
	for i := range seq {
		if i == 2 {
			break
		}
	}
	if st.Items < 3 || st.Items == 100000 {
		t.Errorf("stats after a loop stopped at 2: %+v", st)
	}
}
