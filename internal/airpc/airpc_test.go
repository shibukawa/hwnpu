package airpc

import (
	"context"
	"errors"
	"fmt"
	"image"
	"image/color"
	"iter"
	"net"
	"net/http"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

var errSentinel = errors.New("test: the sentinel")

type typedErr struct{ Code int }

func (e *typedErr) Error() string { return fmt.Sprintf("test: code %d", e.Code) }

func init() {
	RegisterError("test.sentinel", errSentinel)
	RegisterErrorType("test.typed", (*typedErr)(nil))
}

// options is an argument with functions among its fields.
type options struct {
	Scale    int
	Progress func(step, steps int)
	Nested   *nested
}

type nested struct {
	Label  string
	OnDone func(string) error
}

type result struct {
	Sum   int
	Text  string
	Image *image.RGBA
	Took  time.Duration
}

// service is the object the tests call: what a model of ai/vision or ai/text is to hwnpud.
type service struct {
	stopped   atomic.Int64
	cancelled atomic.Int64
	produced  atomic.Int64
}

func (s *service) Add(ctx context.Context, a, b int) (int, error) { return a + b, nil }

func (s *service) Name() string { return "service" }

func (s *service) Pair() (string, int) { return "pair", 2 }

func (s *service) Fail(ctx context.Context, which string) error {
	switch which {
	case "sentinel":
		return fmt.Errorf("wrapped: %w", errSentinel)
	case "typed":
		return fmt.Errorf("wrapped: %w", &typedErr{Code: 7})
	case "canceled":
		return context.Canceled
	}
	return errors.New("plain")
}

// Paint takes a picture behind an interface and options with callbacks.
func (s *service) Paint(ctx context.Context, img image.Image, o options) (*result, error) {
	b := img.Bounds()
	out := image.NewRGBA(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			out.Set(x, y, img.At(x, y))
		}
	}
	for i := range 3 {
		if o.Progress != nil {
			o.Progress(i+1, 3)
		}
	}
	text := ""
	if o.Nested != nil && o.Nested.OnDone != nil {
		if err := o.Nested.OnDone(o.Nested.Label); err != nil {
			text = "refused: " + err.Error()
		}
	}
	return &result{Sum: b.Dx() * b.Dy() * o.Scale, Text: text, Image: out, Took: time.Second}, nil
}

// Each tells fn the words until it returns false.
func (s *service) Each(ctx context.Context, words []string, fn func(string) bool) (int, error) {
	n := 0
	for _, w := range words {
		n++
		if !fn(w) {
			break
		}
	}
	return n, nil
}

// Upper reads a stream of words and gives each in capitals: an input and an output stream.
func (s *service) Upper(ctx context.Context, in iter.Seq2[string, error]) iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
		for w, err := range in {
			if err != nil {
				yield("", err)
				return
			}
			s.produced.Add(1)
			if !yield(strings.ToUpper(w), nil) {
				s.stopped.Add(1)
				return
			}
		}
	}
}

// Count counts up forever, until its reader stops.
func (s *service) Count(ctx context.Context) iter.Seq[int] {
	return func(yield func(int) bool) {
		for i := 0; ; i++ {
			if !yield(i) {
				s.stopped.Add(1)
				return
			}
		}
	}
}

// Wait waits for its context to end.
func (s *service) Wait(ctx context.Context) error {
	<-ctx.Done()
	s.cancelled.Add(1)
	return ctx.Err()
}

// pair is a session over a pipe: the client's end and the service.
func pair(t *testing.T) (*Handle, *service) {
	t.Helper()
	a, b := net.Pipe()
	svc := &service{}
	go Serve(b, nil, func(*Session) any { return svc })
	s := NewClient(a, nil)
	t.Cleanup(func() { s.Close() })
	return s.Object(0).Bind(svc), svc
}

func TestCall(t *testing.T) {
	h, _ := pair(t)
	ctx := context.Background()
	if n, err := Call1[int](h, ctx, "Add", 2, 3); err != nil || n != 5 {
		t.Errorf("Add: %d, %v", n, err)
	}
	if v := Value[string](h, "Name"); v != "service" {
		t.Errorf("Name: %q", v)
	}
	if s, n, err := Call2[string, int](h, ctx, "Pair"); err != nil || s != "pair" || n != 2 {
		t.Errorf("Pair: %q %d %v", s, n, err)
	}
	if _, err := h.Call(ctx, "Nope"); err == nil {
		t.Error("a method that is not there")
	}
	if _, err := h.Call(ctx, "Add", 1); err == nil {
		t.Error("too few arguments")
	}
}

func TestErrors(t *testing.T) {
	h, _ := pair(t)
	ctx := context.Background()
	err := Call0(h, ctx, "Fail", "sentinel")
	if !errors.Is(err, errSentinel) || err.Error() != "wrapped: test: the sentinel" {
		t.Errorf("sentinel: %v", err)
	}
	err = Call0(h, ctx, "Fail", "typed")
	var te *typedErr
	if !errors.As(err, &te) || te.Code != 7 || errors.Is(err, errSentinel) {
		t.Errorf("typed: %v, %+v", err, te)
	}
	if err := Call0(h, ctx, "Fail", "canceled"); !errors.Is(err, context.Canceled) {
		t.Errorf("context.Canceled: %v", err)
	}
	if err := Call0(h, ctx, "Fail", "plain"); err == nil || err.Error() != "plain" || errors.Is(err, errSentinel) {
		t.Errorf("plain: %v", err)
	}
}

func TestCallbacks(t *testing.T) {
	h, _ := pair(t)
	ctx := context.Background()
	img := image.NewNRGBA(image.Rect(0, 0, 4, 2))
	img.Set(1, 1, color.NRGBA{R: 200, A: 255})
	var steps []string
	o := options{Scale: 2, Progress: func(step, steps2 int) { steps = append(steps, fmt.Sprintf("%d/%d", step, steps2)) },
		Nested: &nested{Label: "label", OnDone: func(l string) error { return fmt.Errorf("no %s: %w", l, errSentinel) }}}
	r, err := Call1[*result](h, ctx, "Paint", img, o)
	if err != nil {
		t.Fatal(err)
	}
	if r.Sum != 16 || r.Took != time.Second || r.Image.RGBAAt(1, 1).R != 200 {
		t.Errorf("result %+v", r)
	}
	if strings.Join(steps, " ") != "1/3 2/3 3/3" {
		t.Errorf("progress %q", steps)
	}
	if r.Text != "refused: no label: test: the sentinel" {
		t.Errorf("a callback's error: %q", r.Text)
	}
	// nil functions and pictures are none
	if r, err := Call1[*result](h, ctx, "Paint", image.NewGray(image.Rect(0, 0, 1, 1)), options{}); err != nil || r.Sum != 0 {
		t.Errorf("no callbacks: %+v, %v", r, err)
	}
	// a callback's result
	var seen []string
	n, err := Call1[int](h, ctx, "Each", []string{"a", "b", "c", "d"}, func(w string) bool {
		seen = append(seen, w)
		return w != "b"
	})
	if err != nil || n != 2 || strings.Join(seen, "") != "ab" {
		t.Errorf("Each: %d %q %v", n, seen, err)
	}
}

func TestStreams(t *testing.T) {
	h, svc := pair(t)
	ctx := context.Background()
	words := []string{"one", "two", "three"}
	in := func(yield func(string, error) bool) {
		for _, w := range words {
			if !yield(w, nil) {
				return
			}
		}
	}
	out, err := Call1[iter.Seq2[string, error]](h, ctx, "Upper", iter.Seq2[string, error](in))
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for w, err := range out {
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, w)
	}
	if strings.Join(got, " ") != "ONE TWO THREE" {
		t.Errorf("Upper: %q", got)
	}
	// a stream is read once
	for _, err := range out {
		if err == nil {
			t.Error("a stream read twice")
		}
	}
	// an input stream's error reaches the output
	bad := func(yield func(string, error) bool) { yield("", errSentinel) }
	out, _ = Call1[iter.Seq2[string, error]](h, ctx, "Upper", iter.Seq2[string, error](bad))
	for _, err := range out {
		if !errors.Is(err, errSentinel) {
			t.Errorf("the input's error: %v", err)
		}
	}
	// the reader stops: the producer stops
	count, err := Call1[iter.Seq[int]](h, ctx, "Count")
	if err != nil {
		t.Fatal(err)
	}
	sum := 0
	for i := range count {
		if i == 10 {
			break
		}
		sum += i
	}
	if sum != 45 {
		t.Errorf("sum %d", sum)
	}
	deadline := time.Now().Add(5 * time.Second)
	for svc.stopped.Load() == 0 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if svc.stopped.Load() == 0 {
		t.Error("the producer did not stop")
	}
}

func TestCancel(t *testing.T) {
	h, svc := pair(t)
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	err := Call0(h, ctx, "Wait")
	if !errors.Is(err, context.DeadlineExceeded) || svc.cancelled.Load() != 1 {
		t.Errorf("cancelled: %v (%d)", err, svc.cancelled.Load())
	}
}

func TestSessionEnd(t *testing.T) {
	h, _ := pair(t)
	done := make(chan error, 1)
	go func() { done <- Call0(h, context.Background(), "Wait") }()
	time.Sleep(20 * time.Millisecond)
	h.Session().Close()
	select {
	case err := <-done:
		if !errors.Is(err, ErrClosed) {
			t.Errorf("after the end: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("a call outlived its session")
	}
	if _, err := Call1[int](h, context.Background(), "Add", 1, 2); !errors.Is(err, ErrClosed) {
		t.Errorf("a call after the end: %v", err)
	}
}

func TestHTTP(t *testing.T) {
	sock := filepath.Join(t.TempDir(), "s.sock")
	ln, err := net.Listen("unix", sock)
	if err != nil {
		t.Fatal(err)
	}
	svc := &service{}
	closed := make(chan struct{})
	mux := http.NewServeMux()
	mux.Handle("GET /rpc", Handler("test/1", func(s *Session) any {
		s.OnClose(func() { close(closed) })
		return svc
	}))
	srv := &http.Server{Handler: mux}
	go srv.Serve(ln)
	defer srv.Close()
	ctx := context.Background()
	conn, err := net.Dial("unix", sock)
	if err != nil {
		t.Fatal(err)
	}
	s, err := Dial(ctx, conn, "/rpc", "test/1")
	if err != nil {
		t.Fatal(err)
	}
	if n, err := Call1[int](s.Object(0).Bind(svc), ctx, "Add", 20, 22); err != nil || n != 42 {
		t.Errorf("over HTTP: %d %v", n, err)
	}
	s.Close()
	select {
	case <-closed:
	case <-time.After(5 * time.Second):
		t.Error("the server's end did not see the session end")
	}
	conn, _ = net.Dial("unix", sock)
	if _, err := Dial(ctx, conn, "/rpc", "other/1"); err == nil {
		t.Error("another protocol upgraded")
	}
}
