package flow

import (
	"context"
	"errors"
	"slices"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
	"time"
)

// collect is a sink that keeps what it is given.
type collect[T any] struct {
	mu  sync.Mutex
	got []T
}

func (c *collect[T]) sink(name string) *Snk[T] {
	return Sink[T](name).Do(func(ch Chunk[T]) error {
		c.mu.Lock()
		c.got = append(c.got, ch.Data)
		c.mu.Unlock()
		return nil
	})
}

func TestNetworkChainAndFanOut(t *testing.T) {
	var doubled, all, strs collect[int]
	src := Source[int]("numbers").From(From(1, 2, 3, 4, 5))
	double := Each("double", func(c Chunk[int], emit func(Chunk[int])) error {
		c.Data *= 2
		emit(c)
		return nil
	})
	seen := 0
	tap := Tap("count", func(Chunk[int]) { seen++ })
	// numbers → double → doubled; numbers → count → all; double → strs too
	src.Connect(double)
	src.Connect(tap)
	double.Connect(doubled.sink("doubled"))
	double.Connect(strs.sink("again"), Keep(2))
	tap.Connect(all.sink("all"))
	if err := Run(context.Background(), src); err != nil {
		t.Fatal(err)
	}
	if want := []int{2, 4, 6, 8, 10}; !slices.Equal(doubled.got, want) || !slices.Equal(strs.got, want) {
		t.Errorf("after double: %v and %v, want %v twice", doubled.got, strs.got, want)
	}
	if want := []int{1, 2, 3, 4, 5}; !slices.Equal(all.got, want) || seen != 5 {
		t.Errorf("the other branch: %v (the tap saw %d), want %v", all.got, seen, want)
	}

	var names []string
	for _, n := range Walk(src) {
		line := n.Kind + " " + n.Name
		for _, e := range n.To {
			line += " →" + e.To
			if e.Policy != "" {
				line += "(" + e.Policy + ")"
			}
		}
		names = append(names, line)
	}
	want := []string{"source numbers →double →count", "node double →doubled →again(keep 2)", "sink doubled", "sink again", "node count →all", "sink all"}
	if !slices.Equal(names, want) {
		t.Errorf("the network:\n%s\nwant:\n%s", strings.Join(names, "\n"), strings.Join(want, "\n"))
	}
}

// TestNetworkProgramNode: a node of a program keeps text until a sentence ends, hears a command
// that changes what the node after it does, and ends the network at another.
func TestNetworkProgramNode(t *testing.T) {
	var out collect[string]
	read := 0
	texts := []string{"こん", "にちは。元気", "ですか？", "大文字", "abc。", "終わり", "聞こえない。"}
	src := Source[string]("texts").From(func(yield func(Chunk[string], error) bool) {
		for _, s := range texts {
			read++
			if !yield(Chunk[string]{Data: s}, nil) {
				return
			}
		}
	})
	upper := false
	kept := ""
	sentences := Each("sentences", func(c Chunk[string], emit func(Chunk[string])) error {
		switch c.Data {
		case "大文字": // a command: it changes the node after and is not passed on
			upper = true
			return nil
		case "終わり":
			return Stop
		}
		kept += c.Data
		for {
			i := strings.IndexAny(kept, "。？")
			if i < 0 {
				return nil
			}
			i += len("。")
			emit(Chunk[string]{Data: kept[:i]})
			kept = kept[i:]
		}
	}, func(emit func(Chunk[string])) error {
		if kept != "" {
			emit(Chunk[string]{Data: kept})
		}
		return nil
	})
	shout := Each("shout", func(c Chunk[string], emit func(Chunk[string])) error {
		if upper {
			c.Data = strings.ToUpper(c.Data)
		}
		emit(c)
		return nil
	})
	src.Connect(sentences)
	sentences.Connect(shout)
	shout.Connect(out.sink("out"))
	if err := Run(context.Background(), src); err != nil {
		t.Fatal(err)
	}
	if want := []string{"こんにちは。", "元気ですか？", "ABC。"}; !slices.Equal(out.got, want) {
		t.Errorf("sentences %q, want %q", out.got, want)
	}
	if read != 6 {
		t.Errorf("the source was read %d times, want 6: Stop ends the nodes before", read)
	}

	// without a Stop, what was kept goes out at the end
	kept, upper, out.got = "", false, nil
	src2 := Source[string]("texts").From(From("はい。", "まだ"))
	s2 := Each("sentences", func(c Chunk[string], emit func(Chunk[string])) error {
		kept += c.Data
		if strings.HasSuffix(kept, "。") {
			emit(Chunk[string]{Data: kept})
			kept = ""
		}
		return nil
	}, func(emit func(Chunk[string])) error {
		emit(Chunk[string]{Data: kept})
		return nil
	})
	src2.Connect(s2)
	s2.Connect(out.sink("out"))
	if err := Run(context.Background(), src2); err != nil {
		t.Fatal(err)
	}
	if want := []string{"はい。", "まだ"}; !slices.Equal(out.got, want) {
		t.Errorf("with what was kept at the end: %q, want %q", out.got, want)
	}
}

func TestNetworkErrors(t *testing.T) {
	boom := errors.New("boom")
	run := func(node *Node[int, int], sink *Snk[int]) error {
		src := Source[int]("numbers").From(From(1, 2, 3))
		src.Connect(node)
		node.Connect(sink)
		return Run(context.Background(), src)
	}
	// the error of a node names it
	err := run(Each("picky", func(c Chunk[int], emit func(Chunk[int])) error {
		if c.Data == 2 {
			return boom
		}
		emit(c)
		return nil
	}), Sink[int]("out"))
	var ne *NodeError
	if !errors.Is(err, boom) || !errors.As(err, &ne) || ne.Node != "picky" {
		t.Errorf("a node's error: %v, want boom of picky", err)
	}
	// and a sink's
	err = run(Tap("pass", func(Chunk[int]) {}), Sink[int]("full").Do(func(Chunk[int]) error { return boom }))
	if !errors.As(err, &ne) || ne.Node != "full" {
		t.Errorf("a sink's error: %v, want boom of full", err)
	}
	// a source's own error names the source, through the nodes after it
	src := Source[int]("broken").From(func(yield func(Chunk[int], error) bool) { yield(Chunk[int]{}, boom) })
	src.Connect(Tap("pass", func(Chunk[int]) {}))
	if err := Run(context.Background(), src); !errors.As(err, &ne) || ne.Node != "broken" {
		t.Errorf("a source's error: %v, want boom of broken", err)
	}

	if err := Run(context.Background(), Source[int]("unbound")); err == nil || !strings.Contains(err.Error(), "has no stream") {
		t.Errorf("a source without a stream: %v", err)
	}
	if err := Run(context.Background()); err == nil {
		t.Error("a network without a source ran")
	}
	// a node has one input
	func() {
		defer func() {
			if r := recover(); r == nil || !strings.Contains(r.(string), "already has its input from a") {
				t.Errorf("a second input: %v", r)
			}
		}()
		n := Tap("n", func(Chunk[int]) {})
		Source[int]("a").Connect(n)
		Source[int]("b").Connect(n)
	}()

	// the cancel of the context ends the network with its error
	ctx, cancel := context.WithCancel(context.Background())
	endless := Source[int]("endless").From(func(yield func(Chunk[int], error) bool) {
		for i := 0; ctx.Err() == nil && yield(Chunk[int]{Data: i}, nil); i++ {
		}
	})
	n := 0
	endless.Connect(Sink[int]("count").Do(func(Chunk[int]) error {
		if n++; n == 10 {
			cancel()
		}
		return nil
	}))
	if err := Run(ctx, endless); !errors.Is(err, context.Canceled) {
		t.Errorf("after cancel: %v, want context.Canceled", err)
	}
}

// TestNetworkEdges: the readers of one output do not hold each other back, Keep lets the node
// before go on, and Newest drops what went stale.
func TestNetworkEdges(t *testing.T) {
	ms := time.Millisecond
	synctest.Test(t, func(t *testing.T) {
		ctx := t.Context()
		start := time.Now()
		at := func() time.Duration { return time.Since(start) }
		var mu sync.Mutex
		var fast, slow, kept []time.Duration
		var newest []int
		src := Source[int]("ticks").From(every(ctx, 4, 100*ms)) // chunks at 100, 200, 300, 400 ms
		note := func(to *[]time.Duration, work time.Duration) func(Chunk[int]) error {
			return func(Chunk[int]) error {
				mu.Lock()
				*to = append(*to, at())
				mu.Unlock()
				time.Sleep(work)
				return nil
			}
		}
		src.Connect(Sink[int]("fast").Do(note(&fast, 0)))
		src.Connect(Sink[int]("slow").Do(note(&slow, 250*ms)))
		src.Connect(Sink[int]("kept").Do(note(&kept, 150*ms)), Keep(4))
		src.Connect(Sink[int]("newest").Do(func(c Chunk[int]) error {
			mu.Lock()
			newest = append(newest, c.Data)
			mu.Unlock()
			time.Sleep(250 * ms)
			return nil
		}), Newest())
		if err := Run(ctx, src); err != nil {
			t.Fatal(err)
		}
		if want := []time.Duration{100 * ms, 200 * ms, 300 * ms, 400 * ms}; !slices.Equal(fast, want) {
			t.Errorf("the fast reader read at %v, want %v: the slow one held it back", fast, want)
		}
		if want := []time.Duration{100 * ms, 350 * ms, 600 * ms, 850 * ms}; !slices.Equal(slow, want) {
			t.Errorf("the slow reader read at %v, want %v: every chunk, late", slow, want)
		}
		if want := []time.Duration{100 * ms, 250 * ms, 400 * ms, 550 * ms}; !slices.Equal(kept, want) {
			t.Errorf("the reader behind Keep read at %v, want %v", kept, want)
		}
		if want := []int{0, 2, 3}; !slices.Equal(newest, want) {
			t.Errorf("the reader behind Newest read %v, want %v: the chunk that went stale dropped, the last kept", newest, want)
		}
	})
}
