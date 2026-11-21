package flow

import (
	"context"
	"errors"
	"fmt"
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
	src.To(double)
	src.To(tap)
	double.To(doubled.sink("doubled"))
	double.To(strs.sink("again"), Keep(2))
	tap.To(all.sink("all"))
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
	}, AtEnd(func(emit func(Chunk[string])) error {
		if kept != "" {
			emit(Chunk[string]{Data: kept})
		}
		return nil
	}))
	shout := Each("shout", func(c Chunk[string], emit func(Chunk[string])) error {
		if upper {
			c.Data = strings.ToUpper(c.Data)
		}
		emit(c)
		return nil
	})
	src.To(sentences)
	sentences.To(shout)
	shout.To(out.sink("out"))
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
	}, AtEnd(func(emit func(Chunk[string])) error {
		emit(Chunk[string]{Data: kept})
		return nil
	}))
	src2.To(s2)
	s2.To(out.sink("out"))
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
		src.To(node)
		node.To(sink)
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
	src.To(Tap("pass", func(Chunk[int]) {}))
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
		Source[int]("a").To(n)
		Source[int]("b").To(n)
	}()

	// the cancel of the context ends the network with its error
	ctx, cancel := context.WithCancel(context.Background())
	endless := Source[int]("endless").From(func(yield func(Chunk[int], error) bool) {
		for i := 0; ctx.Err() == nil && yield(Chunk[int]{Data: i}, nil); i++ {
		}
	})
	n := 0
	endless.To(Sink[int]("count").Do(func(Chunk[int]) error {
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
		src.To(Sink[int]("fast").Do(note(&fast, 0)))
		src.To(Sink[int]("slow").Do(note(&slow, 250*ms)))
		src.To(Sink[int]("kept").Do(note(&kept, 150*ms)), Keep(4))
		src.To(Sink[int]("newest").Do(func(c Chunk[int]) error {
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

// TestNetworkInARow: joins written in a row are the same network, a branch starts from the
// place kept, and what does not fit is refused as the network is built.
func TestNetworkInARow(t *testing.T) {
	var lens, shouted collect[int]
	var words collect[string]
	src := Source[string]("words").From(From("a", "bb", "ccc"))
	length := Each("len", func(c Chunk[string], emit func(Chunk[int])) error {
		emit(Chunk[int]{Data: len(c.Data)})
		return nil
	})
	double := Each("double", func(c Chunk[int], emit func(Chunk[int])) error {
		emit(Chunk[int]{Data: 2 * c.Data})
		return nil
	})
	counted := src.To(length)
	counted.To(lens.sink("lens"))
	counted.To(double, Keep(2)).To(shouted.sink("doubled"))
	Chain(src).To(words.sink("words"))
	if err := Run(context.Background(), src); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(lens.got, []int{1, 2, 3}) || !slices.Equal(shouted.got, []int{2, 4, 6}) || !slices.Equal(words.got, []string{"a", "bb", "ccc"}) {
		t.Errorf("the branches gave %v, %v and %v", lens.got, shouted.got, words.got)
	}

	refused := func(what, want string, build func()) {
		t.Helper()
		defer func() {
			if r := recover(); r == nil || !strings.Contains(r.(string), want) {
				t.Errorf("%s: %v, want a panic with %q", what, r, want)
			}
		}()
		build()
	}
	refused("a string into a node of ints", "words gives string and double takes int", func() {
		Chain(Source[string]("words")).To(Each("double", func(Chunk[int], func(Chunk[int])) error { return nil }))
	})
	refused("after a sink", "out is a sink", func() {
		Source[int]("n").To(Sink[int]("out")).To(Sink[int]("more"))
	})
	refused("into a source", "is a source", func() {
		Source[int]("n").To(Sink[int]("out"))
		Chain(Tap("t", func(Chunk[int]) {})).To(Source[int]("again"))
	})
}

func TestCall(t *testing.T) {
	ctx := context.Background()
	upper := Each("upper", func(c Chunk[string], emit func(Chunk[string])) error {
		emit(Chunk[string]{Data: strings.ToUpper(c.Data)})
		return nil
	})
	letters := Each("letters", func(c Chunk[string], emit func(Chunk[int])) error {
		for i := range c.Data { // the answer so far, then the whole answer
			emit(Chunk[int]{Data: i + 1})
		}
		return nil
	})
	count := NewCall[string, int]("count").Then(upper).Then(letters, Keep(2))
	if n, err := count.Do(ctx, "hello"); err != nil || n != 5 {
		t.Errorf("Do = %d, %v; want 5", n, err)
	}
	if all, err := count.All(ctx, "abc"); err != nil || !slices.Equal(all, []int{1, 2, 3}) {
		t.Errorf("All = %v, %v; want 1 2 3", all, err)
	}
	if _, err := count.Do(ctx, ""); !errors.Is(err, ErrEmpty) {
		t.Errorf("a call that gave nothing: %v, want ErrEmpty", err)
	}
	// calls at the same time each have their own streams
	var wg sync.WaitGroup
	got := make([]int, 50)
	for i := range got {
		wg.Go(func() { got[i], _ = count.Do(ctx, strings.Repeat("x", i+1)) })
	}
	wg.Wait()
	for i, n := range got {
		if n != i+1 {
			t.Fatalf("call %d gave %d, want %d", i, n, i+1)
		}
	}
	var names []string
	for _, n := range count.Nodes() {
		names = append(names, n.Kind+" "+n.Name)
	}
	if want := []string{"call count", "node upper", "node letters", "sink count result"}; !slices.Equal(names, want) {
		t.Errorf("the call's nodes %v, want %v", names, want)
	}

	// a node's error names the node
	boom := errors.New("boom")
	failing := NewCall[string, string]("failing").Then(Each("picky", func(Chunk[string], func(Chunk[string])) error { return boom }))
	var ne *NodeError
	if _, err := failing.Do(ctx, "x"); !errors.As(err, &ne) || ne.Node != "picky" {
		t.Errorf("a call's error: %v, want boom of picky", err)
	}
	if _, err := NewCall[string, int]("empty").Do(ctx, "x"); err == nil || !strings.Contains(err.Error(), "has no nodes") {
		t.Errorf("a call without nodes: %v", err)
	}
	wrong := NewCall[string, int]("wrong").Then(Tap("pass", func(Chunk[string]) {}))
	if _, err := wrong.Do(ctx, "x"); err == nil || !strings.Contains(err.Error(), "gives int, and its last node pass gives string") {
		t.Errorf("a call whose last node gives another type: %v", err)
	}
	func() {
		defer func() {
			if r := recover(); r == nil || !strings.Contains(r.(string), "takes int") {
				t.Errorf("a first node of another type: %v", r)
			}
		}()
		NewCall[string, int]("mismatch").Then(Tap("ints", func(Chunk[int]) {}))
	}()
}

// TestWhenIdle: a node that keeps text until a sentence ends lets go of what it holds when
// nothing has come for a while, once for each pause.
func TestWhenIdle(t *testing.T) {
	ms := time.Millisecond
	synctest.Test(t, func(t *testing.T) {
		ctx := t.Context()
		start := time.Now()
		// words at 100, 200 and 300 ms, a pause, then at 1000 and 1100 ms
		src := Source[string]("words").From(func(yield func(Chunk[string], error) bool) {
			for _, w := range []struct {
				at   time.Duration
				word string
			}{{100 * ms, "明日は"}, {200 * ms, "晴れ。"}, {300 * ms, "たぶん"}, {1000 * ms, "ね。"}, {1100 * ms, "では"}} {
				if sleepUntil(ctx, start.Add(w.at)) != nil || !yield(Chunk[string]{Data: w.word}, nil) {
					return
				}
			}
		})
		kept, idles := "", 0
		var got []string
		flush := func(emit func(Chunk[string])) {
			if kept != "" {
				emit(Chunk[string]{Data: kept})
				kept = ""
			}
		}
		sentences := Each("sentences", func(c Chunk[string], emit func(Chunk[string])) error {
			kept += c.Data
			if strings.HasSuffix(kept, "。") {
				flush(emit)
			}
			return nil
		}, WhenIdle(250*ms, func(emit func(Chunk[string])) error {
			idles++
			flush(emit)
			return nil
		}), AtEnd(func(emit func(Chunk[string])) error {
			flush(emit)
			return nil
		}))
		src.To(sentences).To(Sink[string]("out").Do(func(c Chunk[string]) error {
			got = append(got, fmt.Sprintf("%s@%d", c.Data, time.Since(start)/ms))
			return nil
		}))
		if err := Run(ctx, src); err != nil {
			t.Fatal(err)
		}
		// "たぶん" waits for the end of its sentence and goes out 250 ms after it came; "では" at the end
		if want := []string{"明日は晴れ。@200", "たぶん@550", "ね。@1000", "では@1100"}; !slices.Equal(got, want) {
			t.Errorf("sentences %v, want %v", got, want)
		}
		if idles != 1 {
			t.Errorf("idle was called %d times, want once (for the one pause; the end is not a pause)", idles)
		}
	})
}

// TestRoute: a node that sends each chunk to one of its outputs.
func TestRoute(t *testing.T) {
	var commands, texts, all collect[string]
	src := Source[string]("heard").From(From("hello", "!stop", "world", "!faster", "bye"))
	sort := Route("sort", func(c Chunk[string], emit func(to int, c Chunk[string])) error {
		if cmd, ok := strings.CutPrefix(c.Data, "!"); ok {
			c.Data = cmd
			emit(1, c)
			return nil
		}
		emit(0, c)
		if c.Data == "bye" {
			emit(1, Chunk[string]{Data: "(the end)"}) // one chunk to several outputs, another of its own
		}
		return nil
	}, AtEnd(func(emit func(Chunk[string])) error {
		emit(Chunk[string]{Data: "(done)"}) // to the first output
		return nil
	}))
	src.To(sort).To(texts.sink("texts"))
	sort.To(commands.sink("commands"), Keep(1))
	src.To(all.sink("all"))
	if err := Run(context.Background(), src); err != nil {
		t.Fatal(err)
	}
	if want := []string{"hello", "world", "bye", "(done)"}; !slices.Equal(texts.got, want) {
		t.Errorf("the first output %q, want %q", texts.got, want)
	}
	if want := []string{"stop", "faster", "(the end)"}; !slices.Equal(commands.got, want) {
		t.Errorf("the second output %q, want %q", commands.got, want)
	}
	if len(all.got) != 5 {
		t.Errorf("the source's other reader got %d chunks, want 5", len(all.got))
	}

	// an output that is not there
	bad := Source[int]("n").From(From(1))
	bad.To(Route("lost", func(c Chunk[int], emit func(int, Chunk[int])) error { emit(2, c); return nil })).To(Sink[int]("only"))
	var ne *NodeError
	if err := Run(context.Background(), bad); !errors.As(err, &ne) || ne.Node != "lost" || !strings.Contains(err.Error(), "output 2, and has 1") {
		t.Errorf("a chunk to an output that is not there: %v", err)
	}
}

// TestJoins: nodes of two inputs.
func TestJoins(t *testing.T) {
	ctx := context.Background()
	// Both: the first with the first, the second with the second, until either ends
	var pairs collect[string]
	names := Source[string]("names").From(From("a", "b", "c"))
	sizes := Source[int]("sizes").From(From(1, 2))
	both := Both("pair", func(a Chunk[string], b Chunk[int], emit func(Chunk[string])) error {
		emit(Chunk[string]{Data: fmt.Sprintf("%s%d", a.Data, b.Data)})
		return nil
	})
	names.To(both.In1).To(pairs.sink("pairs")) // on from the node the input belongs to
	sizes.To(both.In2)
	if err := Run(ctx, names, sizes); err != nil {
		t.Fatal(err)
	}
	if want := []string{"a1", "b2"}; !slices.Equal(pairs.got, want) {
		t.Errorf("Both gave %q, want %q", pairs.got, want)
	}
	var lines []string
	for _, n := range Walk(names, sizes) {
		line := n.Name
		for _, e := range n.To {
			line += " →" + e.To
		}
		lines = append(lines, line)
	}
	if want := []string{"names →pair (1)", "pair →pairs", "pairs", "sizes →pair (2)"}; !slices.Equal(lines, want) {
		t.Errorf("the network %q, want %q", lines, want)
	}

	// only one input given: the network does not run
	half := Source[string]("names").From(From("a"))
	half.To(Both("pair", func(Chunk[string], Chunk[int], func(Chunk[string])) error { return nil }).In1)
	if err := Run(ctx, half); err == nil || !strings.Contains(err.Error(), "nothing goes into input 2 of pair") {
		t.Errorf("a node of two inputs with one given: %v", err)
	}

	// With: on the beat of the first, with the newest of the second
	ms := time.Millisecond
	synctest.Test(t, func(t *testing.T) {
		ctx := t.Context()
		ticks := Source[int]("ticks").From(every(ctx, 6, 100*ms)) // at 100 ... 600 ms
		found := Source[int]("found").From(every(ctx, 2, 240*ms)) // at 240 and 480 ms: never as a tick comes
		var drawn []string
		draw := With("draw", func(tick Chunk[int], pos Chunk[int], ok bool, emit func(Chunk[string])) error {
			s := fmt.Sprintf("%d:-", tick.Data)
			if ok {
				s = fmt.Sprintf("%d:%d", tick.Data, pos.Data)
			}
			emit(Chunk[string]{Data: s})
			return nil
		})
		ticks.To(draw.In1)
		found.To(draw.In2, Newest())
		draw.To(Sink[string]("screen").Do(func(c Chunk[string]) error {
			drawn = append(drawn, c.Data)
			return nil
		}))
		if err := Run(ctx, ticks, found); err != nil {
			t.Fatal(err)
		}
		if want := []string{"0:-", "1:-", "2:0", "3:0", "4:1", "5:1"}; !slices.Equal(drawn, want) {
			t.Errorf("With drew %q, want %q", drawn, want)
		}
	})
}
