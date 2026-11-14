package decode

import (
	"slices"
	"testing"
)

// smaller gives p attention programs over chunks shorter than its own (Program.Smaller).
func smaller(p Program, q int, chunks ...int) Program {
	for _, c := range chunks {
		p.Smaller = append(p.Smaller, fakeAttentionOf(q, c, false))
	}
	return p
}

// A session whose cache moves into longer chunks as the sequence grows (16, 32, then the
// attention program's 64) attends as one that holds it in chunks of 64 from the start: every call
// is the exact attention while the whole cache is read, and the two-level one past the length
// where it pays, with programs of 1 and 4 tokens per call.
func TestSmallerChunks(t *testing.T) {
	const n = 1200
	tab := newTables(n, 11)
	var out []float32
	s, err := New(smaller(fakeProgram(1, n, &tab, &out, true), 1, 16, 32), fakeModel{})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Add(smaller(fakeProgram(4, n, &tab, &out, true), 4, 16, 32)); err != nil {
		t.Fatal(err)
	}
	if err := s.SetSparse(testSparse()); err != nil {
		t.Fatal(err)
	}
	if s.chunk != 16 {
		t.Fatalf("an empty session holds chunks of %d, want 16", s.chunk)
	}
	pick, _ := s.layout(testSparse(), 4, false)
	var seen []int
	for _, end := range []int{14, 30, 60, 200} {
		steps(t, s, tab, &out, s.Pos(), end, 4, 0, pick)
		seen = append(seen, s.chunk)
	}
	if !slices.Equal(seen, []int{16, 32, 64, 64}) {
		t.Errorf("chunks %v after 14, 30, 60 and 200 tokens and a call's padding, want 16, 32, 64, 64", seen)
	}
	if steps(t, s, tab, &out, s.Pos(), n, 4, 0, pick) == 0 {
		t.Fatal("the two-level attention never ran")
	}
}

// After going back (Rewind, Reset) the next step moves the cache into the shortest chunk that
// holds the sequence again, the rows before it kept; rows put back (a Save's) move it into
// longer ones.
func TestSmallerChunksBack(t *testing.T) {
	const n, q = 600, 4
	a, b := newTables(n, 12), newTables(n, 13)
	tab := a
	var out []float32
	s, err := New(smaller(fakeProgram(q, n, &tab, &out, true), q, 16, 32), fakeModel{})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetSparse(testSparse()); err != nil {
		t.Fatal(err)
	}
	pick, _ := s.layout(testSparse(), q, true)
	steps(t, s, a, &out, 0, 20, q, 0, pick)
	at20 := s.Mark()
	steps(t, s, a, &out, 20, 300, q, 0, pick)
	saved := s.Save()
	if s.chunk != 64 {
		t.Fatalf("chunks of %d at 300 tokens, want 64", s.chunk)
	}
	// back to 20, b's tokens from there: the first step's 24 positions fit in chunks of 32
	s.Rewind(at20)
	mixed := &tables{q: append(slices.Clone(a.q[:20]), b.q[20:]...), k: append(slices.Clone(a.k[:20]), b.k[20:]...), v: append(slices.Clone(a.v[:20]), b.v[20:]...)}
	tab = mixed
	steps(t, s, mixed, &out, 20, 24, q, 0, pick)
	if s.chunk != 32 {
		t.Errorf("chunks of %d after going back to 20 and a step of 4, want 32", s.chunk)
	}
	steps(t, s, mixed, &out, 24, 100, q, 0, pick)
	// all of b from a Reset
	s.Reset()
	tab = b
	steps(t, s, b, &out, 0, 8, q, 0, pick)
	if s.chunk != 16 {
		t.Errorf("chunks of %d after a Reset and 8 tokens, want 16", s.chunk)
	}
	steps(t, s, b, &out, 8, 50, q, 0, pick)
	// a's 300 rows put back, then a's sequence on
	s.Rewind(saved)
	if s.chunk != 64 {
		t.Errorf("chunks of %d with 300 rows put back, want 64", s.chunk)
	}
	tab = a
	steps(t, s, a, &out, 300, n, q, 0, pick)
}

// Rows exported from a session that holds its cache in chunks of 64 go into one that starts in
// chunks of 16 (a prompt kept on disk, Resume), and the other way.
func TestSmallerChunksRows(t *testing.T) {
	const n, q = 400, 4
	tab := newTables(n, 14)
	var out []float32
	whole, err := New(fakeProgram(q, n, &tab, &out, true), fakeModel{})
	if err != nil {
		t.Fatal(err)
	}
	ladder, err := New(smaller(fakeProgram(q, n, &tab, &out, true), q, 16, 32), fakeModel{})
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range []*Session{whole, ladder} {
		if err := s.SetSparse(testSparse()); err != nil {
			t.Fatal(err)
		}
	}
	pick, _ := whole.layout(testSparse(), q, true)
	for _, c := range []struct {
		from, to *Session
		at       int
	}{{whole, ladder, 28}, {ladder, whole, 150}} {
		c.from.Reset()
		steps(t, c.from, tab, &out, 0, c.at, q, 0, pick)
		m := c.from.Save()
		if err := c.to.Resume(m.rows, NewMark(m.pos, m.bufs)); err != nil {
			t.Fatal(err)
		}
		steps(t, c.to, tab, &out, c.at, c.at+120, q, 0, pick)
	}
	if ladder.chunk != 64 {
		t.Errorf("chunks of %d after 148 tokens, want 64", ladder.chunk)
	}
}

// A shorter chunk must divide the next and hold a step.
func TestSmallerChunksChecked(t *testing.T) {
	tab := newTables(64, 15)
	var out []float32
	for _, chunks := range [][]int{{24}, {32, 16}, {2}, {64}} {
		if _, err := New(smaller(fakeProgram(4, 64, &tab, &out, false), 4, chunks...), fakeModel{}); err == nil {
			t.Errorf("shorter chunks %v of 64 for 4 tokens a step taken", chunks)
		}
	}
}
