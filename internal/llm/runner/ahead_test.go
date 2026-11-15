package runner

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"testing"
	"time"
)

// fakeProgram notes its close in a fakeLoader.
type fakeProgram struct {
	Program
	name string
	l    *fakeLoader
}

func (p *fakeProgram) Close() error {
	p.l.mu.Lock()
	defer p.l.mu.Unlock()
	p.l.open--
	p.l.events = append(p.l.events, "close "+p.name)
	return nil
}

// fakeLoader opens fakePrograms, counting those open at once. It fails every load of fail, and
// the first load of once.
type fakeLoader struct {
	mu         sync.Mutex
	events     []string
	open, max  int
	fail, once string
}

func (l *fakeLoader) load(ctx context.Context, name string) (Program, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if name == l.once {
		l.once = ""
		l.events = append(l.events, "fail "+name)
		return nil, errors.New("not now " + name)
	}
	if name == l.fail {
		return nil, errors.New("no " + name)
	}
	l.open++
	l.max = max(l.max, l.open)
	l.events = append(l.events, "open "+name)
	return &fakeProgram{name: name, l: l}, nil
}

// TestInTurn checks that InTurn gives every program once, in order, each closed after its turn:
// one held at a time without ahead, the next loaded before a turn ends with it, and nothing left
// open after a load or a turn fails.
func TestInTurn(t *testing.T) {
	names := []string{"a", "b", "c", "d"}
	for _, ahead := range []bool{false, true} {
		l := &fakeLoader{}
		var got []string
		err := InTurn(context.Background(), names, ahead, l.load, func(i int, p Program) error {
			got = append(got, p.(*fakeProgram).name)
			if ahead && i+1 < len(names) { // the next one loads during this turn
				for deadline := time.Now().Add(5 * time.Second); ; time.Sleep(time.Millisecond) {
					l.mu.Lock()
					n := l.open
					l.mu.Unlock()
					if n == 2 {
						break
					}
					if time.Now().After(deadline) {
						t.Fatal("the next program did not load during a turn")
					}
				}
			}
			return nil
		})
		if err != nil || !slices.Equal(got, names) || l.open != 0 {
			t.Fatalf("ahead %v: %v, ran %v, %d left open", ahead, err, got, l.open)
		}
		if want := map[bool]int{false: 1, true: 2}[ahead]; l.max > want {
			t.Errorf("ahead %v: %d programs open at once, want at most %d", ahead, l.max, want)
		}
		// a failing load, and a failing turn with a load under way
		l = &fakeLoader{fail: "c"}
		if err := InTurn(context.Background(), names, ahead, l.load, func(int, Program) error { return nil }); err == nil || l.open != 0 {
			t.Errorf("ahead %v, no c: %v, %d left open", ahead, err, l.open)
		}
		l = &fakeLoader{}
		err = InTurn(context.Background(), names, ahead, l.load, func(i int, p Program) error {
			if i == 1 {
				return errors.New("stop")
			}
			return nil
		})
		if err == nil || err.Error() != "b: stop" || l.open != 0 {
			t.Errorf("ahead %v, a turn failing: %v, %d left open", ahead, err, l.open)
		}
	}
}

// TestRing checks a Ring over several rounds: every Take is the program asked for, with ahead the
// one after it is the one loading (the first after the last), a Take out of order drops that load,
// and Close leaves nothing open.
func TestRing(t *testing.T) {
	names := []string{"g0", "g1", "g2"}
	for _, ahead := range []bool{false, true} {
		l := &fakeLoader{}
		r := NewRing(names, ahead, l.load)
		for round := 0; round < 3; round++ {
			for i, name := range names {
				p, err := r.Take(context.Background(), i)
				if err != nil || p.(*fakeProgram).name != name {
					t.Fatalf("ahead %v, round %d: program %d is %v (%v)", ahead, round, i, p, err)
				}
				p.Close()
			}
		}
		// out of order: the load ahead (of g0) is not g2
		p, err := r.Take(context.Background(), 2)
		if err != nil || p.(*fakeProgram).name != "g2" {
			t.Fatalf("ahead %v: out of order %v (%v)", ahead, p, err)
		}
		p.Close()
		if err := r.Close(); err != nil || l.open != 0 {
			t.Errorf("ahead %v: %v, %d left open", ahead, err, l.open)
		}
		if l.max > 2 || (!ahead && l.max > 1) {
			t.Errorf("ahead %v: %d programs open at once", ahead, l.max)
		}
		opens := 0
		for _, e := range l.events {
			if e[:4] == "open" {
				opens++
			}
		}
		// 3 rounds of 3 and one more; with ahead one load after each but the dropped ones
		if want := 10; !ahead && opens != want {
			t.Errorf("%d loads, want %d: %v", opens, want, l.events)
		}
		if ahead && opens > 12 {
			t.Errorf("ahead: %d loads: %v", opens, l.events)
		}
	}
	// a load ahead that failed is tried again by the Take of that program
	l := &fakeLoader{once: "g1"}
	r := NewRing(names, true, l.load)
	for i, name := range names {
		p, err := r.Take(context.Background(), i)
		if err != nil || p.(*fakeProgram).name != name {
			t.Fatalf("with a load ahead failing once: program %d is %v (%v)", i, p, err)
		}
		p.Close()
	}
	if r.Close(); l.open != 0 || !slices.Contains(l.events, "fail g1") {
		t.Errorf("%d left open: %v", l.open, l.events)
	}
	// one that keeps failing is reported by the Take of that program, not before
	l = &fakeLoader{fail: "g1"}
	r = NewRing(names, true, l.load)
	p, err := r.Take(context.Background(), 0)
	if err != nil {
		t.Fatal(err)
	}
	p.Close()
	if _, err := r.Take(context.Background(), 1); err == nil {
		t.Error("the failed load ahead gave a program")
	}
	if r.Close(); l.open != 0 {
		t.Errorf("%d left open: %s", l.open, fmt.Sprint(l.events))
	}
}
