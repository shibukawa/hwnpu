package experts

import (
	"maps"
	"path/filepath"
	"slices"
	"testing"
	"time"
)

// fakeHost is a Host of L layers of E experts in memory (no checkpoint), for the placement.
func fakeHost(L, E int) *Host {
	sh := Shape{Hidden: 256, Inter: 256, Experts: E, TopK: 2}
	h := &Host{shape: sh, layers: map[int]*hostLayer{}}
	H, I := sh.Hidden, sh.Inter
	for l := range L {
		h.layers[l] = &hostLayer{gw: make([]byte, E*2*I*H/2), ge: make([]byte, E*2*I*H/32), dw: make([]byte, E*H*I/2), de: make([]byte, E*H*I/32)}
	}
	h.usage = NewUsage(L, E)
	return h
}

// fakeLock stands for mlock: the system's limit on wired memory is not the test's.
func fakeLock(pl *placer) *placer {
	pl.lock = func([]byte) error { return nil }
	pl.unlock = func([]byte) {}
	return pl
}

func wiredSet(pl *placer) []int {
	pl.mu.Lock()
	defer pl.mu.Unlock()
	return slices.Sorted(maps.Keys(pl.wired))
}

func TestPlacement(t *testing.T) {
	const L, E = 2, 4
	h := fakeHost(L, E)
	eb := h.expertBytes()
	for _, sp := range h.spans(0, 1) {
		if len(sp) == 0 {
			t.Fatal("an empty span")
		}
	}

	// this application's counts: layer 0 runs expert 2 most, then 3; layer 1 runs 0
	path := filepath.Join(t.TempDir(), "experts-usage", "chat.bin")
	u := NewUsage(L, E)
	u.Add(0, [][]int{{2, 3}, {2, 3}, {2, 1}})
	u.Add(1, [][]int{{0}, {0}, {0}})
	if err := u.Write(path); err != nil {
		t.Fatal(err)
	}
	// core: layer 1's expert 3; a budget of 3 experts
	pl := fakeLock(newPlacer(h, Placement{Core: [][]int{{}, {3}}, Usage: path, Idle: -1}, 3*eb, L))
	pl.balance(true)
	// the core first (1*E+3 = 7), then by share: layer 1's 0 (share 1) and layer 0's 2 (0.5)
	if got, want := wiredSet(pl), []int{0*E + 2, 1*E + 0, 1*E + 3}; !slices.Equal(got, want) {
		t.Fatalf("wired %v, want %v", got, want)
	}

	// the use moves to layer 0's expert 1: after a few balances it replaces layer 0's 2
	for range 3 {
		h.usage.Add(0, [][]int{{1}, {1}, {1}, {1}, {1}, {1}})
		h.usage.Add(1, [][]int{{0}})
		pl.balance(false)
	}
	if got, want := wiredSet(pl), []int{0*E + 1, 1*E + 0, 1*E + 3}; !slices.Equal(got, want) {
		t.Fatalf("after the use moved: wired %v, want %v", got, want)
	}

	// another application with no counts of its own takes the directory's files together
	other := newPlacer(h, Placement{Usage: filepath.Join(filepath.Dir(path), "agent.bin"), Idle: -1}, eb, L)
	if other.score[1*E+0] == 0 {
		t.Fatal("the directory's counts were not read for a missing file")
	}
	pl.stop()

	// idle: all but the core unwired
	h2 := fakeHost(L, E)
	idle := fakeLock(newPlacer(h2, Placement{Core: [][]int{{1}}, Usage: path, Idle: 20 * time.Millisecond}, 3*eb, L))
	idle.start()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if got := wiredSet(idle); len(got) == 1 && got[0] == 1 && !idle.busy.Load() {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if got := wiredSet(idle); !slices.Equal(got, []int{1}) {
		t.Fatalf("idle: wired %v, want only the core [1]", got)
	}
	idle.stop()
}

func TestWiringBudget(t *testing.T) {
	h := fakeHost(2, 4)
	n := h.bytes()
	for _, c := range []struct {
		env    string
		all    bool
		budget int64
	}{{"0", false, 0}, {"1", true, n}, {"0.0001GB", false, 100000}, {"100g", true, n}} {
		t.Setenv("HWNPU_MOE_WIRE", c.env)
		all, b := h.wiring()
		if all != c.all || b != c.budget {
			t.Errorf("HWNPU_MOE_WIRE=%s: all %v budget %d, want %v %d", c.env, all, b, c.all, c.budget)
		}
	}
}
