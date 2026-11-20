package experts

import (
	"maps"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/shibukawa/hwnpu/internal/mem"
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

func TestPlacementPressure(t *testing.T) {
	const L, E = 2, 4
	h := fakeHost(L, E)
	eb := h.expertBytes()
	path := filepath.Join(t.TempDir(), "u.bin")
	u := NewUsage(L, E)
	u.Add(0, [][]int{{2, 3}, {2, 3}, {2, 1}, {1}})
	u.Add(1, [][]int{{0}, {0}, {1}})
	if err := u.Write(path); err != nil {
		t.Fatal(err)
	}
	pl := fakeLock(newPlacer(h, Placement{Core: [][]int{{}, {3}}, Usage: path, Idle: -1}, 5*eb, L))
	var now mem.Pressure
	pl.pressure = func() (mem.Pressure, bool) { return now, true }
	pl.phys = 100 * eb // a percent of the memory is an expert
	for _, c := range []struct {
		p    mem.Pressure
		want int64
	}{
		{mem.Pressure{Free: 80, Level: 1}, 5 * eb},
		{mem.Pressure{Free: 27, Level: 1}, 2 * eb}, // what leaves 25 % free
		{mem.Pressure{Free: 25, Level: 1}, eb},     // the core alone
		{mem.Pressure{Free: 10, Level: 1}, eb},
		{mem.Pressure{Free: 60, Level: 4}, eb},     // the system's critical level
		{mem.Pressure{Free: 40, Level: 2}, 5 * eb}, // its warning: 35 % stay free
		{mem.Pressure{Free: 37, Level: 1}, 2 * eb}, // and for a while after it
	} {
		now = c.p
		if got := pl.effective(); got != c.want {
			t.Errorf("%+v: budget %d, want %d", c.p, got, c.want)
		}
	}
	pl.warned.Store(0)
	now = mem.Pressure{Free: 80, Level: 1}
	pl.balance(true)
	if got := len(wiredSet(pl)); got != 5 {
		t.Fatalf("free: %d experts wired, want 5", got)
	}
	// eb + 4eb/3: the core and the best by use (layer 1's expert 0, 2/3 of its layer's runs,
	// above layer 0's expert 2, 3/7)
	// 5 experts wired (own bytes: their whole pages) and so little free that the memory free
	// without them, less the 25 % to stay free, is 2 experts and a part
	own := pl.bytes
	short := 25 - int((own-2*eb)/eb)
	now = mem.Pressure{Free: short, Level: 1}
	if got, want := pl.effective(), own-int64(25-short)*eb; got != want || want < 2*eb || want >= 3*eb {
		t.Fatalf("short (%d%% free, %d wired): budget %d, want %d of 2 experts", short, own, got, want)
	}
	pl.follow()
	pl.apply(pl.target())
	if got, want := wiredSet(pl), []int{1*E + 0, 1*E + 3}; !slices.Equal(got, want) {
		t.Fatalf("short: wired %v, want %v", got, want)
	}
	// settled: with the 2 wired and 25 % free the budget stays, and a percent more too
	for _, f := range []int{25, 26} {
		now = mem.Pressure{Free: f, Level: 1}
		if got := pl.effective(); got != pl.budget {
			t.Fatalf("settled at %d%%: budget %d, want %d", f, got, pl.budget)
		}
	}
	now = mem.Pressure{Free: 80, Level: 1}
	pl.balance(false)
	if got := len(wiredSet(pl)); got != 5 {
		t.Fatalf("free again: %d experts wired, want 5", got)
	}
	pl.stop()
}

// The system's warning with more than the core wired lowers the budget's ceiling a step below
// what was wired, so that the next balance does not wire the same again; it rises after ceilHold.
func TestPlacementWarned(t *testing.T) {
	const L, E = 2, 8
	h := fakeHost(L, E)
	eb := h.expertBytes()
	u := NewUsage(L, E)
	for l := range L {
		for e := range E {
			u.Add(l, [][]int{{e}})
		}
	}
	path := filepath.Join(t.TempDir(), "u.bin")
	if err := u.Write(path); err != nil {
		t.Fatal(err)
	}
	place := Placement{Core: [][]int{{0}}, Usage: path, Idle: -1, Ceiling: filepath.Join(t.TempDir(), "experts-usage", "ceiling")}
	pl := fakeLock(newPlacer(h, place, 9*eb, L))
	free := mem.Pressure{Free: 40, Level: 1}
	now, clock := free, time.Unix(1000, 0)
	pl.pressure = func() (mem.Pressure, bool) { return now, true }
	pl.now = func() time.Time { return clock }
	pl.balance(true)
	if got := len(wiredSet(pl)); got != 9 {
		t.Fatalf("free: %d experts wired, want 9", got)
	}
	// the warning: the core alone, and a ceiling 2 experts (a quarter of the 8 above the core) lower
	for _, want := range []int{7, 5} {
		now = mem.Pressure{Free: 24, Level: 2}
		pl.balance(false)
		if got := len(wiredSet(pl)); got != 1 {
			t.Fatalf("warned: %d experts wired, want the core's 1", got)
		}
		now = free
		for range 3 { // it stays there
			clock = clock.Add(30 * time.Second)
			pl.balance(false)
			if got := len(wiredSet(pl)); got != want {
				t.Fatalf("after the warning: %d experts wired, want %d", got, want)
			}
		}
	}
	// the next run starts under the ceiling the first kept
	pl.stop()
	timeNow = func() time.Time { return clock }
	defer func() { timeNow = time.Now }()
	next := fakeLock(newPlacer(fakeHost(L, E), place, 9*eb, L))
	next.pressure = pl.pressure
	next.balance(true)
	if got := len(wiredSet(next)); got != 5 {
		t.Fatalf("the next run: %d experts wired, want 5", got)
	}
	clock = clock.Add(ceilHold)
	next.balance(false)
	if got := len(wiredSet(next)); got != 7 {
		t.Fatalf("after %v: %d experts wired, want 7", ceilHold, got)
	}
	// a warning right after the step up: back to 5, and twice as long before the next step
	now = mem.Pressure{Free: 24, Level: 2}
	clock = clock.Add(30 * time.Second)
	next.balance(false)
	now = free
	clock = clock.Add(ceilHold)
	next.balance(false)
	if got := len(wiredSet(next)); got != 5 {
		t.Fatalf("%v after a warning that followed a step up: %d experts wired, want 5", ceilHold, got)
	}
	clock = clock.Add(ceilHold)
	next.balance(false)
	if got := len(wiredSet(next)); got != 7 {
		t.Fatalf("%v after it: %d experts wired, want 7", 2*ceilHold, got)
	}
	next.stop()
	// and with the time of the ceiling long past, none
	clock = clock.Add(24 * time.Hour)
	last := fakeLock(newPlacer(fakeHost(L, E), place, 9*eb, L))
	last.pressure = pl.pressure
	last.balance(true)
	if got := len(wiredSet(last)); got != 9 {
		t.Fatalf("a day later: %d experts wired, want 9", got)
	}
	last.stop()
	if _, err := os.Stat(place.Ceiling); err == nil {
		t.Fatalf("%s is left with no ceiling", place.Ceiling)
	}
}
