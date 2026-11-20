package experts

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"

	"github.com/shibukawa/hwnpu/internal/mem"
)

// Placement says which of the host's experts stay wired in memory when they do not all fit
// (Host.wiring): the experts any text uses (Core, models.yaml's experts_core), then those this
// application ran most (Usage), up to a budget of bytes. While the model runs, the experts it
// uses less are unwired for those it uses more, and after Idle without a run all but Core are
// unwired: their pages go back to the page cache, which the system gives to other applications.
type Placement struct {
	// Core are each layer's experts wired from the start (models.yaml's experts_core: the
	// experts every kind of text runs, tools/moeusage).
	Core [][]int
	// Usage is this application's counts of the experts each layer ran (Usage), read when the
	// experts open and added to when they close: the model's experts-usage/KEY.bin. When it is
	// missing, the other files of its directory together stand for it.
	Usage string
	// Idle is how long without a run unwires all but Core (0: 5 minutes; below 0: never).
	Idle time.Duration
}

// balanceEvery is how often, at most, the wired experts follow the ones in use; the first time
// after balanceFirst, so that a start with no counts (Core alone) soon wires the ones in use.
// pressureEvery is how often the system's shortage of memory is read, to give wired memory back.
const (
	balanceEvery  = 30 * time.Second
	balanceFirst  = 5 * time.Second
	pressureEvery = 5 * time.Second
)

// The budget follows the system's free memory (mem.Pressure): all of it at freeHigh percent free
// or more, the core alone at freeLow or less and at the system's warning (Level 2 and up), in
// proportion between. The free percent counts the page cache the system can drop, so it falls only
// once the experts not wired are dropped as well: then the wired ones are what the other
// applications lack. A budget that shrinks unwires the experts used least; it grows back at the
// next balance of a run.
const (
	freeHigh = 25
	freeLow  = 10
)

// effective is the budget for the system's pressure now.
func (pl *placer) effective() int64 {
	p, ok := pl.pressure()
	if !ok {
		return pl.max
	}
	core := pl.coreBytes()
	switch {
	case p.Level >= 2 || p.Free <= freeLow:
		return core
	case p.Free >= freeHigh:
		return max(min(pl.max, pl.ceiling), core)
	}
	return max(min(core+(pl.max-core)*int64(p.Free-freeLow)/(freeHigh-freeLow), pl.ceiling), core)
}

// coreBytes is the size of the core's experts, at most the budget asked for.
func (pl *placer) coreBytes() int64 {
	var core int64
	for _, c := range pl.core {
		if c {
			core += pl.h.expertBytes()
		}
	}
	return min(core, pl.max)
}

// The system's warning (Level 2 and up) while more than the core is wired lowers the ceiling of
// the budget to a step below what was wired then, for ceilHold; after it the ceiling rises a step
// at a time. Without it the wiring and the warning chase each other: on a 16 GB M2 with 24–26 %
// free, 4.3 GB wired brought the warning, the core's 1.9 GB took it away (40 % free), and the
// next balance wired the 2.4 GB again, every 30 s.
const ceilHold = 10 * time.Minute

// ceilStep is a step of the ceiling: a quarter of the budget above the core, an expert at least.
func (pl *placer) ceilStep() int64 {
	return max((pl.max-pl.coreBytes())/4, pl.h.expertBytes())
}

// warned lowers the ceiling when the system warns of memory with more than the core wired.
func (pl *placer) warned() {
	p, ok := pl.pressure()
	if !ok || p.Level < 2 {
		return
	}
	pl.mu.Lock()
	wired := int64(len(pl.wired)) * pl.h.expertBytes() // as the budget counts them (spans are whole pages, a little less)
	pl.mu.Unlock()
	core := pl.coreBytes()
	if wired <= core+pl.h.expertBytes() {
		return
	}
	pl.ceiling = max(min(pl.ceiling, wired)-pl.ceilStep(), core)
	pl.ceilAt = pl.now()
	if DebugTime() {
		fmt.Fprintf(os.Stderr, "experts: the system warns of memory with %.1f GB wired: at most %.1f GB for %v\n", float64(wired)/1e9, float64(pl.ceiling)/1e9, ceilHold)
	}
}

// rise raises the ceiling a step when it has held for ceilHold.
func (pl *placer) rise() {
	if pl.ceiling < pl.max && pl.now().Sub(pl.ceilAt) >= ceilHold {
		pl.ceiling = min(pl.ceiling+pl.ceilStep(), pl.max)
		pl.ceilAt = pl.now()
	}
}

// follow sets the budget to effective's, telling a change of more than an expert.
func (pl *placer) follow() {
	b := pl.effective()
	if d := b - pl.budget; d > pl.h.expertBytes() || -d > pl.h.expertBytes() {
		if DebugTime() {
			p, _ := pl.pressure()
			fmt.Fprintf(os.Stderr, "experts: memory %d%% free (pressure %d): budget %.1f → %.1f GB\n", p.Free, p.Level, float64(pl.budget)/1e9, float64(b)/1e9)
		}
	}
	pl.budget = b
}

// wiring is how the experts are wired: all of them, none, or a budget of bytes for
// Placement's choice. HWNPU_MOE_WIRE=1 wires all, 0 none, a size (6GB, 6g, 6.5) that budget;
// otherwise all when they take at most half of the machine's memory, else a quarter of it (4.3
// GB of 16 GiB; it was 3/8 until the prefetch made 3 GB as fast as 6: an M2 12.4–12.6 against
// 12.2–12.5 tokens/s, an M3 16.8–16.9 against 17.2–17.6). The budget then follows the system's
// free memory (placer.effective). Wired, a token's experts cost half as much even with memory to spare (gpt-oss-20b on a
// 32 GB M4: 1.8 → 0.9 ms a layer, 13.5 → 18.0 tokens/s): unwired pages of a mapped file keep
// costing the process faults, and on a 16 GB machine with a desktop's applications open the
// system drops them (an M2: 13.2 tokens/s wired, 7.9 not).
func (h *Host) wiring() (all bool, budget int64) {
	n := h.bytes()
	phys := mem.Physical()
	switch v := strings.TrimSpace(os.Getenv("HWNPU_MOE_WIRE")); v {
	case "":
	case "0":
		return false, 0
	case "1":
		return true, n
	default:
		s := strings.TrimSuffix(strings.TrimSuffix(strings.ToUpper(v), "B"), "G")
		if gb, err := strconv.ParseFloat(s, 64); err == nil && gb > 0 {
			b := int64(gb * 1e9)
			return b >= n, min(b, n)
		}
		fmt.Fprintf(os.Stderr, "experts: HWNPU_MOE_WIRE=%s is not 0, 1 or a size in GB: taken as 1\n", v)
		return true, n
	}
	if phys > 0 && 2*n <= phys {
		return true, n
	}
	return false, phys / 4
}

// bytes is the size of every expert's weights.
func (h *Host) bytes() int64 {
	var n int64
	for _, hl := range h.layers {
		n += int64(len(hl.gw) + len(hl.ge) + len(hl.dw) + len(hl.de))
	}
	return n
}

// expertBytes is the size of one expert's weights.
func (h *Host) expertBytes() int64 {
	H, I := h.shape.Hidden, h.shape.Inter
	return int64(2*I*H/2 + 2*I*H/32 + H*I/2 + H*I/32)
}

// spans are the pages of expert e of layer l: its four ranges (the gate and up blocks and
// scales, the down ones), each shrunk to whole pages so that wiring one expert never wires or
// unwires a page of its neighbours.
func (h *Host) spans(l, e int) [][]byte {
	hl := h.layers[l]
	if hl == nil {
		return nil
	}
	H, I := h.shape.Hidden, h.shape.Inter
	page := os.Getpagesize()
	var out [][]byte
	for _, t := range []struct {
		b []byte
		n int
	}{{hl.gw, 2 * I * H / 2}, {hl.ge, 2 * I * H / 32}, {hl.dw, H * I / 2}, {hl.de, H * I / 32}} {
		b := t.b[e*t.n : (e+1)*t.n]
		addr := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
		lo := (page - int(addr%uintptr(page))) % page            // to the first whole page
		hi := len(b) - int((addr+uintptr(len(b)))%uintptr(page)) // to the end of the last one
		if lo < hi {
			out = append(out, b[lo:hi])
		}
	}
	return out
}

// placer keeps the wired experts in step with the ones in use (Placement).
type placer struct {
	h      *Host
	place  Placement
	budget int64 // now: max, less while the system is short of memory (effective)
	max    int64 // the budget asked for (Host.wiring)
	// ceiling is the most the budget is after the system's warning (warned), from ceilAt
	ceiling int64
	ceilAt  time.Time
	now     func() time.Time // time.Now; a test's own
	// pressure reads how short of memory the system is (mem.PressureNow; a test's own)
	pressure func() (mem.Pressure, bool)
	L, E     int
	core     []bool    // by l*E+e
	score    []float64 // each expert's share of its layer's runs, older runs weighing less
	seen     []uint32  // Host.usage's counts at the last balance

	mu      sync.Mutex // the wired experts and the system's wiring of them
	wired   map[int][][]byte
	bytes   int64
	limited error         // the system's limit on wired memory, met (vm.user_wire_limit), or too little free
	on      []atomic.Bool // by l*E+e: wired, for Run's count of hits
	hits    atomic.Int64  // the runs of a wired expert
	runs    atomic.Int64  // every expert's runs

	lock   func([]byte) error // mlock, and munlock: a test's own
	unlock func([]byte)

	busy    atomic.Bool
	last    atomic.Int64 // the time of the last balance (UnixNano)
	lastRun atomic.Int64 // the time of the last run
	quit    chan struct{}
	wg      sync.WaitGroup
}

func newPlacer(h *Host, p Placement, budget int64, L int) *placer {
	E := h.shape.Experts
	pl := &placer{h: h, place: p, budget: budget, max: budget, ceiling: budget, now: time.Now, pressure: mem.PressureNow, L: L, E: E, core: make([]bool, L*E), score: make([]float64, L*E),
		seen: make([]uint32, L*E), wired: map[int][][]byte{}, quit: make(chan struct{}), lock: mlock, unlock: munlock, on: make([]atomic.Bool, L*E)}
	for l, es := range p.Core {
		for _, e := range es {
			if l < L && e >= 0 && e < E {
				pl.core[l*E+e] = true
			}
		}
	}
	if u := readUsageFor(p.Usage, L, E); u != nil {
		pl.addShares(u.Counts, 1)
	}
	pl.lastRun.Store(time.Now().UnixNano())
	return pl
}

// readUsageFor reads the counts at path, or, when it is missing, its directory's other files
// together; nil for none.
func readUsageFor(path string, L, E int) *Usage {
	if path == "" {
		return nil
	}
	if u, err := ReadUsage(path); err == nil && u.Layers == L && u.Experts == E {
		return u
	} else if err != nil && !errors.Is(err, fs.ErrNotExist) {
		fmt.Fprintf(os.Stderr, "experts: usage: %v\n", err)
	}
	others, _ := filepath.Glob(filepath.Join(filepath.Dir(path), "*.bin"))
	var sum *Usage
	for _, o := range others {
		u, err := ReadUsage(o)
		if err != nil || u.Layers != L || u.Experts != E {
			continue
		}
		if sum == nil {
			sum = NewUsage(L, E)
		}
		sum.Merge(u)
	}
	return sum
}

// addShares adds each expert's share of its layer's counts, weighed w, to the scores.
func (pl *placer) addShares(counts []uint32, w float64) {
	for l := range pl.L {
		c := counts[l*pl.E : (l+1)*pl.E]
		var s float64
		for _, x := range c {
			s += float64(x)
		}
		if s == 0 {
			continue
		}
		for e, x := range c {
			pl.score[l*pl.E+e] += w * float64(x) / s
		}
	}
}

// target is the experts to have wired: Core, then by score, up to the budget.
func (pl *placer) target() map[int]bool {
	eb := pl.h.expertBytes()
	order := make([]int, 0, pl.L*pl.E)
	for i := range pl.L * pl.E {
		if pl.core[i] || pl.score[i] > 0 {
			order = append(order, i)
		}
	}
	slices.SortStableFunc(order, func(a, b int) int {
		if pl.core[a] != pl.core[b] {
			if pl.core[a] {
				return -1
			}
			return 1
		}
		switch {
		case pl.score[a] > pl.score[b]:
			return -1
		case pl.score[a] < pl.score[b]:
			return 1
		}
		return 0
	})
	t := map[int]bool{}
	var n int64
	for _, i := range order {
		if n+eb > pl.budget {
			break
		}
		t[i] = true
		n += eb
	}
	return t
}

// apply unwires the wired experts t leaves out and wires those it adds, until the system's
// limit; it returns the bytes wired and unwired.
func (pl *placer) apply(t map[int]bool) (wired, unwired int64) {
	pl.mu.Lock()
	defer pl.mu.Unlock()
	for i, sp := range pl.wired {
		if !t[i] {
			for _, b := range sp {
				pl.unlock(b)
				unwired += int64(len(b))
			}
			pl.bytes -= spanBytes(sp)
			delete(pl.wired, i)
			pl.on[i].Store(false)
		}
	}
	// the cores first, then by score: the order target chose them in
	add := make([]int, 0, len(t))
	for i := range t {
		if pl.wired[i] == nil {
			add = append(add, i)
		}
	}
	slices.SortFunc(add, func(a, b int) int {
		if pl.core[a] != pl.core[b] {
			if pl.core[a] {
				return -1
			}
			return 1
		}
		if pl.score[a] > pl.score[b] {
			return -1
		}
		if pl.score[a] < pl.score[b] {
			return 1
		}
		return a - b
	})
	pl.limited = nil
	for _, i := range add {
		select {
		case <-pl.quit:
			return
		default:
		}
		sp := pl.h.spans(i/pl.E, i%pl.E)
		var done [][]byte
		var err error
		for _, b := range sp {
			// ENOMEM comes and goes while the model loads (the Neural Engine's programs wired
			// at the same time: an M2 and an M3 wired 0.4–0.6 of 1.9 GB at the first try, and
			// every expert alone): try again for a while
			for try := 0; ; try++ {
				if err = pl.lock(b); err == nil || !errors.Is(err, syscall.ENOMEM) || try == 20 {
					break
				}
				select {
				case <-pl.quit:
					return
				case <-time.After(100 * time.Millisecond):
				}
			}
			if err != nil {
				break
			}
			done = append(done, b)
		}
		if err != nil {
			for _, b := range done {
				pl.unlock(b)
			}
			pl.limited = err
			break
		}
		pl.wired[i] = sp
		pl.on[i].Store(true)
		pl.bytes += spanBytes(sp)
		wired += spanBytes(sp)
	}
	return
}

func spanBytes(sp [][]byte) int64 {
	var n int64
	for _, b := range sp {
		n += int64(len(b))
	}
	return n
}

// start wires the first choice in the background and keeps the choice in step after.
func (pl *placer) start() {
	pl.last.Store(time.Now().Add(balanceFirst - balanceEvery).UnixNano())
	pl.busy.Store(true)
	pl.wg.Add(1)
	go func() {
		defer pl.wg.Done()
		defer pl.busy.Store(false)
		pl.balance(true)
	}()
	idle := pl.place.Idle
	if idle == 0 {
		idle = 5 * time.Minute
	}
	pl.wg.Add(1)
	go func() {
		defer pl.wg.Done()
		every := pressureEvery
		if idle > 0 {
			every = min(every, idle/4)
		}
		tick := time.NewTicker(every)
		defer tick.Stop()
		for {
			select {
			case <-pl.quit:
				return
			case <-tick.C:
			}
			if !pl.busy.CompareAndSwap(false, true) {
				continue
			}
			// the system short of memory: give back what it asks now, not at the next balance
			pl.warned()
			if b := pl.effective(); b < pl.budget-pl.h.expertBytes() {
				pl.follow()
				_, un := pl.apply(pl.target())
				if DebugTime() {
					fmt.Fprintf(os.Stderr, "experts: %.1f GB unwired, %.1f GB kept\n", float64(un)/1e9, float64(pl.bytes)/1e9)
				}
			}
			// idle: all but the core unwired
			if idle > 0 && time.Since(time.Unix(0, pl.lastRun.Load())) >= idle {
				core := map[int]bool{}
				for i, c := range pl.core {
					if c {
						core[i] = true
					}
				}
				pl.mu.Lock()
				extra := len(pl.wired) > len(core)
				pl.mu.Unlock()
				if extra {
					_, un := pl.apply(core)
					if DebugTime() {
						fmt.Fprintf(os.Stderr, "experts: idle %v: %.1f GB unwired, the core's %.1f GB kept\n", idle, float64(un)/1e9, float64(pl.bytes)/1e9)
					}
				}
			}
			pl.busy.Store(false)
		}
	}()
}

// count notes the experts idx of layer l run, for the share of them wired (report).
func (pl *placer) count(l int, idx [][]int) {
	var h, n int64
	for _, row := range idx {
		for _, e := range row {
			if e >= 0 && e < pl.E {
				n++
				if pl.on[l*pl.E+e].Load() {
					h++
				}
			}
		}
	}
	pl.hits.Add(h)
	pl.runs.Add(n)
}

// report prints the share of the experts' runs that found them wired.
func (pl *placer) report() {
	if n := pl.runs.Load(); n > 0 {
		fmt.Fprintf(os.Stderr, "experts: %.1f%% of %d runs on wired experts (%.1f GB wired)\n", 100*float64(pl.hits.Load())/float64(n), n, float64(pl.bytes)/1e9)
	}
}

// ran notes a run of layer: the first layer's runs balance the wiring now and then.
func (pl *placer) ran(first bool) {
	now := time.Now().UnixNano()
	pl.lastRun.Store(now)
	if !first || time.Duration(now-pl.last.Load()) < balanceEvery || !pl.busy.CompareAndSwap(false, true) {
		return
	}
	pl.last.Store(now)
	pl.wg.Add(1)
	go func() {
		defer pl.wg.Done()
		defer pl.busy.Store(false)
		pl.balance(false)
	}()
}

// balance folds the runs since the last balance into the scores (the older ones halved) and
// wires the experts they choose.
func (pl *placer) balance(first bool) {
	if !first {
		u := pl.h.usage
		u.mu.Lock()
		diff := make([]uint32, len(u.Counts))
		for i, c := range u.Counts {
			diff[i] = c - pl.seen[i]
			pl.seen[i] = c
		}
		u.mu.Unlock()
		for i := range pl.score {
			pl.score[i] /= 2
		}
		pl.addShares(diff, 1)
	}
	pl.warned()
	pl.rise()
	pl.follow()
	t0 := time.Now()
	w, un := pl.apply(pl.target())
	if errors.Is(pl.limited, syscall.ENOMEM) {
		// too little free memory to wire the pages now (not the limit, EAGAIN): try again soon
		pl.last.Store(time.Now().Add(balanceFirst - balanceEvery).UnixNano())
	}
	if DebugTime() || first && pl.limited != nil {
		total := pl.h.bytes()
		fmt.Fprintf(os.Stderr, "experts: wired %.1f GB of %.1f (budget %.1f; +%.1f −%.1f in %.1fs)", float64(pl.bytes)/1e9, float64(total)/1e9,
			float64(pl.budget)/1e9, float64(w)/1e9, float64(un)/1e9, time.Since(t0).Seconds())
		if pl.limited != nil {
			fmt.Fprintf(os.Stderr, " (%v)", pl.limited)
		}
		fmt.Fprintln(os.Stderr)
	}
}

// stop ends the balancing and unwires every expert.
func (pl *placer) stop() {
	close(pl.quit)
	pl.wg.Wait()
	if DebugTime() {
		pl.report()
	}
	pl.mu.Lock()
	for _, sp := range pl.wired {
		for _, b := range sp {
			pl.unlock(b)
		}
	}
	pl.wired, pl.bytes = nil, 0
	for i := range pl.on {
		pl.on[i].Store(false)
	}
	pl.mu.Unlock()
}
