package iron

import (
	"bytes"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"syscall"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm/experts"
	"github.com/shibukawa/hwnpu/internal/mem"
)

// A mixture of experts whose layers' tables hold fewer places than experts (Manifest.ExpertsPool,
// gpt-oss's GPTOSS_EXPERT_PLACES). The NPU reads only the pinned banks, so an expert runs from a
// place: the host copies an expert chosen and not in one from the export's experts.bin (mapped,
// in the page cache) over the place of the expert of the layer least used lately, never one
// chosen in the same step or a core expert (models.yaml's experts_core) while another will do.
//
// A decode step's experts are chosen on the device, a layer's routing writing their offsets
// into the next command's scratchpad (id × the expert's bytes: the place of a table of every
// expert). So the step's commands run one at a time: after each, the host reads the choices
// back, copies the missing experts in, and writes the places' offsets over the words before it
// submits the next command. A prompt chunk's are chosen on the host (moeHostRouteMX4): the
// experts in a place run in the first pass, the others in passes after it, each pass's missing
// experts copied in before it.

// poolDecay is how much an expert's lateness of use weighs a token later: half after about 140
// tokens.
const poolDecay = 0.995

// expertPool is the places' state.
type expertPool struct {
	man     *ExpertsPool
	data    []byte   // experts.bin, mapped
	direct  *os.File // experts.bin opened O_DIRECT where it does not fit the page cache (directReads), or nil
	L, E, P int
	places  []poolPlaces
	core    []bool    // by l*E+e
	score   []float64 // by l*E+e: the decayed uses at last
	last    []int64
	tick    int64
	filled  bool // the places hold what at says (false after a load: the banks' zeros)
	init    []float64
	ids     []int32

	// inflight are the copies into a place still running (a guess of the next layer's experts:
	// route.go), by layer and place, nil for none: closed when done
	inflight [][]chan struct{}
	waiting  time.Duration // for them

	hits, misses int64
	copying      time.Duration // the decode steps' copies
	loading      time.Duration // the prompt passes' copies
	loads        int64
}

// poolPlaces is a layer's places: the expert in each (-1 none), each expert's place (-1 none),
// the tables' buffers.
type poolPlaces struct {
	at, of []int
	gu, dn Placement
	lay    PoolLayer
}

// openPool maps the export's experts file.
func (m *Model) openPool() error {
	ep := m.Manifest.ExpertsPool
	if ep == nil {
		return nil
	}
	if ep.Places <= 0 || ep.Places > ep.Experts || len(ep.Layers) == 0 {
		return fmt.Errorf("iron: experts_pool of %d places for %d experts, %d layers", ep.Places, ep.Experts, len(ep.Layers))
	}
	f, err := os.Open(filepath.Join(m.dir, ep.File))
	if err != nil {
		return fmt.Errorf("iron: experts_pool: %w", err)
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return err
	}
	L, E, P := len(ep.Layers), ep.Experts, ep.Places
	p := &expertPool{man: ep, L: L, E: E, P: P, places: make([]poolPlaces, L), core: make([]bool, L*E),
		score: make([]float64, L*E), last: make([]int64, L*E), init: make([]float64, L*E)}
	dec := m.decode.m.Buffers
	for l, lay := range ep.Layers {
		end := lay.Offset + int64(E)*int64(lay.GUBytes+lay.DNBytes)
		if end > st.Size() {
			return fmt.Errorf("iron: experts_pool: layer %d ends at %d, past %s's %d bytes", l, end, ep.File, st.Size())
		}
		gu, ok1 := dec[lay.GU]
		dn, ok2 := dec[lay.DN]
		if !ok1 || !ok2 || gu.Bytes < P*lay.GUBytes || dn.Bytes < P*lay.DNBytes {
			return fmt.Errorf("iron: experts_pool: layer %d's tables %s and %s are not %d places", l, lay.GU, lay.DN, P)
		}
		pl := poolPlaces{at: make([]int, P), of: make([]int, E), gu: gu, dn: dn, lay: lay}
		for i := range pl.at {
			pl.at[i] = -1
		}
		for i := range pl.of {
			pl.of[i] = -1
		}
		p.places[l] = pl
	}
	p.inflight = make([][]chan struct{}, L)
	for l := range p.inflight {
		p.inflight[l] = make([]chan struct{}, P)
	}
	if p.data, err = syscall.Mmap(int(f.Fd()), 0, int(st.Size()), syscall.PROT_READ, syscall.MAP_SHARED); err != nil {
		return fmt.Errorf("iron: experts_pool: %w", err)
	}
	if directReads(st.Size()) {
		if p.direct, err = openDirect(filepath.Join(m.dir, ep.File)); err != nil {
			m.logf("experts: %s not read directly (%v): from the page cache", ep.File, err)
			p.direct = nil
		}
	}
	m.pool = p
	m.logf("experts: %d places of %d a layer (%.1f of %.1f GiB), the others copied in from %s when chosen",
		P, E, float64(int64(L)*int64(P)*int64(ep.Layers[0].GUBytes+ep.Layers[0].DNBytes))/(1<<30),
		float64(int64(L)*int64(E)*int64(ep.Layers[0].GUBytes+ep.Layers[0].DNBytes))/(1<<30), ep.File)
	return nil
}

func (p *expertPool) close() {
	if p.direct != nil {
		p.direct.Close()
		p.direct = nil
	}
	if p.data != nil {
		syscall.Munmap(p.data)
		p.data = nil
	}
}

// PlaceExperts says which experts the places hold first (experts.Placement: Core, then by the
// application's counts) and keeps the core ones in place while others will do. The places are
// filled now.
func (m *Model) PlaceExperts(pl experts.Placement) error {
	p := m.pool
	if p == nil {
		return nil
	}
	clear(p.core)
	for l, es := range pl.Core {
		for _, e := range es {
			if l < p.L && e >= 0 && e < p.E {
				p.core[l*p.E+e] = true
			}
		}
	}
	clear(p.init)
	if u := experts.ReadUsageFor(pl.Usage, p.L, p.E); u != nil {
		for l := range p.L {
			var s float64
			for e := range p.E {
				s += float64(u.Counts[l*p.E+e])
			}
			for e := range p.E {
				if s > 0 {
					// the steady score of an expert of that share of a layer's choices (4 a token)
					p.init[l*p.E+e] = 4 * float64(u.Counts[l*p.E+e]) / s / (1 - poolDecay)
				}
			}
		}
	}
	copy(p.score, p.init)
	clear(p.last)
	p.tick = 0
	p.filled = false
	if m.released {
		return nil // filled on Reload's first call
	}
	return m.fillPool()
}

// fill puts the experts to start with into the places: each layer's core experts, then by
// score, then by index.
func (m *Model) fillPool() error {
	p := m.pool
	if p == nil || p.filled {
		return nil
	}
	m.waitGuesses()
	t0 := time.Now()
	order := make([]int, p.E)
	for l := range p.L {
		pl := &p.places[l]
		for e := range order {
			order[e] = e
		}
		slices.SortStableFunc(order, func(a, b int) int {
			ca, cb := p.core[l*p.E+a], p.core[l*p.E+b]
			if ca != cb {
				if ca {
					return -1
				}
				return 1
			}
			sa, sb := p.score[l*p.E+a], p.score[l*p.E+b]
			switch {
			case sa > sb:
				return -1
			case sa < sb:
				return 1
			}
			return 0
		})
		for i := range pl.of {
			pl.of[i] = -1
		}
		for i := range p.P {
			pl.at[i], pl.of[order[i]] = order[i], i
		}
		if err := m.copyPlaces(l, pl.at); err != nil {
			return err
		}
	}
	p.filled = true
	m.logf("experts: %d places a layer filled in %.2f s", p.P, time.Since(t0).Seconds())
	if _, ok := experts.DebugOpt("check"); ok {
		return m.checkPool()
	}
	return nil
}

// checkPool compares every place's tables with its expert's in the experts file
// (HWNPU_MOE_DEBUG=check).
func (m *Model) checkPool() error {
	m.waitGuesses()
	p := m.pool
	bad := 0
	for l := range p.L {
		pl := &p.places[l]
		lay := pl.lay
		for i, e := range pl.at {
			if e < 0 {
				continue
			}
			gu := m.banks[pl.gu.Argument].Bytes()[pl.gu.Offset+i*lay.GUBytes:][:lay.GUBytes]
			dn := m.banks[pl.dn.Argument].Bytes()[pl.dn.Offset+i*lay.DNBytes:][:lay.DNBytes]
			fg := p.data[lay.Offset+int64(e)*int64(lay.GUBytes):][:lay.GUBytes]
			fd := p.data[lay.Offset+int64(p.E)*int64(lay.GUBytes)+int64(e)*int64(lay.DNBytes):][:lay.DNBytes]
			if !bytes.Equal(gu, fg) || !bytes.Equal(dn, fd) {
				bad++
			}
		}
	}
	m.logf("experts: check of the places: %d of %d differ from %s", bad, p.L*p.P, p.man.File)
	if bad > 0 {
		return fmt.Errorf("iron: %d places differ from their experts", bad)
	}
	return nil
}

// copyPlaces copies each place i's expert es[i] (-1: none) of layer l into its tables, in
// parallel over the places.
func (m *Model) copyPlaces(l int, es []int) error {
	var wg sync.WaitGroup
	errs := make([]error, len(es))
	sem := make(chan struct{}, 8)
	for i, e := range es {
		if e < 0 {
			continue
		}
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			errs[i] = m.copyExpert(l, e, i, 1)
			<-sem
		}()
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}

// copyExpert copies expert e of layer l from the experts file into place i, the tables' bytes
// split between n goroutines, and flushes them to the device.
func (m *Model) copyExpert(l, e, i, n int) error {
	p := m.pool
	pl := &p.places[l]
	lay := pl.lay
	gu, dn := m.banks[pl.gu.Argument], m.banks[pl.dn.Argument]
	if gu == nil || dn == nil {
		return fmt.Errorf("iron: experts_pool: layer %d's tables are not in a bank", l)
	}
	type span struct {
		buf      *hwnpu.Buffer
		dst, src int64
		n        int
	}
	guFile := lay.Offset + int64(e)*int64(lay.GUBytes)
	dnFile := lay.Offset + int64(p.E)*int64(lay.GUBytes) + int64(e)*int64(lay.DNBytes)
	spans := []span{
		{gu, int64(pl.gu.Offset) + int64(i)*int64(lay.GUBytes), guFile, lay.GUBytes},
		{dn, int64(pl.dn.Offset) + int64(i)*int64(lay.DNBytes), dnFile, lay.DNBytes},
	}
	var parts []span
	for _, s := range spans {
		k := max(1, n*s.n/(lay.GUBytes+lay.DNBytes))
		step := (s.n/k + 4095) &^ 4095 // whole blocks for O_DIRECT
		for at := 0; at < s.n; at += step {
			c := min(step, s.n-at)
			parts = append(parts, span{s.buf, s.dst + int64(at), s.src + int64(at), c})
		}
	}
	errs := make([]error, len(parts))
	var wg sync.WaitGroup
	for j, s := range parts {
		wg.Add(1)
		go func() {
			defer wg.Done()
			dst := s.buf.Bytes()[s.dst : s.dst+int64(s.n)]
			if p.direct == nil || !readStaged(p.direct, dst, s.src) {
				copy(dst, p.data[s.src:s.src+int64(s.n)])
			}
			errs[j] = s.buf.SyncToDevice(int(s.dst), s.n)
		}()
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}

// now is an expert's score at the pool's tick.
func (p *expertPool) now(i int) float64 {
	return p.score[i] * math.Pow(poolDecay, float64(p.tick-p.last[i]))
}

// use counts a use of expert i.
func (p *expertPool) use(i int) {
	p.score[i] = p.now(i) + 1
	p.last[i] = p.tick
}

// victim is the place of layer l to copy an expert into: one holding none, else the least
// scored of those whose expert keep does not hold, a core one only when no other is left; -1
// when keep holds every place's.
func (p *expertPool) victim(l int, keep func(e int) bool) int {
	pl := &p.places[l]
	best, bestCore := -1, true
	var low float64
	for i, e := range pl.at {
		if e < 0 {
			return i
		}
		if keep(e) {
			continue
		}
		core := p.core[l*p.E+e]
		s := p.now(l*p.E + e)
		if best < 0 || (bestCore && !core) || (core == bestCore && s < low) {
			best, bestCore, low = i, core, s
		}
	}
	return best
}

// waitPlace waits for a copy still running into place i of layer l.
func (p *expertPool) waitPlace(l, i int) {
	if c := p.inflight[l][i]; c != nil {
		t0 := time.Now()
		<-c
		p.waiting += time.Since(t0)
		p.inflight[l][i] = nil
	}
}

// place puts expert e of layer l into place i (the copy is the caller's).
func (p *expertPool) place(l, e, i int) {
	pl := &p.places[l]
	if old := pl.at[i]; old >= 0 {
		pl.of[old] = -1
	}
	pl.at[i], pl.of[e] = e, i
}

// placeChoices gives layer l's chosen experts ids their places: those not in one copied in over
// the place of the expert least used lately and not chosen.
func (m *Model) placeChoices(l int, ids []int32, places []int) ([]int, error) {
	places = places[:0]
	p := m.pool
	if p == nil {
		for _, id := range ids {
			places = append(places, int(id))
		}
		return places, nil
	}
	pl := &p.places[l]
	chosen := func(e int) bool { return slices.Contains(ids, int32(e)) }
	var copies [][2]int // (expert, place), copied together
	for k, id := range ids {
		if id < 0 || int(id) >= p.E {
			return nil, fmt.Errorf("iron: decode layer %d's choice %d is no expert (%d)", l, k, id)
		}
		e := int(id)
		p.use(l*p.E + e)
		i := pl.of[e]
		if i >= 0 {
			p.hits++
			p.waitPlace(l, i) // a guess's copy: in before the command reads it
		} else {
			p.misses++
			if i = p.victim(l, chosen); i < 0 {
				return nil, fmt.Errorf("iron: decode layer %d: no place for expert %d (%d places)", l, e, p.P)
			}
			p.waitPlace(l, i) // a guess's copy over it first
			p.place(l, e, i)
			copies = append(copies, [2]int{e, i})
		}
		places = append(places, i)
	}
	if len(copies) > 0 {
		t0 := time.Now()
		errs := make([]error, len(copies))
		var wg sync.WaitGroup
		for j, c := range copies {
			wg.Add(1)
			go func() {
				defer wg.Done()
				errs[j] = m.copyExpert(l, c[0], c[1], 8)
			}()
		}
		wg.Wait()
		p.copying += time.Since(t0)
		for _, err := range errs {
			if err != nil {
				return nil, err
			}
		}
	}
	return places, nil
}

// route is the decode step's host work before fed command r: the layer's choices read back,
// given places (placeChoices), the command's words set to them.
func (m *Model) route(r *fedRoute) error {
	p := m.pool
	buf := m.buffer(m.decode, "feedback")
	if err := buf.SyncFromDevice(r.at, r.bytes); err != nil {
		return err
	}
	p.ids = r.ids(buf.Bytes(), p.E, p.ids)
	if m.usage != nil {
		m.usage.AddIDs(r.layer, p.ids)
	}
	places, err := m.placeChoices(r.layer, p.ids, nil)
	if err != nil {
		return err
	}
	values := make(map[string]int64, len(places))
	for k, i := range places {
		values[fmt.Sprintf("e%d_%d", r.layer, k)] = int64(i)
	}
	sp := buf.Bytes()[r.at : r.at+r.bytes]
	if err := writeWords(sp, r.params, values); err != nil {
		return err
	}
	return buf.SyncToDevice(r.at, r.bytes)
}

// PoolStats is the decode steps' choices found in a place and copied in, and the time the
// copies took (zeros without places).
func (m *Model) PoolStats() (hits, misses int64, copying time.Duration) {
	if m.pool == nil {
		return 0, 0, 0
	}
	return m.pool.hits, m.pool.misses, m.pool.copying
}

// loadExperts copies the experts loads (expert, place) of layer l into their places.
func (m *Model) loadExperts(l int, loads [][2]int) error {
	es := make([]int, m.pool.P)
	for i := range es {
		es[i] = -1
	}
	for _, ld := range loads {
		es[ld[1]] = ld[0]
	}
	t0 := time.Now()
	err := m.copyPlaces(l, es)
	m.pool.loading += time.Since(t0)
	m.pool.loads += int64(len(loads))
	return err
}

// poolPasses splits a moe_route_mx4 step's choices ids (top a row of the chunk's n) into passes
// over layer's places (h.Places, fixed slot i place i; the overflow slots' values places): the
// experts in a place first, then those not in one, copied in over the places of experts no pass
// to come needs (the least used lately first, the core ones last) before their pass.
func (m *Model) poolPasses(h *HostStep, rt *moeRoute, ids []int32, n, K int) error {
	m.waitGuesses() // the prompt's passes read and copy over any place
	p := m.pool
	if p == nil || h.Layer < 0 || h.Layer >= p.L || h.Places != p.P || h.Experts != p.E {
		return fmt.Errorf("iron: moe_route_mx4 of layer %d: %d places of %d experts, the pool's %d", h.Layer, h.Places, h.Experts, p.P)
	}
	l, E, P, S, cap, top := h.Layer, h.Experts, h.Places, h.Slots, h.Cap, h.Top
	pl := &p.places[l]
	cnt := make([]int, E)
	for _, e := range ids {
		cnt[e]++
	}
	left := make([]bool, E)
	nleft := 0
	missing := []int{}
	for e, c := range cnt {
		if c > 0 {
			left[e] = true
			nleft++
			p.score[l*E+e] = p.now(l*E+e) + float64(c)
			p.last[l*E+e] = p.tick
			if pl.of[e] < 0 {
				missing = append(missing, e)
			}
		}
	}
	slices.SortStableFunc(missing, func(a, b int) int { return cnt[b] - cnt[a] })
	extra := func(e int) int { return (cnt[e] - 1) / cap }
	passOf, placeOf := make([]int, E), make([]int, E)
	var groups [][]int
	var loads [][][2]int
	for nleft > 0 {
		var g []int
		var ld [][2]int
		ovf := S - P
		for _, e := range pl.at {
			if e >= 0 && left[e] && extra(e) <= ovf {
				g, ovf = append(g, e), ovf-extra(e)
			}
		}
		inPass := make([]bool, E)
		for _, e := range g {
			inPass[e] = true
		}
		for _, e := range missing {
			if !left[e] || pl.of[e] >= 0 || extra(e) > ovf {
				continue
			}
			i := p.victim(l, func(x int) bool { return left[x] })
			if i < 0 {
				break
			}
			p.place(l, e, i)
			ld = append(ld, [2]int{e, i})
			g, ovf = append(g, e), ovf-extra(e)
			inPass[e] = true
		}
		if len(g) == 0 {
			return fmt.Errorf("iron: moe_route_mx4 of layer %d: no pass for the %d experts left", l, nleft)
		}
		for _, e := range g {
			left[e], passOf[e], placeOf[e] = false, len(groups), pl.of[e]
			nleft--
		}
		groups, loads = append(groups, g), append(loads, ld)
	}
	p.tick += int64(n)
	kt := h.KT
	kbs := h.InFeatures / kt
	over := make([]int, E)
	for q, g := range groups {
		ps := moePass{counts: make([]byte, h.Writes[0].Bytes), outRows: S * cap, cap: cap, loads: loads[q]}
		slotPlace, slotRows := make([]int, S), make([]int, S)
		next := P
		for _, e := range g {
			c := cnt[e]
			slotRows[placeOf[e]] = min(cap, c)
			over[e] = next
			for r := cap; r < c; r += cap {
				slotPlace[next], slotRows[next] = placeOf[e], min(cap, c-r)
				next++
			}
			ps.rows += c
		}
		if next > S {
			return fmt.Errorf("iron: moe_route_mx4 of layer %d: %d slots of %d", l, next, S)
		}
		ps.values = make(map[string]int64, S)
		for i, name := range h.SlotValues {
			if name != "" {
				ps.values[name] = int64(slotPlace[i])
			}
		}
		ps.slotRows = slotRows
		for sl, c := range slotRows {
			ps.counts[4*sl], ps.counts[4*sl+1], ps.counts[4*sl+2], ps.counts[4*sl+3] = byte(c), byte(c>>8), byte(c>>16), byte(c>>24)
		}
		rt.passes = append(rt.passes, ps)
	}
	at := make([]int, E)
	for t := 0; t < n; t++ {
		for k := 0; k < top; k++ {
			c := t*top + k
			e := int(ids[c])
			q := passOf[e]
			sl, r := placeOf[e], at[e]%cap
			if j := at[e] / cap; j > 0 {
				sl = over[e] + j - 1
			}
			at[e]++
			rt.pass[c], rt.row[c] = int32(q), int32(sl*cap+r)
			ps := &rt.passes[q]
			for kb := 0; kb < kbs; kb++ {
				dst := ((sl*kbs+kb)*cap + r) * kt
				ps.copies = append(ps.copies, rowCopy{dst: 2 * dst, src: 2 * (t*K + kb*kt), n: 2 * kt})
			}
		}
	}
	return nil
}

// directReads says whether to read an experts file of n bytes with O_DIRECT: when it is half the
// memory or more, the page cache cannot keep it beside the places, and a miss's page faults read
// the disk at about 1.5 GB/s (gpt-oss-120b: 9.4 ms an expert) where one read straight into the
// bank takes 6 GB/s. HWNPU_MOE_DEBUG=direct=1 or 0 decides.
func directReads(n int64) bool {
	if v, ok := experts.DebugOpt("direct"); ok {
		return v != "0"
	}
	phys := mem.Physical()
	return phys > 0 && 2*n >= phys
}

var directOnce sync.Once

// staging holds the buffers direct reads land in: O_DIRECT does not read into the bank's
// mapping (the driver's pages: EFAULT), so a part goes through one and is copied in.
var staging sync.Pool

func readStaged(f *os.File, dst []byte, off int64) bool {
	b, _ := staging.Get().([]byte)
	if cap(b) < len(dst) {
		b = make([]byte, len(dst))
	}
	b = b[:len(dst)]
	ok := readDirect(f, b, off)
	if ok {
		copy(dst, b)
	}
	staging.Put(b)
	return ok
}

func openDirect(path string) (*os.File, error) {
	fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_DIRECT|syscall.O_CLOEXEC, 0)
	if err != nil {
		return nil, err
	}
	return os.NewFile(uintptr(fd), path), nil
}

// readDirect reads len(dst) bytes at off of f into dst, false where O_DIRECT will not (a length
// or an offset off the disk's blocks, a buffer it cannot take): the caller copies instead.
func readDirect(f *os.File, dst []byte, off int64) bool {
	for len(dst) > 0 {
		n, err := syscall.Pread(int(f.Fd()), dst, off)
		if err != nil || n <= 0 {
			directOnce.Do(func() {
				fmt.Fprintf(os.Stderr, "experts: a direct read of %d bytes at %d: %v (copied instead)\n", len(dst), off, err)
			})
			return false
		}
		dst, off = dst[n:], off+int64(n)
	}
	return true
}
