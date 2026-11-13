package iron

import (
	"context"
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"
	"time"
)

// HostStep is a host step of a version that hwnpu runs itself, before command Command
// (tools/amd/iron_export_lm.py's host_steps): a prompt chunk's experts of Qwen3.6, routed on
// the host (tools/amd/iron_moep.py, iron_qwen35.MoEHost). Kind is moe_route (reads the
// router's logits and the normed rows; writes the experts' row counts and x's stream) or
// moe_combine (reads the residual, the experts' outputs and the shared expert's; writes the
// layer's output). gpt-oss's (tools/amd/iron_gptoss.py) are moe_route_mx4, the counts and x
// in iron_gptoss_moe's layout (moeHostRouteMX4), and moe_combine without a shared expert (two
// reads).
type HostStep struct {
	Command int      `json:"command"`
	Kind    string   `json:"kind"`
	Layer   int      `json:"layer"`
	Experts int      `json:"experts"`
	Top     int      `json:"top"`
	Cap     int      `json:"cap"`
	Tokens  int      `json:"tokens"`
	// moe_route_mx4: the in-features of x's stream and of its k-blocks (iron_gptoss_moe's
	// HIDDEN, KT)
	InFeatures int      `json:"in_features,omitempty"`
	KT         int      `json:"kt,omitempty"`
	// moe_route_mx4 by slots (iron_gptoss_moe's SLOTS): a pass's Slots slots of up to Cap rows,
	// slot e < Experts expert e's first Cap rows, the others an expert's further rows, slot i's
	// expert the graph's value SlotValues[i] ("" for a fixed slot); none, a slot an expert
	Slots      int      `json:"slots,omitempty"`
	SlotValues []string `json:"slot_values,omitempty"`
	Reads      []Region `json:"reads"`
	Writes  []Region `json:"writes"`
}

// Region is a buffer's place in an argument, and the tensor it holds.
type Region struct {
	Name     string `json:"name"`
	Argument string `json:"argument"`
	Offset   int    `json:"offset"`
	Bytes    int    `json:"bytes"`
	Shape    []int  `json:"shape"`
	DType    string `json:"dtype"`
}

// moeCounts is the counts' words (iron_moep.COUNTS: each expert's rows, then the padding
// rows' at moePad); kt the in-features of a k-block.
const (
	moeCounts = 512
	moePad    = 256
	moeKT     = 256
)

// moeRoute is a layer's routing of a prompt chunk: each token's experts' weights, the shared
// expert's gate, and for each choice the pass and the row of that pass's output holding it.
// The passes split the chunk so that no expert has more than Cap rows in one (the cores keep
// an expert's rows' accumulators); each pass's output is copied out after its run.
type moeRoute struct {
	w, shared []float32 // (n, top) and (n)
	pass, row []int32   // (n, top)
	passes    []moePass
	outs      [][]byte // each pass's experts' output (8 C rows of the hidden size, bf16)
	// hn is the chunk's rows of the experts' input, as route read it: the command's shared
	// expert reads it again each pass, and the arena lets the experts' output (MoEGroupDown's y)
	// lie over it, so a pass after the first writes it back first
	hn []byte
}

type moePass struct {
	counts []byte // moeCounts int32
	xs     []byte // MoEGroupGateUp's x (bf16) of the pass's rows; the padding rows' after them are not written
	rows   int    // the pass's rows (top a token): the experts' output past them is zero, not read
	// outRows: the rows of the experts' output read after the pass (rows, or every expert's Cap
	// rows where the output is by expert, moe_route_mx4)
	outRows int
	// values: the graph's values the pass sets (moe_route_mx4 by slots: each slot's expert)
	values map[string]int64
	// moe_route_mx4: x written as row copies into its region (xs nil: the slots' rows past their
	// counts are not read), and only each slot's rows of the output read back
	copies   []rowCopy
	slotRows []int
	cap      int
}

// rowCopy is one k-block of one choice's row of x: bytes src.. of the chunk's rows to dst.. of
// x's stream, n bytes.
type rowCopy struct{ dst, src, n int }

func (m *Model) region(v *version, r Region) ([]byte, error) {
	buf := m.buffer(v, r.Argument)
	if buf == nil {
		return nil, fmt.Errorf("iron: %s: no argument %s", r.Name, r.Argument)
	}
	if r.Offset+r.Bytes > buf.Size() {
		return nil, fmt.Errorf("iron: %s past its argument %s", r.Name, r.Argument)
	}
	return buf.Bytes()[r.Offset : r.Offset+r.Bytes], nil
}

// readRegion returns a copy of r's first n bytes once the device's writes are visible: the
// buffers' memory is slow to read a few bytes at a time (a 32-token chunk's top-8 took 1.1 ms
// longer reading the logits in place).
func (m *Model) readRegion(v *version, r Region, n int) ([]byte, error) {
	b, err := m.viewRegion(v, r, n)
	if err != nil {
		return nil, err
	}
	return append([]byte(nil), b...), nil
}

// viewRegion is readRegion's bytes in the buffer itself, for a caller writing them back.
func (m *Model) viewRegion(v *version, r Region, n int) ([]byte, error) {
	b, err := m.region(v, r)
	if err != nil {
		return nil, err
	}
	if n > len(b) {
		return nil, fmt.Errorf("iron: %d bytes of %s's %d", n, r.Name, len(b))
	}
	if err := m.buffer(v, r.Argument).SyncFromDevice(r.Offset, n); err != nil {
		return nil, err
	}
	return b[:n], nil
}

func (m *Model) writeRegion(v *version, r Region, data []byte) error {
	b, err := m.region(v, r)
	if err != nil {
		return err
	}
	if len(data) > len(b) {
		return fmt.Errorf("iron: %d bytes into %s's %d", len(data), r.Name, len(b))
	}
	copy(b, data)
	return m.buffer(v, r.Argument).SyncToDevice(r.Offset, len(data))
}

func bf16At(b []byte, i int) float32 {
	return math.Float32frombits((uint32(b[2*i]) | uint32(b[2*i+1])<<8) << 16)
}

func putBF16(b []byte, i int, x float32) {
	h := bf16(x)
	b[2*i], b[2*i+1] = byte(h), byte(h>>8)
}

func roundBF16(x float32) float32 { return math.Float32frombits(uint32(bf16(x)) << 16) }

// topK puts the indices of vals' len(best) largest into best, largest first, ties to the lower
// index (a stable sort's first len(best)).
func topK(best []int, vals []float32) {
	n := 0
	for i, x := range vals {
		if n == len(best) && x <= vals[best[n-1]] {
			continue
		}
		j := min(n, len(best)-1)
		for j > 0 && x > vals[best[j-1]] {
			best[j] = best[j-1]
			j--
		}
		best[j] = i
		n = min(n+1, len(best))
	}
}

// moeHostRoute runs a moe_route step over the chunk's first n rows, as iron_moep's route_chunk
// and plan (iron_qwen35.MoEHost.moe_route) do, split into passes of at most Cap rows an expert.
func (m *Model) moeHostRoute(v *version, h *HostStep, n int) (*moeRoute, error) {
	if len(h.Reads) != 2 || len(h.Writes) != 2 {
		return nil, fmt.Errorf("iron: moe_route of layer %d: %d reads and %d writes", h.Layer, len(h.Reads), len(h.Writes))
	}
	lg, hn := h.Reads[0], h.Reads[1]
	if lg.DType != "bfloat16" || hn.DType != "bfloat16" || len(lg.Shape) != 2 || len(hn.Shape) != 2 {
		return nil, fmt.Errorf("iron: moe_route of layer %d reads %v %s and %v %s", h.Layer, lg.Shape, lg.DType, hn.Shape, hn.DType)
	}
	C, L, K, E, top := h.Tokens, lg.Shape[1], hn.Shape[1], h.Experts, h.Top
	if n > C || E > moePad || E >= L || K%moeKT != 0 {
		return nil, fmt.Errorf("iron: moe_route of layer %d: %d rows of %d, %d experts, %d logits", h.Layer, n, C, E, L)
	}
	logits, err := m.readRegion(v, lg, 2*n*L)
	if err != nil {
		return nil, err
	}
	rows, err := m.readRegion(v, hn, 2*n*K)
	if err != nil {
		return nil, err
	}
	rt := &moeRoute{w: make([]float32, n*top), shared: make([]float32, n),
		pass: make([]int32, n*top), row: make([]int32, n*top), hn: rows}
	ids := make([]int32, n*top)
	order := make([]int, top)
	vals := make([]float32, L)
	for t := 0; t < n; t++ {
		for i := range vals {
			vals[i] = bf16At(logits, t*L+i)
		}
		topK(order[:top], vals[:E])
		var sum float64
		e := make([]float64, top)
		for k := 0; k < top; k++ {
			ids[t*top+k] = int32(order[k])
			e[k] = math.Exp(float64(vals[order[k]] - vals[order[0]]))
			sum += e[k]
		}
		for k := 0; k < top; k++ {
			rt.w[t*top+k] = roundBF16(float32(e[k] / sum))
		}
		rt.shared[t] = roundBF16(float32(1 / (1 + math.Exp(-float64(vals[E])))))
	}
	// Passes: tokens in order, a new pass where a token would give an expert more than Cap rows.
	var passTokens [][]int
	cnt := make([]int, E)
	for t := 0; t < n; t++ {
		over := len(passTokens) == 0
		for k := 0; k < top && !over; k++ {
			over = cnt[ids[t*top+k]] >= h.Cap
		}
		if over {
			passTokens = append(passTokens, nil)
			clear(cnt)
		}
		p := len(passTokens) - 1
		passTokens[p] = append(passTokens[p], t)
		for k := 0; k < top; k++ {
			cnt[ids[t*top+k]]++
		}
	}
	A := top * C
	for p, tokens := range passTokens {
		// rows by expert: the pass's tokens' choices, then the padding rows filling the A rows
		// (x zero, output zero and unread; the cores read their x past without computing)
		groups := make([][]int, moePad+1) // token*top+k, or -1 for a padding row
		for _, t := range tokens {
			for k := 0; k < top; k++ {
				e := ids[t*top+k]
				groups[e] = append(groups[e], t*top+k)
			}
		}
		for i := top * len(tokens); i < A; i++ {
			groups[moePad] = append(groups[moePad], -1)
		}
		// the cores read the padding rows' x past, so only the real rows' (the first ones, the
		// padding group last) are written: a 32-token chunk's 1/8 of the A rows'
		real := top * len(tokens)
		ps := moePass{counts: make([]byte, 4*moeCounts), xs: make([]byte, 2*2*real*K), rows: real, outRows: real}
		at, x := 0, 0
		for e, list := range groups {
			c := len(list)
			ps.counts[4*e], ps.counts[4*e+1], ps.counts[4*e+2], ps.counts[4*e+3] = byte(c), byte(c>>8), byte(c>>16), byte(c>>24)
			for r, choice := range list {
				if choice >= 0 {
					rt.pass[choice], rt.row[choice] = int32(p), int32(at+r)
				}
			}
			at += c
			if e == moePad {
				break // the padding rows: last, their x unwritten
			}
			for g := 0; g < 2; g++ { // the gate's pass over the k-blocks, then the up's
				for kb := 0; kb < K/moeKT; kb++ {
					for _, choice := range list {
						if choice >= 0 {
							t := choice / top
							copy(ps.xs[2*x:2*(x+moeKT)], rows[2*(t*K+kb*moeKT):2*(t*K+(kb+1)*moeKT)])
						} // a dummy's x stays zero
						x += moeKT
					}
				}
			}
		}
		rt.passes = append(rt.passes, ps)
	}
	return rt, nil
}

// moeHostCombine runs a moe_combine step: out = h + shared gate * the shared expert's + each
// choice's weight times its row of its pass's output, for the chunk's first n rows (zero past
// them), summed in float32 as iron_moep's combine.
func (m *Model) moeHostCombine(v *version, h *HostStep, rt *moeRoute, n int) error {
	if len(h.Reads) < 2 || len(h.Reads) > 3 || len(h.Writes) != 1 || rt == nil {
		return fmt.Errorf("iron: moe_combine of layer %d without its route", h.Layer)
	}
	hr, out := h.Reads[0], h.Writes[0]
	K, top := hr.Shape[1], h.Top
	res, err := m.readRegion(v, hr, 2*n*K)
	if err != nil {
		return err
	}
	resid := append([]byte(nil), res...)
	sh := make([]byte, 2*n*K) // no shared expert (gpt-oss): its gate is 0 too
	if len(h.Reads) == 3 {
		if sh, err = m.readRegion(v, h.Reads[2], 2*n*K); err != nil {
			return err
		}
	}
	o := make([]byte, out.Bytes)
	// the tokens split among the CPUs, each token's sums in the same order as one loop's
	workers := min(runtime.GOMAXPROCS(0), n)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			acc := make([]float32, K)
			for t := lo; t < hi; t++ {
				g := rt.shared[t]
				rr, sr := resid[2*t*K:2*(t+1)*K], sh[2*t*K:2*(t+1)*K]
				for i := range acc {
					acc[i] = bf16At(rr, i) + g*bf16At(sr, i)
				}
				for k := 0; k < top; k++ {
					c := t*top + k
					base := int(rt.row[c]) * K
					y, w := rt.outs[rt.pass[c]][2*base:2*(base+K)], rt.w[c]
					for i := range acc {
						acc[i] += w * bf16At(y, i)
					}
				}
				ot := o[2*t*K : 2*(t+1)*K]
				for i, x := range acc {
					putBF16(ot, i, x)
				}
			}
		}(w*n/workers, (w+1)*n/workers)
	}
	wg.Wait()
	return m.writeRegion(v, out, o)
}

// runMoECommand runs command k of v (an experts' command after a moe_route) once a pass: the
// pass's counts and x written first, its output (combine's second read) copied out after.
func (m *Model) runMoECommand(ctx context.Context, v *version, rt *moeRoute, route, combine *HostStep, run func() error, values map[string]int64) error {
	yr := combine.Reads[1]
	K := combine.Reads[0].Shape[1]
	for i, p := range rt.passes {
		for k, x := range p.values {
			values[k] = x
		}
		if i > 0 { // the last pass's y lies over the shared expert's input
			if err := m.writeRegion(v, route.Reads[1], rt.hn); err != nil {
				return err
			}
		}
		if err := m.writeRegion(v, route.Writes[0], p.counts); err != nil {
			return err
		}
		t0 := time.Now()
		if p.copies != nil {
			if err := m.writeCopies(v, route.Writes[1], rt.hn, p.copies); err != nil {
				return err
			}
		} else if err := m.writeRegion(v, route.Writes[1], p.xs); err != nil {
			return err
		}
		t1 := time.Now()
		if err := run(); err != nil {
			return err
		}
		t2 := time.Now()
		if p.slotRows != nil {
			y, err := m.readSlots(v, yr, p.slotRows, p.cap, K)
			if err != nil {
				return err
			}
			rt.outs = append(rt.outs, y)
			if os.Getenv("HWNPU_IRON_TIMING") != "" {
				fmt.Fprintf(os.Stderr, "moe pass: x %.1f ms, run %.1f ms, y %.1f ms\n", t1.Sub(t0).Seconds()*1000,
					t2.Sub(t1).Seconds()*1000, time.Since(t2).Seconds()*1000)
			}
			continue
		}
		y, err := m.readRegion(v, yr, min(yr.Bytes, 2*p.outRows*K)) // the real rows; the padding rows' are zero
		if err != nil {
			return err
		}
		rt.outs = append(rt.outs, append([]byte(nil), y...))
	}
	return nil
}

// writeCopies writes x's rows into r's region in place (the other rows left as they are: past a
// slot's count, the kernels do not use them), then the region's touched span to the device.
func (m *Model) writeCopies(v *version, r Region, rows []byte, copies []rowCopy) error {
	b, err := m.region(v, r)
	if err != nil {
		return err
	}
	lo, hi := len(b), 0
	for _, c := range copies {
		if c.dst+c.n > len(b) || c.src+c.n > len(rows) {
			return fmt.Errorf("iron: a row of x past %s", r.Name)
		}
		copy(b[c.dst:c.dst+c.n], rows[c.src:c.src+c.n])
		lo, hi = min(lo, c.dst), max(hi, c.dst+c.n)
	}
	if hi <= lo {
		return nil
	}
	return m.buffer(v, r.Argument).SyncToDevice(r.Offset+lo, hi-lo)
}

// readSlots copies out each slot's rows of the experts' output r (slots of cap rows of k bf16),
// the rows past a slot's count left zero.
func (m *Model) readSlots(v *version, r Region, rows []int, cap, k int) ([]byte, error) {
	out := make([]byte, 2*len(rows)*cap*k)
	for s, c := range rows {
		if c == 0 {
			continue
		}
		at := 2 * s * cap * k
		if at+2*c*k > r.Bytes {
			return nil, fmt.Errorf("iron: slot %d past %s", s, r.Name)
		}
		sub := r
		sub.Offset, sub.Bytes = r.Offset+at, 2*c*k
		b, err := m.viewRegion(v, sub, 2*c*k)
		if err != nil {
			return nil, err
		}
		copy(out[at:], b)
	}
	return out, nil
}

// moeHostRouteMX4 runs a moe_route_mx4 step over the chunk's first n rows: each token's top of
// the experts' logits and their softmax, split into passes of at most Cap rows an expert, each
// pass's counts (the x channel's first object: an int32 an expert) and x in iron_gptoss_moe's
// stream_x layout ((experts, in / kt, Cap, kt) bf16); a choice's output row is expert * Cap +
// its row (MX4GroupDown's output by expert).
func (m *Model) moeHostRouteMX4(v *version, h *HostStep, n int) (*moeRoute, error) {
	if len(h.Reads) != 2 || len(h.Writes) != 2 || h.KT <= 0 || h.InFeatures%h.KT != 0 {
		return nil, fmt.Errorf("iron: moe_route_mx4 of layer %d: %d reads, %d writes, kt %d", h.Layer, len(h.Reads), len(h.Writes), h.KT)
	}
	lg, hn := h.Reads[0], h.Reads[1]
	C, L, K, E, top, cap := h.Tokens, lg.Shape[1], hn.Shape[1], h.Experts, h.Top, h.Cap
	in, kt := h.InFeatures, h.KT
	if n > C || E > L || in > K {
		return nil, fmt.Errorf("iron: moe_route_mx4 of layer %d: %d rows of %d, %d experts of %d logits, %d of %d in-features", h.Layer, n, C, E, L, in, K)
	}
	logits, err := m.readRegion(v, lg, 2*n*L)
	if err != nil {
		return nil, err
	}
	rows, err := m.readRegion(v, hn, 2*n*K)
	if err != nil {
		return nil, err
	}
	rt := &moeRoute{w: make([]float32, n*top), shared: make([]float32, n),
		pass: make([]int32, n*top), row: make([]int32, n*top), hn: rows}
	ids := make([]int32, n*top)
	order := make([]int, top)
	vals := make([]float32, L)
	e := make([]float64, top)
	for t := 0; t < n; t++ {
		for i := range vals {
			vals[i] = bf16At(logits, t*L+i)
		}
		topK(order, vals[:E])
		var sum float64
		for k := 0; k < top; k++ {
			ids[t*top+k] = int32(order[k])
			e[k] = math.Exp(float64(vals[order[k]] - vals[order[0]]))
			sum += e[k]
		}
		for k := 0; k < top; k++ {
			rt.w[t*top+k] = roundBF16(float32(e[k] / sum))
		}
	}
	// Passes: each token into the first pass where its experts have room (any pass: combine
	// finds each choice's row), a new pass where none has. Every pass streams the weights again,
	// so fewer is faster: in order, a 256-token chunk took 3 passes at Cap 64 an expert. By slots,
	// an expert's rows take ceil(rows / Cap) slots of the pass's Slots.
	S := E
	if h.Slots > 0 {
		S = h.Slots
		if len(h.SlotValues) != S {
			return nil, fmt.Errorf("iron: moe_route_mx4 of layer %d: %d slots, %d slot values", h.Layer, S, len(h.SlotValues))
		}
	}
	if h.Writes[0].Bytes < 4*S || h.Writes[1].Bytes < 2*S*in*cap {
		return nil, fmt.Errorf("iron: moe_route_mx4 of layer %d: %d slots of %d rows past its counts and x", h.Layer, S, cap)
	}
	if h.Slots > 0 && S < E {
		return nil, fmt.Errorf("iron: moe_route_mx4 of layer %d: %d slots for %d experts", h.Layer, S, E)
	}
	need := func(cnt []int) int { // the slots cnt takes: an expert's fixed one, and one each further Cap rows
		n := E
		for _, c := range cnt {
			if c > cap {
				n += (c - 1) / cap
			}
		}
		return n
	}
	var passTokens [][]int
	var passCnt [][]int
	trial := make([]int, E)
	for t := 0; t < n; t++ {
		p := -1
		for q := range passTokens {
			copy(trial, passCnt[q])
			fits := true
			for k := 0; k < top; k++ {
				trial[ids[t*top+k]]++
			}
			if h.Slots == 0 {
				for k := 0; k < top && fits; k++ {
					fits = trial[ids[t*top+k]] <= cap
				}
			} else {
				fits = need(trial) <= S
			}
			if fits {
				p = q
				break
			}
		}
		if p < 0 {
			passTokens, passCnt = append(passTokens, nil), append(passCnt, make([]int, E))
			p = len(passTokens) - 1
		}
		passTokens[p] = append(passTokens[p], t)
		for k := 0; k < top; k++ {
			passCnt[p][ids[t*top+k]]++
		}
	}
	kbs := in / kt
	for p, tokens := range passTokens {
		ps := moePass{counts: make([]byte, h.Writes[0].Bytes), rows: top * len(tokens), outRows: S * cap, cap: cap}
		// expert ex's first Cap rows in slot ex, its further ones from slot over[ex] on
		over := make([]int, E)
		slotExpert, slotRows := make([]int, S), make([]int, S)
		next := E
		for ex, c := range passCnt[p] {
			slotExpert[ex], slotRows[ex] = ex, min(cap, c)
			over[ex] = next
			for r := cap; r < c; r += cap {
				slotExpert[next], slotRows[next] = ex, min(cap, c-r)
				next++
			}
		}
		if h.Slots > 0 {
			ps.values = make(map[string]int64, S)
			for i, name := range h.SlotValues {
				if name != "" {
					ps.values[name] = int64(slotExpert[i])
				}
			}
		}
		at := make([]int, E)
		for _, t := range tokens {
			for k := 0; k < top; k++ {
				c := t*top + k
				ex := int(ids[c])
				sl, r := ex, at[ex]%cap
				if q := at[ex] / cap; q > 0 {
					sl = over[ex] + q - 1
				}
				at[ex]++
				rt.pass[c], rt.row[c] = int32(p), int32(sl*cap+r)
				for kb := 0; kb < kbs; kb++ {
					dst := ((sl*kbs+kb)*cap + r) * kt
					ps.copies = append(ps.copies, rowCopy{dst: 2 * dst, src: 2 * (t*K + kb*kt), n: 2 * kt})
				}
			}
		}
		ps.slotRows = slotRows
		for sl, c := range slotRows {
			ps.counts[4*sl], ps.counts[4*sl+1], ps.counts[4*sl+2], ps.counts[4*sl+3] = byte(c), byte(c>>8), byte(c>>16), byte(c>>24)
		}
		rt.passes = append(rt.passes, ps)
	}
	return rt, nil
}
