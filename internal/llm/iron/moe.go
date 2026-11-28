package iron

import (
	"context"
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"
	"time"

	devenv "github.com/shibukawa/hwnpu/internal/dev"
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
	Command int    `json:"command"`
	Kind    string `json:"kind"`
	Layer   int    `json:"layer"`
	Experts int    `json:"experts"`
	Top     int    `json:"top"`
	Cap     int    `json:"cap"`
	Tokens  int    `json:"tokens"`
	// moe_route_mx4: the in-features of x's stream and of its k-blocks (iron_gptoss_moe's
	// HIDDEN, KT)
	InFeatures int `json:"in_features,omitempty"`
	KT         int `json:"kt,omitempty"`
	// moe_route_mx4 by slots (iron_gptoss_moe's SLOTS): a pass's Slots slots of up to Cap rows,
	// slot e < Experts expert e's first Cap rows, the others an expert's further rows, slot i's
	// expert the graph's value SlotValues[i] ("" for a fixed slot); none, a slot an expert
	Slots      int      `json:"slots,omitempty"`
	SlotValues []string `json:"slot_values,omitempty"`
	// Places, where below Experts, is the places of the layer's tables (Manifest.ExpertsPool):
	// fixed slot i is place i (moe_route: the counts and x by place), and the experts not in a
	// place are copied in between passes
	Places int `json:"places,omitempty"`
	// LFM2-MoE's prompt chunk (tools/amd/iron_lfm2moe.py: moe_rows_plan, moe_gather and
	// moe_gather_combine): the experts' scores (sigmoid: each logit's sigmoid, the top Top of
	// score + Bias chosen and weighted by score over the chosen's sum, times Scale)
	Score string    `json:"score,omitempty"`
	Bias  []float32 `json:"bias,omitempty"`
	Scale float32   `json:"scale,omitempty"`
	// moe_rows_plan (LFM2-MoE's, iron_lfm2moe's MoERowsHost), ahead of the command that runs a
	// prompt chunk's experts a row at a time: a chunk of up to RowMax rows takes that form
	// (moeRowsPlan), the command run once for every RowTokens rows with the rows' indices in the
	// values RowValues and their experts in ExpertValues; the weights' rows are of Stride
	// elements. A longer chunk leaves the command out and runs the one after moe_gather.
	RowMax       int        `json:"row_max,omitempty"`
	RowTokens    int        `json:"row_tokens,omitempty"`
	Stride       int        `json:"stride,omitempty"`
	RowValues    []string   `json:"row_values,omitempty"`
	ExpertValues [][]string `json:"expert_values,omitempty"`
	// moe_gather (iron_lfm2moe's MoEGatherHost), ahead of the command that runs Slots slots of
	// SlotRows rows, a slot the rows of one expert (the graph's value SlotValues[i]): each
	// expert's rows of the chunk are gathered into slots (moeGatherPlan) and the command run once
	// for every Slots slots (runMoEGather); left out with its command where the rows form ran.
	// moe_gather_combine then adds each row's experts' rows to the residual.
	SlotRows int      `json:"slot_rows,omitempty"`
	Reads    []Region `json:"reads"`
	Writes   []Region `json:"writes"`
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
	// loads: the experts (expert, place) of a layer with fewer places than experts copied in
	// before the pass (pool.go)
	loads [][2]int
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

// expertRouter chooses a row's experts from its row of router logits: route fills chosen's
// experts (the largest first) and their weights, rounded to bf16. softmaxRouter is Qwen3.6's and
// gpt-oss's (iron_moep's route), sigmoidRouter LFM2-MoE's (kernels/moe_route_sigmoid.cc).
type expertRouter interface {
	route(logits []byte, t, L int)
	chosen() (best []int, w []float32)
}

// routed is a router's choice of a row.
type routed struct {
	best []int     // the chosen experts, the largest first
	w    []float32 // their weights
}

func (r *routed) chosen() ([]int, []float32) { return r.best, r.w }

// newRouter is the router of a host step's experts (its Score: "" softmax, or sigmoid).
func newRouter(h *HostStep) (expertRouter, error) {
	switch h.Score {
	case "":
		return newSoftmaxRouter(h.Experts, h.Top), nil
	case "sigmoid":
		if len(h.Bias) != h.Experts {
			return nil, fmt.Errorf("iron: layer %d: %d biases for %d experts", h.Layer, len(h.Bias), h.Experts)
		}
		return newSigmoidRouter(h), nil
	}
	return nil, fmt.Errorf("iron: layer %d: experts scored by %q", h.Layer, h.Score)
}

// softmaxRouter takes the Top largest of the first E logits (ties to the lower index) and
// weights them by their softmax over the chosen alone. vals holds the row's logits after route
// (Qwen3.6's shared expert's gate follows the experts').
type softmaxRouter struct {
	routed
	E    int
	vals []float32
	e    []float64
}

func newSoftmaxRouter(E, top int) *softmaxRouter {
	return &softmaxRouter{routed: routed{best: make([]int, top), w: make([]float32, top)}, E: E, e: make([]float64, top)}
}

func (r *softmaxRouter) route(logits []byte, t, L int) {
	if len(r.vals) != L {
		r.vals = make([]float32, L)
	}
	for i := range r.vals {
		r.vals[i] = bf16At(logits, t*L+i)
	}
	topK(r.best, r.vals[:r.E])
	var sum float64
	for k, o := range r.best {
		r.e[k] = math.Exp(float64(r.vals[o] - r.vals[r.best[0]]))
		sum += r.e[k]
	}
	for k := range r.best {
		r.w[k] = roundBF16(float32(r.e[k] / sum))
	}
}

// routeRows routes the first n rows of logits (rows of L bf16 values): the chosen experts and
// their weights, Top a row; each, where not nil, is called after row t is routed.
func routeRows(r expertRouter, logits []byte, n, L int, each func(t int)) (ids []int32, w []float32) {
	best, rw := r.chosen()
	top := len(best)
	ids, w = make([]int32, n*top), make([]float32, n*top)
	for t := 0; t < n; t++ {
		r.route(logits, t, L)
		for k, e := range best {
			ids[t*top+k], w[t*top+k] = int32(e), rw[k]
		}
		if each != nil {
			each(t)
		}
	}
	return ids, w
}

// byExpert is the choices (row × top + rank, from first) of ids by expert, each expert's in
// the rows' order; size is the experts' count (or the counts' slots, moePad).
func byExpert(ids []int32, size, first int) [][]int {
	by := make([][]int, size)
	for c, e := range ids {
		by[e] = append(by[e], first+c)
	}
	return by
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
	rt := &moeRoute{shared: make([]float32, n), pass: make([]int32, n*top), row: make([]int32, n*top), hn: rows}
	r := newSoftmaxRouter(E, top)
	ids, w := routeRows(r, logits, n, L, func(t int) {
		rt.shared[t] = roundBF16(float32(1 / (1 + math.Exp(-float64(r.vals[E])))))
	})
	rt.w = w
	m.countPrompt(h.Layer, ids)
	if h.Places > 0 && h.Places < E {
		return rt, m.poolRowPasses(h, rt, ids, rows, n, K)
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
	for p, tokens := range passTokens {
		// rows by expert: the pass's tokens' choices (a pass's tokens are consecutive)
		lo, hi := tokens[0]*top, (tokens[len(tokens)-1]+1)*top
		rt.passes = append(rt.passes, rowsPass(rt, p, byExpert(ids[lo:hi], moePad, lo), rows, top*C, K, top))
	}
	return rt, nil
}

// rowsPass is pass p of a moe_route step whose choices (token*top+k) are by the table's expert
// (or place) in groups: the counts, x's stream, and each choice's row of the output (rt's), the
// padding rows filling the a rows after them (x zero, output zero and unread; the cores read
// their x past without computing).
func rowsPass(rt *moeRoute, p int, groups [][]int, rows []byte, a, K, top int) moePass {
	real := 0
	for _, list := range groups {
		real += len(list)
	}
	// the cores read the padding rows' x past, so only the real rows' (the first ones, the
	// padding group last) are written: a 32-token chunk's 1/8 of the A rows'
	ps := moePass{counts: make([]byte, 4*moeCounts), xs: make([]byte, 2*2*real*K), rows: real, outRows: real}
	putCount := func(e, c int) {
		ps.counts[4*e], ps.counts[4*e+1], ps.counts[4*e+2], ps.counts[4*e+3] = byte(c), byte(c>>8), byte(c>>16), byte(c>>24)
	}
	at, x := 0, 0
	for e, list := range groups {
		putCount(e, len(list))
		for r, choice := range list {
			rt.pass[choice], rt.row[choice] = int32(p), int32(at+r)
		}
		at += len(list)
		for g := 0; g < 2; g++ { // the gate's pass over the k-blocks, then the up's
			for kb := 0; kb < K/moeKT; kb++ {
				for _, choice := range list {
					t := choice / top
					copy(ps.xs[2*x:2*(x+moeKT)], rows[2*(t*K+kb*moeKT):2*(t*K+(kb+1)*moeKT)])
					x += moeKT
				}
			}
		}
	}
	putCount(moePad, a-real) // the padding rows: last, their x unwritten
	return ps
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
	combineRows(o, resid, sh, rt.shared, rt.w, n, K, top, func(c int) []byte {
		base := int(rt.row[c]) * K
		return rt.outs[rt.pass[c]][2*base : 2*(base+K)]
	})
	return m.writeRegion(v, out, o)
}

// combineRows writes into out, for each of the first n rows of K bf16 values, the residual's
// row plus (where shared is not nil) gate × the shared expert's row plus each choice's weight ×
// its expert's output row (row(c) for choice c = row × top + rank, K bf16 values), summed in
// float32 in that order and rounded to bf16 once (iron_moep's and iron_lfm2moe's combine). The
// rows split among the CPUs, each row's sums in the same order as one loop's.
func combineRows(out, resid, shared []byte, gate, w []float32, n, K, top int, row func(c int) []byte) {
	workers := min(runtime.GOMAXPROCS(0), max(n, 1))
	var wg sync.WaitGroup
	for j := 0; j < workers; j++ {
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			acc := make([]float32, K)
			for t := lo; t < hi; t++ {
				rr := resid[2*t*K : 2*(t+1)*K]
				if shared != nil {
					g, sr := gate[t], shared[2*t*K:2*(t+1)*K]
					for i := range acc {
						acc[i] = bf16At(rr, i) + g*bf16At(sr, i)
					}
				} else {
					for i := range acc {
						acc[i] = bf16At(rr, i)
					}
				}
				for k := 0; k < top; k++ {
					c := t*top + k
					y, wk := row(c), w[c]
					for i := range acc {
						acc[i] += wk * bf16At(y, i)
					}
				}
				ot := out[2*t*K : 2*(t+1)*K]
				for i, x := range acc {
					putBF16(ot, i, x)
				}
			}
		}(j*n/workers, (j+1)*n/workers)
	}
	wg.Wait()
}

// sigmoidRouter routes rows as LFM2-MoE does (tools/amd/kernels/moe_route_sigmoid.cc): each
// expert's score is the sigmoid of its logit, the Top largest of score + bias are chosen (ties to
// the lower index), and their weights are the scores over the chosen's sum, times the scale,
// rounded to bf16.
type sigmoidRouter struct {
	routed
	bias    []float32
	scale   float32
	s, pick []float32
}

func newSigmoidRouter(h *HostStep) *sigmoidRouter {
	scale := h.Scale
	if scale == 0 {
		scale = 1
	}
	return &sigmoidRouter{routed: routed{best: make([]int, h.Top), w: make([]float32, h.Top)}, bias: h.Bias, scale: scale,
		s: make([]float32, h.Experts), pick: make([]float32, h.Experts)}
}

// route fills best and w for row t of logits (rows of L bf16 values).
func (r *sigmoidRouter) route(logits []byte, t, L int) {
	for i := range r.s {
		r.s[i] = float32(1 / (1 + math.Exp(-float64(bf16At(logits, t*L+i)))))
		r.pick[i] = r.s[i] + r.bias[i]
	}
	topK(r.best, r.pick)
	var sum float32
	for _, e := range r.best {
		sum += r.s[e]
	}
	norm := r.scale / (sum + 1e-6)
	for k, e := range r.best {
		r.w[k] = roundBF16(r.s[e] * norm)
	}
}

// moeRows is the routing of a short prompt chunk whose experts run a row at a time
// (moe_rows_plan): each row's experts, Top a row.
type moeRows struct {
	ids []int32
	n   int
}

// moeRowsPlan runs a moe_rows_plan step for a chunk of n rows: nil for a chunk of more than
// RowMax rows (the commands of every expert run); else the rows' routing, their weights written
// into the weights' buffer (the first Top elements of a row of Stride).
func (m *Model) moeRowsPlan(v *version, h *HostStep, n int) (*moeRows, error) {
	if len(h.Reads) != 1 || len(h.Writes) != 1 || h.Score != "sigmoid" || h.RowTokens < 1 || len(h.RowValues) != h.RowTokens ||
		len(h.ExpertValues) != h.RowTokens || h.Stride < h.Top || h.Top < 1 {
		return nil, fmt.Errorf("iron: moe_rows_plan of layer %d: %d reads, %d writes, score %q, %d rows a pass", h.Layer, len(h.Reads), len(h.Writes), h.Score, h.RowTokens)
	}
	lg, wr := h.Reads[0], h.Writes[0]
	if len(lg.Shape) != 2 || h.Experts > lg.Shape[1] || len(h.Bias) != h.Experts || 2*h.RowMax*h.Stride > wr.Bytes {
		return nil, fmt.Errorf("iron: moe_rows_plan of layer %d: logits %v, %d experts, weights of %d bytes for %d rows", h.Layer, lg.Shape, h.Experts, wr.Bytes, h.RowMax)
	}
	for _, names := range h.ExpertValues {
		if len(names) != h.Top {
			return nil, fmt.Errorf("iron: moe_rows_plan of layer %d: %d experts' values a row, top %d", h.Layer, len(names), h.Top)
		}
	}
	if n > h.RowMax || n > h.Tokens {
		return nil, nil
	}
	L, top := lg.Shape[1], h.Top
	logits, err := m.readRegion(v, lg, 2*n*L)
	if err != nil {
		return nil, err
	}
	r, err := newRouter(h)
	if err != nil {
		return nil, err
	}
	ids, rw := routeRows(r, logits, n, L, nil)
	rows := &moeRows{ids: ids, n: n}
	w := make([]byte, 2*n*h.Stride)
	for c, x := range rw {
		putBF16(w, c/top*h.Stride+c%top, x)
	}
	m.countPrompt(h.Layer, rows.ids)
	return rows, m.writeRegion(v, wr, w)
}

// moeRowsOut ends a layer whose experts ran a row at a time: the n rows the rows form wrote
// (the last read of the layer's moe_gather_combine step) go into the layer's output, which the
// host writes (a step may not write a buffer the host writes too). The output's rows past n
// keep what they held: no row of the chunk reads them.
func (m *Model) moeRowsOut(v *version, h *HostStep, n int) error {
	if len(h.Reads) != 5 || len(h.Writes) != 1 || len(h.Reads[1].Shape) != 2 {
		return fmt.Errorf("iron: moe_gather_combine of layer %d has no rows form's output", h.Layer)
	}
	b, err := m.readRegion(v, h.Reads[4], 2*n*h.Reads[1].Shape[1])
	if err != nil {
		return err
	}
	return m.writeRegion(v, h.Writes[0], b)
}

// runMoERows runs command k of v, a short prompt chunk's experts a row at a time, once for every
// RowTokens rows: the rows' indices and their experts in the values the step names (a last pass
// short of rows runs its last row again, which writes the same output).
func (m *Model) runMoERows(ctx context.Context, v *version, k int, h *HostStep, rows *moeRows, values map[string]int64) error {
	run, err := m.promptRun(v, k)
	if err != nil {
		return err
	}
	defer run.Close()
	ids := make([]int32, 0, h.RowTokens*h.Top)
	for first := 0; first < rows.n; first += h.RowTokens {
		ids = ids[:0]
		for j := 0; j < h.RowTokens; j++ {
			t := min(first+j, rows.n-1)
			values[h.RowValues[j]] = int64(t)
			ids = append(ids, rows.ids[t*h.Top:(t+1)*h.Top]...)
		}
		places, err := m.placePass(h.Layer, ids) // an expert's place: itself where every one has its own
		if err != nil {
			return err
		}
		for j := 0; j < h.RowTokens; j++ {
			for i, name := range h.ExpertValues[j] {
				values[name] = int64(places[j*h.Top+i])
			}
		}
		if err := writeWords(run.Scratchpad(), v.m.Kernels[k].Parameters, values); err != nil {
			return err
		}
		if err := run.Run(ctx); err != nil {
			return fmt.Errorf("iron: prompt command %d (rows from %d): %w", k, first, err)
		}
	}
	return nil
}

// moeGather is the routing of a prompt chunk whose experts' rows are gathered into slots
// (moe_gather): each row's experts and weights, the slots (an expert's rows of the chunk, up to
// SlotRows of them; an expert of more rows has further slots), and for each choice its row among
// the slots' rows in order (at). ys holds the slots' rows out, in that order.
type moeGather struct {
	n     int
	w     []float32 // (n, top)
	at    []int32   // (n, top)
	slots []gatherSlot
	x     []byte // the chunk's rows of the experts' input, as the plan read them
	ys    []byte
}

type gatherSlot struct {
	expert int
	rows   []int32 // the chunk's rows
	first  int     // the slot's first row among the slots' rows
}

// gatherPad is the rows a slot's rows are padded to with zero rows: the GEMM packs its rows in
// blocks, and what a slot held before must not share one with a row that is read.
const gatherPad = 64

// gatherSlots gives each expert the rows that chose it (ids: top experts a row), in slots of up
// to slotRows rows in order of the experts, and each choice's row among the slots' rows.
func gatherSlots(ids []int32, experts, top, slotRows int) ([]gatherSlot, []int32) {
	n := len(ids) / top
	by := byExpert(ids, experts, 0)
	var slots []gatherSlot
	at := make([]int32, n*top)
	first := 0
	for e, choices := range by {
		for len(choices) > 0 {
			c := min(len(choices), slotRows)
			s := gatherSlot{expert: e, rows: make([]int32, c), first: first}
			for r, choice := range choices[:c] {
				s.rows[r] = int32(choice / top)
				at[choice] = int32(first + r)
			}
			slots = append(slots, s)
			first += c
			choices = choices[c:]
		}
	}
	return slots, at
}

// moeGatherPlan runs a moe_gather step for a chunk of n rows: the rows' routing from the
// router's logits (its first read) and the slots of the rows of the experts' input (its second).
func (m *Model) moeGatherPlan(v *version, h *HostStep, n int) (*moeGather, error) {
	if len(h.Reads) != 2 || len(h.Writes) != 1 || h.Score != "sigmoid" || h.Slots < 1 || h.SlotRows < 1 ||
		len(h.SlotValues) != h.Slots || h.Top < 1 {
		return nil, fmt.Errorf("iron: moe_gather of layer %d: %d reads, %d writes, score %q, %d slots of %d rows", h.Layer, len(h.Reads), len(h.Writes), h.Score, h.Slots, h.SlotRows)
	}
	lg, xr, gx := h.Reads[0], h.Reads[1], h.Writes[0]
	if len(lg.Shape) != 2 || len(xr.Shape) != 2 || h.Experts > lg.Shape[1] || len(h.Bias) != h.Experts || n > h.Tokens ||
		n > xr.Shape[0] || gx.Bytes < 2*h.Slots*h.SlotRows*xr.Shape[1] {
		return nil, fmt.Errorf("iron: moe_gather of layer %d: logits %v, x %v, %d rows of %d, slots of %d bytes", h.Layer, lg.Shape, xr.Shape, n, h.Tokens, gx.Bytes)
	}
	L, K, top := lg.Shape[1], xr.Shape[1], h.Top
	logits, err := m.readRegion(v, lg, 2*n*L)
	if err != nil {
		return nil, err
	}
	g := &moeGather{n: n}
	if g.x, err = m.readRegion(v, xr, 2*n*K); err != nil {
		return nil, err
	}
	r, err := newRouter(h)
	if err != nil {
		return nil, err
	}
	var ids []int32
	ids, g.w = routeRows(r, logits, n, L, nil)
	m.countPrompt(h.Layer, ids)
	g.slots, g.at = gatherSlots(ids, h.Experts, top, h.SlotRows)
	g.ys = make([]byte, 2*n*top*K)
	return g, nil
}

// runMoEGather runs command k of v, the gathered form of a prompt chunk's experts, once for
// every Slots slots of g: the slots' rows written into the command's input (h's write; a slot's
// rows padded with zero rows to whole gatherPad), their experts in the values h names, and the
// slots' rows of the command's output (combine's fourth read) copied out after it. A last run
// short of slots gives the others the last slot's expert and no rows.
func (m *Model) runMoEGather(ctx context.Context, v *version, k int, h, combine *HostStep, g *moeGather, values map[string]int64) error {
	if len(combine.Reads) != 5 || combine.Reads[3].Bytes < h.Writes[0].Bytes {
		return fmt.Errorf("iron: moe_gather_combine of layer %d has no slots' output", h.Layer)
	}
	gx, gy := h.Writes[0], combine.Reads[3]
	K, R := h.Reads[1].Shape[1], h.SlotRows
	run, err := m.promptRun(v, k)
	if err != nil {
		return err
	}
	defer run.Close()
	slot := func(r Region, j, rows int) Region { // slot j's first rows of r
		r.Offset += 2 * j * R * K
		r.Bytes = 2 * rows * K
		return r
	}
	each := func(pass []gatherSlot, f func(j int, s gatherSlot) error) error {
		errs := make([]error, len(pass))
		var wg sync.WaitGroup
		for j, s := range pass {
			wg.Add(1)
			go func() {
				defer wg.Done()
				errs[j] = f(j, s)
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
	ids := make([]int32, h.Slots)
	for first := 0; first < len(g.slots); first += h.Slots {
		pass := g.slots[first:min(first+h.Slots, len(g.slots))]
		for j := range ids {
			ids[j] = int32(pass[min(j, len(pass)-1)].expert)
		}
		places, err := m.placePass(h.Layer, ids) // an expert's place: itself where every one has its own
		if err != nil {
			return err
		}
		for j, name := range h.SlotValues {
			values[name] = int64(places[j])
		}
		err = each(pass, func(j int, s gatherSlot) error {
			padded := min(R, (len(s.rows)+gatherPad-1)/gatherPad*gatherPad)
			b := make([]byte, 2*padded*K)
			for r, t := range s.rows {
				copy(b[2*r*K:2*(r+1)*K], g.x[2*int(t)*K:2*(int(t)+1)*K])
			}
			return m.writeRegion(v, slot(gx, j, padded), b)
		})
		if err != nil {
			return err
		}
		if err = writeWords(run.Scratchpad(), v.m.Kernels[k].Parameters, values); err != nil {
			return err
		}
		if err = run.Run(ctx); err != nil {
			return fmt.Errorf("iron: prompt command %d (slots from %d): %w", k, first, err)
		}
		err = each(pass, func(j int, s gatherSlot) error {
			y, err := m.viewRegion(v, slot(gy, j, len(s.rows)), 2*len(s.rows)*K)
			if err != nil {
				return err
			}
			copy(g.ys[2*s.first*K:], y)
			return nil
		})
		if err != nil {
			return err
		}
	}
	return nil
}

// moeGatherCombine runs a moe_gather_combine step after the gathered form: for the chunk's n
// rows, out = h (its second read) + each chosen expert's weight times its row of the slots'
// output, summed in float32 in the order chosen (iron_lfm2moe's MoECombineHost). The output's
// rows past n keep what they held: no row of the chunk reads them.
func (m *Model) moeGatherCombine(v *version, h *HostStep, g *moeGather, n int) error {
	if len(h.Reads) != 5 || len(h.Writes) != 1 || g == nil || g.n != n || len(h.Reads[1].Shape) != 2 {
		return fmt.Errorf("iron: moe_gather_combine of layer %d without its moe_gather", h.Layer)
	}
	K, top := h.Reads[1].Shape[1], h.Top
	res, err := m.readRegion(v, h.Reads[1], 2*n*K)
	if err != nil {
		return err
	}
	o := make([]byte, 2*n*K)
	combineRows(o, res, nil, nil, g.w, n, K, top, func(c int) []byte {
		row := int(g.at[c]) * K
		return g.ys[2*row : 2*(row+K)]
	})
	return m.writeRegion(v, h.Writes[0], o)
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
		if len(p.loads) > 0 {
			t0 := time.Now()
			if err := m.loadExperts(route.Layer, p.loads); err != nil {
				return err
			}
			if devenv.On("iron.timing") {
				fmt.Fprintf(os.Stderr, "moe pass %d of layer %d: %d experts copied in, %.1f ms\n", i, route.Layer, len(p.loads), time.Since(t0).Seconds()*1000)
			}
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
			if devenv.On("iron.timing") {
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
	rt := &moeRoute{shared: make([]float32, n), pass: make([]int32, n*top), row: make([]int32, n*top), hn: rows}
	ids, w := routeRows(newSoftmaxRouter(E, top), logits, n, L, nil)
	rt.w = w
	m.countPrompt(h.Layer, ids)
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
	if h.Places > 0 && h.Places < E {
		if h.Slots <= h.Places {
			return nil, fmt.Errorf("iron: moe_route_mx4 of layer %d: %d slots for %d places", h.Layer, S, h.Places)
		}
		return rt, m.poolPasses(h, rt, ids, n, K)
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
