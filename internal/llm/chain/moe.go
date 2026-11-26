package chain

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/experts"
	"github.com/shibukawa/hwnpu/internal/safetensors"
	"github.com/shibukawa/hwnpu/internal/vecops"
)

// moe runs the mixture of experts layers of a model cut at them (llm.Routed, the manifest's
// Experts) between the parts: after a part that outputs moe_logits_N, the host picks each row's
// experts from the logits, runs the experts' program on the row (moe_x_N) with the chosen
// experts' arrays bound and writes the result into the next part's input moe_o_N. Every layer's
// experts are held in the program's own input arrays (hwnpu.Graph.AllocInput), read once from the
// weights file; the chains that Share one load them once.
type moe struct {
	g       *hwnpu.Graph
	req     *hwnpu.Request
	shape   experts.Shape
	routing experts.Routing
	x, i, r *hwnpu.Buffer
	y       *hwnpu.Buffer
	sArg    int
	guArg   []int
	dnArg   []int
	layers  map[int]*moeLayer
	bufs    []*hwnpu.Buffer
	locked  [][]byte   // the memory HWNPU_MOE_WIRE wired, unwired by close
	mu      sync.Mutex // one row at a time: the chains that share it run in turn
	// the programs of one expert on all the rows of a call, by its rows (experts.BuildRows)
	rows map[int]*moeRows
	// host runs the experts on the CPU from the checkpoint (llm.HostRouted), in place of g
	host *experts.Host
	bias map[int][]float32

	// what the experts took, for HWNPU_DEV's moe.time: the chain's runs, the rows a layer's experts
	// ran on, their time and the time of the calls alone
	Runs, Calls  int
	Time, Called time.Duration
}

// report prints what the experts took with HWNPU_DEV's moe.time.
func (m *moe) report() {
	if m == nil || m.Calls == 0 || !experts.DebugTime() {
		return
	}
	ms := func(d time.Duration, n int) float64 { return float64(d.Microseconds()) / 1e3 / float64(n) }
	fmt.Fprintf(os.Stderr, "chain: experts: %d runs, %d layer rows; %.2f ms a layer row (the program's calls %.2f), %.1f ms a run\n",
		m.Runs, m.Calls, ms(m.Time, m.Calls), ms(m.Called, m.Calls), ms(m.Time, max(m.Runs, 1)))
}

// moeRows is a loaded program of one expert on the rows of a call.
type moeRows struct {
	g                    *hwnpu.Graph
	req                  *hwnpu.Request
	x, i, r, zero        *hwnpu.Buffer
	ys                   [2]*hwnpu.Buffer // the calls' outputs in turn, each the next call's acc
	sArg, gu, dn, acc, y int
}

// minRows is the fewest valid rows a call's experts run on with the program of all its rows
// rather than a row at a time: the rows' choices cover most of a layer's experts, each read once,
// where a row at a time reads its k.
const minRows = 3

type moeLayer struct {
	gu, dn []*hwnpu.Buffer
	scales *hwnpu.Buffer
	bias   []float32
}

// loadMoE loads the experts' program and every layer's experts.
func loadMoE(ctx context.Context, c *Chain, rec *llm.ExpertsRecord, p experts.Placement) (*moe, error) {
	if rec.Host != nil {
		h, err := experts.OpenHostWith(rec.Host, rec.Shape, p)
		if err != nil {
			return nil, fmt.Errorf("chain: the experts: %w", err)
		}
		if experts.DebugTime() {
			fmt.Fprintf(os.Stderr, "chain: the experts of %d layers on the CPU, from %s\n", len(rec.Host.Layers), rec.Host.Checkpoint)
		}
		return &moe{shape: rec.Shape, routing: rec.Routing, host: h}, nil
	}
	g, err := c.gr.Compile(ctx, hwnpu.GraphSource{Path: rec.Program.Path, Units: c.units})
	if err != nil {
		return nil, fmt.Errorf("chain: the experts' program: %w", err)
	}
	m := &moe{g: g, shape: rec.Shape, routing: rec.Routing, layers: map[int]*moeLayer{}}
	if m.req, err = g.NewRequest(); err != nil {
		m.close()
		return nil, err
	}
	arg := func(name string) (hwnpu.ArgInfo, error) {
		a, ok := g.Arg(name)
		if !ok {
			return a, fmt.Errorf("chain: the experts' program has no %s", name)
		}
		return a, nil
	}
	own := func(name string) (*hwnpu.Buffer, error) {
		a, err := arg(name)
		if err != nil {
			return nil, err
		}
		var b *hwnpu.Buffer
		if a.Kind == hwnpu.ArgOutput {
			b, err = c.dev.AllocBuffer(a.Bytes)
		} else {
			b, err = g.AllocInput(a.Index)
		}
		if err != nil {
			return nil, err
		}
		m.bufs = append(m.bufs, b)
		return b, m.req.Bind(a.Index, b)
	}
	for _, p := range []struct {
		b    **hwnpu.Buffer
		name string
	}{{&m.x, "x"}, {&m.i, "i"}, {&m.r, "r"}, {&m.y, "y"}} {
		if *p.b, err = own(p.name); err != nil {
			m.close()
			return nil, err
		}
	}
	a, err := arg("S")
	if err != nil {
		m.close()
		return nil, err
	}
	m.sArg = a.Index
	for k := 0; k < rec.Shape.TopK; k++ {
		gu, err := arg(fmt.Sprintf("gu%d", k))
		if err != nil {
			m.close()
			return nil, err
		}
		dn, err := arg(fmt.Sprintf("dn%d", k))
		if err != nil {
			m.close()
			return nil, err
		}
		m.guArg, m.dnArg = append(m.guArg, gu.Index), append(m.dnArg, dn.Index)
	}
	m.rows = map[int]*moeRows{}
	for _, rr := range rec.Rows {
		mr, err := loadRows(ctx, c, rr.Path)
		if err != nil {
			m.close()
			return nil, err
		}
		m.rows[mr.ys[0].Size()/(2*rec.Shape.Hidden)] = mr
		m.bufs = append(m.bufs, mr.x, mr.i, mr.r, mr.zero, mr.ys[0], mr.ys[1])
	}
	f, err := safetensors.Open(rec.Weights)
	if err != nil {
		m.close()
		return nil, err
	}
	defer f.Close()
	read := func(b *hwnpu.Buffer, name string) error {
		sec, t, err := f.Section(name)
		if err != nil {
			return err
		}
		if int(sec.Size()) != b.Size() {
			return fmt.Errorf("chain: the experts' %s has %d bytes, the program takes %d (%v)", name, sec.Size(), b.Size(), t.Shape)
		}
		_, err = io.ReadFull(sec, b.Bytes())
		return err
	}
	alloc := func(index int) (*hwnpu.Buffer, error) {
		b, err := g.AllocInput(index)
		if err == nil {
			m.bufs = append(m.bufs, b)
		}
		return b, err
	}
	// HWNPU_MOE_WIRE wires the experts' arrays into memory (mlock): short of memory, the system
	// compresses or swaps out the experts no row chose for a while, and a prompt's calls then wait
	// for them (an M2 of 16 GB: 14.3 s to 43.3 s for the same prompt, the binding of the arrays
	// alone growing). Off by default: wiring 7.2 GB of 16 pushes the other applications out. A
	// size wires the arrays in turn up to it (experts.WireEnv).
	set, budget := experts.WireEnv()
	wire := set && budget != 0
	var wired int64
	var wireErr error
	if wire {
		read0 := read
		read = func(b *hwnpu.Buffer, name string) error {
			if err := read0(b, name); err != nil {
				return err
			}
			if wireErr == nil && (budget < 0 || wired+int64(b.Size()) <= budget) {
				if wireErr = mlock(b.Bytes()); wireErr == nil {
					wired += int64(b.Size())
					m.locked = append(m.locked, b.Bytes())
				}
			}
			return nil
		}
	}
	for _, l := range rec.Layers {
		ml := &moeLayer{}
		for e := 0; e < rec.Shape.Experts; e++ {
			gu, err := alloc(m.guArg[0])
			if err == nil {
				err = read(gu, experts.GateUpName(l, e))
			}
			if err != nil {
				m.close()
				return nil, err
			}
			dn, err := alloc(m.dnArg[0])
			if err == nil {
				err = read(dn, experts.DownName(l, e))
			}
			if err != nil {
				m.close()
				return nil, err
			}
			ml.gu, ml.dn = append(ml.gu, gu), append(ml.dn, dn)
		}
		if ml.scales, err = alloc(m.sArg); err == nil {
			err = read(ml.scales, experts.ScalesName(l))
		}
		if err != nil {
			m.close()
			return nil, err
		}
		if rec.Routing.Bias {
			if ml.bias, err = f.Float32(experts.BiasName(l)); err != nil {
				m.close()
				return nil, err
			}
		}
		m.layers[l] = ml
	}
	if wireErr != nil {
		fmt.Fprintf(os.Stderr, "chain: HWNPU_MOE_WIRE: %.2f GiB of the experts wired, the rest not: %v\n", float64(wired)/(1<<30), wireErr)
	}
	if experts.DebugTime() {
		fmt.Fprintf(os.Stderr, "chain: the experts of %d layers held (%.2f GiB wired), placement %v\n", len(rec.Layers), float64(wired)/(1<<30), g.Placement())
	}
	return m, nil
}

// loadRows loads a program of one expert on a call's rows (experts.BuildRows).
func loadRows(ctx context.Context, c *Chain, path string) (*moeRows, error) {
	g, err := c.gr.Compile(ctx, hwnpu.GraphSource{Path: path, Units: c.units})
	if err != nil {
		return nil, fmt.Errorf("chain: the experts' program of rows: %w", err)
	}
	mr := &moeRows{g: g}
	if mr.req, err = g.NewRequest(); err != nil {
		g.Close()
		return nil, err
	}
	idx := map[string]int{}
	for _, n := range []string{"x", "i", "S", "gu", "dn", "r", "acc", "y"} {
		a, ok := g.Arg(n)
		if !ok {
			mr.req.Close()
			g.Close()
			return nil, fmt.Errorf("chain: %s has no %s", path, n)
		}
		idx[n] = a.Index
	}
	mr.sArg, mr.gu, mr.dn, mr.acc, mr.y = idx["S"], idx["gu"], idx["dn"], idx["acc"], idx["y"]
	in := func(n string) (*hwnpu.Buffer, error) {
		b, err := g.AllocInput(idx[n])
		if err == nil {
			err = mr.req.Bind(idx[n], b)
		}
		return b, err
	}
	fail := func(err error) (*moeRows, error) {
		for _, b := range []*hwnpu.Buffer{mr.x, mr.i, mr.r, mr.zero, mr.ys[0], mr.ys[1]} {
			if b != nil {
				b.Close()
			}
		}
		mr.req.Close()
		g.Close()
		return nil, err
	}
	if mr.x, err = in("x"); err != nil {
		return fail(err)
	}
	if mr.i, err = in("i"); err != nil {
		return fail(err)
	}
	if mr.r, err = in("r"); err != nil {
		return fail(err)
	}
	if mr.zero, err = in("acc"); err != nil { // zeros: the first expert's acc
		return fail(err)
	}
	n := g.Args()[mr.y].Bytes
	for j := range mr.ys {
		if mr.ys[j], err = c.dev.AllocBuffer(n); err != nil {
			return fail(err)
		}
	}
	return mr, nil
}

func (m *moe) close() {
	if m == nil {
		return
	}
	if m.host != nil {
		m.host.Close()
		m.host = nil
	}
	for _, mr := range m.rows {
		mr.req.Close()
		mr.g.Close()
	}
	m.rows = nil
	for _, b := range m.locked {
		munlock(b)
	}
	m.locked = nil
	if m.req != nil {
		m.req.Close()
	}
	for _, b := range m.bufs {
		b.Close()
	}
	if m.g != nil {
		m.g.Close()
	}
	m.bufs = nil
}

// run runs layer's experts on the first rows of xs (fp16, rows of hidden) from the router's
// logits (fp16, rows of experts), writing their output into out (fp16, as xs); the other rows of
// out are zeros (padding past the step's tokens, which the host does not keep).
func (m *moe) run(ctx context.Context, layer int, xs, logits, out []byte, rows int) error {
	if m.host != nil {
		return m.runHost(layer, xs, logits, out, rows)
	}
	ml := m.layers[layer]
	if ml == nil {
		return fmt.Errorf("chain: layer %d has no experts held", layer)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	start := time.Now()
	H, E, K := m.shape.Hidden, m.shape.Experts, m.shape.TopK
	if mr := m.rows[len(xs)/(2*H)]; mr != nil && rows >= minRows {
		err := m.runRows(ctx, mr, ml, xs, logits, out, rows)
		m.Calls += rows
		m.Time += time.Since(start)
		return err
	}
	clear(out)
	lg := make([]float32, E)
	idx := make([]int, K)
	w := make([]float32, K)
	fi := make([]float32, K)
	if err := m.req.Bind(m.sArg, ml.scales); err != nil {
		return err
	}
	for row := 0; row < rows; row++ {
		vecops.F16ToF32(lg, logits[2*row*E:2*(row+1)*E])
		m.routing.Choose(lg, ml.bias, idx, w)
		for k, e := range idx {
			if err := m.req.Bind(m.guArg[k], ml.gu[e]); err != nil {
				return err
			}
			if err := m.req.Bind(m.dnArg[k], ml.dn[e]); err != nil {
				return err
			}
			fi[k] = float32(e)
		}
		vecops.F32ToF16(m.i.Bytes(), fi)
		vecops.F32ToF16(m.r.Bytes(), w)
		copy(m.x.Bytes(), xs[2*row*H:2*(row+1)*H])
		for _, b := range []*hwnpu.Buffer{m.x, m.i, m.r} {
			b.SyncToDevice(0, b.Size())
		}
		t0 := time.Now()
		if err := m.req.Run(ctx); err != nil {
			return fmt.Errorf("chain: layer %d's experts: %w", layer, err)
		}
		m.Called += time.Since(t0)
		m.y.SyncFromDevice(0, m.y.Size())
		copy(out[2*row*H:2*(row+1)*H], m.y.Bytes())
	}
	m.Calls += rows
	m.Time += time.Since(start)
	return nil
}

// moeHooks finds the parts after which the host runs a layer's experts: those that output
// moe_logits_N, by part index.
func (t *tier) moeHooks() (map[*Part]int, error) {
	hooks := map[*Part]int{}
	for i, p := range t.parts {
		for _, a := range p.g.Args() {
			if a.Kind == hwnpu.ArgOutput && strings.HasPrefix(a.Name, "moe_logits_") {
				var layer int
				if _, err := fmt.Sscanf(a.Name, "moe_logits_%d", &layer); err != nil {
					return nil, fmt.Errorf("chain: part %d: output %s", i, a.Name)
				}
				for _, n := range []string{fmt.Sprintf("moe_x_%d", layer), fmt.Sprintf("moe_o_%d", layer)} {
					if t.outputs[n] == nil && t.inputs[n] == nil {
						return nil, fmt.Errorf("chain: part %d outputs %s but no part has %s", i, a.Name, n)
					}
				}
				hooks[p] = layer
			}
		}
	}
	return hooks, nil
}

// runExperts runs the experts of layer after its part: the step's rows are those up to the one
// the input last picks (the program that reads prompts), else all.
func (t *tier) runExperts(ctx context.Context, layer int) error {
	xs, lg := t.outputs[fmt.Sprintf("moe_x_%d", layer)], t.outputs[fmt.Sprintf("moe_logits_%d", layer)]
	o := t.inputs[fmt.Sprintf("moe_o_%d", layer)]
	xs.SyncFromDevice(0, xs.Size())
	lg.SyncFromDevice(0, lg.Size())
	if err := t.moe.run(ctx, layer, xs.Bytes(), lg.Bytes(), o.Bytes(), t.stepRows(layer)); err != nil {
		return err
	}
	return o.SyncToDevice(0, o.Size())
}

// stepRows is the rows of the step in the tier's inputs that layer's experts run on: those up to
// the one the input last picks (the program that reads prompts), else all.
func (t *tier) stepRows(layer int) int {
	rows := t.outShape[fmt.Sprintf("moe_x_%d", layer)][1]
	if last := t.inputs["last"]; last != nil {
		b := last.Bytes()
		for j := 0; 2*j+1 < len(b); j++ {
			if b[2*j] != 0 || b[2*j+1] != 0 {
				rows = j + 1
			}
		}
	}
	return rows
}

// hostRows is the most rows one call of the host's experts takes of several steps' (runMany):
// with 8 of 256 experts a row, 512 rows give an expert 16, two full groups of the kernel that
// takes 8 vectors at a time (Qwen3.6-35B-A3B on an M3: 0.139 ms a row at 128 rows a call, 0.089
// at 512, 0.085 at 1024).
const hostRows = 512

// runMany runs layer's experts for several steps: xs and logits are each step's moe_x_N and
// moe_logits_N and rows its rows, and out, of each step's moe_o_N, is written. On the host the
// steps' rows go together, hostRows a call: an expert is decoded once for all the rows of a call
// that chose it, and a step of 128 tokens gives most of 256 experts fewer rows than the kernel
// takes at a time. The values differ from a step at a time in the last bits (which kernel takes a
// row depends on how many rows share its expert, and they sum in different orders).
func (m *moe) runMany(ctx context.Context, layer int, xs, logits, out [][]byte, rows []int) error {
	if m.host == nil {
		for c := range xs {
			if err := m.run(ctx, layer, xs[c], logits[c], out[c], rows[c]); err != nil {
				return err
			}
		}
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	start := time.Now()
	H, E, K := m.shape.Hidden, m.shape.Experts, m.shape.TopK
	type at struct{ step, row int }
	var from []at
	var x, y [][]float32
	var idx [][]int
	var w [][]float32
	lg := make([]float32, E)
	flush := func() error {
		if len(from) == 0 {
			return nil
		}
		if err := m.host.Run(layer, x, idx, w, y); err != nil {
			return fmt.Errorf("chain: layer %d's experts: %w", layer, err)
		}
		for i, a := range from {
			vecops.F32ToF16(out[a.step][2*a.row*H:2*(a.row+1)*H], y[i])
		}
		m.Calls += len(from)
		from, x, y, idx, w = from[:0], x[:0], y[:0], idx[:0], w[:0]
		return nil
	}
	for c := range xs {
		clear(out[c])
		for r := range rows[c] {
			xr, yr := make([]float32, H), make([]float32, H)
			vecops.F16ToF32(xr, xs[c][2*r*H:2*(r+1)*H])
			vecops.F16ToF32(lg, logits[c][2*r*E:2*(r+1)*E])
			ir, wr := make([]int, K), make([]float32, K)
			m.routing.Choose(lg, nil, ir, wr)
			from, x, y, idx, w = append(from, at{c, r}), append(x, xr), append(y, yr), append(idx, ir), append(w, wr)
			if len(from) == hostRows {
				if err := flush(); err != nil {
					return err
				}
			}
		}
	}
	if err := flush(); err != nil {
		return err
	}
	m.Time += time.Since(start)
	m.Called += time.Since(start)
	return nil
}

// runRows runs a call's experts with the program of all its rows: each expert any of the first
// rows chose once, the rows' weights for it (0 for the others) in r, the outputs summed through
// acc.
func (m *moe) runRows(ctx context.Context, mr *moeRows, ml *moeLayer, xs, logits, out []byte, rows int) error {
	H, E, K := m.shape.Hidden, m.shape.Experts, m.shape.TopK
	R := len(xs) / (2 * H)
	lg := make([]float32, E)
	idx := make([]int, K)
	w := make([]float32, K)
	weights := make([][]float32, E) // by expert, a weight a row; nil for an expert no row chose
	for row := 0; row < rows; row++ {
		vecops.F16ToF32(lg, logits[2*row*E:2*(row+1)*E])
		m.routing.Choose(lg, ml.bias, idx, w)
		for k, e := range idx {
			if weights[e] == nil {
				weights[e] = make([]float32, R)
			}
			weights[e][row] = w[k]
		}
	}
	copy(mr.x.Bytes(), xs)
	mr.x.SyncToDevice(0, mr.x.Size())
	if err := mr.req.Bind(mr.sArg, ml.scales); err != nil {
		return err
	}
	acc := mr.zero
	n := 0
	for e, we := range weights {
		if we == nil {
			continue
		}
		y := mr.ys[n%2]
		for _, b := range []struct {
			i int
			b *hwnpu.Buffer
		}{{mr.gu, ml.gu[e]}, {mr.dn, ml.dn[e]}, {mr.acc, acc}, {mr.y, y}} {
			if err := mr.req.Bind(b.i, b.b); err != nil {
				return err
			}
		}
		vecops.F32ToF16(mr.i.Bytes(), []float32{float32(e)})
		vecops.F32ToF16(mr.r.Bytes(), we)
		mr.i.SyncToDevice(0, mr.i.Size())
		mr.r.SyncToDevice(0, mr.r.Size())
		t0 := time.Now()
		if err := mr.req.Run(ctx); err != nil {
			return fmt.Errorf("chain: an expert of %d rows: %w", R, err)
		}
		m.Called += time.Since(t0)
		y.SyncFromDevice(0, y.Size())
		acc = y
		n++
	}
	copy(out, acc.Bytes())
	return nil
}

// runHost runs layer's experts on the CPU (llm.HostRouted).
func (m *moe) runHost(layer int, xs, logits, out []byte, rows int) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	start := time.Now()
	H, E, K := m.shape.Hidden, m.shape.Experts, m.shape.TopK
	x := make([][]float32, rows)
	idx := make([][]int, rows)
	w := make([][]float32, rows)
	y := make([][]float32, rows)
	lg := make([]float32, E)
	for r := range rows {
		x[r], y[r] = make([]float32, H), make([]float32, H)
		vecops.F16ToF32(x[r], xs[2*r*H:2*(r+1)*H])
		vecops.F16ToF32(lg, logits[2*r*E:2*(r+1)*E])
		idx[r], w[r] = make([]int, K), make([]float32, K)
		m.routing.Choose(lg, nil, idx[r], w[r])
	}
	if err := m.host.Run(layer, x, idx, w, y); err != nil {
		return fmt.Errorf("chain: layer %d's experts: %w", layer, err)
	}
	clear(out)
	for r := range rows {
		vecops.F32ToF16(out[2*r*H:2*(r+1)*H], y[r])
	}
	m.Calls += rows
	m.Time += time.Since(start)
	m.Called += time.Since(start)
	return nil
}
