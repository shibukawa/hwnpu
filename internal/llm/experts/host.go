package experts

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"github.com/shibukawa/hwnpu/internal/mem"
	"github.com/shibukawa/hwnpu/internal/mxfp4"
	"github.com/shibukawa/hwnpu/internal/safetensors"
)

// MXFP4 names a layer's experts in a checkpoint of OCP's MXFP4, every expert of the layer in one
// tensor (gpt-oss): the gate and up projections interleaved along their rows (the gate the even
// rows), (experts, 2·inter, hidden/32, 16) bytes and (experts, 2·inter, hidden/32) scales, the
// down projection (experts, hidden, inter/32, 16) and (experts, hidden, inter/32), their biases
// (experts, 2·inter) and (experts, hidden), none when their names are empty (Qwen3.6's experts,
// which hwnpu-dev mxfp4 writes in this form).
//
// Router and Norm name the layer's router (experts, hidden) with its bias RouterBias (none when
// empty) and the norm before it (hidden), its weight plus NormOffset (1 for a zero centered norm),
// from which the host guesses the next layer's experts (prefetcher); gpt-oss's records have none
// and the names are gpt-oss's.
type MXFP4 struct {
	GateUpBlocks string  `json:"gate_up_blocks"`
	GateUpScales string  `json:"gate_up_scales"`
	GateUpBias   string  `json:"gate_up_bias,omitempty"`
	DownBlocks   string  `json:"down_blocks"`
	DownScales   string  `json:"down_scales"`
	DownBias     string  `json:"down_bias,omitempty"`
	Router       string  `json:"router,omitempty"`
	RouterBias   string  `json:"router_bias,omitempty"`
	Norm         string  `json:"norm,omitempty"`
	NormOffset   float64 `json:"norm_offset,omitempty"`
}

// ActGptOss is gpt-oss's gated unit: the gate clamped to Limit, the up to ±Limit,
// gate·sigmoid(Alpha·gate)·(up+1).
const ActGptOss = "gptoss"

// ActSiLU is the SwiGLU of most experts (Qwen3.6): silu(gate)·up.
const ActSiLU = "silu"

// Act is the experts' gated unit.
type Act struct {
	Kind  string  `json:"kind"`
	Alpha float64 `json:"alpha,omitempty"`
	Limit float64 `json:"limit,omitempty"`
}

// HostRecord is the experts the host runs on its CPU from the checkpoint as it is
// (llm.HostRouted): the checkpoint (its model.safetensors.index.json, or one file), each layer's
// tensors and the gated unit.
type HostRecord struct {
	Checkpoint string        `json:"checkpoint"`
	Layers     map[int]MXFP4 `json:"layers"`
	Act        Act           `json:"act"`
}

type hostLayer struct {
	gw, ge, dw, de []byte // every expert's blocks and scales, mapped
	gb, db         []float32
	// the same four tensors in their file, for reading an expert's ranges whole (Host.readIn)
	src [4]io.ReaderAt
	// seen is, by expert, when its ranges were last found in memory or read (Host.fresh), in
	// nanoseconds; made on first use
	seen     []atomic.Int64
	seenOnce sync.Once
}

// residentFor is how long an expert found in memory, or read, is taken to be there without
// asking again: the system drops the pages used longest ago, and a second after its use an
// expert is far from those (Qwen3.6-35B-A3B on 16 GB reads 0.4 GB a second into 6 GB).
const residentFor = time.Second

// fresh reports whether expert e was found in memory or read within residentFor.
func (hl *hostLayer) fresh(e int, now int64) bool {
	return hl.seen != nil && now-hl.seen[e].Load() < int64(residentFor)
}

// saw records that expert e's ranges are in memory now.
func (hl *hostLayer) saw(e, experts int, now int64) {
	hl.seenOnce.Do(func() { hl.seen = make([]atomic.Int64, experts) })
	hl.seen[e].Store(now)
}

// arrays are the layer's four mapped tensors with an expert's bytes in each and its reader.
func (hl *hostLayer) arrays(sh Shape) [4]struct {
	b   []byte
	n   int
	src io.ReaderAt
} {
	H, I := sh.Hidden, sh.Inter
	return [4]struct {
		b   []byte
		n   int
		src io.ReaderAt
	}{{hl.gw, 2 * I * H / 2, hl.src[0]}, {hl.ge, 2 * I * H / 32, hl.src[1]}, {hl.dw, H * I / 2, hl.src[2]}, {hl.de, H * I / 32, hl.src[3]}}
}

// Host runs layers' experts on the CPU (internal/mxfp4), reading their weights from the
// checkpoint mapped into memory: they are not copied, and the system may drop the pages of experts
// no row chose for a while and read them again.
type Host struct {
	shape   Shape
	act     Act
	files   []*safetensors.File
	layers  map[int]*hostLayer
	warm    sync.WaitGroup
	stop    chan struct{}
	mu      sync.Mutex
	locked  [][]byte    // the spans wired (Host.wire), unwired by Close
	usage   *Usage      // how often each expert ran (rows), added by Close to Placement.Usage
	usageTo string      // Placement.Usage
	place   *placer     // the experts wired by use, when they are not all (Placement)
	pre     *prefetcher // the next layer's experts guessed and brought in, when not all are wired
	first   int         // the first layer held
	// zeros in place of the biases of experts that have none (hostLayer.gb, db nil)
	zeroGate, zeroDown []float32

	// for HWNPU_MOE_DEBUG: the bytes of the experts a row ran, and those of them not in memory
	// when it chose them (read from the disk for it)
	// reads: the experts are not all wired, and a run reads the ones it chose that are not in
	// memory whole from the file before it touches them (readIn)
	reads            bool
	debug            bool
	dbgNeed, dbgMiss atomic.Int64
	// the rows' runs whose experts were all in memory and the others', and their time (ns)
	dbgIn, dbgOut, dbgInTime, dbgOutTime atomic.Int64
	dbgRead                              atomic.Int64 // the bytes readChosen read
	times                                *runTimes    // HWNPU_MOE_DEBUG's time: a row's layers by what their experts were
	trace                                *traceFile   // HWNPU_MOE_DEBUG's trace=FILE: every row's experts
}

// OpenHost maps the experts of rec, all of them wired when they fit (Host.wiring).
func OpenHost(rec *HostRecord, sh Shape) (*Host, error) { return OpenHostWith(rec, sh, Placement{}) }

// OpenHostWith maps the experts of rec, wiring them as p says when they do not all fit.
func OpenHostWith(rec *HostRecord, sh Shape, p Placement) (*Host, error) {
	h := &Host{shape: sh, act: rec.Act, layers: map[int]*hostLayer{}, debug: DebugTime()}
	if rec.Act.Kind != ActGptOss && rec.Act.Kind != ActSiLU {
		return nil, fmt.Errorf("experts: the gated unit %q is not one the host runs", rec.Act.Kind)
	}
	files := map[string]*safetensors.File{}
	open := func(name string) (*safetensors.File, error) {
		path, err := shardOf(rec.Checkpoint, name)
		if err != nil {
			return nil, err
		}
		if f := files[path]; f != nil {
			return f, nil
		}
		f, err := safetensors.Open(path)
		if err != nil {
			return nil, err
		}
		files[path] = f
		h.files = append(h.files, f)
		return f, nil
	}
	mapped := func(name string, want int) ([]byte, io.ReaderAt, error) {
		f, err := open(name)
		if err != nil {
			return nil, nil, err
		}
		b, err := f.Map(name)
		if err == nil && len(b) != want {
			err = fmt.Errorf("experts: %s has %d bytes, want %d", name, len(b), want)
		}
		if err != nil {
			return nil, nil, err
		}
		sec, _, err := f.Section(name)
		return b, sec, err
	}
	floats := func(name string, want int) ([]float32, error) {
		if name == "" {
			return nil, nil // no bias
		}
		f, err := open(name)
		if err != nil {
			return nil, err
		}
		v, err := f.Float32(name)
		if err == nil && len(v) != want {
			err = fmt.Errorf("experts: %s has %d values, want %d", name, len(v), want)
		}
		return v, err
	}
	E, H, I := sh.Experts, sh.Hidden, sh.Inter
	for l, n := range rec.Layers {
		hl := &hostLayer{}
		var err error
		for i, t := range []struct {
			b    *[]byte
			name string
			n    int
		}{{&hl.gw, n.GateUpBlocks, E * 2 * I * H / 2}, {&hl.ge, n.GateUpScales, E * 2 * I * H / 32}, {&hl.dw, n.DownBlocks, E * H * I / 2}, {&hl.de, n.DownScales, E * H * I / 32}} {
			if *t.b, hl.src[i], err = mapped(t.name, t.n); err != nil {
				h.Close()
				return nil, err
			}
		}
		if hl.gb, err = floats(n.GateUpBias, E*2*I); err == nil {
			hl.db, err = floats(n.DownBias, E*H)
		}
		if err != nil {
			h.Close()
			return nil, err
		}
		h.layers[l] = hl
	}
	n, first := 0, -1
	for l := range h.layers {
		n = max(n, l+1)
		if first < 0 || l < first {
			first = l
		}
	}
	h.first, h.usage, h.usageTo = first, NewUsage(n, sh.Experts), p.Usage
	h.zeroGate, h.zeroDown = make([]float32, 2*I), make([]float32, H)
	all, budget := h.wiring()
	if v, _ := DebugOpt("read"); !all && v != "0" {
		h.reads = true
	}
	pf, _ := DebugOpt("prefault")
	switch {
	case all:
		h.prefault(true)
	case budget > 0:
		h.place = newPlacer(h, p, budget, n)
		h.place.start()
	case pf != "0" && 2*h.bytes() <= mem.Physical():
		h.prefault(false) // unwired as asked, and they fit: their pages in before the first runs
	}
	if !all {
		// a router as the checkpoint has it in bf16, mapped: read a row at a time by the guess
		// (float32 copies of Qwen3.6-35B-A3B's 40 routers of 256 x 2048 were 84 MB of heap)
		bf16 := func(name string, want int) []byte {
			f, err := open(name)
			if err != nil {
				return nil
			}
			if t, ok := f.Lookup(name); !ok || t.DType != "BF16" || t.Len() != want {
				return nil
			}
			b, err := f.Map(name)
			if err != nil {
				return nil
			}
			return b
		}
		if routers := hostRouters(rec, sh, floats, bf16); routers != nil {
			h.pre = newPrefetcher(h, routers, prefetchN(sh.TopK))
		}
	}
	if DebugTime() {
		h.times = &runTimes{}
	}
	if path, ok := DebugOpt("trace"); ok && path != "" {
		h.trace = openTrace(path)
	}
	if pf == "wait" { // for measuring: every page in before the first run
		h.warm.Wait()
	}
	return h, nil
}

// prefault touches every page of the experts in the background: the first touch of a page a
// process maps costs a fault, read from the disk or not, and a token's experts are new pages
// until most have been chosen once (gpt-oss-20b on an M3: 9.7 ms a layer for experts first
// touched, 1.1 once touched). The cores that run the experts take it over as they go.
// touched takes the prefault's sums, so that its reads are not compiled away.
var touched atomic.Uint32

// With wire (Host.wire) the pages are wired instead (mlock), so that the system does not drop
// them for others when memory is short: gpt-oss-20b's 10.2 GB of experts on a 16 GB machine with
// a desktop's applications open do not stay in the page cache, and a token's experts are read
// from the disk again (an M3: 4.0 ms a layer for them against 1.1).
func (h *Host) prefault(wire bool) {
	var spans [][]byte
	for _, hl := range h.layers {
		spans = append(spans, hl.gw, hl.ge, hl.dw, hl.de)
	}
	h.stop = make(chan struct{})
	n := max(runtime.GOMAXPROCS(0)/2, 1)
	var wired, unwired int64 // under h.mu
	var wireErr error
	for k := range n {
		h.warm.Add(1)
		go func() {
			defer h.warm.Done()
			var sink byte
			for i := k; i < len(spans); i += n {
				b := spans[i]
				if wire {
					err := mlock(b)
					h.mu.Lock()
					if err == nil {
						h.locked = append(h.locked, b)
						wired += int64(len(b))
					} else {
						unwired += int64(len(b))
						wireErr = err
					}
					h.mu.Unlock()
				}
				for j := 0; j < len(b); j += 16 << 10 {
					sink += b[j]
				}
				touched.Add(uint32(sink)) // kept: a sum nobody reads is compiled away
				select {
				case <-h.stop:
					return
				default:
				}
			}
		}()
	}
	debug := DebugTime()
	if debug || wire {
		start := time.Now()
		go func() {
			h.warm.Wait()
			if wire && unwired > 0 {
				// past the system's limit on wired memory (vm.user_wire_limit; the Neural
				// Engine's programs are wired too): the rest stays in the page cache
				fmt.Fprintf(os.Stderr, "experts: wired %.1f of %.1f GB (%v)\n", float64(wired)/1e9, float64(wired+unwired)/1e9, wireErr)
			}
			if debug {
				fmt.Fprintf(os.Stderr, "experts: every page touched in %.1fs\n", time.Since(start).Seconds())
			}
		}()
	}
}

// shardOf is the file of checkpoint that holds tensor name: checkpoint itself, or the one its
// index (model.safetensors.index.json) maps the name to.
func shardOf(checkpoint, name string) (string, error) {
	if filepath.Ext(checkpoint) != ".json" {
		return checkpoint, nil
	}
	raw, err := os.ReadFile(checkpoint)
	if err != nil {
		return "", err
	}
	var idx struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(raw, &idx); err != nil {
		return "", fmt.Errorf("experts: %s: %w", checkpoint, err)
	}
	f, ok := idx.WeightMap[name]
	if !ok {
		return "", fmt.Errorf("experts: %s has no %s", checkpoint, name)
	}
	return filepath.Join(filepath.Dir(checkpoint), f), nil
}

// fileAt is the checkpoint's file b is mapped from, and b's offset in it.
func (h *Host) fileAt(b []byte) (*safetensors.File, int64, bool) {
	for _, f := range h.files {
		if off, ok := f.MappedAt(b); ok {
			return f, off, true
		}
	}
	return nil, 0, false
}

// Close unmaps the checkpoint.
func (h *Host) Close() {
	if h.debug {
		var in, all int64
		for _, hl := range h.layers {
			for _, b := range [][]byte{hl.gw, hl.ge, hl.dw, hl.de} {
				all += int64(len(b))
				n, _ := mem.Resident(b)
				in += n
			}
		}
		line := fmt.Sprintf("experts: %.1f of %.1f GB in memory at the end", float64(in)/1e9, float64(all)/1e9)
		if need := h.dbgNeed.Load(); need > 0 {
			line += fmt.Sprintf("; of the %.1f GB the tokens' experts took, %.1f%% were not in memory when chosen (%.2f GB read for them)",
				float64(need)/1e9, 100*float64(h.dbgMiss.Load())/float64(need), float64(h.dbgMiss.Load())/1e9)
		}
		fmt.Fprintln(os.Stderr, line)
		if in, out := h.dbgIn.Load(), h.dbgOut.Load(); in+out > 0 {
			ms := func(ns, n int64) float64 { return float64(ns) / 1e6 / float64(max(n, 1)) }
			fmt.Fprintf(os.Stderr, "experts: a layer's row with every expert in memory %.2f ms (%d), with some to read %.2f ms (%d, %.2f MB not in memory each); %.2f GB read whole before the runs\n",
				ms(h.dbgInTime.Load(), in), in, ms(h.dbgOutTime.Load(), out), out, float64(h.dbgMiss.Load())/1e6/float64(max(out, 1)), float64(h.dbgRead.Load())/1e9)
		}
	}
	h.trace.close()
	if h.times != nil {
		h.times.report()
		h.times = nil
	}
	if h.pre != nil {
		h.pre.stop()
		h.pre = nil
	}
	if h.place != nil {
		h.place.stop()
		h.place = nil
	}
	if h.usage != nil {
		dbg, _ := DebugOpt("usage")
		for _, p := range []string{h.usageTo, dbg} {
			if p == "" {
				continue
			}
			if err := h.usage.AddTo(p); err != nil {
				fmt.Fprintf(os.Stderr, "experts: usage: %v\n", err)
			}
		}
		h.usage = nil
	}
	if h.stop != nil {
		close(h.stop)
		h.warm.Wait()
		h.stop = nil
	}
	for _, b := range h.locked {
		munlock(b)
	}
	h.locked = nil
	for _, f := range h.files {
		f.Close()
	}
	h.files = nil
}

// gateBias and downBias are expert e's biases of the gate and up products (2 inter) and of the
// down products (hidden): zeros for experts without them (Qwen3.6's; a slice of zeros for every
// expert of its 40 layers was 126 MB of heap).
func (h *Host) gateBias(hl *hostLayer, e int) []float32 {
	if hl.gb == nil {
		return h.zeroGate
	}
	return hl.gb[e*2*h.shape.Inter : (e+1)*2*h.shape.Inter]
}

func (h *Host) downBias(hl *hostLayer, e int) []float32 {
	if hl.db == nil {
		return h.zeroDown
	}
	return hl.db[e*h.shape.Hidden : (e+1)*h.shape.Hidden]
}

// rowPiece is the weights of a piece of work of runRow: 128 rows of Qwen3.6-35B-A3B's gate and up
// products, 512 of its down products.
const rowPiece = 128 * 2048

// runItems is the most rows of experts (an expert on a row) whose products Run holds at once:
// those of a step of 128 tokens with 8 experts a token (12 MB of Qwen3.6-35B-A3B's).
const runItems = 1024

// runRow is Run of one row (a generated token), as one piece of work of the cores' workers, woken
// once a layer: as three Runs over all the experts (the rows' form) each wakes them again, and
// after the Neural Engine's part of a step they are asleep (Qwen3.6-35B-A3B's 8 experts of a
// layer took 1.0–1.1 ms with 1.2 ms between the layers against 0.55 back to back).
//
// The cores take the experts' products in pieces of rowPiece weights, any core any piece: first
// those of the gate and up products, and an expert's down products once its activations are
// there (the core that finishes its last piece computes them). With an expert a core, as it was
// first written, a layer waited for its slowest core: on an M3 (4 cores of performance and 4 of
// efficiency) 8 cores were no faster than 4 (0.49 and 0.46 ms back to back, 1.20 and 0.88 with
// 1.2 ms between the layers), a core of efficiency or one that woke late holding a whole expert.
// The values are Run's (the same kernel a row, the experts added in the same order).
func (h *Host) runRow(hl *hostLayer, x []float32, idx []int, w []float32, out []float32) {
	H, I := h.shape.Hidden, h.shape.Inter
	xv := mxfp4.NewVec(H)
	xv.Quantize(x)
	gb, ge := 2*I*H/2, 2*I*H/32 // an expert's bytes of blocks and scales, gate and up
	db, de := H*I/2, H*I/32
	nbG, nbD := H/mxfp4.Block, I/mxfp4.Block
	alpha, limit := float32(h.act.Alpha), float32(h.act.Limit)
	gptoss := h.act.Kind == ActGptOss
	K := len(idx)
	rowsG, rowsD := max(rowPiece/H, 1), max(rowPiece/I, 1) // the rows of a piece
	piecesG, piecesD := (2*I+rowsG-1)/rowsG, (H+rowsD-1)/rowsD
	type expert struct {
		gu, down     []float32
		hv           *mxfp4.Vec
		nextG, doneG atomic.Int32
		nextD        atomic.Int32
		ready        atomic.Bool // the activations are there
	}
	ex := make([]expert, K)
	for k := range ex {
		ex[k].gu, ex[k].down = make([]float32, 2*I), make([]float32, H)
	}
	var left atomic.Int32
	left.Store(int32(K * (piecesG + piecesD)))
	gate := func(k, p int) {
		e, x := idx[k], &ex[k]
		gw, gs := hl.gw[e*gb:(e+1)*gb], hl.ge[e*ge:(e+1)*ge]
		bias := h.gateBias(hl, e)
		lo, hi := p*rowsG, min((p+1)*rowsG, 2*I)
		mxfp4.Rows(x.gu[lo:hi], gw[lo*nbG*16:hi*nbG*16], gs[lo*nbG:hi*nbG], xv)
		for r := lo; r < hi; r++ {
			x.gu[r] += bias[r]
		}
		if int(x.doneG.Add(1)) < piecesG {
			return
		}
		act, gu := make([]float32, I), x.gu
		if gptoss {
			for j := range I {
				g := min(gu[2*j], limit)
				u := min(max(gu[2*j+1], -limit), limit)
				act[j] = g / (1 + float32(math.Exp(float64(-alpha*g)))) * (u + 1)
			}
		} else {
			for j := range I {
				g := gu[2*j]
				act[j] = g / (1 + float32(math.Exp(float64(-g)))) * gu[2*j+1]
			}
		}
		x.hv = mxfp4.NewVec(I)
		x.hv.Quantize(act)
		x.ready.Store(true)
	}
	down := func(k, p int) {
		e, x := idx[k], &ex[k]
		dw, ds := hl.dw[e*db:(e+1)*db], hl.de[e*de:(e+1)*de]
		lo, hi := p*rowsD, min((p+1)*rowsD, H)
		mxfp4.Rows(x.down[lo:hi], dw[lo*nbD*16:hi*nbD*16], ds[lo*nbD:hi*nbD], x.hv)
	}
	// a piece of expert k if one is free: of its down products when they can begin
	take := func(k int) bool {
		x := &ex[k]
		if x.ready.Load() && int(x.nextD.Load()) < piecesD {
			if p := int(x.nextD.Add(1)) - 1; p < piecesD {
				down(k, p)
				left.Add(-1)
				return true
			}
		}
		if int(x.nextG.Load()) < piecesG {
			if p := int(x.nextG.Add(1)) - 1; p < piecesG {
				gate(k, p)
				left.Add(-1)
				return true
			}
		}
		return false
	}
	mxfp4.Parallel(min(K*(piecesG+piecesD), runtime.GOMAXPROCS(0)), func(i int) {
		// each worker from an expert of its own; with none free it waits for the pieces the
		// others are at, which may free an expert's down products (a piece is 10 to 40 µs)
		for k := i % K; left.Load() > 0; {
			if !take(k) {
				k = (k + 1) % K
			}
		}
	})
	// added by the experts' numbers, as Run adds them
	order := make([]int, len(idx))
	for k := range order {
		order[k] = k
	}
	slices.SortFunc(order, func(a, b int) int { return idx[a] - idx[b] })
	for _, k := range order {
		bias := h.downBias(hl, idx[k])
		wt, y := w[k], ex[k].down
		for c := range H {
			out[c] += wt * (y[c] + bias[c])
		}
	}
}

// readPiece is the most a read of readIn asks for at once: an SSD's largest requests.
const readPiece = 1 << 20

var readBufs = sync.Pool{New: func() any { b := make([]byte, readPiece); return &b }}

// readIn reads the ranges of expert e of layer l that are not all in memory from the file, each
// whole in requests of readPiece, and returns the bytes read. The pages land in the page cache,
// where the mapping finds them. Touching the mapping instead reads them a fault at a time, by
// every core at its own rows: Qwen3.6-35B-A3B on an M3 read 2.2 MB of a layer's experts in 5.9
// ms more than it took with them in memory (0.4 GB/s; the same SSD gives 4.7 GB/s for 1 MiB
// requests, four at a time), and with madvise(WILLNEED) on the ranges first 1.9 MB in 2.6 ms.
// Where what is in memory is not told (mem.Resident), the kernel is asked to read ahead.
func (h *Host) readIn(l, e int) (n int64) {
	hl := h.layers[l]
	now := time.Now().UnixNano()
	if hl == nil || hl.fresh(e, now) {
		return 0
	}
	told := true
	for a := range 4 {
		k, ok := h.readRange(l, e, a)
		n, told = n+k, told && ok
	}
	if told {
		hl.saw(e, h.shape.Experts, now)
	}
	return n
}

// readRange is readIn of one of the expert's four ranges (hostLayer.arrays): the bytes it read,
// and whether the range is in memory now (false where that is not told). Whether it was in
// memory is asked of two of its pages (mem.Probe): a range is read whole and its rows are used
// together, so its pages are there or dropped together, and asking of every page took longer
// than a token's products (2026-11-20, an M3: 47 % of the experts' CPU time against the
// kernel's 19 %).
func (h *Host) readRange(l, e, array int) (n int64, ok bool) {
	hl := h.layers[l]
	if hl == nil {
		return 0, false
	}
	a := hl.arrays(h.shape)[array]
	b := a.b[e*a.n : (e+1)*a.n]
	in, ok := mem.Probe(b, 2)
	if !ok {
		willneed(b)
		return 0, false
	}
	if in {
		return 0, true
	}
	buf := readBufs.Get().(*[]byte)
	defer readBufs.Put(buf)
	for off := 0; off < len(b); off += readPiece {
		k, err := a.src.ReadAt((*buf)[:min(readPiece, len(b)-off)], int64(e*a.n+off))
		n += int64(k)
		if err != nil {
			return n, false
		}
	}
	return n, true
}

// readers are the reads of readChosen at once.
const chosenReaders = 8

// readChosen reads the experts a run chose (the rows of each in byExpert) that are neither wired
// nor in memory, several ranges at once, before the cores run them. Those found there or read
// within residentFor are not asked about.
func (h *Host) readChosen(layer int, byExpert [][]int) {
	E := h.shape.Experts
	hl := h.layers[layer]
	now := time.Now().UnixNano()
	var es []int
	for e, rs := range byExpert {
		if len(rs) > 0 && (h.place == nil || !h.place.on[layer*E+e].Load()) && !hl.fresh(e, now) {
			es = append(es, e)
		}
	}
	if len(es) == 0 {
		return
	}
	var todo [][2]int // expert, range: the blocks first, the longest reads
	for _, a := range []int{0, 2, 1, 3} {
		for _, e := range es {
			todo = append(todo, [2]int{e, a})
		}
	}
	untold := make([]atomic.Bool, E) // an expert with a range of which it could not be told
	one := func(i int) {
		n, ok := h.readRange(layer, todo[i][0], todo[i][1])
		h.dbgRead.Add(n)
		if !ok {
			untold[todo[i][0]].Store(true)
		}
	}
	if len(todo) <= 4*h.shape.TopK {
		// a generated token's: asking is 1 µs a range, less than starting the readers; they
		// read what is not there
		var missing []int
		for i := range todo {
			a := hl.arrays(h.shape)[todo[i][1]]
			e := todo[i][0]
			if in, ok := mem.Probe(a.b[e*a.n:(e+1)*a.n], 2); !ok || !in {
				missing = append(missing, i)
			}
		}
		todo2 := make([][2]int, len(missing))
		for j, i := range missing {
			todo2[j] = todo[i]
		}
		todo = todo2
	}
	if len(todo) > 0 {
		var next atomic.Int32
		var wg sync.WaitGroup
		for range min(chosenReaders, len(todo)) {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for {
					i := int(next.Add(1)) - 1
					if i >= len(todo) {
						return
					}
					one(i)
				}
			}()
		}
		wg.Wait()
	}
	for _, e := range es {
		if !untold[e].Load() {
			hl.saw(e, E, now)
		}
	}
}

// Has reports whether the host holds layer's experts.
func (h *Host) Has(layer int) bool { return h.layers[layer] != nil }

// Run adds layer's experts' weighted outputs to out (rows of hidden): rows[i]'s experts idx[i]
// weighed by w[i]. The rows that chose an expert run it together: its weights are read once.
func (h *Host) Run(layer int, rows [][]float32, idx [][]int, w [][]float32, out [][]float32) error {
	hl := h.layers[layer]
	if hl == nil {
		return fmt.Errorf("experts: layer %d has no experts held", layer)
	}
	E, H, I := h.shape.Experts, h.shape.Hidden, h.shape.Inter
	byExpert := make([][]int, E) // the rows that chose each expert, and at which of their k
	byK := make([][]int, E)
	if h.usage != nil {
		h.usage.Add(layer, idx)
	}
	h.trace.add(layer, idx)
	if h.place != nil {
		h.place.count(layer, idx)
		h.place.ran(layer == h.first)
	}
	if h.debug && len(rows) == 1 {
		var miss int64
		for _, e := range idx[0] {
			for _, b := range h.spans(layer, e) {
				h.dbgNeed.Add(int64(len(b)))
				if in, ok := mem.Resident(b); ok {
					miss += int64(len(b)) - in
				}
			}
		}
		h.dbgMiss.Add(miss)
		start := time.Now()
		defer func() {
			if d := time.Since(start).Nanoseconds(); miss == 0 {
				h.dbgIn.Add(1)
				h.dbgInTime.Add(d)
			} else {
				h.dbgOut.Add(1)
				h.dbgOutTime.Add(d)
			}
		}()
	}
	if h.times != nil && len(rows) == 1 {
		kind, start := h.kind(layer, idx[0]), time.Now()
		defer func() { h.times.add(kind, time.Since(start)) }()
	}
	if h.pre != nil && len(rows) == 1 {
		t := time.Now()
		h.pre.ran(layer, rows[0], idx[0])
		if h.times != nil {
			h.times.guess(time.Since(t))
		}
	}
	for r := range rows {
		for k, e := range idx[r] {
			byExpert[e] = append(byExpert[e], r)
			byK[e] = append(byK[e], k)
		}
	}
	if h.reads {
		h.readChosen(layer, byExpert)
	}
	if v, _ := DebugOpt("row"); len(rows) == 1 && v != "0" {
		h.runRow(hl, rows[0], idx[0], w[0], out[0])
		return nil
	}
	xv := make([]*mxfp4.Vec, len(rows))
	mxfp4.Parallel(len(rows), func(r int) {
		xv[r] = mxfp4.NewVec(H)
		xv[r].Quantize(rows[r])
	})
	type run struct {
		e    int
		gu   [][]float32
		hv   []*mxfp4.Vec
		down [][]float32
	}
	gb, ge := 2*I*H/2, 2*I*H/32 // an expert's bytes of blocks and scales, gate and up
	db, de := H*I/2, H*I/32
	alpha, limit := float32(h.act.Alpha), float32(h.act.Limit)
	gptoss := h.act.Kind == ActGptOss
	// the experts in groups of at most runItems rows of theirs, so that the products held at
	// once are those of a step of a prompt however many rows the call has
	group := func(es []int) {
		var runs []*run
		var jobs []mxfp4.Job
		for _, e := range es {
			rs := byExpert[e]
			ru := &run{e: e}
			vs := make([]*mxfp4.Vec, len(rs))
			for j, r := range rs {
				ru.gu = append(ru.gu, make([]float32, 2*I))
				vs[j] = xv[r]
			}
			runs = append(runs, ru)
			jobs = append(jobs, mxfp4.Job{Out: ru.gu, W: hl.gw[e*gb : (e+1)*gb], E: hl.ge[e*ge : (e+1)*ge], V: vs})
		}
		mxfp4.Run(jobs...)
		jobs = jobs[:0]
		// the activations, a row of an expert each, over the cores
		type item struct {
			ru *run
			j  int
		}
		var items []item
		for _, ru := range runs {
			ru.hv = make([]*mxfp4.Vec, len(ru.gu))
			for j := range ru.gu {
				items = append(items, item{ru, j})
				ru.down = append(ru.down, make([]float32, H))
			}
		}
		mxfp4.Parallel(len(items), func(i int) {
			ru, gu := items[i].ru, items[i].ru.gu[items[i].j]
			bias := h.gateBias(hl, ru.e)
			act := make([]float32, I)
			if gptoss {
				for j := range I {
					g := min(gu[2*j]+bias[2*j], limit)
					u := min(max(gu[2*j+1]+bias[2*j+1], -limit), limit)
					act[j] = g / (1 + float32(math.Exp(float64(-alpha*g)))) * (u + 1)
				}
			} else {
				for j := range I {
					g, u := gu[2*j]+bias[2*j], gu[2*j+1]+bias[2*j+1]
					act[j] = g / (1 + float32(math.Exp(float64(-g)))) * u
				}
			}
			v := mxfp4.NewVec(I)
			v.Quantize(act)
			ru.hv[items[i].j] = v
		})
		for _, ru := range runs {
			jobs = append(jobs, mxfp4.Job{Out: ru.down, W: hl.dw[ru.e*db : (ru.e+1)*db], E: hl.de[ru.e*de : (ru.e+1)*de], V: ru.hv})
		}
		mxfp4.Run(jobs...)
		for _, ru := range runs {
			bias := h.downBias(hl, ru.e)
			for j, r := range byExpert[ru.e] {
				wt := w[r][byK[ru.e][j]]
				o, y := out[r], ru.down[j]
				for c := range H {
					o[c] += wt * (y[c] + bias[c])
				}
			}
		}
	}
	var es []int
	held := 0
	for e, rs := range byExpert {
		if len(rs) == 0 {
			continue
		}
		if held > 0 && held+len(rs) > runItems {
			group(es)
			es, held = es[:0], 0
		}
		es, held = append(es, e), held+len(rs)
	}
	if len(es) > 0 {
		group(es)
	}
	return nil
}
