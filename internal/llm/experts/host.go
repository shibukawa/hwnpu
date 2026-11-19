package experts

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/shibukawa/hwnpu/internal/mxfp4"
	"github.com/shibukawa/hwnpu/internal/safetensors"
)

// MXFP4 names a layer's experts in a checkpoint of OCP's MXFP4, every expert of the layer in one
// tensor (gpt-oss): the gate and up projections interleaved along their rows (the gate the even
// rows), (experts, 2·inter, hidden/32, 16) bytes and (experts, 2·inter, hidden/32) scales, the
// down projection (experts, hidden, inter/32, 16) and (experts, hidden, inter/32), their biases
// (experts, 2·inter) and (experts, hidden).
type MXFP4 struct {
	GateUpBlocks string `json:"gate_up_blocks"`
	GateUpScales string `json:"gate_up_scales"`
	GateUpBias   string `json:"gate_up_bias"`
	DownBlocks   string `json:"down_blocks"`
	DownScales   string `json:"down_scales"`
	DownBias     string `json:"down_bias"`
}

// ActGptOss is gpt-oss's gated unit: the gate clamped to Limit, the up to ±Limit,
// gate·sigmoid(Alpha·gate)·(up+1).
const ActGptOss = "gptoss"

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
}

// Host runs layers' experts on the CPU (internal/mxfp4), reading their weights from the
// checkpoint mapped into memory: they are not copied, and the system may drop the pages of experts
// no row chose for a while and read them again.
type Host struct {
	shape  Shape
	act    Act
	files  []*safetensors.File
	layers map[int]*hostLayer
	warm   sync.WaitGroup
	stop   chan struct{}
	mu     sync.Mutex
	locked [][]byte // the spans HWNPU_MOE_WIRE wired, unwired by Close
}

// OpenHost maps the experts of rec.
func OpenHost(rec *HostRecord, sh Shape) (*Host, error) {
	h := &Host{shape: sh, act: rec.Act, layers: map[int]*hostLayer{}}
	if rec.Act.Kind != ActGptOss {
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
	mapped := func(name string, want int) ([]byte, error) {
		f, err := open(name)
		if err != nil {
			return nil, err
		}
		b, err := f.Map(name)
		if err == nil && len(b) != want {
			err = fmt.Errorf("experts: %s has %d bytes, want %d", name, len(b), want)
		}
		return b, err
	}
	floats := func(name string, want int) ([]float32, error) {
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
		for _, t := range []struct {
			b    *[]byte
			name string
			n    int
		}{{&hl.gw, n.GateUpBlocks, E * 2 * I * H / 2}, {&hl.ge, n.GateUpScales, E * 2 * I * H / 32}, {&hl.dw, n.DownBlocks, E * H * I / 2}, {&hl.de, n.DownScales, E * H * I / 32}} {
			if *t.b, err = mapped(t.name, t.n); err != nil {
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
	if os.Getenv("HWNPU_MOE_PREFAULT") != "0" || os.Getenv("HWNPU_MOE_WIRE") != "" {
		h.prefault(os.Getenv("HWNPU_MOE_WIRE") != "")
		if os.Getenv("HWNPU_MOE_PREFAULT") == "wait" { // for measuring: every page in before the first run
			h.warm.Wait()
		}
	}
	return h, nil
}

// prefault touches every page of the experts in the background: the first touch of a page a
// process maps costs a fault, read from the disk or not, and a token's experts are new pages
// until most have been chosen once (gpt-oss-20b on an M3: 9.7 ms a layer for experts first
// touched, 1.1 once touched). The cores that run the experts take it over as they go.
// touched takes the prefault's sums, so that its reads are not compiled away.
var touched atomic.Uint32

// With wire (HWNPU_MOE_WIRE) the pages are wired instead (mlock), so that the system does not drop
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
	debug := os.Getenv("HWNPU_MOE_DEBUG") != ""
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

// Close unmaps the checkpoint.
func (h *Host) Close() {
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
	for r := range rows {
		for k, e := range idx[r] {
			byExpert[e] = append(byExpert[e], r)
			byK[e] = append(byK[e], k)
		}
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
	var runs []*run
	var jobs []mxfp4.Job
	gb, ge := 2*I*H/2, 2*I*H/32 // an expert's bytes of blocks and scales, gate and up
	for e, rs := range byExpert {
		if len(rs) == 0 {
			continue
		}
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
	db, de := H*I/2, H*I/32
	alpha, limit := float32(h.act.Alpha), float32(h.act.Limit)
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
		bias := hl.gb[ru.e*2*I : (ru.e+1)*2*I]
		act := make([]float32, I)
		for j := range I {
			g := min(gu[2*j]+bias[2*j], limit)
			u := min(max(gu[2*j+1]+bias[2*j+1], -limit), limit)
			act[j] = g / (1 + float32(math.Exp(float64(-alpha*g)))) * (u + 1)
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
		bias := hl.db[ru.e*H : (ru.e+1)*H]
		for j, r := range byExpert[ru.e] {
			wt := w[r][byK[ru.e][j]]
			o, y := out[r], ru.down[j]
			for c := range H {
				o[c] += wt * (y[c] + bias[c])
			}
		}
	}
	return nil
}
