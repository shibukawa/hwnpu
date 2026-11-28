package experts

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"

	"github.com/shibukawa/hwnpu"
	devenv "github.com/shibukawa/hwnpu/internal/dev"
	"github.com/shibukawa/hwnpu/internal/f32"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/mxfp4"
	"github.com/shibukawa/hwnpu/internal/vecops"
)

// hostPrefill runs hot experts of a multi-row prefill batch on the Neural Engine via a single
// weightless grouped fixed-slot program (x: 1xGxMxH, gu: 1xGx2IxH, dn: 1xGxHxI -> y: 1xGxMxH),
// unpacking MXFP4 weights directly to fp16 into ping-pong zero-copy input arrays while colder
// experts with fewer than minRows tokens run concurrently on the CPU. The ping-pong buffers are
// allocated lazily on prefill and released as soon as 1-token decoding begins, so zero extra RAM
// is held during generation.
type hostPrefill struct {
	mu       sync.Mutex
	gr       hwnpu.GraphRunner
	units    string
	G, M     int
	minRows  int
	hasBias  bool
	disabled bool

	g     *hwnpu.Graph
	slots [2]*prefillSlot
}

type prefillSlot struct {
	req                    *hwnpu.Request
	x, gu, dn, rw, gb, db *hwnpu.Buffer
	y                      *hwnpu.Buffer
	all                    []*hwnpu.Buffer
}

func (s *prefillSlot) close() {
	if s == nil {
		return
	}
	if s.req != nil {
		_ = s.req.Close()
		s.req = nil
	}
	for _, b := range s.all {
		if b != nil {
			_ = b.Close()
		}
	}
	s.all = nil
}

// EnablePrefillNPU configures h to offload hot multi-row prefill experts to gr (Core AI's Neural
// Engine) while keeping cold experts and 1-token decoding on the CPU. Set HWNPU_DEV=moe.prefill=0
// (or cpu) to keep all prefill experts on the CPU.
func (h *Host) EnablePrefillNPU(gr hwnpu.GraphRunner, units string) {
	if h == nil || gr == nil || units != "" {
		return
	}
	if v, ok := DebugOpt("prefill"); ok && (devenv.Off(v) || v == "cpu") {
		return
	}
	H, I := h.shape.Hidden, h.shape.Inter
	if H < 256 || I < 128 || H%32 != 0 || I%32 != 0 {
		return
	}
	G, M, minRows := 8, 64, 24
	if 6*H*I > 16<<20 {
		// Large experts (e.g. gpt-oss-20b, 49.8 MB in fp16): 4 experts of 128 rows per call.
		G, M, minRows = 4, 128, 24
	}
	if v, ok := DebugOpt("prefill-min-rows"); ok {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			minRows = n
		}
	}
	hasBias := false
	for _, hl := range h.layers {
		if hl.gb != nil || hl.db != nil {
			hasBias = true
			break
		}
	}
	h.npu = &hostPrefill{
		gr:      gr,
		units:   units,
		G:       G,
		M:       M,
		minRows: minRows,
		hasBias: hasBias,
	}
}

// HasPrefillNPU reports whether Neural Engine prefill offloading is enabled on h.
func (h *Host) HasPrefillNPU() bool {
	if h == nil || h.npu == nil {
		return false
	}
	h.npu.mu.Lock()
	defer h.npu.mu.Unlock()
	return !h.npu.disabled
}

func (p *hostPrefill) releaseSlots() {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	for i := range p.slots {
		if p.slots[i] != nil {
			p.slots[i].close()
			p.slots[i] = nil
		}
	}
}

func (p *hostPrefill) close() {
	if p == nil {
		return
	}
	p.releaseSlots()
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.g != nil {
		_ = p.g.Close()
		p.g = nil
	}
}

func (p *hostPrefill) ensureReady(ctx context.Context, sh Shape, act Act) (bool, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.disabled {
		return false, nil
	}
	if p.g == nil {
		tmpDir, err := os.MkdirTemp("", "hwnpu-moe-prefill-*")
		if err != nil {
			return false, err
		}
		defer os.RemoveAll(tmpDir)
		pkgPath := filepath.Join(tmpDir, "moe_prefill.mlpackage")
		if err := buildGroupedPrefillPkg(pkgPath, p.G, p.M, sh.Hidden, sh.Inter, act, p.hasBias); err != nil {
			p.disabled = true
			if DebugTime() {
				fmt.Fprintf(os.Stderr, "experts: prefill NPU build skipped (%v), using CPU\n", err)
			}
			return false, nil
		}
		g, err := p.gr.Compile(ctx, hwnpu.GraphSource{Path: pkgPath, Units: p.units})
		if err != nil || !g.Placement().OnNPU() {
			if g != nil {
				_ = g.Close()
			}
			p.disabled = true
			if DebugTime() {
				fmt.Fprintf(os.Stderr, "experts: prefill NPU compile skipped (%v), using CPU\n", err)
			}
			return false, nil
		}
		p.g = g
	}
	for i := range p.slots {
		if p.slots[i] == nil {
			sl, err := newPrefillSlot(p.g, p.hasBias)
			if err != nil {
				for j := range p.slots {
					if p.slots[j] != nil {
						p.slots[j].close()
						p.slots[j] = nil
					}
				}
				return false, err
			}
			p.slots[i] = sl
		}
	}
	return true, nil
}

func newPrefillSlot(g *hwnpu.Graph, hasBias bool) (*prefillSlot, error) {
	req, err := g.NewRequest()
	if err != nil {
		return nil, err
	}
	sl := &prefillSlot{req: req}
	bindIn := func(name string) (*hwnpu.Buffer, error) {
		a, ok := g.Arg(name)
		if !ok {
			return nil, fmt.Errorf("experts: prefill program missing %s", name)
		}
		b, err := g.AllocInput(a.Index)
		if err != nil {
			return nil, err
		}
		sl.all = append(sl.all, b)
		return b, req.Bind(a.Index, b)
	}
	if sl.x, err = bindIn("x"); err != nil {
		sl.close()
		return nil, err
	}
	if sl.gu, err = bindIn("gu"); err != nil {
		sl.close()
		return nil, err
	}
	if sl.dn, err = bindIn("dn"); err != nil {
		sl.close()
		return nil, err
	}
	if sl.rw, err = bindIn("rw"); err != nil {
		sl.close()
		return nil, err
	}
	if hasBias {
		if sl.gb, err = bindIn("gb"); err != nil {
			sl.close()
			return nil, err
		}
		if sl.db, err = bindIn("db"); err != nil {
			sl.close()
			return nil, err
		}
	}
	ya, ok := g.Arg("y")
	if !ok {
		sl.close()
		return nil, fmt.Errorf("experts: prefill program missing y")
	}
	if sl.y, err = g.AllocOutput(ya.Index); err != nil {
		sl.close()
		return nil, err
	}
	sl.all = append(sl.all, sl.y)
	if err := req.Bind(ya.Index, sl.y); err != nil {
		sl.close()
		return nil, err
	}
	return sl, nil
}

func buildGroupedPrefillPkg(path string, G, M, H, I int, act Act, hasBias bool) error {
	pkg, err := mlpackage.Create(path)
	if err != nil {
		return err
	}
	b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
	m := &milBuilder{b: b}

	x := m.input("x", mlpackage.Float16, 1, G, M, H)
	gu := m.input("gu", mlpackage.Float16, 1, G, 2*I, H)
	dn := m.input("dn", mlpackage.Float16, 1, G, H, I)
	rw := m.input("rw", mlpackage.Float16, 1, G, M, 1)

	h := m.matmul(x, gu, true, 1, G, M, 2*I)
	if hasBias {
		gb := m.input("gb", mlpackage.Float16, 1, G, 1, 2*I)
		h = m.add(h, gb, 1, G, M, 2*I)
	}
	gate := m.slice(h, 3, 0, I)
	up := m.slice(h, 3, I, 2*I)
	var a *mlpackage.Var
	if act.Kind == ActGptOss {
		lim, nlim := b.ScalarF16(float32(act.Limit)), b.ScalarF16(-float32(act.Limit))
		gate = m.f16("minimum", map[string]any{"x": gate, "y": lim}, 1, G, M, I)
		up = m.f16("maximum", map[string]any{"x": m.f16("minimum", map[string]any{"x": up, "y": lim}, 1, G, M, I), "y": nlim}, 1, G, M, I)
		neg := m.mul(gate, b.ScalarF16(-float32(act.Alpha)))
		den := m.add(m.f16("exp", map[string]any{"x": neg}, 1, G, M, I), b.ScalarF16(1))
		a = m.mul(m.f16("real_div", map[string]any{"x": gate, "y": den}, 1, G, M, I), m.add(up, b.ScalarF16(1)))
	} else {
		a = m.mul(m.silu(gate), up)
	}
	d := m.matmul(a, dn, true, 1, G, M, H)
	if hasBias {
		db := m.input("db", mlpackage.Float16, 1, G, 1, H)
		d = m.add(d, db, 1, G, M, H)
	}
	y, ins, outs := m.finishY(m.mul(d, rw, 1, G, M, H), 1, G, M, H)
	return pkg.Finish(b.Finish(y), 9, ins, outs)
}

func unpackRowsF16(dst, w, e []byte, rows, cols int, deinterleave bool) {
	nb := cols / 32
	half := rows / 2
	rowBytes := cols * 2
	for r := range rows {
		dr := r
		if deinterleave {
			if r&1 == 0 {
				dr = r >> 1
			} else {
				dr = half + (r >> 1)
			}
		}
		mxfp4.DequantizeF16(dst[dr*rowBytes:(dr+1)*rowBytes], w[r*nb*16:(r+1)*nb*16], e[r*nb:(r+1)*nb])
	}
}

// tryRunPrefillNPU runs hot expert slices on the Neural Engine and cold experts concurrently on the
// CPU. Returns (true, nil) when handled, (false, nil) when the batch should run on the CPU path.
func (h *Host) tryRunPrefillNPU(hl *hostLayer, rows [][]float32, byExpert, byK [][]int, w, out [][]float32) (bool, error) {
	p := h.npu
	if p == nil || len(rows) < 64 {
		return false, nil
	}
	G, M, minR := p.G, p.M, p.minRows
	type sliceJob struct {
		e  int
		rs []int
		ks []int
	}
	var sjobs []sliceJob
	var coldExps []int
	for e, rs := range byExpert {
		if len(rs) == 0 {
			continue
		}
		if len(rs) < minR {
			coldExps = append(coldExps, e)
			continue
		}
		for off := 0; off < len(rs); off += M {
			end := min(off+M, len(rs))
			sjobs = append(sjobs, sliceJob{e: e, rs: rs[off:end], ks: byK[e][off:end]})
		}
	}
	if len(sjobs) < G {
		return false, nil
	}
	ctx := context.Background()
	ok, err := p.ensureReady(ctx, h.shape, h.act)
	if err != nil || !ok {
		return false, err
	}

	H, I := h.shape.Hidden, h.shape.Inter
	gbSz, geSz := 2*I*H/2, 2*I*H/32
	dbSz, deSz := H*I/2, H*I/32

	rowsF16 := make([]byte, len(rows)*2*H)
	mxfp4.Parallel(len(rows), func(i int) {
		vecops.F32ToF16(rowsF16[i*2*H:(i+1)*2*H], rows[i])
	})

	// Run cold experts (< minRows) concurrently on the CPU while the Neural Engine runs hot slices.
	var outCPU [][]float32
	var cpuWG sync.WaitGroup
	if len(coldExps) > 0 {
		outCPU = make([][]float32, len(rows))
		for i := range outCPU {
			outCPU[i] = make([]float32, H)
		}
		cpuWG.Add(1)
		go func() {
			defer cpuWG.Done()
			h.runExpertsCPU(hl, coldExps, rows, byExpert, byK, w, outCPU)
		}()
	}

	nbatches := (len(sjobs) + G - 1) / G
	fillSlot := func(s *prefillSlot, batch int) {
		base := batch * G
		xBuf := s.x.Bytes()
		guBuf := s.gu.Bytes()
		dnBuf := s.dn.Bytes()
		rwBuf := s.rw.Bytes()
		var gbBuf, dbBuf []byte
		if p.hasBias {
			gbBuf, dbBuf = s.gb.Bytes(), s.db.Bytes()
		}
		rws := make([]float32, G*M)
		workers := min(G, max(runtime.GOMAXPROCS(0)/2, 2))
		var wg sync.WaitGroup
		for wk := range workers {
			wg.Add(1)
			go func(wk int) {
				defer wg.Done()
				var gbDeint []float32
				if p.hasBias {
					gbDeint = make([]float32, 2*I)
				}
				for g := wk; g < G; g += workers {
					if base+g >= len(sjobs) {
						clear(xBuf[g*M*2*H : (g+1)*M*2*H])
						continue
					}
					sj := sjobs[base+g]
					for j, ri := range sj.rs {
						copy(xBuf[(g*M+j)*2*H:(g*M+j+1)*2*H], rowsF16[ri*2*H:(ri+1)*2*H])
						rws[g*M+j] = w[ri][sj.ks[j]]
					}
					for j := len(sj.rs); j < M; j++ {
						copy(xBuf[(g*M+j)*2*H:(g*M+j+1)*2*H], xBuf[g*M*2*H:(g*M+1)*2*H])
					}
					e := sj.e
					unpackRowsF16(guBuf[g*2*I*H*2:(g+1)*2*I*H*2], hl.gw[e*gbSz:(e+1)*gbSz], hl.ge[e*geSz:(e+1)*geSz], 2*I, H, true)
					unpackRowsF16(dnBuf[g*H*I*2:(g+1)*H*I*2], hl.dw[e*dbSz:(e+1)*dbSz], hl.de[e*deSz:(e+1)*deSz], H, I, false)
					if p.hasBias {
						gb := h.gateBias(hl, e)
						for j := range I {
							gbDeint[j] = gb[2*j]
							gbDeint[I+j] = gb[2*j+1]
						}
						vecops.F32ToF16(gbBuf[g*2*I*2:(g+1)*2*I*2], gbDeint)
						vecops.F32ToF16(dbBuf[g*H*2:(g+1)*H*2], h.downBias(hl, e))
					}
				}
			}(wk)
		}
		wg.Wait()
		vecops.F32ToF16(rwBuf, rws)
		for _, bf := range []*hwnpu.Buffer{s.x, s.gu, s.dn, s.rw, s.gb, s.db} {
			if bf != nil {
				_ = bf.SyncToDevice(0, bf.Size())
			}
		}
	}

	yRowF32 := make([]float32, H)
	p.mu.Lock()
	slots := p.slots
	p.mu.Unlock()

	fillSlot(slots[0], 0)
	var runErr error
	for b := range nbatches {
		cur := slots[b%2]
		done := make(chan error, 1)
		go func(s *prefillSlot) {
			err := s.req.Run(ctx)
			if err == nil {
				err = s.y.SyncFromDevice(0, s.y.Size())
			}
			done <- err
		}(cur)
		if b+1 < nbatches {
			fillSlot(slots[(b+1)%2], b+1)
		}
		if err := <-done; err != nil && runErr == nil {
			runErr = err
		}
		if runErr != nil {
			continue
		}
		yBuf := cur.y.Bytes()
		base := b * G
		for g := range G {
			if base+g >= len(sjobs) {
				break
			}
			sj := sjobs[base+g]
			for j, ri := range sj.rs {
				off := (g*M + j) * 2 * H
				vecops.F16ToF32(yRowF32, yBuf[off:off+2*H])
				f32.Axpy(out[ri], 1, yRowF32)
			}
		}
	}
	cpuWG.Wait()
	if runErr != nil {
		return false, runErr
	}
	if len(coldExps) > 0 {
		for i := range out {
			f32.Axpy(out[i], 1, outCPU[i])
		}
	}
	return true, nil
}
