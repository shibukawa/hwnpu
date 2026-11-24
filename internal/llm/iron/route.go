package iron

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"unsafe"

	"github.com/shibukawa/hwnpu/internal/llm/experts"

	"github.com/shibukawa/hwnpu/internal/f32"
	"github.com/shibukawa/hwnpu/internal/vecops"
)

// A decode step routed on the host (Manifest.ExpertsRoute: gpt-oss's 128 experts, which
// GptOssStep's router does not hold): a layer's command ends after its attention, and before the
// command that runs its experts the host reads the layer's residual, runs RMSNorm and the router
// on the CPU, picks the top experts (placeChoices gives them places), and writes their offsets
// (the command's per-call values e<layer>_<rank>) and routing weights (the RW state).

type hostRouter struct {
	man       *ExpertsRoute
	wt, bias  [][]float32 // by layer: (Hidden, Experts) and (Experts)
	cmd       map[int]int // the command running layer l's experts: l
	x, logits []float32
	vals      []float32
	order     []int
	ids       []int32
	places    []int
	rw        []byte
	// prefetch: the next layer's experts guessed from this layer's residual (its norm's weight
	// and router, as the Mac's experts.prefetcher) and copied into its places while the next
	// command runs, which reads none of them (expertPool.inflight: a place is waited for when it
	// is run or copied over). guess: how many (HWNPU_DEV=moe.prefetch=N, 0 none)
	guess     int
	guess2    int      // two layers ahead (HWNPU_DEV=moe.prefetch2=N)
	fetched   [][]bool // by layer and place: brought in by a guess, not yet run
	raw       []float32
	guessed   []int
	prefetchN int64 // experts brought in by a guess
	prefetchU int64 // of them, run before they left their place
}

func (m *Model) openRouter() error {
	er := m.Manifest.ExpertsRoute
	if er == nil {
		return nil
	}
	E, K := er.Experts, er.Hidden
	// read straight into float32s (this runs on AMD's x86, little-endian as the file): no copy
	// of the file beside them
	all, err := readFloat32s(filepath.Join(m.dir, er.File))
	if err != nil {
		return fmt.Errorf("iron: experts_route: %w", err)
	}
	r := &hostRouter{man: er, cmd: map[int]int{}, x: make([]float32, K), logits: make([]float32, E),
		vals: make([]float32, E), order: make([]int, er.Top), rw: make([]byte, 2*er.Weights),
		raw: make([]float32, K), guess: 4}
	if v, ok := experts.DebugOpt("prefetch"); ok {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			r.guess = n
		}
	}
	if v, ok := experts.DebugOpt("prefetch2"); ok {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			r.guess2 = n
		}
	}
	for l, lay := range er.Layers {
		n := int64(K*E + E)
		if lay.Offset%4 != 0 || lay.Offset/4+n > int64(len(all)) {
			return fmt.Errorf("iron: experts_route: layer %d past %s", l, er.File)
		}
		f := all[lay.Offset/4 : lay.Offset/4+n : lay.Offset/4+n]
		r.wt, r.bias = append(r.wt, f[:K*E]), append(r.bias, f[K*E:])
		for _, k := range []string{lay.Norm, lay.YO, lay.RW} {
			if _, ok := m.decode.m.Buffers[k]; !ok {
				return fmt.Errorf("iron: experts_route: the decode step has no buffer %s", k)
			}
		}
	}
	// the command reading layer l's offsets
	for k, kern := range m.decode.m.Kernels {
		for _, p := range kern.Parameters {
			if p.Form == nil {
				continue
			}
			if l, _, ok := routedValue(p.Form.Value); ok {
				if old, seen := r.cmd[k]; seen && old != l {
					return fmt.Errorf("iron: decode command %d reads layers %d and %d's experts", k, old, l)
				}
				r.cmd[k] = l
			}
		}
	}
	if len(r.cmd) != len(er.Layers) {
		return fmt.Errorf("iron: experts_route: %d commands read the %d layers' experts", len(r.cmd), len(er.Layers))
	}
	m.router = r
	return nil
}

// hostRoute routes layer l of the decode step: its experts' places into values, their weights
// into the RW state.
func (m *Model) hostRoute(l int, values map[string]int64) error {
	r, v := m.router, m.decode
	er := r.man
	lay := er.Layers[l]
	K, top := er.Hidden, er.Top
	yo, err := m.viewRegion(v, regionOf(v, lay.YO), 2*K)
	if err != nil {
		return err
	}
	nw, err := m.region(v, regionOf(v, lay.Norm))
	if err != nil || len(nw) < 2*K {
		return fmt.Errorf("iron: experts_route: layer %d's norm: %v", l, err)
	}
	vecops.BF16ToF32(r.raw, yo[:2*K])
	var ss float64
	for _, a := range r.raw {
		ss += float64(a) * float64(a)
	}
	inv := float32(1 / math.Sqrt(ss/float64(K)+er.Eps))
	r.logitsOf(l, inv, nw)
	topK(r.order, r.vals)
	r.ids = r.ids[:0]
	var sum float64
	e := make([]float64, top)
	for k, o := range r.order {
		r.ids = append(r.ids, int32(o))
		e[k] = math.Exp(float64(r.vals[o] - r.vals[r.order[0]]))
		sum += e[k]
	}
	if m.usage != nil {
		m.usage.AddIDs(l, r.ids)
	}
	if r.places, err = m.placeChoices(l, r.ids, r.places); err != nil {
		return err
	}
	if r.fetched != nil {
		for _, i := range r.places {
			if r.fetched[l][i] {
				r.prefetchU++
				r.fetched[l][i] = false
			}
		}
	}
	if m.pool != nil && r.guess > 0 && l+1 < len(er.Layers) {
		if err := m.guessNext(l+1, inv, r.guess); err != nil {
			return err
		}
	}
	if m.pool != nil && r.guess2 > 0 && l+2 < len(er.Layers) {
		if err := m.guessNext(l+2, inv, r.guess2); err != nil {
			return err
		}
	}
	clear(r.rw)
	for k, i := range r.places {
		values[fmt.Sprintf("e%d_%d", l, k)] = int64(i)
		putBF16(r.rw, k, float32(e[k]/sum))
	}
	return m.writeRegion(v, regionOf(v, lay.RW), r.rw)
}

// logitsOf is router l's logits (rounded to bf16) in r.vals on r.raw, RMSNormed by inv times
// the norm's weight nw (as the device: the normed row in bf16).
func (r *hostRouter) logitsOf(l int, inv float32, nw []byte) {
	E := r.man.Experts
	x := r.x
	for k := range x {
		x[k] = roundBF16(r.raw[k] * inv * bf16At(nw, k))
	}
	lg := r.logits
	copy(lg, r.bias[l])
	wt := r.wt[l]
	for k, a := range x {
		if a != 0 {
			f32.Axpy(lg, a, wt[k*E:(k+1)*E])
		}
	}
	for i := range lg {
		r.vals[i] = roundBF16(lg[i])
	}
}

// guessNext guesses layer l's experts from the layer before's residual (r.raw, inv) with layer
// l's norm and router, and copies the guesses not in a place into layer l's places in the
// background, over the experts least used lately that are not guessed.
func (m *Model) guessNext(l int, inv float32, n int) error {
	r, p, v := m.router, m.pool, m.decode
	nw, err := m.region(v, regionOf(v, r.man.Layers[l].Norm))
	if err != nil {
		return err
	}
	r.logitsOf(l, inv, nw)
	if cap(r.guessed) < n {
		r.guessed = make([]int, n)
	}
	g := r.guessed[:min(n, p.E)]
	topK(g, r.vals)
	if r.fetched == nil {
		r.fetched = make([][]bool, p.L)
		for i := range r.fetched {
			r.fetched[i] = make([]bool, p.P)
		}
	}
	pl := &p.places[l]
	keep := func(e int) bool { return slices.Contains(g, e) }
	for _, e := range g {
		if pl.of[e] >= 0 {
			continue
		}
		i := p.victim(l, keep)
		if i < 0 {
			break
		}
		p.waitPlace(l, i) // an earlier guess's copy into it
		p.place(l, e, i)
		r.fetched[l][i] = true
		r.prefetchN++
		done := make(chan struct{})
		p.inflight[l][i] = done
		go func() {
			defer close(done)
			if err := m.copyExpert(l, e, i, 8); err != nil {
				fmt.Fprintf(os.Stderr, "iron: experts: a guess of layer %d: %v\n", l, err)
			}
		}()
	}
	return nil
}

// waitGuesses waits for every guess's copy (before the places are read whole or filled again).
func (m *Model) waitGuesses() {
	if m.pool == nil {
		return
	}
	for l := range m.pool.inflight {
		for i := range m.pool.inflight[l] {
			m.pool.waitPlace(l, i)
		}
	}
}

// regionOf is buffer name of v as a Region.
func regionOf(v *version, name string) Region {
	p := v.m.Buffers[name]
	return Region{Name: name, Argument: p.Argument, Offset: p.Offset, Bytes: p.Bytes}
}

// readFloat32s reads a file of little-endian float32s into float32s, on a little-endian machine
// without a copy of the file's bytes beside them.
func readFloat32s(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if fi.Size()%4 != 0 {
		return nil, fmt.Errorf("%s: %d bytes is not float32s", path, fi.Size())
	}
	v := make([]float32, fi.Size()/4)
	if len(v) == 0 {
		return v, nil
	}
	b := unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), 4*len(v))
	if _, err := io.ReadFull(f, b); err != nil {
		return nil, err
	}
	if binary.NativeEndian.Uint16([]byte{1, 0}) != 1 { // big-endian: turn each around
		for i := range v { // element i's own bytes, read before they are written
			v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		}
	}
	return v, nil
}
