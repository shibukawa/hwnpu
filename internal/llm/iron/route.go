package iron

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"

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
}

func (m *Model) openRouter() error {
	er := m.Manifest.ExpertsRoute
	if er == nil {
		return nil
	}
	E, K := er.Experts, er.Hidden
	b, err := os.ReadFile(filepath.Join(m.dir, er.File))
	if err != nil {
		return fmt.Errorf("iron: experts_route: %w", err)
	}
	r := &hostRouter{man: er, cmd: map[int]int{}, x: make([]float32, K), logits: make([]float32, E),
		vals: make([]float32, E), order: make([]int, er.Top), rw: make([]byte, 2*er.Weights)}
	for l, lay := range er.Layers {
		n := int64(4 * (K*E + E))
		if lay.Offset+n > int64(len(b)) {
			return fmt.Errorf("iron: experts_route: layer %d past %s", l, er.File)
		}
		f := make([]float32, K*E+E)
		for i := range f {
			f[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[lay.Offset+int64(4*i):]))
		}
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
	K, E, top := er.Hidden, er.Experts, er.Top
	yo, err := m.viewRegion(v, regionOf(v, lay.YO), 2*K)
	if err != nil {
		return err
	}
	nw, err := m.region(v, regionOf(v, lay.Norm))
	if err != nil || len(nw) < 2*K {
		return fmt.Errorf("iron: experts_route: layer %d's norm: %v", l, err)
	}
	x := r.x
	vecops.BF16ToF32(x, yo[:2*K])
	var ss float64
	for _, a := range x {
		ss += float64(a) * float64(a)
	}
	inv := float32(1 / math.Sqrt(ss/float64(K)+er.Eps))
	for k := range x {
		// as the device's RMSNorm: the normed row in bf16
		x[k] = roundBF16(x[k] * inv * bf16At(nw, k))
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
	clear(r.rw)
	for k, i := range r.places {
		values[fmt.Sprintf("e%d_%d", l, k)] = int64(i)
		putBF16(r.rw, k, float32(e[k]/sum))
	}
	return m.writeRegion(v, regionOf(v, lay.RW), r.rw)
}

// regionOf is buffer name of v as a Region.
func regionOf(v *version, name string) Region {
	p := v.m.Buffers[name]
	return Region{Name: name, Argument: p.Argument, Offset: p.Offset, Bytes: p.Bytes}
}
