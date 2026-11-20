package iron

import (
	"encoding/binary"
	"fmt"
	"strconv"
	"strings"

	"github.com/shibukawa/hwnpu/internal/llm/experts"
)

// Counting the experts a mixture of experts runs (experts.Usage, the Mac's file and tools:
// tools/moeusage). A prompt chunk's experts are chosen on the host (moeHostRoute,
// moeHostRouteMX4), which counts them there. A decode step's are chosen on the device: its
// routing writes each chosen expert's offsets into the next command's scratchpad (Fed), id ×
// scale + bias as the word's form says, so after the step the host reads one word a choice
// back and divides (fedRoute.ids).

// fedRoute is a fed command of the decode step: the words of its scratchpad that hold layer's
// chosen experts' offsets (params: every one of them; by rank, the one the host reads back).
type fedRoute struct {
	command, layer int
	at, bytes      int     // the scratchpad's buffer in the feedback argument
	params         []Param // the words naming e<layer>_<rank>
	rank           []routedWord
}

// routedWord is a word holding choice rank's offset: at byte at of the feedback argument, id ×
// scale + bias, held shifted left by shift.
type routedWord struct {
	at          int
	scale, bias int64
	shift       uint
}

// ExpertsShape is the layers and experts of the model's mixture of experts (from its prompt
// chunk's host steps), or zeros for a model without one.
func (m *Model) ExpertsShape() (layers, experts int) {
	for _, v := range m.versions() {
		if v == nil {
			continue
		}
		for _, h := range v.m.Host {
			if strings.HasPrefix(h.Kind, "moe_route") {
				layers, experts = max(layers, h.Layer+1), max(experts, h.Experts)
			}
		}
	}
	return layers, experts
}

// CountExperts has the model count the experts each layer runs into u (nil: stop counting),
// the prompt chunks' and the decode steps'. u's shape must be ExpertsShape's.
func (m *Model) CountExperts(u *experts.Usage) error {
	if u == nil {
		m.usage = nil
		return nil
	}
	L, E := m.ExpertsShape()
	if L == 0 {
		return fmt.Errorf("iron: the model has no mixture of experts to count")
	}
	if u.Layers != L || u.Experts != E {
		return fmt.Errorf("iron: counts of %d×%d experts for a model of %d×%d", u.Layers, u.Experts, L, E)
	}
	if err := m.findRoutes(); err != nil {
		return err
	}
	m.usage = u
	return nil
}

// findRoutes finds the decode step's fed commands' words that hold the chosen experts' offsets
// (parameters whose form's value is e<layer>_<rank>, as iron_gptoss.py and iron_qwen35.py name
// them), once.
func (m *Model) findRoutes() error {
	v := m.decode
	if v == nil || m.routes != nil {
		return nil
	}
	routes := []fedRoute{}
	for _, f := range v.m.Fed {
		if f.Command < 0 || f.Command >= len(v.m.Kernels) {
			return fmt.Errorf("iron: fed command %d of %d", f.Command, len(v.m.Kernels))
		}
		buf, ok := v.m.Buffers[f.Buffer]
		if !ok {
			return fmt.Errorf("iron: fed command %d: no buffer %s", f.Command, f.Buffer)
		}
		r := fedRoute{command: f.Command, layer: -1, at: buf.Offset, bytes: buf.Bytes}
		for _, p := range v.m.Kernels[f.Command].Parameters {
			if p.Form == nil || p.Index < f.First || p.Index >= f.First+f.Words {
				continue
			}
			l, rank, ok := routedValue(p.Form.Value)
			if !ok {
				continue
			}
			if r.layer >= 0 && l != r.layer {
				return fmt.Errorf("iron: fed command %d holds layers %d and %d's experts", f.Command, r.layer, l)
			}
			r.layer = l
			fm := p.Form
			if fm.Scale <= 0 || fm.Down != 0 || fm.Mul != 1 || fm.Add != 0 || fm.Post != 0 || fm.Lo != nil || fm.Hi != nil || fm.Mod != nil {
				return fmt.Errorf("iron: %s of fed command %d is not id × scale + bias", p.Name, f.Command)
			}
			w := routedWord{at: buf.Offset + 4*p.Index, scale: fm.Scale, bias: fm.Bias}
			switch p.Kind {
			case "core":
				w.shift = 2
			case "addr":
			default:
				return fmt.Errorf("iron: %s of fed command %d is of kind %q", p.Name, f.Command, p.Kind)
			}
			r.params = append(r.params, p)
			for len(r.rank) <= rank {
				r.rank = append(r.rank, routedWord{at: -1})
			}
			if r.rank[rank].at < 0 {
				r.rank[rank] = w
			}
		}
		if r.layer < 0 {
			continue
		}
		for i, w := range r.rank {
			if w.at < 0 {
				return fmt.Errorf("iron: fed command %d names no choice %d of layer %d", f.Command, i, r.layer)
			}
		}
		routes = append(routes, r)
	}
	m.routes = routes
	return nil
}

// routedValue parses e<layer>_<rank>.
func routedValue(s string) (layer, rank int, ok bool) {
	a, b, found := strings.Cut(strings.TrimPrefix(s, "e"), "_")
	if !found || !strings.HasPrefix(s, "e") {
		return 0, 0, false
	}
	l, err1 := strconv.Atoi(a)
	r, err2 := strconv.Atoi(b)
	return l, r, err1 == nil && err2 == nil && l >= 0 && r >= 0
}

// ids reads route's chosen experts back from the feedback argument b (synced from the device),
// -1 for a word that is not an expert of E (no step ran yet).
func (r *fedRoute) ids(b []byte, E int, ids []int32) []int32 {
	ids = ids[:0]
	for _, w := range r.rank {
		x := int64(binary.LittleEndian.Uint32(b[w.at:])>>w.shift) - w.bias
		id := int32(-1)
		if x%w.scale == 0 && x/w.scale >= 0 && x/w.scale < int64(E) {
			id = int32(x / w.scale)
		}
		ids = append(ids, id)
	}
	return ids
}

// countPrompt counts a prompt chunk's choices of layer (ids: top a row).
func (m *Model) countPrompt(layer int, ids []int32) {
	if m.usage != nil {
		m.usage.AddIDs(layer, ids)
	}
}

// countDecode reads the decode step's routed words back and counts the experts they name (a
// model whose experts have places counts them as it routes: pool.route).
func (m *Model) countDecode() error {
	if m.usage == nil || len(m.routes) == 0 || m.pool != nil {
		return nil
	}
	buf := m.buffer(m.decode, "feedback")
	if buf == nil {
		return nil
	}
	lo, hi := m.routes[0].at, m.routes[0].at
	for _, r := range m.routes {
		lo, hi = min(lo, r.at), max(hi, r.at+r.bytes)
	}
	if err := buf.SyncFromDevice(lo, hi-lo); err != nil {
		return err
	}
	var ids []int32
	for i := range m.routes {
		r := &m.routes[i]
		ids = r.ids(buf.Bytes(), m.usage.Experts, ids)
		m.usage.AddIDs(r.layer, ids)
	}
	return nil
}
