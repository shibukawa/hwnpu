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
// back and divides (routedWords).

// routedWord is a word of a fed command's scratchpad that holds a chosen expert's offset:
// choice rank of layer, at byte at of the feedback argument.
type routedWord struct {
	layer, rank int
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
		m.usage, m.routed = nil, nil
		return nil
	}
	L, E := m.ExpertsShape()
	if L == 0 {
		return fmt.Errorf("iron: the model has no mixture of experts to count")
	}
	if u.Layers != L || u.Experts != E {
		return fmt.Errorf("iron: counts of %d×%d experts for a model of %d×%d", u.Layers, u.Experts, L, E)
	}
	words, err := m.routedWords()
	if err != nil {
		return err
	}
	m.usage, m.routed = u, words
	return nil
}

// routedWords finds the decode step's words that hold the chosen experts' offsets: of each
// fed command, the first word of each choice (a parameter whose form's value is e<layer>_<rank>,
// as iron_gptoss.py and iron_qwen35.py name them).
func (m *Model) routedWords() ([]routedWord, error) {
	v := m.decode
	if v == nil {
		return nil, nil
	}
	var out []routedWord
	seen := map[[2]int]bool{}
	for _, f := range v.m.Fed {
		if f.Command < 0 || f.Command >= len(v.m.Kernels) {
			return nil, fmt.Errorf("iron: fed command %d of %d", f.Command, len(v.m.Kernels))
		}
		buf, ok := v.m.Buffers[f.Buffer]
		if !ok {
			return nil, fmt.Errorf("iron: fed command %d: no buffer %s", f.Command, f.Buffer)
		}
		for _, p := range v.m.Kernels[f.Command].Parameters {
			if p.Form == nil || p.Index < f.First || p.Index >= f.First+f.Words {
				continue
			}
			l, r, ok := routedValue(p.Form.Value)
			if !ok || seen[[2]int{l, r}] {
				continue
			}
			fm := p.Form
			if fm.Scale <= 0 || fm.Down != 0 || fm.Mul != 1 || fm.Add != 0 || fm.Post != 0 || fm.Lo != nil || fm.Hi != nil || fm.Mod != nil {
				return nil, fmt.Errorf("iron: %s of fed command %d is not id × scale + bias", p.Name, f.Command)
			}
			w := routedWord{layer: l, rank: r, at: buf.Offset + 4*p.Index, scale: fm.Scale, bias: fm.Bias}
			switch p.Kind {
			case "core":
				w.shift = 2
			case "addr":
			default:
				return nil, fmt.Errorf("iron: %s of fed command %d is of kind %q", p.Name, f.Command, p.Kind)
			}
			seen[[2]int{l, r}] = true
			out = append(out, w)
		}
	}
	return out, nil
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

// countPrompt counts a prompt chunk's choices of layer (ids: top a row).
func (m *Model) countPrompt(layer int, ids []int32) {
	if m.usage != nil {
		m.usage.AddIDs(layer, ids)
	}
}

// countDecode reads the decode step's routed words back and counts the experts they name.
func (m *Model) countDecode() error {
	if m.usage == nil || len(m.routed) == 0 {
		return nil
	}
	buf := m.buffer(m.decode, "feedback")
	if buf == nil {
		return nil
	}
	lo, hi := m.routed[0].at, m.routed[0].at+4
	for _, w := range m.routed {
		lo, hi = min(lo, w.at), max(hi, w.at+4)
	}
	if err := buf.SyncFromDevice(lo, hi-lo); err != nil {
		return err
	}
	b := buf.Bytes()
	E := int64(m.usage.Experts)
	for _, w := range m.routed {
		x := int64(binary.LittleEndian.Uint32(b[w.at:])>>w.shift) - w.bias
		if x%w.scale != 0 || x/w.scale < 0 || x/w.scale >= E {
			continue // not a step's word (none ran yet)
		}
		m.usage.AddIDs(w.layer, []int32{int32(x / w.scale)})
	}
	return nil
}
