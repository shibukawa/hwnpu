package llm

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// Target is a device whose NPU takes a model only when its stored weights are under a size. A
// larger model is split into parts that run one after another, the output y of each part bound as
// the input x of the next. Splitting costs no speed: on the M3, 1.81 GiB of lut4 in 2 or 4 parts
// ran at the parts' own times added up (PROBE.md).
type Target struct {
	Name string
	// MaxBytes is the size of a model's weight file the NPU no longer accepts: a larger model is
	// placed on the CPU as a whole.
	MaxBytes int64
	// Measured is false for a limit that is assumed and still to be measured with npuprobe limit.
	Measured bool
	Note     string
}

// Targets are the known devices. The limit is per model, not on the total loaded.
var Targets = []Target{
	{Name: "m1", MaxBytes: 1 << 30, Note: "assumed half the M4's; to be measured with npuprobe limit"},
	{Name: "m2", MaxBytes: 1 << 30, Note: "assumed as the M3's; to be measured with npuprobe limit"},
	{Name: "m3", MaxBytes: 1 << 30, Measured: true, Note: "npuprobe limit: 0.99 GiB on the Neural Engine and 1.02 GiB not, in every format"},
	{Name: "m4", MaxBytes: 2e9, Measured: true, Note: "2 GB (a 4 B model in lut4) on the Neural Engine"},
}

// LookupTarget returns the named target. "any" is the smallest limit of all, for a model that
// runs on every known device.
func LookupTarget(name string) (Target, error) {
	if name == "any" {
		t := Target{Name: "any", Measured: true}
		var assumed []string
		for _, k := range Targets {
			if t.MaxBytes == 0 || k.MaxBytes < t.MaxBytes {
				t.MaxBytes = k.MaxBytes
			}
			if !k.Measured {
				t.Measured = false
				assumed = append(assumed, k.Name)
			}
		}
		t.Note = "the smallest limit of the known devices"
		if len(assumed) > 0 {
			t.Note += "; the limits of " + strings.Join(assumed, " and ") + " are assumed, to be measured with npuprobe limit"
		}
		return t, nil
	}
	var names []string
	for _, t := range Targets {
		if t.Name == name {
			return t, nil
		}
		names = append(names, t.Name)
	}
	return Target{}, fmt.Errorf("llm: unknown target %q; known: any, %s", name, strings.Join(names, ", "))
}

// NeuralEngineMaxDim is the largest size of one tensor dimension the Neural Engine takes: a model
// with a longer axis, such as attention over a KV cache of 16384 tokens and the step's own,
// has those operations placed on the CPU (M3, macOS 27.0.1, npuprobe kv).
const NeuralEngineMaxDim = 16384

// SplitMargin is the share of a target's limit that the estimated weights of a part may fill.
// The estimate misses the blob headers and alignment, a few hundred bytes per tensor.
const SplitMargin = 0.97

// Part is a run of layers built as one program: Start and Layers as in Options. The first part
// holds what comes before layer 0 and the last part the final norm and output layer.
type Part struct {
	Start, Layers int
	// Bytes is the estimated size of the part's weights.
	Bytes int64
}

var layerIndex = regexp.MustCompile(`(?:^|\.)layers?\.(\d+)\.`)

// EstimateBytes estimates the stored weight size of each layer of a model built with opts, and of
// what comes before the first layer (pre) and after the last (post), from the tensor shapes. The
// host keeps the embedding table, except when a causal LM ties it to the output layer, and the
// tables of the model's TokenInputs.
func EstimateBytes(m Model, opts Options) (pre int64, layers []int64, post int64) {
	layers = make([]int64, m.Layers())
	_, logits := m.Output()
	tensors := m.Tensors()
	_, ownHead := tensors["lm_head.weight"]
	host := map[string]bool{}
	if l, ok := m.(TokenInputs); ok {
		for _, lk := range l.TokenInputs() {
			host[lk.Table] = true
		}
	}
	prologue, _ := m.(Prologue)
	for name, shape := range tensors {
		if name == m.Embedding() && (!logits || ownHead) || host[name] {
			continue
		}
		n := tensorBytes(name, shape, opts)
		if s := layerIndex.FindStringSubmatch(name); s != nil {
			if i, err := strconv.Atoi(s[1]); err == nil && i < len(layers) {
				layers[i] += n
				continue
			}
		}
		// Outside the layers: a causal LM's final norm and output layer, an encoder's embedding
		// stage.
		if logits && (prologue == nil || !prologue.BeforeLayers(name)) {
			post += n
		} else {
			pre += n
		}
	}
	return pre, layers, post
}

// tensorBytes is the stored size of a tensor as blocks.Weight writes a matrix (lutN: N bit
// indices and a 2^N entry fp16 table per Group rows and ColBlock columns) and other tensors (fp16).
func tensorBytes(name string, shape []int, o Options) int64 {
	n := int64(1)
	for _, d := range shape {
		n *= int64(d)
	}
	bits := map[string]int64{"": 4, "lut4": 4, "lut6": 6, "lut8": 8}[o.FormatOf(name)]
	if len(shape) == 2 && bits > 0 {
		rows, cols := int64(shape[0]), int64(shape[1])
		group, block := int64(o.Group), int64(o.ColBlock)
		if group <= 0 {
			group = 16
		}
		if block <= 0 {
			block = cols
		}
		return (n*bits+7)/8 + (rows+group-1)/group*((cols+block-1)/block)*(1<<bits)*2
	}
	return 2 * n
}

// PlanSplit divides the model into parts whose estimated weights stay under budget. With parts 0
// it takes the fewest parts that fit; otherwise exactly that many (or fewer when there are fewer
// layers). The layers are spread so that the largest part is as small as it can be.
func PlanSplit(m Model, opts Options, budget int64, parts int) ([]Part, error) {
	pre, layers, post := EstimateBytes(m, opts)
	cost := append([]int64(nil), layers...)
	if len(cost) == 0 {
		return []Part{{Bytes: pre + post}}, nil
	}
	cost[0] += pre
	cost[len(cost)-1] += post
	var total, largest int64
	for _, c := range cost {
		total += c
		largest = max(largest, c)
	}
	if parts <= 0 {
		if largest > budget {
			return nil, fmt.Errorf("llm: one layer (with the model's ends) needs %.2f GiB, over the %.2f GiB a part may hold", gib(largest), gib(budget))
		}
		parts = len(fill(cost, budget, 0))
	}
	parts = min(parts, len(cost))
	// The smallest cap that filling fits in parts.
	lo, hi := largest, total
	for lo < hi {
		mid := lo + (hi-lo)/2
		if len(fill(cost, mid, 0)) <= parts {
			hi = mid
		} else {
			lo = mid + 1
		}
	}
	return fill(cost, lo, parts), nil
}

// fill puts layers in each part up to cap. With n above 0 it also starts a new part whenever the
// layers left are just enough to give each of the n parts one, so that it makes exactly n.
func fill(cost []int64, cap int64, n int) []Part {
	var out []Part
	for i, c := range cost {
		if len(out) == 0 || out[len(out)-1].Bytes+c > cap || n-len(out) >= len(cost)-i {
			out = append(out, Part{Start: i})
		}
		p := &out[len(out)-1]
		p.Layers++
		p.Bytes += c
	}
	return out
}

func gib(n int64) float64 { return float64(n) / (1 << 30) }
