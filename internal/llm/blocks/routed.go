package blocks

import (
	"fmt"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// RoutedBuilder is an architecture's part builder for a mixture of experts model cut by
// llm.PlanRouted (gptoss, qwen35moe, lfm2moe). Structs embedding *Graph satisfy GraphRef.
type RoutedBuilder interface {
	GraphRef() *Graph
	IsRouted(layer int) bool
	IsPaged(layer int) bool
	// Mixer builds the first half of layer (input norm, token mixer, residual add) on h.
	Mixer(layer int, h *mlpackage.Var) *mlpackage.Var
	// DenseFFN builds the second half of a non-routed layer (FFN norm, dense MLP, residual add).
	DenseFFN(layer int, h *mlpackage.Var) *mlpackage.Var
	// BeforeAttention emits the outputs of a part ending before layer's host-paged attention.
	BeforeAttention(layer int, h *mlpackage.Var)
	// AfterAttention reads layer's host-paged attention output (attn_o_N and any gate) and applies
	// its output projection before the residual add to h.
	AfterAttention(layer int) *mlpackage.Var
	// MoEHead emits moe_x_N and moe_logits_N for layer on h, returning the residual entering the
	// host's experts (h itself, or h plus a shared expert's output).
	MoEHead(layer int, h *mlpackage.Var) *mlpackage.Var
	// FinalNorm applies the model's final norm before the output head.
	FinalNorm(h *mlpackage.Var) *mlpackage.Var
}

// NormalizeRoutedOptions fills default quantization options, resolves opts.Layers, validates common
// Paged/LastRow/NoLayers invariants for a llm.Routed model, and reports whether the part is layer 0
// up to its router (first) or the post-attention half of a host-paged routed layer (afterAttn).
func NormalizeRoutedOptions(name string, opts *llm.Options, numLayers int, defaultFormat string, defaultFormats []string, isRouted, isPaged func(int) bool) (first, afterAttn bool, err error) {
	if opts.Group == 0 {
		opts.Group = 16
	}
	if opts.Format == "" {
		opts.Format = defaultFormat
	}
	if len(defaultFormats) > 0 {
		opts.FormatFor = append(append([]string(nil), defaultFormats...), opts.FormatFor...)
	}
	noLayers := opts.Layers == llm.NoLayers
	first = noLayers && opts.Start == 0 && isRouted(0)
	afterAttn = noLayers && !first
	if !noLayers && (opts.Layers <= 0 || opts.Start+opts.Layers > numLayers) {
		opts.Layers = numLayers - opts.Start
	}
	if opts.Paged > 0 && opts.Cache <= 0 {
		return false, false, fmt.Errorf("%s: a program split at its attention (Paged) needs a KV cache", name)
	}
	if opts.LastRow && opts.Cache <= 0 {
		return false, false, fmt.Errorf("%s: the logits of the last row alone (LastRow) are of a program with a KV cache", name)
	}
	if afterAttn && (!isPaged(opts.Start) || !isRouted(opts.Start)) {
		return false, false, fmt.Errorf("%s: a part of no layers from layer %d is the rest of an attention layer the host attends for, up to its experts", name, opts.Start)
	}
	return first, afterAttn, nil
}

// BuildRouted writes a part of a mixture of experts model cut by llm.PlanRouted using m.
func BuildRouted[B RoutedBuilder](m B, first, afterAttn bool, seq, hidden, numLayers, vocab int, head string) (ins, outs []mlpackage.Feature, err error) {
	g := m.GraphRef()
	opts := g.Opts
	start, end := opts.Start, opts.Start+opts.Layers
	err = Run(func() {
		x := g.B.Input("x", mlpackage.Float16, 1, seq, hidden)
		h := x
		var y *mlpackage.Var
		switch {
		case first:
			y = g.IdentityY(m.MoEHead(0, m.Mixer(0, h)))
		case afterAttn:
			h = g.Op("add", map[string]any{"x": m.AfterAttention(start), "y": h}, h.Shape...)
			y = g.IdentityY(m.MoEHead(start, h))
		default:
			i := start
			if m.IsRouted(start) {
				if end != start+1 {
					g.Fail("the part of layers %d to %d holds a layer's experts: a part is one layer's step (llm.PlanRouted)", start, end-1)
				}
				h = g.Op("add", map[string]any{"x": h, "y": g.Input(fmt.Sprintf("moe_o_%d", start), 1, seq, hidden)}, h.Shape...)
				i++
			}
			for ; i < end; i++ {
				if m.IsRouted(i) || m.IsPaged(i) {
					g.Fail("layer %d's experts or attention are inside the part of layers %d to %d: a part ends where they begin (llm.PlanRouted)", i, start, end-1)
				}
				h = m.DenseFFN(i, m.Mixer(i, h))
			}
			switch {
			case end < numLayers && m.IsPaged(end):
				m.BeforeAttention(end, h)
				y = g.IdentityY(h)
			case end < numLayers:
				if !m.IsRouted(end) {
					g.Fail("the part of layers %d to %d ends at a dense layer", start, end-1)
				}
				y = g.IdentityY(m.MoEHead(end, m.Mixer(end, h)))
			case opts.NoHead:
				y = g.IdentityY(h)
			default:
				if opts.LastRow {
					h = g.LastRow(h, g.Input("last", 1, 1, seq))
				}
				h = m.FinalNorm(h)
				if opts.Hidden {
					y = g.IdentityY(h)
				} else {
					y = g.Linear(h, head, "y", vocab)
				}
			}
		}
		ins, outs = g.Features(x, y)
	})
	return ins, outs, err
}
