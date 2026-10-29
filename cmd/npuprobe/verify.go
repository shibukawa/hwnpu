package main

import (
	"fmt"
	"slices"
)

// verify runs every format at the given layer counts on the NPU with K tokens per call, the call
// that checks K drafted tokens in speculative decoding. While the time per call stays near that of
// one token, the call is bound by reading the weights and checking K drafts costs about as much
// as generating one token.
func (p *prober) verify(rep *report, formats []string, layers, tokens []int, save func()) {
	for _, n := range layers {
		for _, f := range formats {
			fmt.Printf("verify, %s %d layers:\n", f, n)
			for _, k := range tokens {
				if p.ctx.Err() != nil {
					return
				}
				r := p.isolated(childSpec{Kind: "try", Format: f, Layers: n, Tokens: k, Units: "npu"})[0]
				logResult(r)
				rep.Verify = append(rep.Verify, r)
			}
			save()
		}
	}
}

// knees tells, for each format and layer count run on the NPU, whether the largest K reached the
// compute limit and at how many tokens per call reading the weights and computing take as long.
// From K to the next K the time per token falls by K/next while the call is bound by reading the
// weights and stays while it is bound by computing; past halfway between the two, the compute has
// run out.
func knees(rows []result) []string {
	type key struct {
		format string
		layers int
	}
	groups := map[key][]result{}
	var keys []key
	for _, r := range rows {
		if !r.OnNPU || r.MsPerCall == 0 {
			continue
		}
		k := key{r.Format, r.Layers}
		if groups[k] == nil {
			keys = append(keys, k)
		}
		groups[k] = append(groups[k], r)
	}
	var out []string
	for _, k := range keys {
		g := groups[k]
		slices.SortFunc(g, func(a, b result) int { return a.Tokens - b.Tokens })
		if len(g) < 3 || g[0].Tokens != 1 {
			continue
		}
		var peak float64
		for _, r := range g {
			peak = max(peak, r.tflops())
		}
		a, b := g[len(g)-2], g[len(g)-1]
		fall := (b.MsPerCall / float64(b.Tokens)) / (a.MsPerCall / float64(a.Tokens))
		if fall < (1+float64(a.Tokens)/float64(b.Tokens))/2 {
			out = append(out, fmt.Sprintf("%s, %d layers: still bound by reading the weights at K=%d (%.1f TFLOPS); a larger -tokens would show where the compute runs out.", k.format, k.layers, b.Tokens, b.tflops()))
			continue
		}
		knee := g[0].MsPerCall * peak * 1e9 / (2 * float64(g[0].Params))
		out = append(out, fmt.Sprintf("%s, %d layers: at most %.1f TFLOPS. Reading the weights once (%.1f ms at K=1) takes as long as computing about %.0f tokens at that rate: up to about that many tokens per call cost little more than one, and beyond it the time grows with K.", k.format, k.layers, peak, g[0].MsPerCall, knee))
	}
	return out
}

// baseMs is the time per call of the one token run of the same format and layers, or 0.
func baseMs(rows []result, r result) float64 {
	for _, b := range rows {
		if b.Format == r.Format && b.Layers == r.Layers && b.Tokens == 1 && b.Error == "" {
			return b.MsPerCall
		}
	}
	return 0
}
