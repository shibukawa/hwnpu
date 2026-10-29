package main

import (
	"fmt"
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

// baseMs is the time per call of the one token run of the same format and layers, or 0.
func baseMs(rows []result, r result) float64 {
	for _, b := range rows {
		if b.Format == r.Format && b.Layers == r.Layers && b.Tokens == 1 && b.Error == "" {
			return b.MsPerCall
		}
	}
	return 0
}
