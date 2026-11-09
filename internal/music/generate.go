package music

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"sort"

	"github.com/shibukawa/hwnpu/internal/llm"
)

// LM steps the depthformer: one temporal step and Codebooks depth steps a frame, with its caches
// on the host.
type LM struct {
	w               llm.Weights
	temporal, depth *step
	frame           int
	// the rings of the temporal layers' self- and cross-attention keys and values, by input name
	// (kv_k_s0, ...): Window slots of tHeads x 128, and which slots hold a frame
	rings map[string][]float32
	used  []bool
	prev  []int // the previous frame's tokens (SOS: all 0)
}

func newLM(w llm.Weights, temporal, depth *step) *LM {
	l := &LM{w: w, temporal: temporal, depth: depth, rings: map[string][]float32{}, used: make([]bool, Window), prev: make([]int, Codebooks)}
	for i := range tLayers {
		for _, kind := range []string{"s", "c"} {
			for _, kv := range []string{"k", "v"} {
				l.rings[fmt.Sprintf("kv_%s_%s%d", kv, kind, i)] = make([]float32, tHeads*Window*headDim)
			}
		}
	}
	return l
}

// Sampler picks a codebook's token from its logits (the range of the codebook alone).
type Sampler func(logits []float32) int

// TopK samples from the k most likely with the temperature, by the Gumbel-max trick as
// _sample_categorical_with_temperature does.
func TopK(k int, temperature float64, r *rand.Rand) Sampler {
	return func(logits []float32) int {
		idx := make([]int, len(logits))
		for i := range idx {
			idx[i] = i
		}
		sort.Slice(idx, func(a, b int) bool { return logits[idx[a]] > logits[idx[b]] })
		kth := logits[idx[min(k, len(idx))-1]]
		best, bv := 0, math.Inf(-1)
		for i, v := range logits {
			if v < kth {
				continue
			}
			g := -math.Log(-math.Log(r.Float64()*(1-2e-12) + 1e-12))
			if s := float64(v) + g*temperature; s > bv {
				best, bv = i, s
			}
		}
		return best
	}
}

// Step generates a frame: the temporal step from the previous frame's tokens and enc (the
// encoder's output of the frame), then the codebooks one by one with pick (nil: the forced
// tokens). It returns the frame's tokens (unique codes) and, when keep, the soft-capped logits of
// each codebook's range (Codebooks x CodebookSize).
func (l *LM) Step(enc []float32, pick Sampler, forced []int, keep bool) ([]int, []float32, error) {
	x, err := EmbedTokens(l.w, l.prev)
	if err != nil {
		return nil, nil, err
	}
	in := map[string][]float32{"x": x, "enc": enc}
	mask := make([]float32, Window+1)
	for j := range Window {
		if !l.used[j] {
			mask[j] = -1e4
		}
	}
	in["self_mask"], in["cross_mask"] = mask, mask
	for k, v := range l.rings {
		in[k] = v
	}
	out, err := l.temporal.run(in)
	if err != nil {
		return nil, nil, err
	}
	// write the frame's keys and values into the rings
	slot := l.frame % Window
	for k, ring := range l.rings {
		nv := out[k+"_new"]
		for h := range tHeads {
			copy(ring[(h*Window+slot)*headDim:(h*Window+slot+1)*headDim], nv[h*headDim:(h+1)*headDim])
		}
	}
	l.used[slot] = true
	l.frame++
	tout := slices.Clone(out["y"])
	// the depth steps
	dcache := map[string][]float32{}
	for i := range dLayers {
		for _, kv := range []string{"k", "v"} {
			dcache[fmt.Sprintf("kv_%s_d%d", kv, i)] = make([]float32, dHeads*depthCache*headDim)
		}
	}
	tokens := make([]int, Codebooks)
	var logits []float32
	dx := tout
	for c := range Codebooks {
		din := map[string][]float32{"x": dx}
		dm := make([]float32, depthCache+1)
		for j := range depthCache {
			if j >= c {
				dm[j] = -1e4
			}
		}
		din["mask"] = dm
		for k, v := range dcache {
			din[k] = v
		}
		dout, err := l.depth.run(din)
		if err != nil {
			return nil, nil, err
		}
		if c < depthCache {
			for k, cache := range dcache {
				nv := dout[k+"_new"]
				for h := range dHeads {
					copy(cache[(h*depthCache+c)*headDim:(h*depthCache+c+1)*headDim], nv[h*headDim:(h+1)*headDim])
				}
			}
		}
		lo := ReservedTokens + c*CodebookSize
		rng := make([]float32, CodebookSize)
		for j := range rng {
			rng[j] = float32(math.Tanh(float64(dout["y"][lo+j])/SoftCap) * SoftCap)
		}
		if keep {
			logits = append(logits, rng...)
		}
		if pick != nil {
			tokens[c] = lo + pick(rng)
		} else {
			tokens[c] = forced[c]
		}
		if dx, err = EmbedTokens(l.w, tokens[c:c+1]); err != nil {
			return nil, nil, err
		}
	}
	l.prev = tokens
	return tokens, logits, nil
}

// Codes are a frame's tokens as SpectroStream's codes (0 to 1023 each).
func Codes(tokens []int) []int {
	out := make([]int, len(tokens))
	for i, t := range tokens {
		out[i] = ((t-ReservedTokens)%CodebookSize + CodebookSize) % CodebookSize
	}
	return out
}
