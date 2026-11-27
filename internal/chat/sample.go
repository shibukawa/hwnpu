package chat

import (
	"container/heap"
	"math"
	"math/rand/v2"
	"sort"

	"github.com/shibukawa/hwnpu/internal/vecops"
)

// Sampling chooses the next token from the logits.
type Sampling struct {
	// Temperature 0 picks the most likely token.
	Temperature float64
	// TopK and TopP keep the K most likely tokens, then the fewest of them whose
	// probabilities add up to TopP. 0 turns either off.
	TopK int
	TopP float64
	// Repetition divides the logits of the tokens already generated in this answer (positive
	// logits; it multiplies negative ones), as Hugging Face's repetition_penalty does. 1 or 0
	// turns it off. The prompt is left out: an agent copies names from it.
	Repetition float64

	// softcap is the soft cap c the engine's session leaves to sampling (decode.Session.Softcap):
	// the logits are the program's y, the model's are c*tanh(y/c). It is monotonic, so the most
	// likely tokens are the same, and pick applies it to the values it uses alone.
	softcap float64
}

// Histogram parameters for O(V) temperature / top-p sampling without sorting the full vocabulary:
// 2048 buckets of width 1/64 cover scaled distances z = (top - l)/T in [0, 32). Beyond z = 32,
// exp(-32) ~= 1.27e-14 so even 262,144 tokens contribute at most ~3.3e-9 to the denominator.
// Inside each bucket of width 1/64, a degree-3 Taylor polynomial multiplied by histExp[b]
// evaluates exp(-z) to ~2.5e-9 relative accuracy without calling math.Exp per vocabulary entry.
const (
	histBuckets  = 2048
	histInvDelta = 64.0
	histDelta    = 1.0 / histInvDelta
	histMaxZ     = float64(histBuckets) * histDelta // 32.0
)

var histExp [histBuckets]float64

func init() {
	for i := range histExp {
		histExp[i] = math.Exp(-float64(i) * histDelta)
	}
}

// capped is the logit after the soft cap.
func (s Sampling) capped(y float32) float64 {
	if s.softcap <= 0 {
		return float64(y)
	}
	return s.softcap * math.Tanh(float64(y)/s.softcap)
}

// uncapped is the logit before the soft cap whose capped value is l (capped's inverse): a value at
// or past the cap, which no logit reaches, is far past every logit.
func (s Sampling) uncapped(l float64) float32 {
	c := s.softcap
	switch {
	case c <= 0:
		return float32(l)
	case l >= c:
		return float32(math.MaxFloat32)
	case l <= -c:
		return -float32(math.MaxFloat32)
	}
	return float32(c * math.Atanh(l/c))
}

// rawLowerBound returns a sound float32 lower bound on raw logits whose capped value is >= lo:
// any raw logit strictly below the returned value is guaranteed to have capped(v) < lo.
func (s Sampling) rawLowerBound(lo float64) float32 {
	if s.softcap <= 0 {
		r := float32(lo)
		if float64(r) > lo {
			r = math.Nextafter32(r, float32(math.Inf(-1)))
		}
		return r
	}
	c := s.softcap
	// Leave a 2-ulp margin around c so float32/float64 rounding near the cap never over-prunes.
	ulp := float64(math.Nextafter32(float32(c), float32(math.Inf(1))) - float32(c))
	adj := lo - 2*ulp
	if adj <= -c {
		return float32(math.Inf(-1))
	}
	if adj >= c {
		return float32(math.Inf(1))
	}
	r := float32(c * math.Atanh(adj/c))
	return math.Nextafter32(r, float32(math.Inf(-1)))
}

// pick chooses a token. It changes logits.
func (s Sampling) pick(logits []float32, seen []int32, rng *rand.Rand) int32 {
	if len(logits) == 0 {
		return 0
	}
	if s.Repetition > 0 && s.Repetition != 1 {
		done := map[int32]bool{}
		for _, t := range seen {
			if t < 0 || int(t) >= len(logits) || done[t] {
				continue
			}
			done[t] = true
			if s.softcap <= 0 {
				if l := logits[t]; l > 0 {
					logits[t] = l / float32(s.Repetition)
				} else {
					logits[t] = l * float32(s.Repetition)
				}
				continue
			}
			// on the capped logit, as the model's are; written back before the cap, so that the
			// order with the others stays
			if l := s.capped(logits[t]); l > 0 {
				logits[t] = s.uncapped(l / s.Repetition)
			} else {
				logits[t] = s.uncapped(l * s.Repetition)
			}
		}
	}
	if s.Temperature <= 0 {
		return argmax(logits)
	}
	if s.TopK > 0 && s.TopK < len(logits) {
		return s.pickTopK(logits, rng)
	}
	return s.pickHistogram(logits, rng)
}

func (s Sampling) pickTopK(logits []float32, rng *rand.Rand) int32 {
	idx := topK(logits, s.TopK)
	if len(idx) == 0 {
		return 0
	}
	top := s.capped(logits[idx[0]])
	if math.IsInf(top, 1) {
		// Multiple +Inf logits would produce Inf - Inf = NaN in softmax: sample uniformly among them.
		n := 1
		for n < len(idx) && math.IsInf(s.capped(logits[idx[n]]), 1) {
			n++
		}
		return int32(idx[rng.IntN(n)])
	}
	if math.IsInf(top, -1) {
		return int32(idx[rng.IntN(len(idx))])
	}
	ps := make([]float64, len(idx))
	var sum float64
	for i, j := range idx {
		if s.softcap > 0 {
			ps[i] = math.Exp((s.capped(logits[j]) - top) / s.Temperature)
		} else {
			ps[i] = math.Exp(float64(logits[j]-logits[idx[0]]) / s.Temperature)
		}
		sum += ps[i]
	}
	if s.TopP > 0 && s.TopP < 1 {
		var acc float64
		for i, p := range ps {
			if acc += p; acc >= s.TopP*sum {
				ps, idx, sum = ps[:i+1], idx[:i+1], acc
				break
			}
		}
	}
	r := rng.Float64() * sum
	for i, p := range ps {
		if r -= p; r <= 0 {
			return int32(idx[i])
		}
	}
	return int32(idx[len(idx)-1])
}

// pickHistogram performs O(V) temperature and nucleus (Top-P) sampling using a 2048-bucket
// histogram over [top - 32*T, top] and per-256-element block maxima to prune low blocks (and
// skip softcap tanh on pruned elements). Only the single boundary bucket crossing Top-P is sorted.
func (s Sampling) pickHistogram(logits []float32, rng *rand.Rand) int32 {
	const block = 256
	n := len(logits)
	nBlocks := (n + block - 1) / block

	var blockHiStack [1024]float32
	var blockHi []float32
	if nBlocks <= len(blockHiStack) {
		blockHi = blockHiStack[:nBlocks]
	} else {
		blockHi = make([]float32, nBlocks)
	}

	// Pass 1: find each block's maximum and the global maximum, ignoring NaNs.
	maxRaw := float32(math.Inf(-1))
	found := false
	for b := range nBlocks {
		at := b * block
		end := min(at+block, n)
		_, hi, ok := vecops.MinMax(logits[at:end])
		if !ok || hi == -math.MaxFloat32 {
			// Scalar fallback for blocks containing NaN or only -Inf.
			bFound := false
			hi = float32(math.Inf(-1))
			for _, v := range logits[at:end] {
				if v != v {
					continue
				}
				if !bFound || v > hi {
					hi = v
					bFound = true
				}
			}
			if !bFound {
				blockHi[b] = float32(math.NaN())
				continue
			}
		}
		blockHi[b] = hi
		if !found || hi > maxRaw {
			maxRaw = hi
			found = true
		}
	}
	if !found {
		return 0
	}

	top := s.capped(maxRaw)
	if math.IsInf(top, 1) {
		// Sample uniformly among all +Inf logits (avoids Inf - Inf = NaN).
		var infs []int
		for b := range nBlocks {
			if !math.IsInf(float64(blockHi[b]), 1) {
				continue
			}
			at := b * block
			end := min(at+block, n)
			for i := at; i < end; i++ {
				if math.IsInf(float64(logits[i]), 1) {
					infs = append(infs, i)
				}
			}
		}
		return int32(infs[rng.IntN(len(infs))])
	}
	if math.IsInf(top, -1) {
		// All non-NaN logits are -Inf: sample uniformly among them.
		var valid []int
		for i, v := range logits {
			if v == v {
				valid = append(valid, i)
			}
		}
		return int32(valid[rng.IntN(len(valid))])
	}

	span := histMaxZ * s.Temperature
	invT := 1.0 / s.Temperature
	if span <= 0 || math.IsInf(invT, 1) {
		// Temperature underflows span to 0: sample uniformly among tokens tied at maxRaw.
		var ties []int
		for b := range nBlocks {
			if !(blockHi[b] >= maxRaw) {
				continue
			}
			at := b * block
			end := min(at+block, n)
			for i := at; i < end; i++ {
				if logits[i] == maxRaw {
					ties = append(ties, i)
				}
			}
		}
		return int32(ties[rng.IntN(len(ties))])
	}

	minCapped := top - span
	minRaw := s.rawLowerBound(minCapped)

	// Pass 2: accumulate 2048-bucket probability histogram over blocks reaching minRaw.
	var bucketSum [histBuckets]float64
	var bucketCount [histBuckets]int32
	var totalSum float64

	if s.softcap <= 0 {
		for b := range nBlocks {
			if !(blockHi[b] >= minRaw) {
				continue
			}
			at := b * block
			end := min(at+block, n)
			for _, v := range logits[at:end] {
				if !(v >= minRaw) {
					continue
				}
				z := (top - float64(v)) * invT
				if z < 0 {
					z = 0
				}
				scaled := z * histInvDelta
				bi := int(scaled)
				if bi >= histBuckets {
					continue
				}
				r := (scaled - float64(bi)) * histDelta
				w := histExp[bi] * (1.0 - r*(1.0-r*(0.5-r*(1.0/6.0))))
				bucketSum[bi] += w
				bucketCount[bi]++
				totalSum += w
			}
		}
	} else {
		for b := range nBlocks {
			if !(blockHi[b] >= minRaw) {
				continue
			}
			at := b * block
			end := min(at+block, n)
			for _, v := range logits[at:end] {
				if !(v >= minRaw) {
					continue
				}
				l := s.capped(v)
				z := (top - l) * invT
				if z < 0 {
					z = 0
				}
				scaled := z * histInvDelta
				bi := int(scaled)
				if bi >= histBuckets {
					continue
				}
				r := (scaled - float64(bi)) * histDelta
				w := histExp[bi] * (1.0 - r*(1.0-r*(0.5-r*(1.0/6.0))))
				bucketSum[bi] += w
				bucketCount[bi]++
				totalSum += w
			}
		}
	}

	// Helper: sample a single token from bucket tb using target mass rem in [0, bucketSum[tb]].
	sampleBucket := func(tb int, rem float64) int32 {
		// Sound lower bound for bucket tb (with 1 extra bucket slack for FP rounding).
		loRaw := s.rawLowerBound(top - float64(tb+2)*histDelta*s.Temperature)
		lastIdx := 0
		hasLast := false
		for b := range nBlocks {
			if !(blockHi[b] >= loRaw) {
				continue
			}
			at := b * block
			end := min(at+block, n)
			for i := at; i < end; i++ {
				v := logits[i]
				if !(v >= loRaw) {
					continue
				}
				l := s.capped(v)
				z := (top - l) * invT
				if z < 0 {
					z = 0
				}
				scaled := z * histInvDelta
				bi := int(scaled)
				if bi != tb {
					continue
				}
				r := (scaled - float64(bi)) * histDelta
				w := histExp[bi] * (1.0 - r*(1.0-r*(0.5-r*(1.0/6.0))))
				lastIdx = i
				hasLast = true
				if rem -= w; rem <= 0 {
					return int32(i)
				}
			}
		}
		if hasLast {
			return int32(lastIdx)
		}
		return argmax(logits)
	}

	// No Top-P filtering: pick one bucket directly, then sample inside it.
	if s.TopP <= 0 || s.TopP >= 1 {
		r := rng.Float64() * totalSum
		var acc float64
		tb := -1
		for b := range histBuckets {
			if bucketCount[b] == 0 {
				continue
			}
			tb = b
			if acc+bucketSum[b] >= r {
				return sampleBucket(b, r-acc)
			}
			acc += bucketSum[b]
		}
		if tb >= 0 {
			return sampleBucket(tb, bucketSum[tb])
		}
		return argmax(logits)
	}

	// Top-P nucleus filtering: find the single boundary bucket where cumulative mass crosses TopP.
	target := s.TopP * totalSum
	var prefixSum float64
	cutBucket := -1
	lastBucket := -1
	for b := range histBuckets {
		if bucketCount[b] == 0 {
			continue
		}
		lastBucket = b
		if prefixSum+bucketSum[b] >= target {
			cutBucket = b
			break
		}
		prefixSum += bucketSum[b]
	}
	if cutBucket < 0 {
		cutBucket = lastBucket
		if cutBucket < 0 {
			return argmax(logits)
		}
		prefixSum -= bucketSum[cutBucket]
	}

	// Collect and sort only the boundary bucket's tokens to get the exact nucleus cutoff.
	type cand struct {
		idx int
		l   float64
		w   float64
	}
	cands := make([]cand, 0, bucketCount[cutBucket])
	cutLoRaw := s.rawLowerBound(top - float64(cutBucket+2)*histDelta*s.Temperature)
	for b := range nBlocks {
		if !(blockHi[b] >= cutLoRaw) {
			continue
		}
		at := b * block
		end := min(at+block, n)
		for i := at; i < end; i++ {
			v := logits[i]
			if !(v >= cutLoRaw) {
				continue
			}
			l := s.capped(v)
			z := (top - l) * invT
			if z < 0 {
				z = 0
			}
			scaled := z * histInvDelta
			bi := int(scaled)
			if bi != cutBucket {
				continue
			}
			r := (scaled - float64(bi)) * histDelta
			w := histExp[bi] * (1.0 - r*(1.0-r*(0.5-r*(1.0/6.0))))
			cands = append(cands, cand{idx: i, l: l, w: w})
		}
	}
	sort.Slice(cands, func(i, j int) bool {
		if cands[i].l != cands[j].l {
			return cands[i].l > cands[j].l
		}
		return cands[i].idx < cands[j].idx
	})
	var cutSum float64
	keep := len(cands)
	for i, c := range cands {
		cutSum += c.w
		if prefixSum+cutSum >= target {
			keep = i + 1
			break
		}
	}
	cands = cands[:keep]
	finalSum := prefixSum + cutSum

	r := rng.Float64() * finalSum
	if r < prefixSum {
		var acc float64
		tb := -1
		for b := range cutBucket {
			if bucketCount[b] == 0 {
				continue
			}
			tb = b
			if acc+bucketSum[b] >= r {
				return sampleBucket(b, r-acc)
			}
			acc += bucketSum[b]
		}
		if tb >= 0 {
			return sampleBucket(tb, bucketSum[tb])
		}
	}
	rem := r - prefixSum
	for _, c := range cands {
		if rem -= c.w; rem <= 0 {
			return int32(c.idx)
		}
	}
	if len(cands) > 0 {
		return int32(cands[len(cands)-1].idx)
	}
	return argmax(logits)
}

// argmax is the index of the first largest logit, ignoring NaNs. A block whose largest is not
// above the best so far is passed over whole (vecops.MinMax, as topK's): 248320 logits 0.2 -> 0.03 ms on an M2.
func argmax(logits []float32) int32 {
	if len(logits) == 0 {
		return 0
	}
	best := 0
	bestVal := logits[0]
	const block = 256
	for at := 0; at < len(logits); at += block {
		end := min(at+block, len(logits))
		if _, hi, ok := vecops.MinMax(logits[at:end]); ok && bestVal == bestVal && !(hi > bestVal) {
			continue
		}
		for i := at; i < end; i++ {
			v := logits[i]
			if v != v {
				continue
			}
			if bestVal != bestVal || v > bestVal {
				best = i
				bestVal = v
			}
		}
	}
	return int32(best)
}

// topK returns the indices of the k largest non-NaN logits, largest first (all when k is 0).
func topK(logits []float32, k int) []int {
	if k <= 0 || k >= len(logits) {
		idx := make([]int, len(logits))
		for i := range idx {
			idx[i] = i
		}
		sort.Slice(idx, func(a, b int) bool {
			va, vb := logits[idx[a]], logits[idx[b]]
			if va != va {
				return false
			}
			if vb != vb {
				return true
			}
			return va > vb
		})
		return idx
	}
	h := &minHeap{l: logits}
	h.idx = make([]int, 0, k)
	for i := range k {
		heap.Push(h, i)
	}
	// least is the smallest of the k kept. Nearly every logit is below it, so a block whose
	// largest is not above it is passed over whole (vecops.MinMax: a vector loop).
	least := logits[h.idx[0]]
	const block = 256
	for at := k; at < len(logits); at += block {
		end := min(at+block, len(logits))
		if _, hi, ok := vecops.MinMax(logits[at:end]); ok && least == least && hi <= least {
			continue
		}
		for i := at; i < end; i++ {
			v := logits[i]
			if v != v {
				continue
			}
			if least != least || v > least {
				h.idx[0] = i
				heap.Fix(h, 0)
				least = logits[h.idx[0]]
			}
		}
	}
	// Drop any NaNs that entered the initial k elements when fewer than k non-NaN logits exist.
	idx := h.idx[:0]
	for _, i := range h.idx {
		if logits[i] == logits[i] {
			idx = append(idx, i)
		}
	}
	sort.Slice(idx, func(a, b int) bool { return logits[idx[a]] > logits[idx[b]] })
	return idx
}

type minHeap struct {
	l   []float32
	idx []int
}

func (h *minHeap) Len() int { return len(h.idx) }
func (h *minHeap) Less(i, j int) bool {
	a, b := h.l[h.idx[i]], h.l[h.idx[j]]
	if a != a {
		return b == b // NaN is smallest so it is evicted first
	}
	if b != b {
		return false
	}
	return a < b
}
func (h *minHeap) Swap(i, j int) { h.idx[i], h.idx[j] = h.idx[j], h.idx[i] }
func (h *minHeap) Push(x any)    { h.idx = append(h.idx, x.(int)) }
func (h *minHeap) Pop() any {
	x := h.idx[len(h.idx)-1]
	h.idx = h.idx[:len(h.idx)-1]
	return x
}
