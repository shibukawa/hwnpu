package anima

import (
	"cmp"
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu"
	hcache "github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/imagegen/anima/mil"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
)

// rangeProgram is a group of the DiT's blocks that first runs its mil.BlockRanges on the same
// inputs.
type rangeProgram struct {
	runner.Program
	ranges runner.Program
	note   func(y []float32)
}

func (p *rangeProgram) Run(ctx context.Context, in map[string][]float32) ([]float32, error) {
	y, err := p.ranges.Run(ctx, in)
	if err != nil {
		return nil, fmt.Errorf("ranges: %w", err)
	}
	p.note(y)
	return p.Program.Run(ctx, in)
}

// TestBlockRanges draws pictures with an export on the NPU and prints the ranges of what the
// DiT's norms take in every block (mil.BlockRanges), over every DiT call: whether a checkpoint's
// values fit fp16 as the norms are written (design.md 16.201). The queries' and keys' RMS is
// also given over what their projection's weights give for white input of unit variance (the
// scale mil folds into the weights).
//
// HWNPU_ANIMA_EXPORT is the export (a .coreai directory), HWNPU_ANIMA_DIR the Anima directory
// its checkpoint is in, HWNPU_ANIMA_RANGES where the range programs are kept (compiled the first
// time: minutes each at 1024 x 1024), HWNPU_ANIMA_PROMPTS the prompts, | between them, drawn
// with seeds 1, 2... HWNPU_ANIMA_STEPS the steps (default: the model's).
func TestBlockRanges(t *testing.T) {
	export, root := os.Getenv("HWNPU_ANIMA_EXPORT"), os.Getenv("HWNPU_ANIMA_DIR")
	if export == "" || root == "" {
		t.Skip("HWNPU_ANIMA_EXPORT and HWNPU_ANIMA_DIR")
	}
	ctx := context.Background()
	m, err := ReadProgramManifest(export)
	if err != nil {
		t.Fatal(err)
	}
	dir := cmp.Or(os.Getenv("HWNPU_ANIMA_RANGES"), t.TempDir())
	w, err := mil.OpenWeights(root, m.Dit)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	group := mil.DitLayers / len(m.Blocks)
	qchunk := mil.DitQChunk(m.Width, m.Height, 512)
	ext := llm.ProgramExt(m.Backend)
	var paths []string
	for first := 0; first < mil.DitLayers; first += group {
		p := mil.BlockRanges(m.Width, m.Height, first, group, qchunk)
		path := filepath.Join(dir, p.Name+ext)
		paths = append(paths, path)
		if _, err := os.Stat(path); err == nil {
			continue
		}
		if _, err := llm.WritePart(path, p, w, llm.Options{Backend: m.Backend, Format: m.Formats["dit"]}, nil); err != nil {
			t.Fatal(err)
		}
	}
	// the RMS of a head of a projection for white input of unit variance
	expect := func(name string) []float64 {
		v, shape, err := w.Get(name)
		if err != nil || len(shape) != 2 {
			t.Fatalf("%s: %v %v", name, shape, err)
		}
		heads, d := 16, shape[0]/16
		out := make([]float64, heads)
		for h := range out {
			var s float64
			for _, x := range v[h*d*shape[1] : (h+1)*d*shape[1]] {
				s += float64(x) * float64(x)
			}
			out[h] = math.Sqrt(s / float64(d))
		}
		return out
	}

	dev, err := hwnpu.Open(ctx, hwnpu.WithBackend(llm.BackendCoreAI), hwnpu.WithCacheDir(hcache.Root()))
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	p, err := Open(ctx, dev, export, os.Stderr)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	pe := p.eng.(*progEngine)

	// what a block's norms took: the residual stream at its 3 LayerNorms together, and its 4
	// queries and keys, each the largest magnitude, the largest and smallest RMS, and for the
	// queries and keys the largest and smallest RMS over the weights' (ratio)
	type span struct{ abs, hi, lo, rhi, rlo float64 }
	newSpan := func() span { return span{lo: math.Inf(1), rlo: math.Inf(1)} }
	names := []string{"residual", "self q", "self k", "cross q", "cross k"}
	stats := make([][5]span, mil.DitLayers)
	// the largest product of an attention before the softmax's sum divides it (1: the
	// cross-attention's; 0: none, the self-attention divides first): what
	// milpart.HeadAttention's deferred needs within fp16
	prods := make([][2]float64, mil.DitLayers)
	for i := range stats {
		for j := range stats[i] {
			stats[i][j] = newSpan()
		}
	}
	weights := []string{"", "self_attn.q_proj", "self_attn.k_proj", "cross_attn.q_proj", "cross_attn.k_proj"}
	for g, path := range paths {
		rp, err := runner.Open(ctx, dev, path)
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		defer rp.Close()
		if !rp.OnNPU() {
			t.Fatalf("%s is not all on the NPU", path)
		}
		first := g * group
		var exp [][5][]float64
		for b := first; b < first+group; b++ {
			var e [5][]float64
			for j := 1; j < 5; j++ {
				e[j] = expect(fmt.Sprintf("dit.blocks.%d.%s.weight", b, weights[j]))
			}
			exp = append(exp, e)
		}
		note := func(y []float32) {
			if len(y) != group*mil.RangeValues {
				t.Fatalf("group %d: %d range values", g, len(y))
			}
			for b := 0; b < group; b++ {
				v := y[b*mil.RangeValues:]
				at := 0
				take := func(j, heads int) {
					s := &stats[first+b][j]
					for h := 0; h < heads; h++ {
						abs, hi, lo := float64(v[at+h]), float64(v[at+heads+h]), float64(v[at+2*heads+h])
						s.abs, s.hi, s.lo = max(s.abs, abs), max(s.hi, hi), min(s.lo, lo)
						if heads > 1 {
							s.rhi, s.rlo = max(s.rhi, hi/exp[b][j][h]), min(s.rlo, lo/exp[b][j][h])
						}
					}
					at += 3 * heads
				}
				prod := func(j int) {
					prods[first+b][j] = max(prods[first+b][j], float64(v[at]))
					at++
				}
				// BlockRanges' order
				take(0, 1)
				take(1, 16)
				take(2, 16)
				take(0, 1)
				take(3, 16)
				take(4, 16)
				prod(1)
				take(0, 1)
			}
		}
		pe.blocks[g].p = &rangeProgram{Program: pe.blocks[g].p, ranges: rp, note: note}
	}

	prompts := strings.Split(cmp.Or(os.Getenv("HWNPU_ANIMA_PROMPTS"), "1girl, solo, silver hair, red eyes, school uniform, cherry blossoms, outdoors, smile"), "|")
	steps := p.DefaultSteps()
	fmt.Sscan(os.Getenv("HWNPU_ANIMA_STEPS"), &steps)
	for i, prompt := range prompts {
		cond, err := p.Encode(ctx, prompt)
		if err != nil {
			t.Fatal(err)
		}
		guidance := p.DefaultGuidance()
		var uncond []float32
		if guidance != 1 {
			if uncond, err = p.EncodeNegative(ctx, ""); err != nil {
				t.Fatal(err)
			}
		}
		if _, err := p.Denoise(ctx, cond, uncond, guidance, p.Noise(uint64(i+1)), Sigmas(steps, 3), p.DefaultSampling(), nil); err != nil {
			t.Fatal(err)
		}
		t.Logf("drew %q", prompt)
	}

	var all [5]span
	for j := range all {
		all[j] = newSpan()
	}
	var b strings.Builder
	fmt.Fprintf(&b, "\n%-5s", "block")
	for _, n := range names {
		fmt.Fprintf(&b, " | %-34s", n+": max |x|, RMS lo..hi (ratio)")
	}
	b.WriteString(" | products: (self), cross\n")
	var allProd [2]float64
	for i, st := range stats {
		fmt.Fprintf(&b, "%-5d", i)
		for j, s := range st {
			a := &all[j]
			a.abs, a.hi, a.lo, a.rhi, a.rlo = max(a.abs, s.abs), max(a.hi, s.hi), min(a.lo, s.lo), max(a.rhi, s.rhi), min(a.rlo, s.rlo)
			if j == 0 {
				fmt.Fprintf(&b, " | %8.3g %8.3g..%-8.3g        ", s.abs, s.lo, s.hi)
			} else {
				fmt.Fprintf(&b, " | %6.3g %6.3g..%-6.3g (%4.2f..%-5.2f)", s.abs, s.lo, s.hi, s.rlo, s.rhi)
			}
		}
		fmt.Fprintf(&b, " | %8.3g %8.3g\n", prods[i][0], prods[i][1])
		allProd[0], allProd[1] = max(allProd[0], prods[i][0]), max(allProd[1], prods[i][1])
	}
	fmt.Fprintf(&b, "%-5s", "all")
	for j, s := range all {
		if j == 0 {
			fmt.Fprintf(&b, " | %8.3g %8.3g..%-8.3g        ", s.abs, s.lo, s.hi)
		} else {
			fmt.Fprintf(&b, " | %6.3g %6.3g..%-6.3g (%4.2f..%-5.2f)", s.abs, s.lo, s.hi, s.rlo, s.rhi)
		}
	}
	fmt.Fprintf(&b, " | %8.3g %8.3g\n", allProd[0], allProd[1])
	t.Log(b.String())
	if m := max(allProd[0], allProd[1]); m > 65504.0/8 {
		t.Errorf("an attention's product before the division reaches %g: under 8 times from fp16's largest", m)
	}
}
