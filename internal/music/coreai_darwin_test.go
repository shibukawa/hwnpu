package music

import (
	"context"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
)

// TestCoreAIAgainstCPU checks the Core AI programs (HWNPU_MRT_COREAI set; HWNPU_MRT_DIR, the
// model's directory: the models' magenta-realtime-2 by default) against the Core ML packages run
// with the CPU interpreter, Core ML's Neural Engine beside them: the language model's logits
// forced along the CPU's greedy frames (the KL at temperature 1.3 and the same most likely token),
// MusicCoCa's text embeddings and SpectroStream's decoder on the same input (cosines).
func TestCoreAIAgainstCPU(t *testing.T) {
	if os.Getenv("HWNPU_MRT_COREAI") == "" {
		t.Skip("HWNPU_MRT_COREAI is not set")
	}
	dir := os.Getenv("HWNPU_MRT_DIR")
	if dir == "" {
		home, _ := os.UserCacheDir()
		dir = filepath.Join(home, "hwnpu", "models", "magenta-realtime-2")
	}
	ctx := context.Background()
	setup := Setup{Dir: dir, Repo: "google/magenta-realtime-2", Text: true}
	setup.Backend = llm.BackendCoreML
	pml, err := Prepare(ctx, setup)
	if err != nil {
		t.Fatal(err)
	}
	setup.Backend = llm.BackendCoreAI
	pai, err := Prepare(ctx, setup)
	if err != nil {
		t.Fatal(err)
	}
	ai, _, err := llm.OpenDevice(ctx, llm.BackendCoreAI, "")
	if err != nil {
		t.Fatal(err)
	}
	defer ai.Close()
	ml, _, err := llm.OpenDevice(ctx, llm.BackendCoreML, "")
	if err != nil {
		t.Fatal(err)
	}
	defer ml.Close()
	type side struct {
		name string
		dev  *hwnpu.Device
		p    Paths
	}
	sides := []side{{"cpu", nil, pml}, {"coreml", ml, pml}, {"coreai", ai, pai}}

	// the language model: the CPU's greedy frames, then each device forced along them
	lw, err := OpenWeights(map[string]string{pml.Checkpoint: "params/"})
	if err != nil {
		t.Fatal(err)
	}
	defer lw.Close()
	enc, err := Encode(lw, NewConditioning([]int{418, 333, 522, 788, 451, 547, 82, 482, 202, 929, 860, 360}, Guidance{Style: 3, Notes: 5, Drums: 1}, DrumsMasked))
	if err != nil {
		t.Fatal(err)
	}
	const frames = 25
	var tokens [][]int
	var ref [][]float32
	for _, s := range sides {
		ts, err := openStep(ctx, s.dev, s.p.Temporal)
		if err != nil {
			t.Fatal(err)
		}
		ds, err := openStep(ctx, s.dev, s.p.Depth)
		if err != nil {
			t.Fatal(err)
		}
		l := newLM(lw, ts, ds)
		var kl float64
		agree := 0
		for f := range frames {
			var pick Sampler
			var forced []int
			if s.dev == nil {
				pick = func(lg []float32) int { return argmax(lg) }
			} else {
				forced = tokens[f]
			}
			tok, lg, err := l.Step(enc, pick, forced, true)
			if err != nil {
				t.Fatal(err)
			}
			if s.dev == nil {
				tokens, ref = append(tokens, tok), append(ref, append([]float32(nil), lg...))
				continue
			}
			for cb := range Codebooks {
				w, g := ref[f][cb*CodebookSize:(cb+1)*CodebookSize], lg[cb*CodebookSize:(cb+1)*CodebookSize]
				kl += klDiv(w, g, 1.3)
				if argmax(g) == argmax(w) {
					agree++
				}
			}
		}
		ts.Close()
		ds.Close()
		if s.dev != nil {
			t.Logf("LM %s (on the NPU: %v %v): mean KL %.4f nats, the same most likely token in %d of %d", s.name, ts.onNPU, ds.onNPU, kl/(frames*Codebooks), agree, frames*Codebooks)
		}
	}

	// MusicCoCa's text embeddings
	texts := []string{"disco funk", "jazz piano", "lo-fi hip hop", "heavy metal", "string quartet"}
	var embs [][]float32
	for _, s := range sides {
		st, err := OpenStyler(ctx, s.dev, s.p)
		if err != nil {
			t.Fatal(err)
		}
		worst := 1.0
		for i, text := range texts {
			e, err := st.Text(text)
			if err != nil {
				t.Fatal(err)
			}
			if s.dev == nil {
				embs = append(embs, e)
				continue
			}
			worst = math.Min(worst, cosine(embs[i], e))
		}
		st.Close()
		if s.dev != nil {
			t.Logf("MusicCoCa text %s: the lowest cosine of %d texts %.6f", s.name, len(texts), worst)
		}
	}

	// SpectroStream's decoder on embeddings of the scale its input has
	W := DecodeFrames + decodeContext
	r := rand.New(rand.NewPCG(1, 2))
	x := make([]float32, EmbeddingSize*W)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	var y0 []float32
	for _, s := range sides {
		dec, err := openStep(ctx, s.dev, s.p.Decoder)
		if err != nil {
			t.Fatal(err)
		}
		in := map[string][]float32{"x": x}
		mask := make([]float32, W)
		for i := range mask {
			mask[i] = 1
		}
		ssMasks(in, mask)
		out, err := dec.run(in)
		if err != nil {
			t.Fatal(err)
		}
		y := append([]float32(nil), out["y"]...)
		dec.Close()
		if s.dev == nil {
			y0 = y
			continue
		}
		t.Logf("decoder %s (on the NPU: %v): cosine %.6f", s.name, dec.onNPU, cosine(y0, y))
	}
}

func cosine(a, b []float32) float64 {
	var ab, aa, bb float64
	for i := range a {
		ab += float64(a[i]) * float64(b[i])
		aa += float64(a[i]) * float64(a[i])
		bb += float64(b[i]) * float64(b[i])
	}
	return ab / math.Sqrt(aa*bb)
}
