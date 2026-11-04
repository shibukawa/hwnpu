package iron

import (
	"context"
	"math"
	"os"
	"testing"

	"github.com/shibukawa/hwnpu"
)

// A model with recurrent states (LFM3's short convolutions) goes back to a position by its rows
// and the slot of its states there (StateAt): after the sequence ran past it, within the same
// chunk and into the next, reading the same tokens again from it gives the same logits. The
// positions are one before a chunk boundary, on it, and past it. Needs an LFM3 export in
// HWNPU_IRON_RECURRENT and the NPU.
func TestStateRewindOnDevice(t *testing.T) {
	dir := os.Getenv("HWNPU_IRON_RECURRENT")
	if dir == "" {
		t.Skip("HWNPU_IRON_RECURRENT is not set")
	}
	ctx := context.Background()
	dev, err := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
	if err != nil {
		t.Skip(err)
	}
	defer dev.Close()
	m, err := Open(ctx, dev, dir, t.Logf)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if !m.Resumable() || len(m.Manifest.Checkpoints) == 0 {
		t.Fatalf("%s: no recurrent states listed (resumable %v)", dir, m.Resumable())
	}
	C := m.Manifest.Config.PrefillChunk
	toks := make([]int32, 2*C+40)
	for i := range toks {
		toks[i] = int32(1000 + (i*7919)%20000)
	}
	// the logits after each of the 24 tokens from each mark, in one run from the start
	marks := []int{C - 1, C, C + 5}
	const after = 24
	want := map[int][][]float32{}
	if _, err := m.Prompt(ctx, toks[:marks[0]], 0); err != nil {
		t.Fatal(err)
	}
	pos := marks[0]
	states := map[int][]byte{}
	for pos < marks[len(marks)-1]+after+C/2 {
		for _, k := range marks {
			if pos == k {
				if states[k], err = m.StateAt(pos); err != nil {
					t.Fatal(err)
				}
			}
		}
		logits, err := m.Step(ctx, toks[pos], pos)
		if err != nil {
			t.Fatal(err)
		}
		for _, k := range marks {
			if pos >= k && pos < k+after {
				want[k] = append(want[k], logits)
			}
		}
		pos++
	}
	for _, k := range marks {
		if err := m.SetStateAt(k, states[k]); err != nil {
			t.Fatal(err)
		}
		for i := range after {
			logits, err := m.Step(ctx, toks[k+i], k+i)
			if err != nil {
				t.Fatal(err)
			}
			if cos := cosine(logits, want[k][i]); cos < 0.9999 {
				t.Fatalf("back at %d, step %d: cosine %.6f with the first run", k, i, cos)
			}
		}
		t.Logf("back at %d: %d steps as the first run", k, after)
	}
	// a prompt chunk from a mark, across the boundary after it, as the steps
	k := marks[0]
	if err := m.SetStateAt(k, states[k]); err != nil {
		t.Fatal(err)
	}
	logits, err := m.Prompt(ctx, toks[k:k+after], k)
	if err != nil {
		t.Fatal(err)
	}
	if cos := cosine(logits, want[k][after-1]); cos < 0.999 {
		t.Fatalf("a prompt from %d across the boundary: cosine %.6f with the steps", k, cos)
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
