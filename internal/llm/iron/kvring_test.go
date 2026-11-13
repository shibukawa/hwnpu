package iron

import (
	"context"
	"os"
	"slices"
	"testing"

	"github.com/shibukawa/hwnpu"
)

// A decode step whose sliding layers keep a ring of their window (rows.go's kvJoin.ring) goes
// back to a position as one with a row a position: after the sequence ran past it by more than
// the ring, the steps from it again give the same logits, bit for bit (the ring is filled from
// the keys and values, each position in its own slot), and so do the steps after the rows are
// exported, the sequence run on, and the rows imported again. Needs a gpt-oss export in
// HWNPU_IRON_KVRING and the NPU.
func TestKVRingRewindOnDevice(t *testing.T) {
	dir := os.Getenv("HWNPU_IRON_KVRING")
	if dir == "" {
		t.Skip("HWNPU_IRON_KVRING is not set")
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
	if !m.rings {
		t.Fatalf("%s: no ring among the decode step's arrays", dir)
	}
	const prompt, last, after = 40, 460, 12
	marks := []int{100, 215, 300} // before the ring is full of the steps' rows, and after
	toks := make([]int32, last+after)
	for i := range toks {
		toks[i] = int32(1000 + (i*7919)%20000)
	}
	if _, err := m.Prompt(ctx, toks[:prompt], 0); err != nil {
		t.Fatal(err)
	}
	want := map[int][][]float32{}
	var rows []byte
	for pos := prompt; pos < last; pos++ {
		if pos == marks[2] {
			if rows, err = m.ExportRows(0, pos); err != nil {
				t.Fatal(err)
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
	}
	again := func(k int, what string) {
		for i := range after {
			logits, err := m.Step(ctx, toks[k+i], k+i)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(logits, want[k][i]) {
				t.Fatalf("%s at %d, step %d: cosine %.6f with the first run", what, k, i, cosine(logits, want[k][i]))
			}
		}
		t.Logf("%s at %d: %d steps as the first run", what, k, after)
	}
	for _, k := range marks {
		again(k, "back")
	}
	// the rows as they were exported (before the steps from marks[2]), after other steps
	again(marks[0], "back")
	if err := m.ImportRows(0, rows); err != nil {
		t.Fatal(err)
	}
	again(marks[2], "the rows imported")
	// a prompt chunk after steps reads their rows, and the steps after it the chunk's
	if _, err := m.Prompt(ctx, toks[marks[2]+after:marks[2]+after+20], marks[2]+after); err != nil {
		t.Fatal(err)
	}
	a, err := m.Step(ctx, toks[marks[2]+after+20], marks[2]+after+20)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.Step(ctx, toks[marks[0]], marks[0]); err != nil { // elsewhere, then back
		t.Fatal(err)
	}
	b, err := m.Step(ctx, toks[marks[2]+after+20], marks[2]+after+20)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(a, b) {
		t.Fatalf("a step after a prompt chunk, again after a step elsewhere: cosine %.6f", cosine(a, b))
	}
}
