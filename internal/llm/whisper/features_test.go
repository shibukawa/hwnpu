package whisper

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// Features gives WhisperFeatureExtractor's frames (testdata/mel.json, tools/ref/
// whisper_mel_golden.py): those of the clip and, past it, of the zeros it is padded with to 30
// seconds.
func TestFeatures(t *testing.T) {
	raw, err := os.ReadFile("testdata/mel.json")
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Samples []float32
		Frames  [][]float32
		Last    []float32
		Count   int
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	const mels = 128
	got := Features(g.Samples, mels)
	if len(got) != g.Count*mels || g.Count != 3000 {
		t.Fatalf("%d values, want %d frames of %d", len(got), g.Count, mels)
	}
	worst := 0.0
	cmp := func(f int, want []float32) {
		for m, v := range want {
			worst = math.Max(worst, math.Abs(float64(got[f*mels+m]-v)))
		}
	}
	for f, want := range g.Frames {
		cmp(f, want)
	}
	for f := len(g.Frames); f < g.Count; f++ {
		cmp(f, g.Last)
	}
	if worst > 1e-4 {
		t.Errorf("worst difference %g from the extractor's frames", worst)
	}
	t.Logf("worst difference %.2g", worst)
}
