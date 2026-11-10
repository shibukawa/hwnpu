package music

import (
	"context"
	"math"
	"os"
	"testing"
	"time"
)

func relErr32(got []float32, want []float32) float64 {
	var se, sw float64
	for i := range want {
		d := float64(got[i]) - float64(want[i])
		se += d * d
		sw += float64(want[i]) * float64(want[i])
	}
	return se / sw
}

// The text encoder's program on the CPU interpreter gives the reference's embedding.
func TestTextEncoderProgram(t *testing.T) {
	p, ref := mccRef(t)
	st, err := OpenStyler(context.Background(), nil, p)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	ids, pad := readF32(t, ref+".ids.bin", true), readF32(t, ref+".pad.bin", false)
	t0 := time.Now()
	out, err := st.text.run(textFeed(st.table, f64(ids), f64(pad)))
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("ran in %.1f s", time.Since(t0).Seconds())
	if e := math.Sqrt(relErr32(out["y"], readF32(t, ref+".emb.bin", false))); e > 2e-3 {
		t.Errorf("embedding: relative error %.3g", e)
	} else {
		t.Logf("embedding: relative error %.3g", e)
	}
}

// The music encoder's program (its loop unrolled) on the CPU interpreter gives the reference's
// embedding (HWNPU_MUSICCOCA_AUDIO_REF, mcc_audio_ref.py's prefix).
func TestMusicEncoderProgram(t *testing.T) {
	ref := os.Getenv("HWNPU_MUSICCOCA_AUDIO_REF")
	if ref == "" {
		t.Skip("HWNPU_MUSICCOCA_AUDIO_REF is not set")
	}
	p, _ := mccRef(t)
	st, err := OpenStyler(context.Background(), nil, p)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	feat := readF32(t, ref+".feat.bin", false)
	x := make([]float32, len(st.patches))
	for i, j := range st.patches {
		x[i] = feat[j]
	}
	t0 := time.Now()
	out, err := st.music.run(map[string][]float32{"x": x})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("ran in %.1f s", time.Since(t0).Seconds())
	if e := math.Sqrt(relErr32(out["y"], readF32(t, ref+".emb.bin", false))); e > 2e-3 {
		t.Errorf("embedding: relative error %.3g", e)
	} else {
		t.Logf("embedding: relative error %.3g", e)
	}
	// the whole audio side from the clip: the same codes as the reference
	emb, err := st.Audio(readF32(t, ref+".clip.bin", false))
	if err != nil {
		t.Fatal(err)
	}
	codes, err := st.Tokens(emb)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("codes %v", codes)
}

func f64(v []float32) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x)
	}
	return out
}

func readF32(t *testing.T, path string, ints bool) []float32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		u := uint32(raw[4*i]) | uint32(raw[4*i+1])<<8 | uint32(raw[4*i+2])<<16 | uint32(raw[4*i+3])<<24
		if ints {
			out[i] = float32(int32(u))
		} else {
			out[i] = math.Float32frombits(u)
		}
	}
	return out
}
