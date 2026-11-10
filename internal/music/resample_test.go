package music

import (
	"context"
	"math"
	"os"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm/lfm3audio"
)

// Resample gives resampy's samples (HWNPU_RESAMPY: a 48 kHz stereo WAV's mean, HWNPU_STYLE_WAV,
// resampled to 16 kHz by resampy) and so the reference's style codes (HWNPU_MUSICCOCA_AUDIO_REF).
func TestResampleStyle(t *testing.T) {
	wavPath, ref := os.Getenv("HWNPU_STYLE_WAV"), os.Getenv("HWNPU_MUSICCOCA_AUDIO_REF")
	if wavPath == "" || ref == "" {
		t.Skip()
	}
	raw, _ := os.ReadFile(wavPath)
	ch, rate, err := lfm3audio.DecodeWAV(raw)
	if err != nil || rate != SampleRate {
		t.Fatal(err, rate)
	}
	mono := make([]float32, len(ch[0]))
	for i := range mono {
		mono[i] = (ch[0][i] + ch[1][i]) / 2
	}
	x := Resample(mono, SampleRate, ClipRate)
	want := readF32(t, os.Getenv("HWNPU_RESAMPY"), false)
	n := min(len(x), len(want))
	t.Logf("%d vs %d samples: relative error %.3g", len(x), len(want), math.Sqrt(relErr32(x[:n], want[:n])))
	p, _ := mccRef(t)
	st, err := OpenStyler(context.Background(), nil, p)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	a, _ := st.Audio(x)
	b, _ := st.Audio(want)
	t.Logf("embeddings: relative error %.3g", math.Sqrt(relErr32(a, b)))
	ca, _ := st.Tokens(a)
	cb, _ := st.Tokens(b)
	t.Logf("codes %v vs %v", ca, cb)
}
