package engine

import (
	"bytes"
	"math"
	"slices"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm/lfm3audio"
	isp "github.com/shibukawa/hwnpu/internal/speech"
	"github.com/shibukawa/hwnpu/internal/speechwire"
)

// TestSpeechwireWAV checks that speechwire's copies read and write WAV files as the engine's
// packages do.
func TestSpeechwireWAV(t *testing.T) {
	const rate = 24000
	pcm := make([]float32, rate/3)
	for i := range pcm {
		pcm[i] = float32(0.7 * math.Sin(2*math.Pi*330*float64(i)/rate))
	}
	var a, b bytes.Buffer
	if err := speechwire.WriteWAV(&a, pcm, rate); err != nil {
		t.Fatal(err)
	}
	if err := isp.WriteWAV(&b, pcm, rate); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a.Bytes(), b.Bytes()) {
		t.Fatal("speechwire.WriteWAV writes otherwise than internal/speech's")
	}
	for _, to := range []int{rate, 16000, 44100} {
		x, err1 := speechwire.ReadWAVAt(a.Bytes(), to)
		y, err2 := lfm3audio.ReadWAVAt(a.Bytes(), to)
		if err1 != nil || err2 != nil || !slices.Equal(x, y) {
			t.Errorf("at %d: speechwire.ReadWAVAt (%v) reads otherwise than lfm3audio's (%v)", to, err1, err2)
		}
	}
	if _, err := speechwire.ReadWAVAt([]byte("not a wav"), rate); err == nil {
		t.Error("read what is not a WAV file")
	}
}
