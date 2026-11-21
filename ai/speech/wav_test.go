package speech

import (
	"bytes"
	"math"
	"testing"

	"github.com/shibukawa/hwnpu/ai/flow"
)

func TestWAVAndChunks(t *testing.T) {
	const rate = 24000
	pcm := make([]float32, rate/2)
	for i := range pcm {
		pcm[i] = float32(0.5 * math.Sin(2*math.Pi*440*float64(i)/rate))
	}
	var buf bytes.Buffer
	if err := WriteWAV(&buf, pcm, rate); err != nil {
		t.Fatal(err)
	}
	same, err := ReadWAV(buf.Bytes(), rate)
	if err != nil || len(same) != len(pcm) {
		t.Fatalf("read back at the same rate: %v, %d samples, want %d", err, len(same), len(pcm))
	}
	for i := range pcm {
		if math.Abs(float64(same[i]-pcm[i])) > 1e-3 {
			t.Fatalf("sample %d is %v, want %v", i, same[i], pcm[i])
		}
	}
	// resampled for the transcriber
	at16, err := ReadWAV(buf.Bytes(), TranscribeRate)
	if err != nil || len(at16) < TranscribeRate/2-2 || len(at16) > TranscribeRate/2+2 {
		t.Fatalf("read back at 16 kHz: %v, %d samples", err, len(at16))
	}

	// a recording as a microphone's frames: the last one shorter and final
	got, err := flow.Collect(Chunks(make([]float32, 3500), 1000, 1000))
	if err != nil || len(got) != 4 {
		t.Fatalf("chunks: %v, %d", err, len(got))
	}
	if last := got[3]; len(last.Data) != 500 || !last.Final || last.Start.Seconds() != 3 || got[0].Final {
		t.Errorf("the last chunk: %d samples at %v, final %v", len(last.Data), last.Start, last.Final)
	}
}
