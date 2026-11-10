package tflite

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/onnx"
)

// MusicCoCa's audio preprocessor and music encoder on the host give the reference's features
// and embedding (HWNPU_MUSICCOCA_AUDIO_REF, mcc_audio_ref.py's prefix).
func TestMusicCoCaAudio(t *testing.T) {
	dir, ref := os.Getenv("HWNPU_MUSICCOCA"), os.Getenv("HWNPU_MUSICCOCA_AUDIO_REF")
	if dir == "" || ref == "" {
		t.Skip("HWNPU_MUSICCOCA and HWNPU_MUSICCOCA_AUDIO_REF are not set")
	}
	var meta struct {
		Tokens []int `json:"tokens"`
		Shape  []int `json:"feat_shape"`
	}
	raw, _ := os.ReadFile(ref + ".json")
	json.Unmarshal(raw, &meta)
	prep, err := Load(filepath.Join(dir, "audio_preprocessor.tflite"))
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("prep: %v -> %v, %d nodes", prep.Inputs, prep.Outputs, len(prep.Graph.Nodes))
	clip := readBin[float32](t, ref+".clip.bin")
	t0 := time.Now()
	feat := run(t, prep, tensor(clip, onnx.Float, 1, len(clip)))
	t.Logf("prep: %.1f s", time.Since(t0).Seconds())
	if e := relErr(feat, readBin[float32](t, ref+".feat.bin")); e > 1e-4 {
		t.Errorf("features: relative error %g", e)
	} else {
		t.Logf("features: relative error %.3g", e)
	}
	enc, err := Load(filepath.Join(dir, "music_encoder.tflite"))
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("enc: %v -> %v, %d nodes", enc.Inputs, enc.Outputs, len(enc.Graph.Nodes))
	f32 := make([]float32, len(feat))
	for i, v := range feat {
		f32[i] = float32(v)
	}
	t0 = time.Now()
	emb := run(t, enc, tensor(f32, onnx.Float, meta.Shape...))
	t.Logf("encoder: %.1f s", time.Since(t0).Seconds())
	if e := relErr(emb, readBin[float32](t, ref+".emb.bin")); e > 1e-4 {
		t.Errorf("embedding: relative error %g", e)
	} else {
		t.Logf("embedding: relative error %.3g", e)
	}
}
