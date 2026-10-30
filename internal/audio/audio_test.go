package audio

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3audio"
)

const tinyAudio = `{"model_type": "lfm3_audio", "lfm": {"model_type": "lfm3", "hidden_size": 48},
	"encoder": {"dim_in": 16, "sliding_window": 3, "lfm": {"hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
		"num_key_value_heads": 2, "intermediate_size": 48, "norm_eps": 1e-5, "conv_L_cache": 3,
		"layer_types": ["conv", "full_attention"], "rope_theta": 1000000.0, "subsampling_conv_channels": 4}}}`

// writeTiny writes a tiny LFM3-Audio config and converts its audio encoder for the given clip
// lengths in frames, as hwnpu-convert does.
func writeTiny(t *testing.T, frames ...int) (string, []string) {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(tinyAudio), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := lfm3audio.ParseConfig([]byte(tinyAudio))
	if err != nil {
		t.Fatal(err)
	}
	var pkgs []string
	for _, f := range frames {
		out := filepath.Join(dir, fmt.Sprintf("audio-%d.mlpackage", f))
		path, err := llm.Convert(lfm3audio.Arch, cfg, llm.NewSynthetic(cfg), out, "none", 0,
			llm.Options{Format: "fp16", Params: map[string]string{"frames": fmt.Sprint(f)}}, io.Discard)
		if err != nil {
			t.Fatal(err)
		}
		pkgs = append(pkgs, path)
	}
	return dir, pkgs
}

func tone(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(0.3*math.Sin(2*math.Pi*440*float64(i)/16000) + 0.1*math.Sin(float64(i)*0.37))
	}
	return s
}

// A clip runs on the shortest program that holds it, and its embeddings are the same on a
// longer one; a clip longer than every program is refused.
func TestEncoder(t *testing.T) {
	dir, pkgs := writeTiny(t, 80, 40)
	e, err := Load(context.Background(), nil, dir, pkgs)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	clip := tone(16000 * 3 / 10) // 0.3 s: 31 frames, 4 tokens
	rows, n, err := e.Encode(context.Background(), clip)
	if err != nil {
		t.Fatal(err)
	}
	if n != lfm3audio.Tokens(31) || len(rows) != n*48 {
		t.Fatalf("%d tokens, %d values", n, len(rows))
	}
	long, err := Load(context.Background(), nil, dir, pkgs[:1])
	if err != nil {
		t.Fatal(err)
	}
	defer long.Close()
	rows80, _, err := long.Encode(context.Background(), clip)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range rows {
		if math.Abs(float64(v-rows80[i])) > 1e-5 {
			t.Errorf("row value %d is %g on 40 frames, %g on 80", i, v, rows80[i])
			break
		}
	}
	if _, _, err := e.Encode(context.Background(), tone(16000)); !errors.Is(err, ErrTooLong) {
		t.Errorf("a clip of 1 s on programs of 0.8 s: %v", err)
	}
}

func TestDecode(t *testing.T) {
	var data bytes.Buffer
	for i := 0; i < 160; i++ {
		binary.Write(&data, binary.LittleEndian, int16(i*100))
	}
	var b bytes.Buffer
	b.WriteString("RIFF")
	binary.Write(&b, binary.LittleEndian, uint32(36+data.Len()))
	b.WriteString("WAVEfmt ")
	for _, v := range []any{uint32(16), uint16(1), uint16(1), uint32(16000), uint32(32000), uint16(2), uint16(16)} {
		binary.Write(&b, binary.LittleEndian, v)
	}
	b.WriteString("data")
	binary.Write(&b, binary.LittleEndian, uint32(data.Len()))
	b.Write(data.Bytes())
	enc := base64.StdEncoding.EncodeToString(b.Bytes())
	for _, in := range []string{enc, "data:audio/wav;base64," + enc} {
		s, err := Decode(in)
		if err != nil || len(s) != 160 || math.Abs(float64(s[1])-100.0/32768) > 1e-6 {
			t.Errorf("decoded %d samples, %v", len(s), err)
		}
	}
	if _, err := Decode("!!"); err == nil {
		t.Error("bad base64 decoded")
	}
}
