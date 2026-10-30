package chat

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/shibukawa/hwnpu/internal/audio"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3audio"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3vl"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
	"github.com/shibukawa/hwnpu/internal/vision"
)

// a tokenizer of a and b, with the image tokens
const imageTokenizer = `{
  "added_tokens": [{"id": 40, "content": "<|vision_start|>", "special": true}, {"id": 41, "content": "<|vision_end|>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
    {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false},
    {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}]},
  "model": {"type": "BPE", "ignore_merges": false, "byte_fallback": false, "vocab": {"a": 0, "b": 1}, "merges": []}
}`

// the vision encoder of a tiny LFM3-VL whose text model is tinyLFM3 (hidden size 64)
const imageVL = `{"model_type": "lfm3_vl", "projector_hidden_act": "gelu", "projector_bias": true, "projector_hidden_size": 32,
	"text_config": {"model_type": "lfm3", "hidden_size": 64},
	"vision_config": {"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2,
		"num_key_value_heads": 1, "head_dim": 8, "patch_size": 4, "temporal_patch_size": 2, "in_channels": 3,
		"conv_kernel_size": 3, "norm_eps": 1e-5, "layer_types": ["conv", "full_attention"], "rope_parameters": {"rope_theta": 100}}}`

func pngURL(t *testing.T, shade uint8) string {
	t.Helper()
	img := image.NewNRGBA(image.Rect(0, 0, 20, 20))
	for y := 0; y < 20; y++ {
		for x := 0; x < 20; x++ {
			img.Set(x, y, color.NRGBA{uint8(x*11) + shade, uint8(y * 7), shade, 255})
		}
	}
	var b bytes.Buffer
	if err := png.Encode(&b, img); err != nil {
		t.Fatal(err)
	}
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(b.Bytes())
}

// An image becomes <|vision_start|>, a token per row of its embeddings, <|vision_end|> where its
// part was. The same image keeps its token ids, so that the cache is reused for it; another
// image gets others, and one no longer kept is encoded again with new ids.
func TestImageTokens(t *testing.T) {
	dir := t.TempDir()
	files := map[string]string{"tokenizer.json": imageTokenizer, "config.json": imageVL,
		"preprocessor_config.json": `{"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "size": {"shortest_edge": 64, "longest_edge": 4096}}`}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	vcfg, err := lfm3vl.ParseConfig([]byte(imageVL))
	if err != nil {
		t.Fatal(err)
	}
	pkg, err := llm.Convert(lfm3vl.Arch, vcfg, llm.NewSynthetic(vcfg), filepath.Join(dir, "vision.mlpackage"), "none", 0,
		llm.Options{Format: "fp16", Params: map[string]string{"image": "16x16"}}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	vis, err := vision.Load(context.Background(), nil, dir, []string{pkg})
	if err != nil {
		t.Fatal(err)
	}
	defer vis.Close()
	tok, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := lfm3.ParseConfig([]byte(tinyLFM3))
	if err != nil {
		t.Fatal(err)
	}
	e := &Engine{Tok: tok, model: cfg, emb: map[int32][]float32{}, nextID: -1, vis: vis}
	defer func(n int) { maxImages = n }(maxImages)
	maxImages = 2

	var msg Message
	if err := json.Unmarshal([]byte(`{"role": "user", "content": [{"type": "text", "text": "ab"}, {"type": "image_url", "image_url": {"url": "`+pngURL(t, 0)+`"}}, {"type": "text", "text": "ba"}]}`), &msg); err != nil {
		t.Fatal(err)
	}
	if msg.Content.Text != "ab"+MediaMarker+"ba" || len(msg.Content.Media) != 1 || msg.Content.Media[0].Audio {
		t.Fatalf("content %q, media %v", msg.Content.Text, msg.Content.Media)
	}
	a, b, c := msg.Content.Media[0], Media{Data: pngURL(t, 50)}, Media{Data: pngURL(t, 100)}
	tokens := func(media ...Media) []int32 {
		t.Helper()
		ids, err := e.Tokens(context.Background(), msg.Content.Text, media)
		if err != nil {
			t.Fatal(err)
		}
		return ids
	}
	if got, want := tokens(a), []int32{0, 1, 40, -1, -2, -3, -4, 41, 1, 0}; !slices.Equal(got, want) {
		t.Errorf("tokens %v, want %v", got, want)
	}
	rows, _, err := vis.Encode(context.Background(), mustDecode(t, a.Data))
	if err != nil {
		t.Fatal(err)
	}
	if r, ok := e.imageRow(-2); !ok || !slices.Equal(r, rows[64:128]) {
		t.Error("the second token of the image is not its second row")
	}
	if got := tokens(a); got[3] != -1 {
		t.Errorf("the same image again: %v", got)
	}
	if got := tokens(b); got[3] != -5 {
		t.Errorf("another image: %v", got)
	}
	tokens(c) // a is no longer kept
	if _, ok := e.imageRow(-1); ok {
		t.Error("the rows of an image put away are still found")
	}
	if got := tokens(a); got[3] != -13 {
		t.Errorf("an image encoded again: %v", got)
	}
	if _, err := e.Tokens(context.Background(), msg.Content.Text, nil); !errors.Is(err, ErrImage) {
		t.Errorf("a marker without its image: %v", err)
	}
	e.vis, e.images = nil, nil
	if _, err := e.Tokens(context.Background(), msg.Content.Text, []Media{a}); !errors.Is(err, ErrImage) {
		t.Errorf("an image without a vision encoder: %v", err)
	}
	if _, err := e.Tokens(context.Background(), msg.Content.Text, []Media{{Audio: true, Data: "AAAA"}}); !errors.Is(err, ErrImage) {
		t.Errorf("a clip without an audio encoder: %v", err)
	}
}

func mustDecode(t *testing.T, url string) image.Image {
	t.Helper()
	img, err := vision.DecodeDataURL(url)
	if err != nil {
		t.Fatal(err)
	}
	return img
}

// a clip of 0.1 s as base64 16 bit WAV at 16 kHz
func wavBase64() string {
	var data bytes.Buffer
	for i := 0; i < 1600; i++ {
		binary.Write(&data, binary.LittleEndian, int16(8000*math.Sin(float64(i)*0.17)))
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
	return base64.StdEncoding.EncodeToString(b.Bytes())
}

// A clip becomes its rows alone in LFM3-Audio's prompt, and between the image tokens in
// LFM3-Omni's.
func TestAudioTokens(t *testing.T) {
	const enc = `"sliding_window": 3, "%s": {"hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2,
		"intermediate_size": 32, "norm_eps": 1e-5, "conv_L_cache": 3, "layer_types": ["full_attention"], "rope_theta": 1000000.0,
		"subsampling_conv_channels": 2}`
	for _, c := range []struct {
		config string
		wrap   bool
	}{
		{`{"model_type": "lfm3_audio", "lfm": {"hidden_size": 64}, "encoder": {"dim_in": 16, ` + fmt.Sprintf(enc, "lfm") + `}}`, false},
		{`{"model_type": "lfm3_omni", "text_config": {"hidden_size": 64}, "audio_encoder_config": {"num_mel_bins": 16, ` + fmt.Sprintf(enc, "text_config") + `}}`, true},
	} {
		dir := t.TempDir()
		for name, body := range map[string]string{"tokenizer.json": imageTokenizer, "config.json": c.config} {
			if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		acfg, err := lfm3audio.ParseConfig([]byte(c.config))
		if err != nil {
			t.Fatal(err)
		}
		pkg, err := llm.Convert(lfm3audio.Arch, acfg, llm.NewSynthetic(acfg), filepath.Join(dir, "audio.mlpackage"), "none", 0,
			llm.Options{Format: "fp16", Params: map[string]string{"frames": "16"}}, io.Discard)
		if err != nil {
			t.Fatal(err)
		}
		aud, err := audio.Load(context.Background(), nil, dir, []string{pkg})
		if err != nil {
			t.Fatal(err)
		}
		tok, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
		if err != nil {
			t.Fatal(err)
		}
		cfg, _ := lfm3.ParseConfig([]byte(tinyLFM3))
		e := &Engine{Tok: tok, model: cfg, emb: map[int32][]float32{}, nextID: -1, aud: aud}
		ids, err := e.Tokens(context.Background(), "a"+MediaMarker+"b", []Media{{Audio: true, Data: wavBase64()}})
		aud.Close()
		if err != nil {
			t.Fatal(err)
		}
		want := []int32{0, -1, -2, 1} // 0.1 s: 11 frames, 2 tokens
		if c.wrap {
			want = []int32{0, 40, -1, -2, 41, 1}
		}
		if !slices.Equal(ids, want) {
			t.Errorf("omni %v: tokens %v, want %v", c.wrap, ids, want)
		}
	}
}
