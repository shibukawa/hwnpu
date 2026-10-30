package chat

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"io"
	"maps"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3vl"
	"github.com/shibukawa/hwnpu/internal/vision"
)

// fetch downloads a missing file, goes on from a download cut off, and leaves a file that is
// there alone.
func TestFetch(t *testing.T) {
	body := strings.Repeat("0123456789", 1000)
	var ranges []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/org/m/resolve/main/f.bin" {
			http.NotFound(w, r)
			return
		}
		ranges = append(ranges, r.Header.Get("Range"))
		http.ServeContent(w, r, "f.bin", time.Time{}, strings.NewReader(body))
	}))
	defer srv.Close()
	defer func(u string) { hubURL = u }(hubURL)
	hubURL = srv.URL + "/"

	dir := t.TempDir()
	path := filepath.Join(dir, "f.bin")
	if err := os.WriteFile(path+".part", []byte(body[:4321]), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := fetch(context.Background(), "org/m", "f.bin", path, io.Discard); err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(path)
	if string(got) != body {
		t.Errorf("got %d bytes, want the %d of the file", len(got), len(body))
	}
	if _, err := os.Stat(path + ".part"); !os.IsNotExist(err) {
		t.Error("the .part file is left")
	}
	if err := fetch(context.Background(), "org/m", "f.bin", path, io.Discard); err != nil {
		t.Fatal(err)
	}
	if len(ranges) != 1 || ranges[0] != "bytes=4321-" {
		t.Errorf("requests with ranges %q, want one for bytes=4321-", ranges)
	}
	if err := fetch(context.Background(), "org/m", "none.bin", filepath.Join(dir, "none.bin"), io.Discard); err == nil {
		t.Error("no error for a file the repository does not have")
	}
}

// Prepare converts the vision encoder of an LFM3-VL checkpoint besides its text model, and
// nothing more for a text only one.
func TestPrepareEncoders(t *testing.T) {
	vl := `{"model_type": "lfm3_vl", "tie_word_embeddings": true, "projector_hidden_act": "gelu", "projector_bias": true, "projector_hidden_size": 32,
	"text_config": {"model_type": "lfm3", ` + tinyLFM3[1:] + `,` + imageVL[strings.Index(imageVL, `"vision_config"`):]
	vcfg, err := lfm3vl.ParseConfig([]byte(vl))
	if err != nil {
		t.Fatal(err)
	}
	tcfg, err := lfm3.ParseConfig([]byte(vl))
	if err != nil {
		t.Fatal(err)
	}
	vlWeights := llm.NewSynthetic(vcfg)
	maps.Copy(vlWeights, llm.NewSynthetic(tcfg))
	text := `{"model_type": "lfm3", ` + tinyLFM3[1:]
	tcfg, err = lfm3.ParseConfig([]byte(text))
	if err != nil {
		t.Fatal(err)
	}
	files := map[string][]byte{
		"/org/vl/resolve/main/config.json":           []byte(vl),
		"/org/vl/resolve/main/tokenizer.json":        []byte(imageTokenizer),
		"/org/vl/resolve/main/model.safetensors":     safetensorsOf(t, vlWeights),
		"/org/vl/resolve/main/processor_config.json": []byte(`{"image_processor": {"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "size": {"shortest_edge": 64, "longest_edge": 4096}}}`),
		"/org/text/resolve/main/config.json":         []byte(text),
		"/org/text/resolve/main/tokenizer.json":      []byte(imageTokenizer),
		"/org/text/resolve/main/model.safetensors":   safetensorsOf(t, llm.NewSynthetic(tcfg)),
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, ok := files[r.URL.Path]
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Write(b)
	}))
	defer srv.Close()
	defer func(u string) { hubURL = u }(hubURL)
	hubURL = srv.URL + "/"
	ctx := context.Background()

	dir := filepath.Join(t.TempDir(), "vl")
	c, err := Prepare(ctx, Setup{Dir: dir, Repo: "org/vl", Context: 64, Log: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Packages) != 2 || len(c.Vision) != 1 || len(c.Audio) != 0 {
		t.Fatalf("packages %q, vision %q, audio %q", c.Packages, c.Vision, c.Audio)
	}
	if _, err := os.Stat(filepath.Join(dir, "processor_config.json")); err != nil {
		t.Error("the image processor's settings are not downloaded:", err)
	}
	vis, err := vision.Load(ctx, nil, dir, c.Vision)
	if err != nil {
		t.Fatal(err)
	}
	vis.Close()

	c, err = Prepare(ctx, Setup{Dir: filepath.Join(t.TempDir(), "text"), Repo: "org/text", Context: 64, Log: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Packages) != 2 || len(c.Vision) != 0 || len(c.Audio) != 0 {
		t.Errorf("text only: packages %q, vision %q, audio %q", c.Packages, c.Vision, c.Audio)
	}
}

// safetensorsOf writes a checkpoint of the synthetic weights in F32.
func safetensorsOf(t *testing.T, w llm.Synthetic) []byte {
	t.Helper()
	header := map[string]any{}
	var data []byte
	for _, name := range slices.Sorted(maps.Keys(w)) {
		v, shape, err := w.Get(name)
		if err != nil {
			t.Fatal(err)
		}
		start := len(data)
		for _, f := range v {
			data = binary.LittleEndian.AppendUint32(data, math.Float32bits(f))
		}
		header[name] = map[string]any{"dtype": "F32", "shape": shape, "data_offsets": []int{start, len(data)}}
	}
	h, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	out := binary.LittleEndian.AppendUint64(nil, uint64(len(h)))
	return append(append(out, h...), data...)
}
