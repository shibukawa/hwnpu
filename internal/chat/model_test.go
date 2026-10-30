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
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3vl"
	"github.com/shibukawa/hwnpu/internal/llm/ovrun"
	"github.com/shibukawa/hwnpu/internal/ovcpu"
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

// With the intel backend Prepare writes OpenVINO IR in the model's intel directory, in Intel's
// default format and without the image encoder, and the programs step through a prompt on
// OpenVINO's CPU plugin (needs HWNPU_OV_PYTHON for that) as the Core ML ones do on the CPU
// interpreter.
func TestPrepareIntel(t *testing.T) {
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
	weights := llm.NewSynthetic(vcfg)
	maps.Copy(weights, llm.NewSynthetic(tcfg))
	files := map[string][]byte{
		"/org/vl/resolve/main/config.json":       []byte(vl),
		"/org/vl/resolve/main/tokenizer.json":    []byte(imageTokenizer),
		"/org/vl/resolve/main/model.safetensors": safetensorsOf(t, weights),
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

	dir := filepath.Join(t.TempDir(), "vl")
	c, err := Prepare(context.Background(), Setup{Dir: dir, Repo: "org/vl", Context: 16, Draft: 4, Backend: "intel", Log: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Packages) != 2 || len(c.Vision) != 0 || c.Backend != "intel" {
		t.Fatalf("packages %q, vision %q, backend %q", c.Packages, c.Vision, c.Backend)
	}
	for _, p := range c.Packages {
		if filepath.Dir(p) != filepath.Join(dir, "intel") || llm.ProgramBackend(p) != "intel" {
			t.Errorf("%s: not an Intel program in %s", p, filepath.Join(dir, "intel"))
		}
		man, err := llm.ReadManifest(p)
		if err != nil {
			t.Fatal(err)
		}
		if f := man.Meta[llm.MetaFormat]; f != "int8" {
			t.Errorf("%s: format %q, want int8", p, f)
		}
	}
	py := ovcpu.Python()
	if py == "" {
		t.Skip("HWNPU_OV_PYTHON is not set")
	}
	var progs []decode.Program
	for _, p := range c.Packages {
		prog, err := ovrun.Program(py, p, ovcpu.Options{})
		if err != nil {
			t.Fatal(err)
		}
		progs = append(progs, prog)
	}
	for name := range progs[1].In {
		if strings.HasPrefix(name, "kv_") || strings.HasPrefix(name, "win_") {
			progs[1].In[name] = progs[0].In[name] // one cache, as chain.Share binds it
		}
	}
	s, err := decode.New(progs[0], tcfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Add(progs[1]); err != nil {
		t.Fatal(err)
	}
	w := llm.NewSynthetic(tcfg)
	emb, err := llm.Embed(w, tcfg, []int32{3, 1, 4, 1, 5, 9})
	if err != nil {
		t.Fatal(err)
	}
	var rows [][]float32
	for i := 0; i < 6; i++ {
		rows = append(rows, emb[i*tcfg.HiddenSize:(i+1)*tcfg.HiddenSize])
	}
	logits, err := s.Step(rows)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("logit %v", v)
		}
	}
}
