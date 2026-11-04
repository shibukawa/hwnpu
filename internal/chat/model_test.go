package chat

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
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
	c, err := Prepare(ctx, Setup{Dir: dir, Repo: "org/vl", Context: 64, Backend: "coreml", Log: io.Discard})
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

	c, err = Prepare(ctx, Setup{Dir: filepath.Join(t.TempDir(), "text"), Repo: "org/text", Context: 64, Backend: "coreml", Log: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Packages) != 2 || len(c.Vision) != 0 || len(c.Audio) != 0 {
		t.Errorf("text only: packages %q, vision %q, audio %q", c.Packages, c.Vision, c.Audio)
	}

	// for Intel's NPU: OpenVINO IR in intel/, each with its list of parts, the vision encoder too
	dir = filepath.Join(t.TempDir(), "vl")
	c, err = Prepare(ctx, Setup{Dir: dir, Repo: "org/vl", Context: 64, Backend: "intel", Log: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Packages) != 2 || len(c.Vision) != 1 || c.Backend != "intel" {
		t.Fatalf("intel: packages %q, vision %q, backend %q", c.Packages, c.Vision, c.Backend)
	}
	for _, p := range append(c.Packages, c.Vision...) {
		if filepath.Dir(p) != filepath.Join(dir, "intel") || !strings.HasSuffix(p, ".parts.json") {
			t.Errorf("intel: %s", p)
		}
		man, err := llm.ReadManifest(p)
		if err != nil {
			t.Fatal(err)
		}
		if man.Backend != "intel" || !strings.HasSuffix(man.Parts[0].Path, ".xml") {
			t.Errorf("intel: %s: backend %q, parts %v", p, man.Backend, man.Parts)
		}
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

func TestTiersAuto(t *testing.T) {
	if got, err := Tiers("auto"); err != nil || !slices.Equal(got, []int{TierAuto}) {
		t.Errorf("Tiers(auto) = %v, %v", got, err)
	}
	if got, err := Tiers(" auto "); err != nil || !slices.Equal(got, []int{TierAuto}) {
		t.Errorf("Tiers( auto ) = %v, %v", got, err)
	}
	for format, want := range map[string][]int{"lut4": {4096}, "lut6": nil, "lut8": nil, "fp16": nil} {
		if got := autoTiers(format, "coreml"); !slices.Equal(got, want) {
			t.Errorf("autoTiers(%s, coreml) = %v, want %v", format, got, want)
		}
	}
	for _, format := range []string{"int4", "int8"} {
		if got := autoTiers(format, llm.BackendIntel); !slices.Equal(got, []int{4096}) {
			t.Errorf("autoTiers(%s, intel) = %v, want 4096", format, got)
		}
	}
}

func TestTiersNone(t *testing.T) {
	for _, s := range []string{"", "none", " none "} {
		if got, err := Tiers(s); err != nil || got != nil {
			t.Errorf("Tiers(%q) = %v, %v", s, got, err)
		}
	}
	if got, err := Tiers("2048, 4096"); err != nil || len(got) != 2 || got[0] != 2048 || got[1] != 4096 {
		t.Errorf("Tiers(\"2048, 4096\") = %v, %v", got, err)
	}
	if _, err := Tiers("0"); err == nil {
		t.Error("Tiers(\"0\") is accepted")
	}
}

// Prepare converts a model that can be split at its attention that way, with the longest context
// it was trained for and no tiers, its chunks no longer than the context, for the Neural Engine
// and for Intel's NPU (in IR); asked not to, it converts it whole with the context asked for.
func TestPreparePaged(t *testing.T) {
	text := `{"model_type": "lfm3", "max_position_embeddings": 96, ` + tinyLFM3[1:]
	tcfg, err := lfm3.ParseConfig([]byte(text))
	if err != nil {
		t.Fatal(err)
	}
	files := map[string][]byte{
		"/org/text/resolve/main/config.json":       []byte(text),
		"/org/text/resolve/main/tokenizer.json":    []byte(imageTokenizer),
		"/org/text/resolve/main/model.safetensors": safetensorsOf(t, llm.NewSynthetic(tcfg)),
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
	dir := filepath.Join(t.TempDir(), "text")
	c, err := Prepare(ctx, Setup{Dir: dir, Repo: "org/text", Draft: 8, Tiers: []int{TierAuto}, Backend: "coreml", Format: "lut4", Log: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Tiers) != 1 || len(c.Packages) != 2 {
		t.Fatalf("tiers %q, packages %q", c.Tiers, c.Packages)
	}
	for i, q := range []int{8, 64} {
		p := c.Packages[i]
		if want := fmt.Sprintf("lfm3-lut4-grouped-p96-c96-q%d.parts.json", q); filepath.Base(p) != want {
			t.Errorf("package %s, want %s", filepath.Base(p), want)
		}
		man, err := llm.ReadManifest(p)
		if err != nil {
			t.Fatal(err)
		}
		if man.Paged != 96 || man.Cache != 96 || man.Seq != q || man.Attention == nil || len(man.Parts) != 2 {
			t.Errorf("%s: paged %d, cache %d, seq %d, %d parts", p, man.Paged, man.Cache, man.Seq, len(man.Parts))
		}
	}
	c, err = Prepare(ctx, Setup{Dir: dir, Repo: "org/text", Context: 64, Paged: PagedOff, Backend: "coreml", Format: "lut4", Log: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	if want := "lfm3-lut4-grouped-c64-q1"; !strings.HasPrefix(filepath.Base(c.Packages[0]), want) {
		t.Errorf("not split: %s, want %s", c.Packages[0], want)
	}
	c, err = Prepare(ctx, Setup{Dir: dir, Repo: "org/text", Backend: "intel", Log: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	if want := "lfm3-int8-grouped-p96-c96-q1.parts.json"; filepath.Base(c.Packages[0]) != want {
		t.Errorf("intel: %s, want %s", filepath.Base(c.Packages[0]), want)
	}
	man, err := llm.ReadManifest(c.Packages[0])
	if err != nil || man.Paged != 96 || man.Backend != llm.BackendIntel || man.Attention == nil || filepath.Ext(man.Attention.Path) != ".xml" {
		t.Errorf("intel: %v, %+v", err, man)
	}
}

// Prepare converts the program that reads prompts beside the one that generates unless the two
// would overflow the Neural Engine's compile cache (here made smaller than the tiny model's
// weights) or -prefill says otherwise; then the one that generates reads the prompts.
func TestPreparePrefill(t *testing.T) {
	text := `{"model_type": "lfm3", "max_position_embeddings": 96, ` + tinyLFM3[1:]
	tcfg, err := lfm3.ParseConfig([]byte(text))
	if err != nil {
		t.Fatal(err)
	}
	files := map[string][]byte{
		"/org/text/resolve/main/config.json":       []byte(text),
		"/org/text/resolve/main/tokenizer.json":    []byte(imageTokenizer),
		"/org/text/resolve/main/model.safetensors": safetensorsOf(t, llm.NewSynthetic(tcfg)),
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
	defer func(n int64) { compileCacheBytes = n }(compileCacheBytes)
	ctx := context.Background()
	dir := filepath.Join(t.TempDir(), "text")
	for _, c := range []struct {
		prefill string
		cache   int64
		want    []string
	}{
		{"", 4 << 30, []string{"q8", "q64"}},
		{PrefillAuto, 1024, []string{"q8"}},
		{PrefillOn, 1024, []string{"q8", "q64"}},
		{PrefillOff, 4 << 30, []string{"q8"}},
	} {
		compileCacheBytes = c.cache
		cfg, err := Prepare(ctx, Setup{Dir: dir, Repo: "org/text", Draft: 8, Prefill: c.prefill, Backend: "coreml", Format: "lut4", Log: io.Discard})
		if err != nil {
			t.Fatal(err)
		}
		var got []string
		for _, p := range cfg.Packages {
			got = append(got, strings.TrimSuffix(strings.TrimPrefix(filepath.Base(p), "lfm3-lut4-grouped-p96-c96-"), ".parts.json"))
		}
		if !slices.Equal(got, c.want) {
			t.Errorf("-prefill %q with a compile cache of %d: programs %v, want %v", c.prefill, c.cache, got, c.want)
		}
	}
	if _, err := Prepare(ctx, Setup{Dir: dir, Repo: "org/text", Draft: 8, Prefill: "sometimes", Backend: "coreml", Log: io.Discard}); err == nil {
		t.Error("-prefill sometimes is accepted")
	}
}
