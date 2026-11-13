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
	"github.com/shibukawa/hwnpu/internal/palette"
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
	// what an M1's Neural Engine holds beside the model's programs: the vision encoder's weights,
	// in fp16 as converted (a real checkpoint's estimates match the files: Qwen3.5-2B's
	// multi-token prediction layer in lut6 0.400 GiB, as on the disk)
	_, vm, err := llm.LoadConfig(c.ModelConfig, "lfm3-vision")
	if err != nil {
		t.Fatal(err)
	}
	pre, layers, post := estimate(vm, "fp16")
	want := pre + post
	for _, l := range layers {
		want += l
	}
	if n := besideBytes(c.ModelConfig, "", "lut4", "coreml"); n != want || n == 0 {
		t.Errorf("the programs beside the model are estimated at %d bytes, the vision encoder's in fp16 are %d", n, want)
	}

	c, err = Prepare(ctx, Setup{Dir: filepath.Join(t.TempDir(), "text"), Repo: "org/text", Context: 64, Backend: "coreml", Log: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	if n := besideBytes(c.ModelConfig, "", "", "coreml"); n != 0 {
		t.Errorf("text only: %d bytes of programs beside the model", n)
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
	// the program that reads prompts takes a chunk (at most 128) per call and computes the logits
	// after the last token alone
	for i, name := range []string{"q8", "q96-last"} {
		q := []int{8, 96}[i]
		p := c.Packages[i]
		if want := fmt.Sprintf("lfm3-r1-lut4-grouped-p96-c96-%s.parts.json", name); filepath.Base(p) != want {
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
	if want := "lfm3-r1-lut4-grouped-c64-q1"; !strings.HasPrefix(filepath.Base(c.Packages[0]), want) {
		t.Errorf("not split: %s, want %s", c.Packages[0], want)
	}
	c, err = Prepare(ctx, Setup{Dir: dir, Repo: "org/text", Backend: "intel", Log: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	if want := "lfm3-r1-int8-grouped-p96-c96-q1.parts.json"; filepath.Base(c.Packages[0]) != want {
		t.Errorf("intel: %s, want %s", filepath.Base(c.Packages[0]), want)
	}
	man, err := llm.ReadManifest(c.Packages[0])
	if err != nil || man.Paged != 96 || man.Backend != llm.BackendIntel || man.Attention == nil || filepath.Ext(man.Attention.Path) != ".xml" {
		t.Errorf("intel: %v, %+v", err, man)
	}
}

// With -codes, Prepare converts programs of their own (named by the codes) whose lutN weights
// take the chosen tables, and leaves the programs without codes as they were; codes are refused
// for a format that is not a lookup table.
func TestPrepareCodes(t *testing.T) {
	text := `{"model_type": "lfm3", "max_position_embeddings": 96, ` + tinyLFM3[1:]
	tcfg, err := lfm3.ParseConfig([]byte(text))
	if err != nil {
		t.Fatal(err)
	}
	syn := llm.NewSynthetic(tcfg)
	files := map[string][]byte{
		"/org/text/resolve/main/config.json":       []byte(text),
		"/org/text/resolve/main/tokenizer.json":    []byte(imageTokenizer),
		"/org/text/resolve/main/model.safetensors": safetensorsOf(t, syn),
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
	// codes for every feed-forward weight: each row's entry i % 16 of a table of 16 rows
	codes := t.TempDir()
	index := map[string]any{}
	for _, name := range slices.Sorted(maps.Keys(syn)) {
		v, shape, err := syn.Get(name)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(name, "feed_forward.w") || len(shape) != 2 || shape[0]%16 != 0 {
			continue
		}
		key := palette.WeightKey(v, shape[0], shape[1])
		idx := make([]byte, len(v))
		for i := range idx {
			idx[i] = byte(i % 16)
		}
		tab := make([]byte, shape[0]/16*16*4)
		for i := range shape[0] / 16 * 16 {
			binary.LittleEndian.PutUint32(tab[4*i:], math.Float32bits(float32(i%16)/100-0.08))
		}
		os.WriteFile(filepath.Join(codes, key+".idx"), idx, 0o644)
		os.WriteFile(filepath.Join(codes, key+".tab"), tab, 0o644)
		index[key] = map[string]any{"name": name, "fmt": "lut4", "shape": shape, "group": 16}
	}
	b, _ := json.Marshal(index)
	os.WriteFile(filepath.Join(codes, "index.json"), b, 0o644)
	if len(index) == 0 {
		t.Fatal("no feed-forward weights")
	}
	ctx := context.Background()
	dir := filepath.Join(t.TempDir(), "text")
	var log strings.Builder
	c, err := Prepare(ctx, Setup{Dir: dir, Repo: "org/text", Draft: 8, Backend: "coreml", Format: "lut4", Codes: codes, Log: &log})
	if err != nil {
		t.Fatal(err)
	}
	h, err := codesHash(codes)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range c.Packages {
		if !strings.HasPrefix(filepath.Base(p), "lfm3-r1-lut4-grouped-gptq-"+h+"-") {
			t.Errorf("package %s is not named by the codes (%s)", filepath.Base(p), h)
		}
	}
	if want := fmt.Sprintf("-codes: %d weights took the chosen tables", len(index)); strings.Count(log.String(), want) != len(c.Packages) {
		t.Errorf("log %q, want %q for each of %d programs", log.String(), want, len(c.Packages))
	}
	plain, err := Prepare(ctx, Setup{Dir: dir, Repo: "org/text", Draft: 8, Backend: "coreml", Format: "lut4", Log: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	if slices.Equal(plain.Packages, c.Packages) || strings.Contains(plain.Packages[0], "gptq") {
		t.Errorf("without codes: %q, with: %q", plain.Packages, c.Packages)
	}
	if _, err := Prepare(ctx, Setup{Dir: dir, Repo: "org/text", Backend: "coreml", Format: "fp16", Codes: codes, Log: io.Discard}); err == nil {
		t.Error("codes are taken for fp16")
	}
	// -format-for: programs of their own, with the rule's tensors in its format
	ff, err := Prepare(ctx, Setup{Dir: dir, Repo: "org/text", Draft: 8, Backend: "coreml", Format: "lut4", Codes: codes, FormatFor: []string{"self_attn=lut8"}, Log: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	if want := "lfm3-r1-lut4-grouped-gptq-" + h + "-self_attn.lut8-"; !strings.HasPrefix(filepath.Base(ff.Packages[0]), want) {
		t.Errorf("package %s, want %s...", filepath.Base(ff.Packages[0]), want)
	}
	bytes := func(p string) (n int64) {
		man, err := llm.ReadManifest(p)
		if err != nil {
			t.Fatal(err)
		}
		for _, part := range man.Parts {
			n += part.WeightBytes
		}
		return n
	}
	if a, b := bytes(ff.Packages[0]), bytes(c.Packages[0]); a <= b {
		t.Errorf("with the attention in lut8 the weights are %d bytes, not more than lut4's %d", a, b)
	}
	if _, err := Prepare(ctx, Setup{Dir: dir, Repo: "org/text", Backend: "coreml", Format: "lut4", FormatFor: []string{"lut8"}, Log: io.Discard}); err == nil {
		t.Error("-format-for lut8 (no SUBSTRING=) is taken")
	}
}

// Prepare converts the program that reads prompts beside the one that generates unless the two
// would overflow the Neural Engine's compile cache (here made smaller than the tiny model's
// weights), hold more than an M1's Neural Engine holds (neuralEngineHold, likewise), or -prefill
// says otherwise; then the one that generates reads the prompts.
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
	defer func(f func() int64) { neuralEngineHold = f }(neuralEngineHold)
	ctx := context.Background()
	dir := filepath.Join(t.TempDir(), "text")
	for _, c := range []struct {
		prefill     string
		cache, hold int64
		want        []string
	}{
		{"", 4 << 30, 0, []string{"q8", "q96-last"}},
		{PrefillAuto, 1024, 0, []string{"q8"}},
		{PrefillOn, 1024, 0, []string{"q8", "q96-last"}},
		{PrefillOff, 4 << 30, 0, []string{"q8"}},
		{PrefillAuto, 4 << 30, m1HoldBytes, []string{"q8", "q96-last"}}, // an M1 holds the tiny model
		{PrefillAuto, 4 << 30, 1024, []string{"q8"}},                    // an M1 that would not
		{PrefillOn, 4 << 30, 1024, []string{"q8", "q96-last"}},
	} {
		compileCacheBytes = c.cache
		neuralEngineHold = func() int64 { return c.hold }
		var log strings.Builder
		cfg, err := Prepare(ctx, Setup{Dir: dir, Repo: "org/text", Draft: 8, Prefill: c.prefill, Backend: "coreml", Format: "lut4", Log: &log})
		if err != nil {
			t.Fatal(err)
		}
		if slow := strings.Contains(log.String(), "expect it to be slow"); slow != (c.hold == 1024) {
			t.Errorf("-prefill %q, an M1 holding %d: warned of the slowdown %v, log:\n%s", c.prefill, c.hold, slow, log.String())
		}
		var got []string
		for _, p := range cfg.Packages {
			got = append(got, strings.TrimSuffix(strings.TrimPrefix(filepath.Base(p), "lfm3-r1-lut4-grouped-p96-c96-"), ".parts.json"))
		}
		if !slices.Equal(got, c.want) {
			t.Errorf("-prefill %q with a compile cache of %d and an M1 holding %d: programs %v, want %v", c.prefill, c.cache, c.hold, got, c.want)
		}
	}
	if _, err := Prepare(ctx, Setup{Dir: dir, Repo: "org/text", Draft: 8, Prefill: "sometimes", Backend: "coreml", Log: io.Discard}); err == nil {
		t.Error("-prefill sometimes is accepted")
	}
}

// onePrograms counts the programs loaded beside the model's (the multi-token prediction layer's,
// the encoders') against an M1's Neural Engine, and not against the compile cache of another chip.
func TestOneProgramsBeside(t *testing.T) {
	cfg, err := lfm3.ParseConfig([]byte(`{"model_type": "lfm3", "max_position_embeddings": 96, ` + tinyLFM3[1:]))
	if err != nil {
		t.Fatal(err)
	}
	var model llm.Model = cfg
	gen, read := programBytes(model, "lut4", false)
	if gen <= 0 || read <= 0 {
		t.Fatalf("the programs' weights are %d and %d bytes", gen, read)
	}
	defer func(n int64) { compileCacheBytes = n }(compileCacheBytes)
	defer func(f func() int64) { neuralEngineHold = f }(neuralEngineHold)
	compileCacheBytes = gen + read
	for _, c := range []struct {
		hold, others int64
		one          bool
	}{
		{0, 0, false},
		{0, 1 << 40, false}, // no limit on what the Neural Engine holds: the compile cache alone
		{gen + read, 0, false},
		{gen + read, 1, true},
		{gen + read - 1, 0, true},
	} {
		neuralEngineHold = func() int64 { return c.hold }
		if one, why := onePrograms(model, "lut4", "coreml", PrefillAuto, 8, 1, false, false, c.others); one != c.one {
			t.Errorf("holding %d with %d beside: one program %v (%s), want %v", c.hold, c.others, one, why, c.one)
		}
	}
}

// On Core AI the program that reads prompts shares the weights of the one that generates on the
// Neural Engine (sharesWeights): an M1 that holds too little for two programs on Core ML keeps both,
// and what they hold is counted once.
func TestOneProgramsShared(t *testing.T) {
	cfg, err := lfm3.ParseConfig([]byte(`{"model_type": "lfm3", "max_position_embeddings": 96, ` + tinyLFM3[1:]))
	if err != nil {
		t.Fatal(err)
	}
	var model llm.Model = cfg
	gen, read := programBytes(model, "lut4", false)
	defer func(n int64) { compileCacheBytes = n }(compileCacheBytes)
	defer func(f func() int64) { neuralEngineHold = f }(neuralEngineHold)
	compileCacheBytes = gen // over the compile cache too
	neuralEngineHold = func() int64 { return gen + read - 1 }
	if one, _ := onePrograms(model, "lut4", "coreml", PrefillAuto, 8, 2, false, false, 0); !one {
		t.Error("Core ML: two tiers of two programs over an M1's hold and the compile cache, want one program")
	}
	if one, why := onePrograms(model, "lut4", "coreml", PrefillAuto, 8, 2, false, true, 0); one {
		t.Errorf("Core AI: one program (%s), want both: they share their weights", why)
	}
	if one, _ := onePrograms(model, "lut4", "coreml", PrefillOff, 8, 1, false, true, 0); !one {
		t.Error("Core AI with -prefill off: want one program")
	}
	if h := heldBytes(gen, read, 2, false, true); h != gen {
		t.Errorf("Core AI holds %d, want the weights once (%d)", h, gen)
	}
	if h := heldBytes(gen, read, 2, false, false); h != 2*(gen+read) {
		t.Errorf("Core ML holds %d, want %d", h, 2*(gen+read))
	}
	if !sharesWeights("coreai") || sharesWeights("coreml") || sharesWeights("") {
		t.Error("only Core AI shares the weights")
	}
}

// besideBytes counts a checkpoint's multi-token prediction layer, which the drafts load beside the
// model, and nothing for a checkpoint without one.
func TestBesideBytesMTP(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(tinyQwen35MTP), 0o644); err != nil {
		t.Fatal(err)
	}
	if n := besideBytes(path, "qwen35-mtp", "lut8", "coreml"); n <= 0 {
		t.Errorf("the multi-token prediction layer is estimated at %d bytes", n)
	}
	if n := besideBytes(path, "", "lut8", "coreml"); n != 0 {
		t.Errorf("without drafts from it: %d bytes", n)
	}
}

func TestGB(t *testing.T) {
	for _, c := range []struct {
		n, total int64
		want     string
	}{{2110, -1, "2.11 kB"}, {470e6, 3.46e9, "0.47 / 3.46 GB"}, {111.23e6, 253.81e6, "111.23 / 253.81 MB"}, {3.46e9, -1, "3.46 GB"}} {
		if got := gb(c.n, c.total); got != c.want {
			t.Errorf("gb(%d, %d) = %q, want %q", c.n, c.total, got, c.want)
		}
	}
}
