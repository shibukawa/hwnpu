// Command chromafake writes Chroma's programs (internal/imagegen/chroma/mil) with random weights
// of the checkpoint's shapes, read from the safetensors headers alone (curl -r of each file's
// first 8 bytes and its header), so that the programs can be lowered to Core AI and checked
// (aicheck run -cpu) without the 27 GB of weights.
//
//	go run ./tools/coreai/chromafake -repo DIR -headers DIR -out DIR [-size 1024] [-parts text,mods,doubles,singles,vae]
//
// DIR of -repo holds the repository's config.json files (transformer/, text_encoder/, vae/);
// -headers the headers as PREFIX_NAME.json (transformer_..., text_encoder_..., vae_...).
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu/internal/imagegen/chroma/mil"
	"github.com/shibukawa/hwnpu/internal/llm"
)

type fake struct{ shapes map[string][]int }

func (f *fake) load(dir, file, prefix string) error {
	b, err := os.ReadFile(filepath.Join(dir, file))
	if err != nil {
		return err
	}
	var h map[string]json.RawMessage
	if err := json.Unmarshal(b, &h); err != nil {
		return err
	}
	for k, v := range h {
		if k == "__metadata__" {
			continue
		}
		var t struct{ Shape []int }
		if err := json.Unmarshal(v, &t); err != nil {
			return err
		}
		f.shapes[prefix+k] = t.Shape
	}
	return nil
}

// Get is random: a matrix (or a convolution) N(0, 1/fan_in), a norm's weight near 1, a bias
// small, T5's position bias and embeddings N(0, 1).
func (f *fake) Get(name string) ([]float32, []int, error) {
	shape, ok := f.shapes[name]
	if !ok {
		return nil, nil, fmt.Errorf("chromafake: no tensor %s", name)
	}
	shape = append([]int(nil), shape...)
	if len(shape) == 4 && shape[2] == 1 && shape[3] == 1 {
		shape = shape[:2]
	}
	n := 1
	for _, d := range shape {
		n *= d
	}
	h := 0
	for _, c := range name {
		h = h*31 + int(c)
	}
	rng := rand.New(rand.NewSource(int64(h)))
	v := make([]float32, n)
	std, mean := 0.02, 0.0
	switch {
	case strings.Contains(name, "norm") && len(shape) == 1 && strings.HasSuffix(name, "weight"),
		strings.Contains(name, "layer_norm") || strings.HasSuffix(name, "final_layer_norm.weight"):
		std, mean = 0.1, 1
	case strings.Contains(name, "relative_attention_bias") || strings.Contains(name, "shared") || strings.Contains(name, "embed_tokens"):
		std = 1
	case len(shape) >= 2:
		fan := n / shape[0]
		std = 1 / math.Sqrt(float64(fan))
	}
	for i := range v {
		v[i] = float32(mean + std*rng.NormFloat64())
	}
	return v, shape, nil
}

func (f *fake) Rows(name string, begin, end int) ([]float32, error) {
	v, shape, err := f.Get(name)
	if err != nil {
		return nil, err
	}
	cols := len(v) / shape[0]
	return v[begin*cols : end*cols], nil
}

func (f *fake) Source() string { return "chromafake" }
func (f *fake) Close() error   { return nil }

func main() {
	repo := flag.String("repo", "", "the repository's config files")
	headers := flag.String("headers", "", "the safetensors headers")
	out := flag.String("out", "", "where to write the packages")
	size := flag.Int("size", 1024, "the picture's size")
	parts := flag.String("parts", "text,mods,doubles,singles,vae", "the parts to write")
	all := flag.Bool("all", false, "every group (default: the first and the last of each kind)")
	flag.Parse()
	if err := run(*repo, *headers, *out, *size, *parts, *all); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(repo, headers, out string, size int, parts string, all bool) error {
	f := &fake{shapes: map[string][]int{}}
	ents, err := os.ReadDir(headers)
	if err != nil {
		return err
	}
	for _, e := range ents {
		n := e.Name()
		if !strings.HasSuffix(n, ".json") {
			continue
		}
		prefix := map[string]string{"transformer": "dit.", "text": "t5.", "vae": "vae."}[strings.SplitN(n, "_", 2)[0]]
		if prefix == "" {
			continue
		}
		if err := f.load(headers, n, prefix); err != nil {
			return err
		}
	}
	dc, err := mil.ReadDit(repo, size, size)
	if err != nil {
		return err
	}
	tc, err := mil.ReadT5(repo)
	if err != nil {
		return err
	}
	vc, err := mil.ReadVae(repo)
	if err != nil {
		return err
	}
	want := map[string]bool{}
	for _, p := range strings.Split(parts, ",") {
		want[p] = true
	}
	pick := func(groups [][2]int) [][2]int {
		if all || len(groups) <= 2 {
			return groups
		}
		return [][2]int{groups[0], groups[len(groups)-1]}
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}
	write := func(p *mil.Part, format string, rules []string) error {
		path := filepath.Join(out, p.Name+".mlpackage")
		if _, err := os.Stat(path); err == nil {
			fmt.Println("kept", path)
			return nil
		}
		if _, err := llm.WritePart(path, p, f, llm.Options{Backend: llm.BackendCoreML, Format: format, FormatFor: rules}, nil); err != nil {
			return fmt.Errorf("%s: %w", p.Name, err)
		}
		fmt.Println("wrote", path, format)
		return nil
	}
	// the defaults of chroma.ConvertOptions: T5 lut8 with wo fp16 in groups of 3, the DiT lut8
	// (doubles 4, singles 8), the approximator and the VAE fp16
	if want["text"] {
		for _, g := range pick(mil.TextGroups(tc.Layers, 3)) {
			if err := write(mil.Text(tc, dc, g[0], g[1], dc.TextRows), "lut8", []string{"wo=fp16"}); err != nil {
				return err
			}
		}
	}
	if want["mods"] {
		if err := write(mil.Mods(dc), "fp16", nil); err != nil {
			return err
		}
	}
	if want["doubles"] {
		for _, g := range pick(mil.DoubleGroups(dc, 4)) {
			if err := write(mil.Doubles(dc, g[0], g[1]), "lut8", nil); err != nil {
				return err
			}
		}
	}
	if want["singles"] {
		for _, g := range pick(mil.SingleGroups(dc, 8)) {
			if err := write(mil.Singles(dc, g[0], g[1]), "lut8", nil); err != nil {
				return err
			}
		}
	}
	if want["vae"] {
		for i := range mil.VaeStages(vc, size, size) {
			if err := write(mil.VaeProgram(vc, size, size, i), "fp16", nil); err != nil {
				return err
			}
		}
	}
	return nil
}
