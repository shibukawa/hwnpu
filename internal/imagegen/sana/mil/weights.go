// Package mil describes Sana and Sana-Sprint (NVlabs, Efficient-Large-Model/*_diffusers) as MIL
// programs, which hwnpu writes as Core ML packages for Apple's Neural Engine (llm.WritePart): the
// parts of tools/amd/iron_sana.py's graphs, Gemma 2 2B in groups of layers, the caption
// projection, the DiT's time embedding, groups of blocks and final layer, and DC-AE's decoder in
// stages.
//
// They run in fp16. Every norm first divides each row by its largest magnitude (milpart's
// ScaledRMSNorm and ScaledLayerNorm), and the linear attentions' sums over the tokens are taken
// of keys scaled down and of query rows divided by their largest value (the quotient does not
// change), so that nothing overflows (design.md 16.145).
package mil

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/shibukawa/hwnpu/internal/safetensors"
)

// Weights reads a Sana repository's checkpoints under one set of names: "dit." and the
// transformer's, "vae." and the VAE's, "gemma." and Gemma 2's (text_encoder/). A 1 x 1
// convolution's weight (out, in, 1, 1) reads as the matrix (out, in).
type Weights struct {
	files []source
}

type source struct {
	prefix string
	f      *safetensors.File
	path   string
}

// OpenWeights opens root/transformer, root/vae and text/text_encoder (text is the repository
// that holds Gemma 2: root, or any other Sana's, the same files), each one file or the shards of
// a model.safetensors.index.json. text "" leaves Gemma 2 out.
func OpenWeights(root, text string) (*Weights, error) {
	w := &Weights{}
	dirs := []struct{ prefix, dir, file string }{
		{"dit.", filepath.Join(root, "transformer"), "diffusion_pytorch_model"},
		{"vae.", filepath.Join(root, "vae"), "diffusion_pytorch_model"},
	}
	if text != "" {
		dirs = append(dirs, struct{ prefix, dir, file string }{"gemma.", filepath.Join(text, "text_encoder"), "model"})
	}
	for _, d := range dirs {
		paths, err := shards(d.dir, d.file)
		if err != nil {
			w.Close()
			return nil, err
		}
		for _, p := range paths {
			f, err := safetensors.Open(p)
			if err != nil {
				w.Close()
				return nil, err
			}
			w.files = append(w.files, source{d.prefix, f, p})
		}
	}
	return w, nil
}

// shards are the safetensors files of dir: base.safetensors, or those base.safetensors.index.json
// names.
func shards(dir, base string) ([]string, error) {
	one := filepath.Join(dir, base+".safetensors")
	if _, err := os.Stat(one); err == nil {
		return []string{one}, nil
	}
	b, err := os.ReadFile(filepath.Join(dir, base+".safetensors.index.json"))
	if err != nil {
		return nil, fmt.Errorf("sana: %s holds no %s.safetensors: %w", dir, base, err)
	}
	var idx struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(b, &idx); err != nil {
		return nil, fmt.Errorf("sana: %s: %w", dir, err)
	}
	seen := map[string]bool{}
	var out []string
	for _, f := range idx.WeightMap {
		if !seen[f] {
			seen[f] = true
			out = append(out, filepath.Join(dir, f))
		}
	}
	sort.Strings(out)
	return out, nil
}

func (w *Weights) find(name string) (*source, string, error) {
	for i := range w.files {
		s := &w.files[i]
		if rest, ok := strings.CutPrefix(name, s.prefix); ok {
			if _, ok := s.f.Lookup(rest); ok {
				return s, rest, nil
			}
		}
	}
	return nil, "", fmt.Errorf("sana: no checkpoint holds %s", name)
}

// Has reports whether a checkpoint holds name.
func (w *Weights) Has(name string) bool {
	_, _, err := w.find(name)
	return err == nil
}

// Get returns a tensor as float32 and its shape.
func (w *Weights) Get(name string) ([]float32, []int, error) {
	s, full, err := w.find(name)
	if err != nil {
		return nil, nil, err
	}
	t, _ := s.f.Lookup(full)
	v, err := s.f.Float32(full)
	shape := t.Shape
	if len(shape) == 4 && shape[2] == 1 && shape[3] == 1 {
		shape = shape[:2]
	}
	return v, shape, err
}

// Rows returns rows [begin, end) of a matrix.
func (w *Weights) Rows(name string, begin, end int) ([]float32, error) {
	s, full, err := w.find(name)
	if err != nil {
		return nil, err
	}
	t, _ := s.f.Lookup(full)
	cols := 1
	for _, d := range t.Shape[1:] {
		cols *= d
	}
	return s.f.Float32Range(full, begin*cols, end*cols)
}

// Source names the checkpoints.
func (w *Weights) Source() string {
	var names []string
	for _, s := range w.files {
		names = append(names, filepath.Base(filepath.Dir(s.path))+"/"+filepath.Base(s.path))
	}
	return "safetensors " + strings.Join(names, ", ")
}

// Close closes the files.
func (w *Weights) Close() error {
	for _, s := range w.files {
		s.f.Close()
	}
	return nil
}
