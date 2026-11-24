package mil

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	kmil "github.com/shibukawa/bonsaiai/internal/imagegen/flux2klein/mil"
	"github.com/shibukawa/bonsaiai/internal/safetensors"
)

// Weights reads a FLUX.3 klein repository's checkpoints under one set of names: "dit." and the
// transformer's (transformer/, diffusers' names), "text." and Qwen3-VL's (text_encoder/:
// model.language_model.*), "vae." and the VAE's.
//
// A block's ff.linear_in.weight is read in parts, #gate (its first half of rows) and #up (the
// second: Flux3SwiGLU's silu(x[:half]) * x[half:]). The DiT's query and key projections and their
// norms come with each head's rows permuted (kmil.RopePerm), so that the rotary embedding's pairs,
// FLUX's (2i, 2i + 1), are blocks.ApplyRope's (i, i + head_dim / 2).
type Weights struct {
	files []source
	dit   Dit
}

type source struct {
	prefix string
	f      *safetensors.File
	path   string
}

// OpenWeights opens root's transformer/, text_encoder/ and vae/ (each one file or the shards of an
// index.json) for the DiT of c's shape. Those of parts missing from root are left out.
func OpenWeights(root string, c Dit) (*Weights, error) {
	w := &Weights{dit: c}
	for _, d := range []struct{ prefix, dir, file string }{
		{"dit.", "transformer", "diffusion_pytorch_model"},
		{"text.", "text_encoder", "model"},
		{"vae.", "vae", "diffusion_pytorch_model"},
	} {
		dir := filepath.Join(root, d.dir)
		if _, err := os.Stat(dir); err != nil {
			continue
		}
		paths, err := shards(dir, d.file)
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
		return nil, fmt.Errorf("flux3klein: %s holds no %s.safetensors: %w", dir, base, err)
	}
	var idx struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(b, &idx); err != nil {
		return nil, fmt.Errorf("flux3klein: %s: %w", dir, err)
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
	return nil, "", fmt.Errorf("flux3klein: no checkpoint holds %s", name)
}

// Has reports whether a checkpoint holds name (not a part).
func (w *Weights) Has(name string) bool {
	_, _, err := w.find(name)
	return err == nil
}

// Roped reports whether the DiT's tensor (or part) name has its rows in RopePerm's order.
func Roped(name string) bool {
	if !strings.HasPrefix(name, "dit.") {
		return false
	}
	for _, s := range []string{"attn.to_q.", "attn.to_k.", "attn.norm_q.", "attn.norm_k."} {
		if strings.Contains(name, s) {
			return true
		}
	}
	return false
}

// Get returns a tensor (or a part, NAME#gate or NAME#up of a linear_in) as float32 and its shape.
func (w *Weights) Get(name string) ([]float32, []int, error) {
	if base, part, ok := strings.Cut(name, "#"); ok {
		return w.getPart(base, part)
	}
	s, full, err := w.find(name)
	if err != nil {
		return nil, nil, err
	}
	t, _ := s.f.Lookup(full)
	v, err := s.f.Float32(full)
	if err != nil {
		return nil, nil, err
	}
	shape := append([]int(nil), t.Shape...)
	if len(shape) == 4 && shape[2] == 1 && shape[3] == 1 {
		shape = shape[:2]
	}
	if Roped(name) {
		v = kmil.PermuteRows(v, shape[0], w.dit.HeadDim)
	}
	return v, shape, nil
}

func (w *Weights) getPart(base, part string) ([]float32, []int, error) {
	if strings.HasSuffix(base, "context_embedder.weight") && (part == "c0" || part == "c1") {
		// its columns in two halves: 20480 in-features are past the Neural Engine's 16384 a
		// dimension, so Ctx takes the eight hidden states as two inputs of four
		v, shape, err := w.Get(base)
		if err != nil {
			return nil, nil, err
		}
		rows, cols := shape[0], shape[1]
		half := cols / 2
		lo := 0
		if part == "c1" {
			lo = half
		}
		out := make([]float32, rows*half)
		for r := 0; r < rows; r++ {
			copy(out[r*half:(r+1)*half], v[r*cols+lo:r*cols+lo+half])
		}
		return out, []int{rows, half}, nil
	}
	if !strings.HasSuffix(base, "linear_in.weight") || (part != "gate" && part != "up") {
		return nil, nil, fmt.Errorf("flux3klein: %s#%s: no such part", base, part)
	}
	s, full, err := w.find(base)
	if err != nil {
		return nil, nil, err
	}
	t, _ := s.f.Lookup(full)
	rows, cols := t.Shape[0], t.Shape[1]
	half := rows / 2
	lo := 0
	if part == "up" {
		lo = half
	}
	v, err := s.f.Float32Range(full, lo*cols, (lo+half)*cols)
	if err != nil {
		return nil, nil, err
	}
	return v, []int{half, cols}, nil
}

// Rows returns rows [begin, end) of a matrix (not of a permuted one, nor of a part).
func (w *Weights) Rows(name string, begin, end int) ([]float32, error) {
	s, full, err := w.find(name)
	if err != nil {
		return nil, err
	}
	if Roped(name) {
		return nil, fmt.Errorf("flux3klein: %s is read whole (its rows are permuted)", name)
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
