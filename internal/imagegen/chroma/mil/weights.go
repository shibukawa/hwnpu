// Package mil describes Chroma1 (lodestones/Chroma1-Flash, Chroma1-HD) as MIL programs, which
// hwnpu writes as Core ML packages for Apple's Neural Engine (llm.WritePart): T5-XXL's encoder in
// groups of layers (the last ending with the final norm and the DiT's context_embedder), the
// approximator (the modulation rows of a timestep), the DiT's double-stream and single-stream
// blocks in groups (the first embedding the latents' patches, the last ending with norm_out and
// proj_out) and FLUX's VAE decoder in stages. They are the parts of tools/amd/iron_chroma.py and
// iron_chroma_parts.py.
//
// They run in fp16. T5's residual stream (3e5 in its outlier channels) and the DiT's are carried
// divided by a constant (T5Scale, DitScale): every sublayer reads them through a norm, which does
// not see the scale, and adds a projection whose weight takes it. Every norm first divides each
// row by its largest magnitude (milpart's ScaledRMSNorm and ScaledLayerNorm).
package mil

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/shibukawa/hwnpu/internal/imagegen/milpart"
	"github.com/shibukawa/hwnpu/internal/safetensors"
)

// Part is one program of Chroma (milpart.Part).
type Part = milpart.Part

type graph = milpart.Graph

func part(name string, body func(g *graph)) *Part {
	return &Part{Name: name, Model: "chroma", Body: body}
}

// Weights reads a Chroma repository's checkpoints under one set of names: "dit." and the
// transformer's, "t5." and T5's (text_encoder/), "vae." and the VAE's. The DiT's query and key
// projections (and their norms) come with each head's rows permuted (RopePerm), so that the
// rotary embedding's pairs, FLUX's (2i, 2i + 1), are blocks.ApplyRope's (i, i + 64).
type Weights struct {
	files []source
	// HeadDim is the DiT's head size the permutation is of (128).
	HeadDim int
}

type source struct {
	prefix string
	f      *safetensors.File
	path   string
}

// OpenWeights opens root/transformer, root/text_encoder and root/vae, each one file or the shards
// of an index.json. text "" (or vae "") leaves T5 (the VAE) out; otherwise they name the
// repositories holding them (root, or another Chroma's: the same files).
func OpenWeights(root, text, vae string) (*Weights, error) {
	w := &Weights{HeadDim: 128}
	dirs := []struct{ prefix, dir, file string }{{"dit.", filepath.Join(root, "transformer"), "diffusion_pytorch_model"}}
	if text != "" {
		dirs = append(dirs, struct{ prefix, dir, file string }{"t5.", filepath.Join(text, "text_encoder"), "model"})
	}
	if vae != "" {
		dirs = append(dirs, struct{ prefix, dir, file string }{"vae.", filepath.Join(vae, "vae"), "diffusion_pytorch_model"})
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
		return nil, fmt.Errorf("chroma: %s holds no %s.safetensors: %w", dir, base, err)
	}
	var idx struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(b, &idx); err != nil {
		return nil, fmt.Errorf("chroma: %s: %w", dir, err)
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
	return nil, "", fmt.Errorf("chroma: no checkpoint holds %s", name)
}

// Has reports whether a checkpoint holds name.
func (w *Weights) Has(name string) bool {
	_, _, err := w.find(name)
	return err == nil
}

// roped reports whether the DiT's tensor name has its rows in RopePerm's order.
func roped(name string) bool {
	if !strings.HasPrefix(name, "dit.") {
		return false
	}
	for _, s := range []string{"attn.to_q.", "attn.to_k.", "attn.add_q_proj.", "attn.add_k_proj.",
		"attn.norm_q.", "attn.norm_k.", "attn.norm_added_q.", "attn.norm_added_k."} {
		if strings.Contains(name, s) {
			return true
		}
	}
	return false
}

// Get returns a tensor as float32 and its shape.
func (w *Weights) Get(name string) ([]float32, []int, error) {
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
	if roped(name) {
		v = PermuteRows(v, shape[0], w.HeadDim)
	}
	return v, shape, nil
}

// RopePerm is the order of a head's d values that makes FLUX's rotary pairs (2i, 2i + 1)
// blocks.ApplyRope's (i, i + d/2): new position i holds old 2i, new i + d/2 old 2i + 1.
func RopePerm(d int) []int {
	p := make([]int, d)
	for i := 0; i < d/2; i++ {
		p[i], p[i+d/2] = 2*i, 2*i+1
	}
	return p
}

// PermuteRows is v (rows x cols, rows a multiple of d) with each block of d rows in RopePerm's
// order.
func PermuteRows(v []float32, rows, d int) []float32 {
	cols := len(v) / rows
	p := RopePerm(d)
	out := make([]float32, len(v))
	for h := 0; h < rows/d; h++ {
		for i, j := range p {
			copy(out[(h*d+i)*cols:(h*d+i+1)*cols], v[(h*d+j)*cols:(h*d+j+1)*cols])
		}
	}
	return out
}

// Rows returns rows [begin, end) of a matrix (not of a permuted one).
func (w *Weights) Rows(name string, begin, end int) ([]float32, error) {
	s, full, err := w.find(name)
	if err != nil {
		return nil, err
	}
	if roped(name) {
		return nil, fmt.Errorf("chroma: %s is read whole (its rows are permuted)", name)
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
