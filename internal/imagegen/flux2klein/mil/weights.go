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

// Weights reads a FLUX.2 klein repository's checkpoints under one set of names: "dit." and the
// transformer's, "text." and Qwen3's (text_encoder/), "vae." and the VAE's.
//
// The DiT's fused matrices are read in parts, named NAME#PART: a single block's
// attn.to_qkv_mlp_proj.weight#q, #k, #v, #gate and #up (its rows), its attn.to_out.weight#attn
// and #mlp (its columns over [attention, MLP]), a double block's ff.linear_in.weight#gate and #up
// (and ff_context's). The DiT's query and key projections (and their norms) come with each head's
// rows permuted (RopePerm), so that the rotary embedding's pairs, FLUX's (2i, 2i + 1), are
// blocks.ApplyRope's (i, i + head_dim / 2).
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
// index.json) for the DiT of c's shape. Those of parts missing from root are left out (a check of
// one part).
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
		return nil, fmt.Errorf("flux2klein: %s holds no %s.safetensors: %w", dir, base, err)
	}
	var idx struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(b, &idx); err != nil {
		return nil, fmt.Errorf("flux2klein: %s: %w", dir, err)
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
	return nil, "", fmt.Errorf("flux2klein: no checkpoint holds %s", name)
}

// Has reports whether a checkpoint holds name (not a part).
func (w *Weights) Has(name string) bool {
	_, _, err := w.find(name)
	return err == nil
}

// roped reports whether the DiT's tensor (or part) name has its rows in RopePerm's order.
func roped(name string) bool {
	if !strings.HasPrefix(name, "dit.") {
		return false
	}
	if strings.HasSuffix(name, "to_qkv_mlp_proj.weight#q") || strings.HasSuffix(name, "to_qkv_mlp_proj.weight#k") {
		return true
	}
	for _, s := range []string{"attn.to_q.", "attn.to_k.", "attn.add_q_proj.", "attn.add_k_proj.",
		"attn.norm_q.", "attn.norm_k.", "attn.norm_added_q.", "attn.norm_added_k."} {
		if strings.Contains(name, s) {
			return true
		}
	}
	return false
}

// rowParts are the parts of the fused matrices cut by rows: the suffix and the part's rows
// [lo, hi) in units of the hidden width E and the MLP's M (lo = a E + b M).
func (w *Weights) rowPart(name string) (base string, lo, hi int, ok bool) {
	base, p, found := strings.Cut(name, "#")
	if !found {
		return "", 0, 0, false
	}
	E, M := w.dit.Hidden, w.dit.MLP
	switch {
	case strings.HasSuffix(base, "to_qkv_mlp_proj.weight"):
		at := map[string][2]int{"q": {0, E}, "k": {E, 2 * E}, "v": {2 * E, 3 * E}, "gate": {3 * E, 3*E + M}, "up": {3*E + M, 3*E + 2*M}}
		if r, ok := at[p]; ok {
			return base, r[0], r[1], true
		}
	case strings.HasSuffix(base, "linear_in.weight"):
		at := map[string][2]int{"gate": {0, M}, "up": {M, 2 * M}}
		if r, ok := at[p]; ok {
			return base, r[0], r[1], true
		}
	}
	return "", 0, 0, false
}

// Get returns a tensor (or a part, NAME#PART) as float32 and its shape.
func (w *Weights) Get(name string) ([]float32, []int, error) {
	if base, part, ok := strings.Cut(name, "#"); ok {
		return w.getPart(name, base, part)
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
	if roped(name) {
		v = PermuteRows(v, shape[0], w.dit.HeadDim)
	}
	return v, shape, nil
}

func (w *Weights) getPart(name, base, part string) ([]float32, []int, error) {
	if b, lo, hi, ok := w.rowPart(name); ok {
		s, full, err := w.find(b)
		if err != nil {
			return nil, nil, err
		}
		t, _ := s.f.Lookup(full)
		cols := t.Shape[1]
		v, err := s.f.Float32Range(full, lo*cols, hi*cols)
		if err != nil {
			return nil, nil, err
		}
		if roped(name) {
			v = PermuteRows(v, hi-lo, w.dit.HeadDim)
		}
		return v, []int{hi - lo, cols}, nil
	}
	if strings.HasSuffix(base, "attn.to_out.weight") && (part == "attn" || part == "mlp") {
		v, shape, err := w.Get(base)
		if err != nil {
			return nil, nil, err
		}
		E := w.dit.Hidden
		rows, cols := shape[0], shape[1]
		lo, hi := 0, E
		if part == "mlp" {
			lo, hi = E, cols
		}
		out := make([]float32, rows*(hi-lo))
		for r := 0; r < rows; r++ {
			copy(out[r*(hi-lo):(r+1)*(hi-lo)], v[r*cols+lo:r*cols+hi])
		}
		return out, []int{rows, hi - lo}, nil
	}
	return nil, nil, fmt.Errorf("flux2klein: %s: no such part", name)
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

// Rows returns rows [begin, end) of a matrix (not of a permuted one, nor of a part).
func (w *Weights) Rows(name string, begin, end int) ([]float32, error) {
	s, full, err := w.find(name)
	if err != nil {
		return nil, err
	}
	if roped(name) {
		return nil, fmt.Errorf("flux2klein: %s is read whole (its rows are permuted)", name)
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
