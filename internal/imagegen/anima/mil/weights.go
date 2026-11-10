// Package mil describes Anima (circlestone-labs/Anima) as MIL programs, which hwnpu writes as
// Core ML packages for Apple's Neural Engine and lowers to OpenVINO IR for Intel's NPU
// (llm.WritePart): the same parts as the IRON graphs of tools/amd/iron_anima.py, the text
// encoder and LLM adapter, the DiT in groups of blocks and the VAE decoder in stages.
//
// These run in fp16, where the DiT's residual stream (up to 2.7e5 in blocks 12 to 20) and
// Qwen3's (up to 7e3) do not fit or overflow when squared. So the DiT holds its residual
// stream divided by DitScale (the patch embedding and the gates scaled down, the norms scale
// free), and every norm over the hidden width first divides each row by its largest magnitude
// (norms do not change with the scale of their input), so no square overflows.
package mil

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu/internal/safetensors"
)

// Weights reads an Anima directory's checkpoints under one set of names: "dit." and the
// diffusion model's names (model.diffusion_model., or net. in anima-base-v1.0 and the previews,
// left out; the LLM adapter's llm_adapter. among them), "qwen." and Qwen3-0.6B's (model. left out), "vae." and the diffusers VAE's.
type Weights struct {
	files []source
}

type source struct {
	prefix string // the names this file answers to
	strip  string // the file's own prefix, left out of the names
	f      *safetensors.File
	path   string
}

// OpenWeights opens the checkpoints in root: diffusion_models/dit (the official split file),
// text_encoders/qwen_3_06b_base.safetensors and diffusers/vae/diffusion_pytorch_model.safetensors.
func OpenWeights(root, dit string) (*Weights, error) {
	w := &Weights{}
	for _, s := range []struct{ prefix, strip, path string }{
		{"dit.", "model.diffusion_model.", filepath.Join(root, "diffusion_models", dit)},
		{"qwen.", "model.", filepath.Join(root, "text_encoders", "qwen_3_06b_base.safetensors")},
		{"vae.", "", filepath.Join(root, "diffusers", "vae", "diffusion_pytorch_model.safetensors")},
	} {
		f, err := safetensors.Open(s.path)
		if err != nil {
			w.Close()
			return nil, err
		}
		if s.prefix == "dit." {
			s.strip = DitPrefix(f.Names())
		}
		w.files = append(w.files, source{s.prefix, s.strip, f, s.path})
	}
	return w, nil
}

// DitPrefix is the prefix of a diffusion model's tensor names: model.diffusion_model. (the
// Turbo and Aesthetic files, as ComfyUI saves them) or net. (anima-base-v1.0 and the previews,
// as Cosmos trains them); the names past it are the same.
func DitPrefix(names []string) string {
	for _, n := range names {
		if strings.HasPrefix(n, "net.") {
			return "net."
		}
	}
	return "model.diffusion_model."
}

func (w *Weights) find(name string) (*source, string, error) {
	for i := range w.files {
		s := &w.files[i]
		if rest, ok := strings.CutPrefix(name, s.prefix); ok {
			full := s.strip + rest
			if _, ok := s.f.Lookup(full); !ok {
				return nil, "", fmt.Errorf("anima: %s has no tensor %s", s.path, full)
			}
			return s, full, nil
		}
	}
	return nil, "", fmt.Errorf("anima: no checkpoint holds %s", name)
}

// Get returns a tensor as float32 and its shape.
func (w *Weights) Get(name string) ([]float32, []int, error) {
	s, full, err := w.find(name)
	if err != nil {
		return nil, nil, err
	}
	t, _ := s.f.Lookup(full)
	v, err := s.f.Float32(full)
	return v, t.Shape, err
}

// Rows returns rows [begin, end) of a matrix.
func (w *Weights) Rows(name string, begin, end int) ([]float32, error) {
	s, full, err := w.find(name)
	if err != nil {
		return nil, err
	}
	t, _ := s.f.Lookup(full)
	if len(t.Shape) != 2 {
		return nil, fmt.Errorf("anima: %s is no matrix", name)
	}
	return s.f.Float32Range(full, begin*t.Shape[1], end*t.Shape[1])
}

// Source names the checkpoints.
func (w *Weights) Source() string {
	var names []string
	for _, s := range w.files {
		names = append(names, filepath.Base(s.path))
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
