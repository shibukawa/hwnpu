package llm

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu/internal/gguf"
	"github.com/shibukawa/hwnpu/internal/safetensors"
)

// OpenWeights opens a safetensors or GGUF checkpoint of the architecture by its extension.
func OpenWeights(path string, a *Arch) (Weights, error) {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".safetensors":
		f, err := safetensors.Open(path)
		if err != nil {
			return nil, err
		}
		return stWeights{f: f, path: path, arch: a}, nil
	case ".gguf":
		if a.GGUFName == nil {
			return nil, fmt.Errorf("%s: GGUF checkpoints are not supported", a.Name)
		}
		f, err := gguf.Open(path)
		if err != nil {
			return nil, err
		}
		if got := f.String("general.architecture"); got != a.Name {
			f.Close()
			return nil, fmt.Errorf("%s: %s is a %q model", a.Name, path, got)
		}
		return ggufWeights{f: f, path: path, arch: a}, nil
	}
	return nil, fmt.Errorf("%s: %s: expected a .safetensors or .gguf file", a.Name, path)
}

type stWeights struct {
	f    *safetensors.File
	path string
	arch *Arch
}

func (w stWeights) Get(name string) ([]float32, []int, error) {
	t, ok := w.f.Lookup(name)
	if !ok {
		return nil, nil, fmt.Errorf("%s: %s has no tensor %s", w.arch.Name, w.path, name)
	}
	v, err := w.f.Float32(name)
	return v, t.Shape, err
}

func (w stWeights) Rows(name string, begin, end int) ([]float32, error) {
	t, ok := w.f.Lookup(name)
	if !ok || len(t.Shape) != 2 {
		return nil, fmt.Errorf("%s: %s has no matrix %s", w.arch.Name, w.path, name)
	}
	return w.f.Float32Range(name, begin*t.Shape[1], end*t.Shape[1])
}

func (w stWeights) Names() []string { return w.f.Names() }

func (w stWeights) Shape(name string) ([]int, bool) {
	t, ok := w.f.Lookup(name)
	return t.Shape, ok
}

func (w stWeights) Source() string { return "safetensors " + filepath.Base(w.path) }
func (w stWeights) Close() error   { return w.f.Close() }

type ggufWeights struct {
	f    *gguf.File
	path string
	arch *Arch
}

func (w ggufWeights) Get(name string) ([]float32, []int, error) {
	gn := w.arch.GGUFName(name)
	if gn == "" {
		return nil, nil, fmt.Errorf("%s: no GGUF name for %s", w.arch.Name, name)
	}
	t, ok := w.f.Lookup(gn)
	if !ok {
		return nil, nil, fmt.Errorf("%s: %s has no tensor %s (%s)", w.arch.Name, w.path, gn, name)
	}
	v, err := w.f.Float32(gn)
	shape := t.Shape
	if w.arch.GGUFShape != nil {
		shape = w.arch.GGUFShape(name, shape)
	}
	return v, shape, err
}

func (w ggufWeights) Rows(name string, begin, end int) ([]float32, error) {
	gn := w.arch.GGUFName(name)
	t, ok := w.f.Lookup(gn)
	if gn == "" || !ok || len(t.Shape) != 2 {
		return nil, fmt.Errorf("%s: %s has no matrix %s", w.arch.Name, w.path, name)
	}
	return w.f.Float32Range(gn, begin*t.Shape[1], end*t.Shape[1])
}

func (w ggufWeights) Source() string {
	return fmt.Sprintf("gguf %s (%s)", filepath.Base(w.path), w.f.String("general.name"))
}
func (w ggufWeights) Close() error { return w.f.Close() }
