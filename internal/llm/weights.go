package llm

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/shibukawa/hwnpu/internal/gguf"
	"github.com/shibukawa/hwnpu/internal/safetensors"
)

// OpenWeights opens a safetensors or GGUF checkpoint of the architecture by its extension. A
// checkpoint in several safetensors files is opened by its index (model.safetensors.index.json),
// or by the name it would have in one file (model.safetensors) when only the index is there.
func OpenWeights(path string, a *Arch) (Weights, error) {
	if strings.HasSuffix(path, ".safetensors.index.json") {
		return openSharded(path, a)
	}
	if strings.HasSuffix(path, ".safetensors") {
		if _, err := os.Stat(path); errors.Is(err, fs.ErrNotExist) {
			if _, err := os.Stat(path + ".index.json"); err == nil {
				return openSharded(path+".index.json", a)
			}
		}
	}
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

// shardedWeights is a safetensors checkpoint in several files, each tensor in the one its index
// names.
type shardedWeights struct {
	files map[string]*safetensors.File // by tensor name
	all   []*safetensors.File
	path  string
	arch  *Arch
}

func openSharded(index string, a *Arch) (Weights, error) {
	raw, err := os.ReadFile(index)
	if err != nil {
		return nil, err
	}
	var idx struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(raw, &idx); err != nil {
		return nil, fmt.Errorf("%s: %s: %w", a.Name, index, err)
	}
	if len(idx.WeightMap) == 0 {
		return nil, fmt.Errorf("%s: %s has no weight_map", a.Name, index)
	}
	w := &shardedWeights{files: map[string]*safetensors.File{}, path: index, arch: a}
	opened := map[string]*safetensors.File{}
	for _, name := range slices.Sorted(maps.Keys(idx.WeightMap)) {
		file := idx.WeightMap[name]
		f, ok := opened[file]
		if !ok {
			if f, err = safetensors.Open(filepath.Join(filepath.Dir(index), file)); err != nil {
				w.Close()
				return nil, err
			}
			opened[file] = f
			w.all = append(w.all, f)
		}
		if _, ok := f.Lookup(name); !ok {
			w.Close()
			return nil, fmt.Errorf("%s: %s names %s for %s, which does not hold it", a.Name, index, file, name)
		}
		w.files[name] = f
	}
	return w, nil
}

func (w *shardedWeights) Get(name string) ([]float32, []int, error) {
	f, ok := w.files[name]
	if !ok {
		return nil, nil, fmt.Errorf("%s: %s has no tensor %s", w.arch.Name, w.path, name)
	}
	t, _ := f.Lookup(name)
	v, err := f.Float32(name)
	return v, t.Shape, err
}

func (w *shardedWeights) Rows(name string, begin, end int) ([]float32, error) {
	f, ok := w.files[name]
	if !ok {
		return nil, fmt.Errorf("%s: %s has no matrix %s", w.arch.Name, w.path, name)
	}
	t, _ := f.Lookup(name)
	if len(t.Shape) != 2 {
		return nil, fmt.Errorf("%s: %s has no matrix %s", w.arch.Name, w.path, name)
	}
	return f.Float32Range(name, begin*t.Shape[1], end*t.Shape[1])
}

func (w *shardedWeights) Names() []string { return slices.Sorted(maps.Keys(w.files)) }

func (w *shardedWeights) Shape(name string) ([]int, bool) {
	f, ok := w.files[name]
	if !ok {
		return nil, false
	}
	t, _ := f.Lookup(name)
	return t.Shape, true
}

func (w *shardedWeights) Source() string {
	return fmt.Sprintf("safetensors %s (%d files)", filepath.Base(w.path), len(w.all))
}

func (w *shardedWeights) Close() error {
	var errs []error
	for _, f := range w.all {
		errs = append(errs, f.Close())
	}
	return errors.Join(errs...)
}
