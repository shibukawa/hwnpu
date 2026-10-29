package llm

import (
	"fmt"
	"hash/fnv"
	"maps"
	"slices"
)

// Synthetic is a checkpoint of deterministic pseudo random tensors with the names and shapes a
// model reads, for testing a conversion without the real weights.
type Synthetic map[string][]int

// NewSynthetic makes the synthetic checkpoint of a model.
func NewSynthetic(m Model) Synthetic { return Synthetic(m.Tensors()) }

// Get returns values uniform in [-0.1, 0.1) that depend only on the tensor's name.
func (w Synthetic) Get(name string) ([]float32, []int, error) {
	shape, ok := w[name]
	if !ok {
		return nil, nil, fmt.Errorf("synthetic: no tensor %s", name)
	}
	n := 1
	for _, s := range shape {
		n *= s
	}
	h := fnv.New64a()
	h.Write([]byte(name))
	seed := h.Sum64()
	v := make([]float32, n)
	for i := range v {
		seed = seed*6364136223846793005 + 1442695040888963407
		v[i] = float32(int64(seed>>40)-(1<<23)) / (1 << 23) * 0.2
	}
	return v, shape, nil
}

func (w Synthetic) Rows(name string, begin, end int) ([]float32, error) {
	v, shape, err := w.Get(name)
	if err != nil {
		return nil, err
	}
	if len(shape) != 2 {
		return nil, fmt.Errorf("synthetic: %s is not a matrix", name)
	}
	return v[begin*shape[1] : end*shape[1]], nil
}

func (w Synthetic) Names() []string { return slices.Sorted(maps.Keys(w)) }
func (w Synthetic) Source() string  { return "synthetic" }
func (w Synthetic) Close() error    { return nil }

func (w Synthetic) Shape(name string) ([]int, bool) {
	s, ok := w[name]
	return s, ok
}

// lister is a checkpoint that can list its tensors without reading them.
type lister interface {
	Names() []string
	Shape(name string) ([]int, bool)
}

// Inventory compares the tensors a model reads with a checkpoint's, by Hugging Face name:
// missing ones the checkpoint lacks, unused ones the conversion ignores (often a sign of a
// missed part of the architecture), and shape mismatches. It needs a checkpoint that can list
// its tensors (safetensors); for others ok is false.
func Inventory(m Model, w Weights) (missing, unused, mismatched []string, ok bool) {
	l, ok := w.(lister)
	if !ok {
		return nil, nil, nil, false
	}
	want := m.Tensors()
	have := map[string]bool{}
	for _, n := range l.Names() {
		have[n] = true
		if _, used := want[n]; !used {
			unused = append(unused, n)
		}
	}
	for _, n := range slices.Sorted(maps.Keys(want)) {
		if !have[n] {
			missing = append(missing, n)
			continue
		}
		if shape, _ := l.Shape(n); !slices.Equal(shape, want[n]) {
			mismatched = append(mismatched, fmt.Sprintf("%s %v, expected %v", n, shape, want[n]))
		}
	}
	slices.Sort(unused)
	return missing, unused, mismatched, true
}
