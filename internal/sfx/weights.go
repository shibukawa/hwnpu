package sfx

import (
	"fmt"
	"strings"

	"github.com/shibukawa/hwnpu/internal/safetensors"
	"github.com/shibukawa/hwnpu/internal/torchzip"
)

// Weights reads MOSS-SoundEffect's checkpoints as llm.Weights: the DAC VAE's .pth by its
// state_dict paths ("state_dict.decoder..."), safetensors files by their names with a prefix of
// one's own before them ("dit." for the transformer's, "qwen." for the text encoder's).
type Weights struct {
	pth  []*torchzip.File
	st   []*safetensors.File
	pre  []string // st[i]'s prefix
	desc string
}

// OpenWeights opens the files: .pth paths as they are, safetensors as prefix=path.
func OpenWeights(pth []string, st map[string][]string) (*Weights, error) {
	w := &Weights{desc: "OpenMOSS-Team/MOSS-SoundEffect-v2.0"}
	for _, p := range pth {
		f, err := torchzip.Open(p)
		if err != nil {
			w.Close()
			return nil, err
		}
		w.pth = append(w.pth, f)
	}
	for prefix, paths := range st {
		for _, p := range paths {
			f, err := safetensors.Open(p)
			if err != nil {
				w.Close()
				return nil, err
			}
			w.st = append(w.st, f)
			w.pre = append(w.pre, prefix)
		}
	}
	return w, nil
}

func (w *Weights) Get(name string) ([]float32, []int, error) {
	for _, f := range w.pth {
		if t, ok := f.Lookup(name); ok {
			v, err := f.Float32(name)
			return v, t.Shape, err
		}
	}
	for i, f := range w.st {
		rest, ok := strings.CutPrefix(name, w.pre[i])
		if !ok {
			continue
		}
		if t, ok := f.Lookup(rest); ok {
			v, err := f.Float32(rest)
			return v, t.Shape, err
		}
	}
	return nil, nil, fmt.Errorf("sfx: no tensor %s", name)
}

func (w *Weights) Rows(name string, begin, end int) ([]float32, error) {
	for i, f := range w.st {
		rest, ok := strings.CutPrefix(name, w.pre[i])
		if !ok {
			continue
		}
		if t, ok := f.Lookup(rest); ok {
			cols := t.Shape[len(t.Shape)-1]
			return f.Float32Range(rest, begin*cols, end*cols)
		}
	}
	v, shape, err := w.Get(name)
	if err != nil {
		return nil, err
	}
	cols := shape[len(shape)-1]
	return v[begin*cols : end*cols], nil
}

func (w *Weights) Source() string { return w.desc }

func (w *Weights) Close() error {
	for _, f := range w.pth {
		f.Close()
	}
	for _, f := range w.st {
		f.Close()
	}
	return nil
}
