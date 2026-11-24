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

// Mapped is a tensor's values as Get gives them, in its safetensors file mapped (read only, valid
// until Close) when it is stored as float32, else read as Get reads it.
func (w *Weights) Mapped(name string) ([]float32, error) {
	for i, f := range w.st {
		rest, ok := strings.CutPrefix(name, w.pre[i])
		if !ok {
			continue
		}
		if _, ok := f.Lookup(rest); ok {
			if v, ok, err := f.MapFloat32(rest); ok || err != nil {
				return v, err
			}
			break
		}
	}
	v, _, err := w.Get(name)
	return v, err
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

// WriteTables writes the named tensors of the safetensors files to out as they are stored (the
// text encoder's bf16 stays bf16), by their names here (the prefix included): OpenWeights opens
// out with the prefix "" and finds them by the same names.
func (w *Weights) WriteTables(names []string, out string) error {
	var ts []safetensors.Out
	for _, name := range names {
		found := false
		for i, f := range w.st {
			rest, ok := strings.CutPrefix(name, w.pre[i])
			if !ok {
				continue
			}
			if _, ok := f.Lookup(rest); !ok {
				continue
			}
			sec, t, err := f.Section(rest)
			if err != nil {
				return err
			}
			ts = append(ts, safetensors.Out{Name: name, DType: t.DType, Shape: t.Shape, Size: sec.Size(), Data: sec})
			found = true
			break
		}
		if !found {
			return fmt.Errorf("sfx: no tensor %s", name)
		}
	}
	return safetensors.Write(out, ts, map[string]string{"format": "pt", "hwnpu": "host tables", "source": w.desc})
}

func (w *Weights) Close() error {
	for _, f := range w.pth {
		f.Close()
	}
	for _, f := range w.st {
		f.Close()
	}
	return nil
}
