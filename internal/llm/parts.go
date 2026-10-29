package llm

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Part is a run of consecutive layers of a model split into several packages. The Neural Engine
// takes less than 1 GiB of weights per package (M3), so a larger model is loaded as its parts, the
// output of each part bound as the input of the next. The limit counts the whole package: parts
// written as the functions of one package (Core ML's multifunction models) all fall to the CPU
// once the package is over it, so every part is a package of its own.
type Part struct {
	Package string `json:"package"` // path relative to the directory of the parts
	Start   int    `json:"start"`
	Layers  int    `json:"layers"`
	// WeightBytes is the size of the part's weight file.
	WeightBytes int64 `json:"weight_bytes"`
}

// PartsFile lists the parts of a model in its directory.
const PartsFile = "parts.json"

// PartsList is the content of PartsFile.
type PartsList struct {
	Source string `json:"source"`
	Format string `json:"format"`
	Seq    int    `json:"seq"`
	Cache  int    `json:"cache,omitempty"` // KV cache length (Options.Cache)
	Parts  []Part `json:"parts"`
}

// SplitLayers divides layers into n runs whose sizes differ by at most one layer.
func SplitLayers(layers, n int) []Part {
	parts := make([]Part, n)
	start := 0
	for i := range parts {
		k := layers / n
		if i < layers%n {
			k++
		}
		parts[i] = Part{Package: fmt.Sprintf("part_%d.mlpackage", i), Start: start, Layers: k}
		start += k
	}
	return parts
}

// WriteParts writes m split into n parts as packages in dir (replaced), with PartsFile listing
// them. The first part includes the embedding stage and takes the model's inputs; the others take
// x, the hidden state before their first layer, and any further inputs the model has (such as
// mask); the last part ends with the final norm and the output layer. opts.Start and opts.Layers
// must be zero. With opts.Cache the model must be Cached, and each part has the cache inputs and
// outputs of its own layers. describe fills in each package's description and metadata.
func WriteParts(dir string, m Model, w Weights, opts Options, n int, describe func(*mlpackage.Package, Part)) (*PartsList, error) {
	if opts.Start != 0 || opts.Layers != 0 || opts.NoHead {
		return nil, fmt.Errorf("llm: parts cover the whole model; Start, Layers and NoHead must not be set")
	}
	if n < 1 || n > m.Layers() {
		return nil, fmt.Errorf("llm: %d parts of %d layers", n, m.Layers())
	}
	if err := os.RemoveAll(dir); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	cm, cached := m.(Cached)
	if opts.Cache > 0 && !cached {
		return nil, fmt.Errorf("llm: the model cannot be built with a KV cache")
	}
	list := &PartsList{Source: w.Source(), Format: opts.Format, Seq: opts.Seq, Cache: opts.Cache, Parts: SplitLayers(m.Layers(), n)}
	for i := range list.Parts {
		p := &list.Parts[i]
		pkg, err := mlpackage.Create(filepath.Join(dir, p.Package))
		if err != nil {
			return nil, err
		}
		if describe != nil {
			describe(pkg, *p)
		}
		o := opts
		o.Start, o.Layers = p.Start, p.Layers
		b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
		var ins, outs []mlpackage.Feature
		var y mlpackage.Feature
		if opts.Cache > 0 {
			ins, outs, err = cm.BuildCached(b, w, o)
		} else if ins, y, err = m.Build(b, w, o); err == nil {
			outs = []mlpackage.Feature{y}
		}
		if err != nil {
			return nil, fmt.Errorf("part %d (layers %d to %d): %w", i, p.Start, p.Start+p.Layers-1, err)
		}
		p.WeightBytes = pkg.Blobs.Size()
		var vars []*mlpackage.Var
		for _, f := range outs {
			vars = append(vars, &mlpackage.Var{Name: f.Name})
		}
		if err := pkg.Finish(b.Finish(vars...), 9, ins, outs); err != nil {
			return nil, err
		}
	}
	js, _ := json.MarshalIndent(list, "", "  ")
	return list, os.WriteFile(filepath.Join(dir, PartsFile), append(js, '\n'), 0o644)
}

// ReadParts reads the parts of a model: a directory written by WriteParts, or one package, which
// is then its only part.
func ReadParts(path string) (dir string, list *PartsList, err error) {
	raw, err := os.ReadFile(filepath.Join(path, PartsFile))
	if os.IsNotExist(err) {
		return filepath.Dir(path), &PartsList{Parts: []Part{{Package: filepath.Base(path)}}}, nil
	}
	if err != nil {
		return "", nil, err
	}
	list = &PartsList{}
	if err := json.Unmarshal(raw, list); err != nil || len(list.Parts) == 0 {
		return "", nil, fmt.Errorf("llm: %s: not a list of parts", filepath.Join(path, PartsFile))
	}
	return path, list, nil
}
