package llm

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Manifest is out.parts.json, written by hwnpu-convert for a split model: the parts in the
// order they run, each part's y bound as the next part's x.
type Manifest struct {
	Model    string       `json:"model"`
	Source   string       `json:"source"`
	Format   string       `json:"format"`
	Seq      int          `json:"seq"`
	Cache    int          `json:"cache,omitempty"` // KV cache length (Options.Cache)
	Target   string       `json:"target"`
	MaxBytes int64        `json:"max_bytes"` // the target's limit on a part's weight file
	Parts    []PartRecord `json:"parts,omitempty"`
	// Tiers are the same model converted with KV caches of increasing length (hwnpu-convert
	// -cache a,b,c). A step costs in proportion to the cache's length, so a sequence starts on the
	// shortest tier and moves to the next when it fills (internal/llm/chain). With tiers, Cache
	// and Parts are empty.
	Tiers []Tier `json:"tiers,omitempty"`
}

// Tier is one cache length of a tiered model.
type Tier struct {
	Cache int          `json:"cache"`
	Parts []PartRecord `json:"parts"`
}

// Tiered returns the model's tiers in order of cache length: Tiers, or the single tier Cache and
// Parts describe.
func (m *Manifest) Tiered() []Tier {
	if len(m.Tiers) > 0 {
		return m.Tiers
	}
	return []Tier{{Cache: m.Cache, Parts: m.Parts}}
}

// PartRecord is one part in a Manifest.
type PartRecord struct {
	Path        string          `json:"path"` // relative to the manifest
	Start       int             `json:"start"`
	Layers      int             `json:"layers"`
	WeightBytes int64           `json:"weight_bytes"`
	Inputs      []FeatureRecord `json:"inputs"`
	Output      FeatureRecord   `json:"output"`
	// Outputs besides y: a program with a KV cache outputs the new keys, values and windows.
	State []FeatureRecord `json:"state_outputs,omitempty"`
}

// FeatureRecord describes an input or output.
type FeatureRecord struct {
	Name  string `json:"name"`
	DType string `json:"dtype"`
	Shape []int  `json:"shape"`
}

func records(fs ...mlpackage.Feature) []FeatureRecord {
	var out []FeatureRecord
	for _, f := range fs {
		out = append(out, FeatureRecord{f.Name, strings.ToLower(f.DType.String()), f.Shape})
	}
	return out
}

// WritePart writes the program for opts (one part when opts.Start or opts.Layers is set) to the
// package at path. With opts.Cache the model must be Cached, and the part has the cache inputs and
// outputs of its own layers. describe fills in the package's description and metadata.
func WritePart(path string, m Model, w Weights, opts Options, describe func(*mlpackage.Package)) (PartRecord, error) {
	pkg, err := mlpackage.Create(path)
	if err != nil {
		return PartRecord{}, err
	}
	if describe != nil {
		describe(pkg)
	}
	b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
	var ins, outs []mlpackage.Feature
	if opts.Cache > 0 {
		cm, ok := m.(Cached)
		if !ok {
			return PartRecord{}, fmt.Errorf("llm: the model cannot be built with a KV cache")
		}
		ins, outs, err = cm.BuildCached(b, w, opts)
	} else {
		var y mlpackage.Feature
		if ins, y, err = m.Build(b, w, opts); err == nil {
			outs = []mlpackage.Feature{y}
		}
	}
	if err != nil {
		return PartRecord{}, err
	}
	var vars []*mlpackage.Var
	for _, f := range outs {
		vars = append(vars, &mlpackage.Var{Name: f.Name})
	}
	if err := pkg.Finish(b.Finish(vars...), 9, ins, outs); err != nil {
		return PartRecord{}, err
	}
	rec := PartRecord{Path: path, Start: opts.Start, Layers: opts.Layers, Inputs: records(ins...), Output: records(outs[0])[0], State: records(outs[1:]...)}
	if info, err := os.Stat(filepath.Join(path, "Data", "com.apple.CoreML", "weights", "weight.bin")); err == nil {
		rec.WeightBytes = info.Size()
	}
	return rec, nil
}

// WriteManifest writes m to path (out.parts.json); the parts' paths are made relative to it.
func WriteManifest(path string, m *Manifest) error {
	relative := func(parts []PartRecord) {
		for i := range parts {
			if rel, err := filepath.Rel(filepath.Dir(path), parts[i].Path); err == nil {
				parts[i].Path = rel
			}
		}
	}
	relative(m.Parts)
	for _, t := range m.Tiers {
		relative(t.Parts)
	}
	js, _ := json.MarshalIndent(m, "", "  ")
	return os.WriteFile(path, append(js, '\n'), 0o644)
}

// ReadManifest reads the parts of a model: a manifest (out.parts.json), or one package, which is
// then its only part. The parts' paths are returned relative to the working directory.
func ReadManifest(path string) (*Manifest, error) {
	if !strings.HasSuffix(path, ".json") {
		return &Manifest{Parts: []PartRecord{{Path: path}}}, nil
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	m := &Manifest{}
	if err := json.Unmarshal(raw, m); err != nil || len(m.Parts) == 0 && len(m.Tiers) == 0 {
		return nil, fmt.Errorf("llm: %s is not a list of parts", path)
	}
	absolute := func(parts []PartRecord) {
		for i := range parts {
			parts[i].Path = filepath.Join(filepath.Dir(path), parts[i].Path)
		}
	}
	absolute(m.Parts)
	for i, t := range m.Tiers {
		if len(t.Parts) == 0 || i > 0 && t.Cache <= m.Tiers[i-1].Cache {
			return nil, fmt.Errorf("llm: %s: tiers must have parts and increasing caches", path)
		}
		absolute(t.Parts)
	}
	return m, nil
}
