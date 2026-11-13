package llm

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/shibukawa/hwnpu/internal/coreai/ailower"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/ovir"
	"github.com/shibukawa/hwnpu/internal/ovlower"
)

// Manifest is out.parts.json, written by hwnpu-dev convert for a split model: the parts in the
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
	// Tiers are the same model converted with KV caches of increasing length (hwnpu-dev convert
	// -cache a,b,c). A step costs in proportion to the cache's length, so a sequence starts on the
	// shortest tier and moves to the next when it fills (internal/llm/chain). With tiers, Cache
	// and Parts are empty.
	Tiers []Tier `json:"tiers,omitempty"`
	// Paged is the chunk length of a model split at its attention (Options.Paged), and Attention
	// the program without weights that attends over one chunk for Seq query rows
	// (internal/llm/paged); Cache is then the longest sequence the host's cache holds.
	Paged     int         `json:"paged,omitempty"`
	Attention *PartRecord `json:"attention,omitempty"`
	// Mass is the program without weights that weighs one chunk of bucket summaries for the
	// two-level attention (decode.Sparse, paged.BuildMass), of a model split at its attention.
	// Conversions before it have none (AddMass).
	Mass *PartRecord `json:"mass,omitempty"`
	// KVOnly marks a model split at its attention converted with Options.KVOnly: its parts end
	// with the last layer that writes a cache, and the program outputs no logits.
	KVOnly bool `json:"kv_only,omitempty"`
	// Backend is the runtime the parts are written for (Options.Backend; empty for coreml).
	Backend string `json:"backend,omitempty"`
	// Meta is the metadata of the parts (MetaArch, MetaParam...), also in each part's own
	// metadata: a package's user defined metadata or an IR's rt_info.
	Meta map[string]string `json:"meta,omitempty"`
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

// markSoftcap writes into the part's metadata the soft cap its logits leave to the host
// (MetaSoftcap), whatever describe wrote: without it the host would take them as capped.
func markSoftcap(pkg *mlpackage.Package, m Model, opts Options) {
	if lc, ok := m.(LogitSoftcap); ok {
		if c := lc.HostSoftcap(opts); c > 0 {
			if pkg.Metadata == nil {
				pkg.Metadata = map[string]string{}
			}
			pkg.Metadata[MetaSoftcap] = strconv.FormatFloat(c, 'g', -1, 64)
		}
	}
}

// WritePart writes the program for opts (one part when opts.Start or opts.Layers is set) to the
// package at path. With opts.Cache the model must be Cached, and the part has the cache inputs and
// outputs of its own layers. A Cached model is built with BuildCached also without a cache, so
// that a part keeps the outputs a later part takes by name (internal/llm/chain). describe fills
// in the package's description and metadata. With opts.Backend "intel" the part is OpenVINO IR:
// path is its .xml. With "coreai" it is a Core AI program: path is its .aimodel.
func WritePart(path string, m Model, w Weights, opts Options, describe func(*mlpackage.Package)) (PartRecord, error) {
	switch opts.Backend {
	case BackendIntel:
		return writeIRPart(path, m, w, opts, describe)
	case BackendCoreAI:
		return writeAIPart(path, m, w, opts, describe)
	}
	pkg, err := mlpackage.Create(path)
	if err != nil {
		return PartRecord{}, err
	}
	if describe != nil {
		describe(pkg)
	}
	markSoftcap(pkg, m, opts)
	b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
	ins, outs, err := buildPart(b, m, w, opts)
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

// buildPart builds the program for opts into b: Build, or BuildCached for a Cached model.
func buildPart(b *mlpackage.Builder, m Model, w Weights, opts Options) (ins, outs []mlpackage.Feature, err error) {
	cm, cached := m.(Cached)
	if opts.Cache > 0 && !cached {
		return nil, nil, fmt.Errorf("llm: the model cannot be built with a KV cache")
	}
	if cached {
		return cm.BuildCached(b, w, opts)
	}
	ins, y, err := m.Build(b, w, opts)
	if err != nil {
		return nil, nil, err
	}
	return ins, []mlpackage.Feature{y}, nil
}

// writeAIPart is WritePart for Core AI: the program is built as MIL, its weights in a temporary
// weight file next to path, and lowered to a Core AI program (internal/coreai/ailower) at path
// (.aimodel), the weights in its bytecode. No Core ML package is written. The description and
// metadata that describe fills in go into the asset's MetaFile.
func writeAIPart(path string, m Model, w Weights, opts Options, describe func(*mlpackage.Package)) (PartRecord, error) {
	pkg := &mlpackage.Package{Metadata: map[string]string{}}
	if describe != nil {
		describe(pkg)
	}
	markSoftcap(pkg, m, opts)
	size, ins, outs, err := writeAI(path, pkg.Metadata, func(b *mlpackage.Builder) ([]mlpackage.Feature, []mlpackage.Feature, error) {
		return buildPart(b, m, w, opts)
	})
	if err == nil && pkg.Description != "" {
		pkg.Metadata[MetaDescription] = pkg.Description
		err = writeAIMeta(path, pkg.Metadata)
	}
	if err != nil {
		return PartRecord{}, err
	}
	return PartRecord{Path: path, Start: opts.Start, Layers: opts.Layers, WeightBytes: size, Inputs: records(ins...), Output: records(outs[0])[0], State: records(outs[1:]...)}, nil
}

// WriteAI writes the program build makes as a Core AI program at path (.aimodel), with meta in
// its MetaFile, for a program that is not a Model (WritePart). It returns the size of the
// bytecode.
func WriteAI(path string, meta map[string]string, build func(*mlpackage.Builder) (ins, outs []mlpackage.Feature, err error)) (int64, error) {
	size, _, _, err := writeAI(path, meta, build)
	return size, err
}

// MetaFile is the file of a Core AI program hwnpu writes that holds its metadata (as a Core ML
// package's user-defined metadata, ReadMeta).
const MetaFile = "hwnpu-meta.json"

func writeAIMeta(path string, meta map[string]string) error {
	if len(meta) == 0 {
		return nil
	}
	b, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(path, MetaFile), append(b, '\n'), 0o644)
}

func writeAI(path string, meta map[string]string, build func(*mlpackage.Builder) (ins, outs []mlpackage.Feature, err error)) (int64, []mlpackage.Feature, []mlpackage.Feature, error) {
	tmp := strings.TrimSuffix(path, filepath.Ext(path)) + ".milblob.tmp"
	blobs, err := mlpackage.NewBlobWriter(tmp)
	if err != nil {
		return 0, nil, nil, err
	}
	defer os.Remove(tmp)
	b := mlpackage.NewBuilder("CoreML8", blobs)
	ins, outs, err := build(b)
	if err != nil {
		blobs.Close()
		return 0, nil, nil, err
	}
	var vars []*mlpackage.Var
	for _, f := range outs {
		vars = append(vars, &mlpackage.Var{Name: f.Name})
	}
	fn := b.Function(vars...)
	if err := blobs.Close(); err != nil {
		return 0, nil, nil, err
	}
	f, err := os.Open(tmp)
	if err != nil {
		return 0, nil, nil, err
	}
	defer f.Close()
	prog, err := ailower.Lower(fn, f)
	if err != nil {
		return 0, nil, nil, fmt.Errorf("%s: %w", filepath.Base(path), err)
	}
	if err := os.RemoveAll(path); err != nil {
		return 0, nil, nil, err
	}
	if err := prog.WriteAsset(path); err != nil {
		return 0, nil, nil, err
	}
	if err := writeAIMeta(path, meta); err != nil {
		return 0, nil, nil, err
	}
	return int64(len(prog.Bytecode)), ins, outs, nil
}

// writeIRPart is WritePart for Intel's NPU: the program is built as MIL, its weights in a
// temporary weight file next to path, and lowered to OpenVINO IR (path, .xml, and its .bin).
// The description and metadata that describe fills in go into the IR's rt_info.
func writeIRPart(path string, m Model, w Weights, opts Options, describe func(*mlpackage.Package)) (PartRecord, error) {
	pkg := &mlpackage.Package{Metadata: map[string]string{}}
	if describe != nil {
		describe(pkg)
	}
	markSoftcap(pkg, m, opts)
	size, ins, outs, err := writeIR(path, pkg, func(b *mlpackage.Builder) ([]mlpackage.Feature, []mlpackage.Feature, error) {
		return buildPart(b, m, w, opts)
	})
	if err != nil {
		return PartRecord{}, err
	}
	return PartRecord{Path: path, Start: opts.Start, Layers: opts.Layers, WeightBytes: size, Inputs: records(ins...), Output: records(outs[0])[0], State: records(outs[1:]...)}, nil
}

// WriteIR writes the program build makes as OpenVINO IR at path (.xml, and its .bin), with meta
// in its rt_info, for a program that is not a Model (WritePart). It returns the size of the
// weights.
func WriteIR(path string, meta map[string]string, build func(*mlpackage.Builder) (ins, outs []mlpackage.Feature, err error)) (int64, error) {
	size, _, _, err := writeIR(path, &mlpackage.Package{Metadata: meta}, build)
	return size, err
}

// writeIR builds a program as MIL, its weights in a temporary weight file next to path, and
// lowers it to OpenVINO IR (path, .xml, and its .bin), with the description and metadata of pkg
// in its rt_info. It returns the size of the weights and the program's inputs and outputs.
func writeIR(path string, pkg *mlpackage.Package, build func(*mlpackage.Builder) (ins, outs []mlpackage.Feature, err error)) (int64, []mlpackage.Feature, []mlpackage.Feature, error) {
	base := strings.TrimSuffix(path, filepath.Ext(path))
	tmp := base + ".milblob.tmp"
	blobs, err := mlpackage.NewBlobWriter(tmp)
	if err != nil {
		return 0, nil, nil, err
	}
	defer os.Remove(tmp)
	b := mlpackage.NewBuilder("CoreML8", blobs)
	ins, outs, err := build(b)
	if err != nil {
		blobs.Close()
		return 0, nil, nil, err
	}
	var vars []*mlpackage.Var
	for _, f := range outs {
		vars = append(vars, &mlpackage.Var{Name: f.Name})
	}
	fn := b.Function(vars...)
	if err := blobs.Close(); err != nil {
		return 0, nil, nil, err
	}
	f, err := os.Open(tmp)
	if err != nil {
		return 0, nil, nil, err
	}
	defer f.Close()
	meta := pkg.Metadata
	if meta == nil {
		meta = map[string]string{}
	}
	if pkg.Description != "" {
		meta[MetaDescription] = pkg.Description
	}
	size, err := ovlower.WriteFiles(path, filepath.Base(base), fn, f, meta)
	return size, ins, outs, err
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
	for _, r := range []**PartRecord{&m.Attention, &m.Mass} {
		if *r != nil {
			a := []PartRecord{**r}
			relative(a)
			*r = &a[0]
		}
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
	for _, r := range []*PartRecord{m.Attention, m.Mass} {
		if r != nil {
			r.Path = filepath.Join(filepath.Dir(path), r.Path)
		}
	}
	for i, t := range m.Tiers {
		if len(t.Parts) == 0 || i > 0 && t.Cache <= m.Tiers[i-1].Cache {
			return nil, fmt.Errorf("llm: %s: tiers must have parts and increasing caches", path)
		}
		absolute(t.Parts)
	}
	return m, nil
}

// ReadMeta returns the metadata of a model's parts (MetaArch, MetaParam...): the manifest's, or
// else the first part's own, a package's user defined metadata or an IR's rt_info.
func ReadMeta(m *Manifest) (map[string]string, error) {
	if len(m.Meta) > 0 {
		return m.Meta, nil
	}
	parts := m.Tiered()[0].Parts
	if len(parts) == 0 {
		return nil, errors.New("llm: no parts")
	}
	path := parts[0].Path
	if strings.EqualFold(filepath.Ext(path), ".xml") {
		raw, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		rt, err := ovir.ReadRTInfo(raw)
		if err != nil {
			return nil, err
		}
		meta := map[string]string{}
		for k, v := range rt {
			if strings.HasPrefix(k, "com.github.shibukawa.hwnpu.") {
				meta[k] = v
			}
		}
		return meta, nil
	}
	if strings.EqualFold(filepath.Ext(path), ".aimodel") {
		meta := map[string]string{}
		raw, err := os.ReadFile(filepath.Join(path, MetaFile))
		if errors.Is(err, os.ErrNotExist) {
			return meta, nil
		}
		if err == nil {
			err = json.Unmarshal(raw, &meta)
		}
		return meta, err
	}
	pkg, err := mlpackage.ReadModel(path)
	if err != nil {
		return nil, err
	}
	return pkg.GetDescription().GetMetadata().GetUserDefined(), nil
}
