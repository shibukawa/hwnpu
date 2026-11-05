package llm

import (
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm/paged"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Convert writes the model as one package at out, or, over the Neural Engine size limit of the
// target chip (Targets; "none" never splits) or when nParts asks for more, as parts
// out.partN.mlpackage listed in order by out.parts.json. It returns the path that loads the
// whole: out or the manifest. Progress and warnings go to log. With opts.Backend "intel" the
// parts are OpenVINO IR (out.xml, out.partN.xml), and out.parts.json is always written.
func Convert(a *Arch, model Model, w Weights, out, targetName string, nParts int, opts Options, log io.Writer) (string, error) {
	return ConvertTiers(a, model, w, out, targetName, nParts, []int{opts.Cache}, opts, log)
}

// ConvertTiers is Convert with the model written once per KV cache length in caches (increasing),
// as the tiers of out.parts.json: out.cN.mlpackage or out.cN.partK.mlpackage for each. A step
// costs in proportion to the cache's length, so a sequence runs on the shortest tier and moves up
// as it grows (internal/llm/chain). With one length it is Convert.
func ConvertTiers(a *Arch, model Model, w Weights, out, targetName string, nParts int, caches []int, opts Options, log io.Writer) (string, error) {
	if len(caches) == 0 {
		caches = []int{opts.Cache}
	}
	if opts.Paged > 0 {
		if len(caches) > 1 {
			return "", fmt.Errorf("llm: a model split at its attention (-paged) costs in proportion to the tokens used and has no tiers")
		}
		opts.Cache = caches[0]
		return convertPaged(a, model, w, out, targetName, opts, log)
	}
	intel := opts.Backend == "intel"
	switch opts.Backend {
	case "", "coreml":
	case "intel":
		if err := checkIntelFormats(opts); err != nil {
			return "", err
		}
	default:
		return "", fmt.Errorf("llm: unknown backend %q (coreml or intel)", opts.Backend)
	}
	for i, c := range caches {
		if i > 0 && c <= caches[i-1] {
			return "", fmt.Errorf("llm: cache lengths must increase: %v", caches)
		}
		lens, chunked := CacheChunks(c)
		if c > 0 && !chunked && c+opts.Seq > NeuralEngineMaxDim && !intel {
			fmt.Fprintf(log, "warning: attention over a cache of %d and %d tokens per call spans %d keys, over the Neural Engine's %d per dimension; it will run on the CPU. Use -cache %d or less, or more than %d (in chunks)\n",
				c, opts.Seq, c+opts.Seq, NeuralEngineMaxDim, NeuralEngineMaxDim-opts.Seq, CacheChunkOver)
		}
		if c > 0 && chunked {
			fmt.Fprintf(log, "note: the cache of %d is held in %d chunks of up to %d tokens\n", c, len(lens), CacheChunkLen)
		}
	}
	tiered := len(caches) > 1
	plan := []Part{{Layers: opts.Layers}}
	var target Target
	var err error
	if targetName != "none" {
		if target, err = LookupTargetFor(opts.Backend, targetName); err != nil {
			return "", err
		}
		if !target.Measured {
			fmt.Fprintf(log, "note: the %s limit (%.2f GiB) is not measured yet: %s\n", target.Name, float64(target.MaxBytes)/(1<<30), target.Note)
		}
	}
	if opts.Layers == 0 && (targetName != "none" || nParts > 1) {
		budget := int64(math.MaxInt64)
		if target.MaxBytes > 0 {
			budget = int64(SplitMargin * float64(target.MaxBytes))
		}
		if plan, err = PlanSplit(model, opts, budget, nParts); err != nil {
			return "", err
		}
	}
	ext := ".mlpackage"
	if intel {
		ext = ".xml"
	}
	base := strings.TrimSuffix(strings.TrimSuffix(out, ".mlpackage"), ".xml")
	man := Manifest{Model: a.Title, Source: w.Source(), Format: opts.Format, Seq: opts.Seq, Target: target.Name, MaxBytes: target.MaxBytes}
	if intel {
		man.Backend, man.Meta = opts.Backend, metadata(a, w, opts)
	}
	for _, cache := range caches {
		opts.Cache = cache
		tierBase := base
		if tiered {
			tierBase = fmt.Sprintf("%s.c%d", base, cache)
		}
		var recs []PartRecord
		for i, p := range plan {
			path := tierBase + ext
			if len(plan) > 1 {
				path = fmt.Sprintf("%s.part%d%s", tierBase, i+1, ext)
				opts.Start, opts.Layers = p.Start, p.Layers
			}
			rec, err := convertPart(a, model, w, path, opts, log)
			if err != nil {
				return "", err
			}
			if target.MaxBytes > 0 && rec.WeightBytes >= target.MaxBytes {
				msg := fmt.Sprintf("%s: weights of %.3f GiB are over the %s limit of %.3f GiB, so the NPU will not take it", path, float64(rec.WeightBytes)/(1<<30), target.Name, float64(target.MaxBytes)/(1<<30))
				if nParts == 0 {
					return "", fmt.Errorf("%s; try -parts %d", msg, len(plan)+1)
				}
				fmt.Fprintln(log, "warning:", msg)
			}
			recs = append(recs, rec)
		}
		if tiered {
			man.Tiers = append(man.Tiers, Tier{Cache: cache, Parts: recs})
		} else {
			man.Cache, man.Parts = cache, recs
		}
	}
	if !tiered && len(plan) == 1 && !intel {
		return out, nil
	}
	if err := WriteManifest(base+".parts.json", &man); err != nil {
		return "", err
	}
	if tiered {
		fmt.Fprintf(log, "%s: %d tiers of %d parts for %s; a sequence starts on the shortest cache and moves up\n", filepath.Base(base+".parts.json"), len(caches), len(plan), target.Name)
	} else {
		fmt.Fprintf(log, "%s: %d parts for %s, run in this order with each y bound as the next x\n", filepath.Base(base+".parts.json"), len(plan), target.Name)
	}
	return base + ".parts.json", nil
}

// The package metadata keys of the architecture's name and of its settings (MetaParam + the
// Param's name), as the program was built with them, and of the checkpoint, the weight format and
// the description (the IR's; a package has a field of its own).
const (
	MetaArch        = "com.github.shibukawa.hwnpu.arch"
	MetaParam       = "com.github.shibukawa.hwnpu.param."
	MetaSource      = "com.github.shibukawa.hwnpu.source"
	MetaFormat      = "com.github.shibukawa.hwnpu.format"
	MetaDescription = "com.github.shibukawa.hwnpu.description"
)

// metadata is what a part's metadata records of how it was built.
func metadata(a *Arch, w Weights, opts Options) map[string]string {
	m := map[string]string{MetaSource: w.Source(), MetaFormat: opts.Format, MetaArch: a.Name}
	for _, p := range a.Params {
		m[MetaParam+p.Name] = a.Param(opts, p.Name)
	}
	return m
}

// IntelFormat is the weight format for Intel's NPU that stands in for a Neural Engine one: int4
// for lut4 (the 4 bit format each reads fast) and int8 for the finer tables, which the
// architectures choose where 4 bits lose too much. Other formats are returned as they are.
func IntelFormat(f string) string {
	switch f {
	case "", "lut4":
		return "int4"
	case "lut6", "lut8":
		return "int8"
	}
	return f
}

// checkIntelFormats rejects lookup table weights, which Intel's NPU cannot run.
func checkIntelFormats(o Options) error {
	formats := []string{o.Format}
	for _, r := range o.FormatFor {
		_, f, _ := strings.Cut(r, "=")
		formats = append(formats, f)
	}
	for _, f := range formats {
		if strings.HasPrefix(f, "lut") || f == "" {
			return fmt.Errorf("llm: weight format %q does not run on Intel's NPU; use int4, int4-g128, int8 or fp16 (%s for %s)", f, IntelFormat(f), f)
		}
	}
	return nil
}

// convertPart writes one program to path, its weights shared with an identical program's next to
// it (shareWeights).
func convertPart(a *Arch, model Model, w Weights, path string, opts Options, log io.Writer) (PartRecord, error) {
	start := time.Now()
	stop := console.Spin(log, filepath.Base(path)) // a part takes up to a minute: show it goes on
	rec, err := WritePart(path, model, w, opts, func(pkg *mlpackage.Package) {
		pkg.Description = fmt.Sprintf("%s from %s, seq %d, %s (hwnpu-convert)", a.Title, w.Source(), opts.Seq, opts.Format)
		if opts.Start > 0 || opts.Layers > 0 {
			pkg.Description += fmt.Sprintf(", layers %d-%d", opts.Start, opts.Start+opts.Layers-1)
		}
		if opts.Cache > 0 {
			pkg.Description += fmt.Sprintf(", KV cache %d", opts.Cache)
		}
		for k, v := range metadata(a, w, opts) {
			pkg.Metadata[k] = v
		}
	})
	if err != nil {
		stop()
		io.WriteString(log, "\n")
		return rec, err
	}
	shareWeights(path)
	size := rec.WeightBytes
	if opts.Backend != "intel" {
		size = 0
		filepath.Walk(path, func(_ string, info os.FileInfo, _ error) error {
			if info != nil && !info.IsDir() {
				size += info.Size()
			}
			return nil
		})
	}
	var inputs []string
	for _, in := range rec.Inputs {
		inputs = append(inputs, fmt.Sprintf("%s %v", in.Name, in.Shape))
	}
	if len(inputs) > 4 {
		inputs = append(inputs[:2], fmt.Sprintf("and %d more", len(inputs)-2))
	}
	stop()
	fmt.Fprintf(log, ": input %s, output %s %v, %.2f GiB, %.0fs\n", strings.Join(inputs, ", "), rec.Output.Name, rec.Output.Shape, float64(size)/(1<<30), time.Since(start).Seconds())
	return rec, nil
}

// convertPaged writes a Paged model split at its attention (Options.Paged): the parts PlanPaged
// gives, out.partN.mlpackage, and the program without weights that attends over one chunk of the
// host's cache, out.attn.mlpackage, listed by out.parts.json, which it returns. For Intel's NPU
// they are OpenVINO IR, out.partN.xml and out.attn.xml.
func convertPaged(a *Arch, model Model, w Weights, out, targetName string, opts Options, log io.Writer) (string, error) {
	pm, ok := model.(Paged)
	if !ok {
		return "", fmt.Errorf("llm: %s cannot be split at its attention (-paged)", a.Title)
	}
	intel := opts.Backend == BackendIntel
	switch {
	case opts.Backend != "" && opts.Backend != "coreml" && !intel:
		return "", fmt.Errorf("llm: unknown backend %q (coreml or intel)", opts.Backend)
	case intel && checkIntelFormats(opts) != nil:
		return "", checkIntelFormats(opts)
	case opts.Cache <= 0:
		return "", fmt.Errorf("llm: -paged needs -cache, the longest sequence")
	case opts.Seq < 1 || opts.Seq > opts.Paged:
		return "", fmt.Errorf("llm: -paged %d takes 1 to %d tokens per call, not %d", opts.Paged, opts.Paged, opts.Seq)
	case opts.Paged > NeuralEngineMaxDim && !intel:
		return "", fmt.Errorf("llm: chunks of %d are over the Neural Engine's %d per dimension", opts.Paged, NeuralEngineMaxDim)
	case opts.Start != 0 || opts.Layers != 0:
		return "", fmt.Errorf("llm: -paged converts the whole model")
	}
	var target Target
	budget := int64(math.MaxInt64)
	if targetName != "none" {
		var err error
		if target, err = LookupTargetFor(opts.Backend, targetName); err != nil {
			return "", err
		}
		if !target.Measured {
			fmt.Fprintf(log, "note: the %s limit (%.2f GiB) is not measured yet: %s\n", target.Name, float64(target.MaxBytes)/(1<<30), target.Note)
		}
		budget = int64(SplitMargin * float64(target.MaxBytes))
	}
	plan, err := PlanPaged(model, opts, budget)
	if err != nil {
		return "", err
	}
	ext := ".mlpackage"
	if intel {
		ext = ".xml"
	}
	base := strings.TrimSuffix(strings.TrimSuffix(out, ".mlpackage"), ".xml")
	man := Manifest{Model: a.Title, Source: w.Source(), Format: opts.Format, Seq: opts.Seq, Cache: opts.Cache, Target: target.Name, MaxBytes: target.MaxBytes, Paged: opts.Paged}
	if intel {
		man.Backend, man.Meta = opts.Backend, metadata(a, w, opts)
	}
	for i, p := range plan {
		path := fmt.Sprintf("%s.part%d%s", base, i+1, ext)
		o := opts
		o.Start, o.Layers = p.Start, p.Layers
		rec, err := convertPart(a, model, w, path, o, log)
		if err != nil {
			return "", err
		}
		if target.MaxBytes > 0 && rec.WeightBytes >= target.MaxBytes {
			return "", fmt.Errorf("%s: weights of %.3f GiB are over the %s limit of %.3f GiB", path, float64(rec.WeightBytes)/(1<<30), target.Name, float64(target.MaxBytes)/(1<<30))
		}
		man.Parts = append(man.Parts, rec)
	}
	form := "grouped"
	if a.Param(opts, "attn") == "tile" {
		form = "tile"
	}
	apath := base + ".attn" + ext
	desc := fmt.Sprintf("%s attention over a chunk of %d cached tokens, %d per call, %s (hwnpu-convert)", a.Title, opts.Paged, opts.Seq, form)
	var ins, outs []mlpackage.Feature
	if intel {
		_, ins, outs, err = writeIR(apath, &mlpackage.Package{Description: desc}, func(b *mlpackage.Builder) ([]mlpackage.Feature, []mlpackage.Feature, error) {
			return paged.Build(b, pm.PagedShape(), form, opts.Seq, opts.Paged)
		})
	} else {
		ins, outs, err = paged.Write(apath, pm.PagedShape(), form, opts.Seq, opts.Paged, desc)
	}
	if err != nil {
		return "", err
	}
	man.Attention = &PartRecord{Path: apath, Inputs: records(ins...), Output: records(outs[0])[0], State: records(outs[1:]...)}
	if man.Mass, err = writeMass(base, ext, pm.PagedShape(), form, opts.Seq, opts.Paged, a.Title); err != nil {
		return "", err
	}
	if err := WriteManifest(base+".parts.json", &man); err != nil {
		return "", err
	}
	fmt.Fprintf(log, "%s: %d parts split at the attention of layers %v, which the host runs over its cache of up to %d tokens in chunks of %d (%s)\n",
		filepath.Base(base+".parts.json"), len(plan), pm.PagedLayers(), opts.Cache, opts.Paged, filepath.Base(apath))
	return base + ".parts.json", nil
}

// writeMass writes the mass program of a model split at its attention, base.mass.mlpackage, or
// for Intel's NPU (ext .xml) base.mass.xml.
func writeMass(base, ext string, sh paged.Shape, form string, seq, chunk int, title string) (*PartRecord, error) {
	path := base + ".mass" + ext
	desc := fmt.Sprintf("%s weights of a chunk of %d bucket summaries, %d per call, %s (hwnpu-convert)", title, chunk, seq, form)
	var ins, outs []mlpackage.Feature
	var err error
	if ext == ".xml" {
		_, ins, outs, err = writeIR(path, &mlpackage.Package{Description: desc}, func(b *mlpackage.Builder) ([]mlpackage.Feature, []mlpackage.Feature, error) {
			return paged.BuildMass(b, sh, form, seq, chunk)
		})
	} else {
		ins, outs, err = paged.WriteMass(path, sh, form, seq, chunk, desc)
	}
	if err != nil {
		return nil, err
	}
	return &PartRecord{Path: path, Inputs: records(ins...), Output: records(outs[0])[0]}, nil
}

// AddMass adds the mass program of the two-level attention (decode.Sparse) to a model split at
// its attention converted before there was one: it has no weights, so it is written from the
// attention program's shape, next to it, in the attention's form ("grouped" or "tile", Build),
// with the model's score scale (paged.Shape.Scale: 0 for 1/sqrt(head_dim), 1 for Gemma 4), as IR
// for a manifest of Intel's NPU. It does nothing for a manifest that has it, or that is not split
// at its attention.
func AddMass(path, form string, scale float64) error {
	man, err := ReadManifest(path)
	if err != nil {
		return err
	}
	if man.Paged == 0 || man.Attention == nil || man.Mass != nil {
		return nil
	}
	var q, k []int
	for _, in := range man.Attention.Inputs {
		switch in.Name {
		case "q":
			q = in.Shape
		case "k":
			k = in.Shape
		}
	}
	if len(q) != 4 || len(k) != 4 {
		return fmt.Errorf("llm: %s: the attention program has no q and k of 4 dimensions", path)
	}
	base, ext := strings.TrimSuffix(path, ".parts.json"), ".mlpackage"
	if man.Backend == BackendIntel {
		ext = ".xml"
	}
	if man.Mass, err = writeMass(base, ext, paged.Shape{Heads: q[1], KVHeads: k[1], HeadDim: q[3], Scale: scale}, form, q[2], k[2], man.Model); err != nil {
		return err
	}
	return WriteManifest(path, man)
}
