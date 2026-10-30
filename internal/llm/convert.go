package llm

import (
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

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
		fmt.Fprintf(log, "%s: %d tiers of %d parts for %s; a sequence starts on the shortest cache and moves up\n", base+".parts.json", len(caches), len(plan), target.Name)
	} else {
		fmt.Fprintf(log, "%s: %d parts for %s, run in this order with each y bound as the next x\n", base+".parts.json", len(plan), target.Name)
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

// convertPart writes one program to path.
func convertPart(a *Arch, model Model, w Weights, path string, opts Options, log io.Writer) (PartRecord, error) {
	start := time.Now()
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
		return rec, err
	}
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
	fmt.Fprintf(log, "%s: input %s, output %s %v, %.2f GiB, %.0fs\n", path, strings.Join(inputs, ", "), rec.Output.Name, rec.Output.Shape, float64(size)/(1<<30), time.Since(start).Seconds())
	return rec, nil
}
