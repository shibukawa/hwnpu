// hwnpu-convert turns a Hugging Face checkpoint into a Core ML package for the Neural Engine,
// in Go alone (no coremltools). The supported architectures are a whitelist (internal/llm/archs);
// -archs lists them.
//
//	hwnpu-convert -config DIR/config.json -weights DIR/model.safetensors -o out.mlpackage [-seq 32] [-format lut4]
//
// The architecture is taken from config.json (model_type, then architectures) unless -arch names
// it. The weights may also be a GGUF (its blocks are dequantized first). The program's input is
// x, the fp16 token embeddings (1, seq, hidden), and the output y the logits (1, seq, vocab) or,
// for an encoder, the last hidden state; the embedding table itself stays with the host. Encoders
// take a second input, mask (1, seq): 1 for tokens, 0 for padding. Settings of one architecture only (such as LFM3's
// -post-scale) are flags too, and are ignored by the other architectures.
//
// The Neural Engine takes a model only when its weight file is under a size that depends on the
// chip (1 GiB on the M3, 2 GB on the M4); a larger one is placed on the CPU as a whole. -target
// names the chip (default any: the smallest known limit), and a model over its limit is split
// into the fewest parts under it, balanced by size: out.part1.mlpackage, out.part2.mlpackage, …
// and out.parts.json, which lists them in order. Each part takes x, the hidden state entering its
// first layer (the token embeddings for the first part), and the other inputs such as the mask,
// and its y is the next part's x; the last part's y is the model's output. On the M3 the parts
// run as fast as the whole would (PROBE.md, npuprobe split). -parts sets the number of parts,
// and -target none turns splitting off.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/llm"
	_ "github.com/shibukawa/hwnpu/internal/llm/archs"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

func main() {
	config := flag.String("config", "", "config.json of the model")
	weights := flag.String("weights", "", "checkpoint: .safetensors (bf16) or .gguf")
	out := flag.String("o", "", "output .mlpackage directory (replaced)")
	arch := flag.String("arch", "", "architecture (default: detected from config.json)")
	list := flag.Bool("archs", false, "list the supported architectures and exit")
	seq := flag.Int("seq", 32, "fixed sequence length")
	layers := flag.Int("layers", 0, "keep only the first N layers and output their hidden state (0: whole model)")
	hidden := flag.Bool("hidden", false, "output the final hidden state instead of logits")
	target := flag.String("target", "any", "chip whose Neural Engine size limit decides the split: "+targetNames()+", any (the smallest limit) or none (never split)")
	parts := flag.Int("parts", 0, "number of parts (0: the fewest under the -target limit)")
	var opts llm.Options
	setOpts := llm.BindFlags(flag.CommandLine, &opts, "lut4")
	flag.Parse()
	if *list {
		for _, a := range llm.Archs() {
			fmt.Printf("%s\t%s (model_type %s)\n", a.Name, a.Title, strings.Join(a.ModelTypes, ", "))
		}
		fmt.Println("\ntargets (-target):")
		for _, t := range llm.Targets {
			fmt.Printf("%s\t%.2f GiB per model\t%s\n", t.Name, float64(t.MaxBytes)/(1<<30), t.Note)
		}
		return
	}
	if *config == "" || *weights == "" || *out == "" {
		flag.Usage()
		os.Exit(2)
	}
	if *parts < 0 || *parts > 1 && *layers > 0 {
		fmt.Fprintln(os.Stderr, "hwnpu-convert: -parts must be 0 or more, and cannot split a model truncated with -layers")
		os.Exit(2)
	}
	setOpts()
	opts.Seq, opts.Layers, opts.Hidden = *seq, *layers, *hidden
	if err := run(*config, *arch, *weights, *out, *target, *parts, opts); err != nil {
		fmt.Fprintln(os.Stderr, "hwnpu-convert:", err)
		os.Exit(1)
	}
}

func targetNames() string {
	var n []string
	for _, t := range llm.Targets {
		n = append(n, t.Name)
	}
	return strings.Join(n, ", ")
}

// manifest is out.parts.json: the parts of a split model in the order they run.
type manifest struct {
	Model    string       `json:"model"`
	Source   string       `json:"source"`
	Format   string       `json:"format"`
	Seq      int          `json:"seq"`
	Target   string       `json:"target"`
	MaxBytes int64        `json:"max_bytes"` // the target's limit on a part's weight file
	Parts    []partRecord `json:"parts"`
}

type partRecord struct {
	Path        string    `json:"path"` // relative to the manifest
	Start       int       `json:"start"`
	Layers      int       `json:"layers"`
	WeightBytes int64     `json:"weight_bytes"`
	Inputs      []feature `json:"inputs"`
	Output      feature   `json:"output"`
}

type feature struct {
	Name  string `json:"name"`
	DType string `json:"dtype"`
	Shape []int  `json:"shape"`
}

func features(fs ...mlpackage.Feature) []feature {
	var out []feature
	for _, f := range fs {
		out = append(out, feature{f.Name, strings.ToLower(f.DType.String()), f.Shape})
	}
	return out
}

func run(config, arch, weights, out, targetName string, nParts int, opts llm.Options) error {
	a, model, err := llm.LoadConfig(config, arch)
	if err != nil {
		return err
	}
	w, err := llm.OpenWeights(weights, a)
	if err != nil {
		return err
	}
	defer w.Close()

	plan := []llm.Part{{Layers: opts.Layers}}
	var target llm.Target
	if targetName != "none" {
		if target, err = llm.LookupTarget(targetName); err != nil {
			return err
		}
		if !target.Measured {
			fmt.Printf("note: the %s limit (%.2f GiB) is not measured yet: %s\n", target.Name, float64(target.MaxBytes)/(1<<30), target.Note)
		}
	}
	if opts.Layers == 0 && (targetName != "none" || nParts > 1) {
		budget := int64(math.MaxInt64)
		if target.MaxBytes > 0 {
			budget = int64(llm.SplitMargin * float64(target.MaxBytes))
		}
		if plan, err = llm.PlanSplit(model, opts, budget, nParts); err != nil {
			return err
		}
	}
	base := strings.TrimSuffix(out, ".mlpackage")
	man := manifest{Model: a.Title, Source: w.Source(), Format: opts.Format, Seq: opts.Seq, Target: target.Name, MaxBytes: target.MaxBytes}
	for i, p := range plan {
		path := out
		if len(plan) > 1 {
			path = fmt.Sprintf("%s.part%d.mlpackage", base, i+1)
			opts.Start, opts.Layers = p.Start, p.Layers
		}
		rec, err := buildPart(a, model, w, path, opts)
		if err != nil {
			return err
		}
		if target.MaxBytes > 0 && rec.WeightBytes >= target.MaxBytes {
			msg := fmt.Sprintf("%s: weights of %.3f GiB are over the %s limit of %.3f GiB, so the Neural Engine will not take it", path, float64(rec.WeightBytes)/(1<<30), target.Name, float64(target.MaxBytes)/(1<<30))
			if nParts == 0 {
				return fmt.Errorf("%s; try -parts %d", msg, len(plan)+1)
			}
			fmt.Println("warning:", msg)
		}
		if len(plan) > 1 {
			rec.Path = filepath.Base(path)
			rec.Start, rec.Layers = p.Start, p.Layers
			man.Parts = append(man.Parts, rec)
		}
	}
	if len(plan) > 1 {
		js, _ := json.MarshalIndent(man, "", "  ")
		if err := os.WriteFile(base+".parts.json", append(js, '\n'), 0o644); err != nil {
			return err
		}
		fmt.Printf("%s: %d parts for %s, run in this order with each y bound as the next x\n", base+".parts.json", len(plan), target.Name)
	}
	return nil
}

// buildPart writes one program to path.
func buildPart(a *llm.Arch, model llm.Model, w llm.Weights, path string, opts llm.Options) (partRecord, error) {
	start := time.Now()
	pkg, err := mlpackage.Create(path)
	if err != nil {
		return partRecord{}, err
	}
	pkg.Description = fmt.Sprintf("%s from %s, seq %d, %s (hwnpu-convert)", a.Title, w.Source(), opts.Seq, opts.Format)
	if opts.Start > 0 || opts.Layers > 0 {
		pkg.Description += fmt.Sprintf(", layers %d-%d", opts.Start, opts.Start+opts.Layers-1)
	}
	pkg.Metadata["com.github.shibukawa.hwnpu.source"] = w.Source()
	pkg.Metadata["com.github.shibukawa.hwnpu.format"] = opts.Format
	b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
	ins, y, err := model.Build(b, w, opts)
	if err != nil {
		return partRecord{}, err
	}
	if err := pkg.Finish(b.Finish(&mlpackage.Var{Name: y.Name}), 9, ins, []mlpackage.Feature{y}); err != nil {
		return partRecord{}, err
	}
	var size int64
	filepath.Walk(path, func(_ string, info os.FileInfo, _ error) error {
		if info != nil && !info.IsDir() {
			size += info.Size()
		}
		return nil
	})
	rec := partRecord{Inputs: features(ins...), Output: features(y)[0]}
	if info, err := os.Stat(filepath.Join(path, "Data", "com.apple.CoreML", "weights", "weight.bin")); err == nil {
		rec.WeightBytes = info.Size()
	}
	var inputs []string
	for _, in := range ins {
		inputs = append(inputs, fmt.Sprintf("%s %v", in.Name, in.Shape))
	}
	fmt.Printf("%s: input %s, output %s %v, %.2f GiB, %.0fs\n", path, strings.Join(inputs, ", "), y.Name, y.Shape, float64(size)/(1<<30), time.Since(start).Seconds())
	return rec, nil
}
