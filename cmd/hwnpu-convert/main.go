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
// take a second input, mask (1, seq): 1 for tokens, 0 for padding. -parts N splits the layers
// into N packages in the directory -o, chained x to y (internal/llm/chain loads them). Settings of one architecture only (such as LFM3's
// -post-scale) are flags too, and are ignored by the other architectures.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/llm"
	_ "github.com/shibukawa/hwnpu/internal/llm/archs"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/mlspec"
)

func main() {
	config := flag.String("config", "", "config.json of the model")
	weights := flag.String("weights", "", "checkpoint: .safetensors (bf16) or .gguf")
	out := flag.String("o", "", "output .mlpackage directory (replaced)")
	arch := flag.String("arch", "", "architecture (default: detected from config.json)")
	list := flag.Bool("archs", false, "list the supported architectures and exit")
	seq := flag.Int("seq", 32, "fixed sequence length")
	cache := flag.Int("cache", 0, "build a causal LM to run with a KV cache of this many tokens held by the host (-seq is then the tokens per call, 1 for decoding); needs -parts")
	layers := flag.Int("layers", 0, "keep only the first N layers and output their hidden state (0: whole model)")
	hidden := flag.Bool("hidden", false, "output the final hidden state instead of logits")
	parts := flag.Int("parts", 0, "split the layers into N parts: -o is then a directory of packages part_0.mlpackage ... and parts.json. The Neural Engine takes less than 1 GiB per package (M3), so a larger model runs in parts, and smaller parts use less memory")
	var opts llm.Options
	setOpts := llm.BindFlags(flag.CommandLine, &opts, "lut4")
	flag.Parse()
	if *list {
		for _, a := range llm.Archs() {
			fmt.Printf("%s\t%s (model_type %s)\n", a.Name, a.Title, strings.Join(a.ModelTypes, ", "))
		}
		return
	}
	if *config == "" || *weights == "" || *out == "" {
		flag.Usage()
		os.Exit(2)
	}
	setOpts()
	opts.Seq, opts.Layers, opts.Hidden, opts.Cache = *seq, *layers, *hidden, *cache
	if *cache > 0 && *parts == 0 {
		fmt.Fprintln(os.Stderr, "hwnpu-convert: -cache writes a directory of parts; give -parts (1 for one part)")
		os.Exit(2)
	}
	if *parts > 0 && *layers > 0 {
		fmt.Fprintln(os.Stderr, "hwnpu-convert: -parts splits the whole model; it does not go with -layers")
		os.Exit(2)
	}
	if err := run(*config, *arch, *weights, *out, opts, *parts); err != nil {
		fmt.Fprintln(os.Stderr, "hwnpu-convert:", err)
		os.Exit(1)
	}
}

func run(config, arch, weights, out string, opts llm.Options, parts int) error {
	a, model, err := llm.LoadConfig(config, arch)
	if err != nil {
		return err
	}
	w, err := llm.OpenWeights(weights, a)
	if err != nil {
		return err
	}
	defer w.Close()
	start := time.Now()
	describe := func(pkg *mlpackage.Package) {
		pkg.Description = fmt.Sprintf("%s from %s, seq %d, %s (hwnpu-convert)", a.Title, w.Source(), opts.Seq, opts.Format)
		pkg.Metadata["com.github.shibukawa.hwnpu.source"] = w.Source()
		pkg.Metadata["com.github.shibukawa.hwnpu.format"] = opts.Format
	}
	var ins []mlpackage.Feature
	var y mlpackage.Feature
	if parts > 0 {
		list, err := llm.WriteParts(out, model, w, opts, parts, func(pkg *mlpackage.Package, p llm.Part) {
			describe(pkg)
			pkg.Description += fmt.Sprintf(", layers %d to %d", p.Start, p.Start+p.Layers-1)
		})
		if err != nil {
			return err
		}
		for _, p := range list.Parts {
			fmt.Printf("  %s: layers %d to %d, %.2f GiB\n", p.Package, p.Start, p.Start+p.Layers-1, float64(p.WeightBytes)/(1<<30))
		}
		// The inputs of the first part and the output of the last.
		first, last := filepath.Join(out, list.Parts[0].Package), filepath.Join(out, list.Parts[len(list.Parts)-1].Package)
		if ins, _, err = features(first); err != nil {
			return err
		}
		if _, y, err = features(last); err != nil {
			return err
		}
	} else {
		pkg, err := mlpackage.Create(out)
		if err != nil {
			return err
		}
		describe(pkg)
		b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
		if ins, y, err = model.Build(b, w, opts); err != nil {
			return err
		}
		if err := pkg.Finish(b.Finish(&mlpackage.Var{Name: y.Name}), 9, ins, []mlpackage.Feature{y}); err != nil {
			return err
		}
	}
	var size int64
	filepath.Walk(out, func(_ string, info os.FileInfo, _ error) error {
		if info != nil && !info.IsDir() {
			size += info.Size()
		}
		return nil
	})
	var inputs []string
	for _, in := range ins {
		inputs = append(inputs, fmt.Sprintf("%s %v", in.Name, in.Shape))
	}
	fmt.Printf("%s: input %s, output %s %v, %.2f GiB, %.0fs\n", out, strings.Join(inputs, ", "), y.Name, y.Shape, float64(size)/(1<<30), time.Since(start).Seconds())
	return nil
}

// features reads the inputs and the first output of a written package.
func features(path string) (ins []mlpackage.Feature, out mlpackage.Feature, err error) {
	m, err := mlpackage.ReadModel(path)
	if err != nil {
		return nil, out, err
	}
	conv := func(fd *mlspec.FeatureDescription) mlpackage.Feature {
		f := mlpackage.Feature{Name: fd.GetName()}
		for _, d := range fd.GetType().GetMultiArrayType().GetShape() {
			f.Shape = append(f.Shape, int(d))
		}
		return f
	}
	for _, fd := range m.GetDescription().GetInput() {
		ins = append(ins, conv(fd))
	}
	if outs := m.GetDescription().GetOutput(); len(outs) > 0 {
		out = conv(outs[0])
	}
	return ins, out, nil
}
