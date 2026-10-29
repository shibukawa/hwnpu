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
	opts.Seq, opts.Layers, opts.Hidden = *seq, *layers, *hidden
	if err := run(*config, *arch, *weights, *out, opts); err != nil {
		fmt.Fprintln(os.Stderr, "hwnpu-convert:", err)
		os.Exit(1)
	}
}

func run(config, arch, weights, out string, opts llm.Options) error {
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
	pkg, err := mlpackage.Create(out)
	if err != nil {
		return err
	}
	pkg.Description = fmt.Sprintf("%s from %s, seq %d, %s (hwnpu-convert)", a.Title, w.Source(), opts.Seq, opts.Format)
	pkg.Metadata["com.github.shibukawa.hwnpu.source"] = w.Source()
	pkg.Metadata["com.github.shibukawa.hwnpu.format"] = opts.Format
	b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
	ins, y, err := model.Build(b, w, opts)
	if err != nil {
		return err
	}
	if err := pkg.Finish(b.Finish(&mlpackage.Var{Name: y.Name}), 9, ins, []mlpackage.Feature{y}); err != nil {
		return err
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
