// hwnpu-convert turns a Hugging Face checkpoint into a Core ML package for the Neural Engine,
// in Go alone (no coremltools). Only the LFM3 architecture is supported so far.
//
//	hwnpu-convert -config DIR/config.json -weights DIR/model.safetensors -o out.mlpackage [-seq 32] [-format lut4]
//
// The weights may also be a GGUF (its blocks are dequantized first). The program's input is x,
// the fp16 token embeddings (1, seq, hidden), and the output y the logits (1, seq, vocab); the
// embedding table itself stays with the host.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/llm/lfm3"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

func main() {
	config := flag.String("config", "", "config.json of the model")
	weights := flag.String("weights", "", "checkpoint: .safetensors (bf16) or .gguf")
	out := flag.String("o", "", "output .mlpackage directory (replaced)")
	seq := flag.Int("seq", 32, "fixed sequence length")
	layers := flag.Int("layers", 0, "keep only the first N layers and output their hidden state (0: whole model)")
	hidden := flag.Bool("hidden", false, "output the final hidden state instead of logits")
	format := flag.String("format", "lut4", "weight format: lut4 (4 bit palettized) or fp16")
	group := flag.Int("group", 16, "rows per lookup table for lut4")
	colBlock := flag.Int("colblock", 0, "columns per lookup table for lut4 (0: whole rows)")
	keep := flag.String("keep-fp16", "", "comma separated substrings of tensor names to keep in fp16, e.g. embed_tokens")
	postScale := flag.Float64("post-scale", 64, "power of two multiplied into the weights of out_proj, o_proj and w2 and absorbed by the sandwich norms (Neural Engine accuracy); 1 disables")
	quant := flag.String("quant", "uniform", "lut4 table fitting: uniform, kmeans or kmeans-minmax")
	convMode := flag.String("conv", "taps", "short convolution as shifted multiply adds (taps) or one conv op (conv)")
	flag.Parse()
	if *config == "" || *weights == "" || *out == "" {
		flag.Usage()
		os.Exit(2)
	}
	opts := lfm3.Options{Seq: *seq, Layers: *layers, Hidden: *hidden, Format: *format, Group: *group, ColBlock: *colBlock, ConvMode: *convMode, Quant: *quant, PostScale: float32(*postScale)}
	if *keep != "" {
		opts.KeepFP16 = strings.Split(*keep, ",")
	}
	if err := run(*config, *weights, *out, opts); err != nil {
		fmt.Fprintln(os.Stderr, "hwnpu-convert:", err)
		os.Exit(1)
	}
}

func run(config, weights, out string, opts lfm3.Options) error {
	cfg, err := lfm3.LoadConfig(config)
	if err != nil {
		return err
	}
	w, err := lfm3.OpenWeights(weights)
	if err != nil {
		return err
	}
	defer w.Close()
	start := time.Now()
	pkg, err := mlpackage.Create(out)
	if err != nil {
		return err
	}
	pkg.Description = fmt.Sprintf("LFM3 from %s, seq %d, %s (hwnpu-convert)", w.Source(), opts.Seq, opts.Format)
	pkg.Metadata["com.github.shibukawa.hwnpu.source"] = w.Source()
	pkg.Metadata["com.github.shibukawa.hwnpu.format"] = opts.Format
	b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
	in, y, err := lfm3.Build(b, cfg, w, opts)
	if err != nil {
		return err
	}
	if err := pkg.Finish(b.Finish(&mlpackage.Var{Name: y.Name}), 9, []mlpackage.Feature{in}, []mlpackage.Feature{y}); err != nil {
		return err
	}
	var size int64
	filepath.Walk(out, func(_ string, info os.FileInfo, _ error) error {
		if info != nil && !info.IsDir() {
			size += info.Size()
		}
		return nil
	})
	fmt.Printf("%s: input %s %v, output %s %v, %.2f GiB, %.0fs\n", out, in.Name, in.Shape, y.Name, y.Shape, float64(size)/(1<<30), time.Since(start).Seconds())
	return nil
}
