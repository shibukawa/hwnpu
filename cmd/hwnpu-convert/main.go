// hwnpu-convert turns a Hugging Face checkpoint into a Core ML package for the Neural Engine,
// in Go alone (no coremltools). The supported architectures are a whitelist (internal/llm/archs);
// -archs lists them.
//
//	hwnpu-convert -config DIR/config.json -weights DIR/model.safetensors -o out.mlpackage [-seq 32] [-format lut4]
//
// The architecture is taken from config.json (model_type, then architectures) unless -arch names
// it. The weights may also be a GGUF (its blocks are dequantized first). The program's input is
// x, the fp16 token embeddings (1, seq, hidden), and the output y the logits (1, seq, vocab) or,
// for an encoder or a d1 decision model, the last hidden state; the embedding table itself stays
// with the host. Encoders and d1 models take a second input, mask (1, seq): 1 for tokens, 0 for
// padding. Settings of one architecture only (such as LFM3's -post-scale) are flags too, and are
// ignored by the other architectures.
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
//
// Some checkpoints hold more than one program. LFM3-VL's config.json selects its text model
// (lfm3); -arch lfm3-vision converts its vision encoder instead, for one image size (-image
// WIDTHxHEIGHT, default 512x512; -seq is not used): x is the image's patches and y its
// embeddings in the text model's input (internal/vision runs it). LFM3-Audio's and LFM3-Omni's
// audio encoder is -arch lfm3-audio-encoder, for clips of up to -frames mel frames (10 ms each,
// default 3000): x is the clip's log mel frames, mask marks them, y is the embeddings
// (internal/audio runs it).
//
// -cache N builds a causal LM to run with a KV cache of N tokens held by the host
// (internal/llm/decode): -seq is then the tokens per call (1 to generate), and each part takes the
// cached keys, values and convolution windows of its own layers as inputs and outputs the new
// ones. -cache a,b,c writes a tier per length, and generation starts on the shortest tier and
// moves up as the sequence grows (internal/llm/chain). cmd/hwnpu-generate runs it.
package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/shibukawa/hwnpu/internal/llm"
	_ "github.com/shibukawa/hwnpu/internal/llm/archs"
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
	cache := flag.String("cache", "0", "build a causal LM to run with a KV cache of this many tokens held by the host; -seq is then the tokens per call (1 to generate). Several lengths, comma separated and increasing, write a tier per length: a sequence runs on the shortest and moves up as it grows")
	var opts llm.Options
	setOpts := llm.BindFlags(flag.CommandLine, &opts, "")
	flag.Parse()
	if *list {
		for _, a := range llm.Archs() {
			if len(a.ModelTypes) > 0 {
				fmt.Printf("%s\t%s (model_type %s)\n", a.Name, a.Title, strings.Join(a.ModelTypes, ", "))
			} else {
				fmt.Printf("%s\t%s (architectures %s)\n", a.Name, a.Title, strings.Join(a.Architectures, ", "))
			}
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
	var caches []int
	for _, s := range strings.Split(*cache, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil || n < 0 || len(caches) > 0 && (n <= caches[len(caches)-1] || caches[0] == 0) {
			fmt.Fprintln(os.Stderr, "hwnpu-convert: -cache takes a length, or increasing lengths separated by commas")
			os.Exit(2)
		}
		caches = append(caches, n)
	}
	opts.Cache = caches[0]
	if err := run(*config, *arch, *weights, *out, *target, *parts, caches, opts); err != nil {
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

func run(config, arch, weights, out, targetName string, nParts int, caches []int, opts llm.Options) error {
	a, model, err := llm.LoadConfig(config, arch)
	if err != nil {
		return err
	}
	w, err := llm.OpenWeights(weights, a)
	if err != nil {
		return err
	}
	defer w.Close()
	if opts.Format == "" {
		opts.Format = a.DefaultFormat()
	}
	_, err = llm.ConvertTiers(a, model, w, out, targetName, nParts, caches, opts, os.Stdout)
	return err
}
