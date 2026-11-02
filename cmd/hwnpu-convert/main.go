// hwnpu-convert turns a Hugging Face checkpoint into a Core ML package for the Neural Engine,
// in Go alone (no coremltools). The supported architectures are a whitelist (internal/llm/archs);
// -archs lists them.
//
//	hwnpu-convert -config DIR/config.json -weights DIR/model.safetensors -o out.mlpackage [-seq 32] [-format lut4]
//	hwnpu-convert -backend intel -config DIR/config.json -weights DIR/model.safetensors -o out.xml [-format int4]
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
//
// -format q4nx writes 4 bit weights with a bf16 scale and minimum per 32 columns, the format of
// AMD's NPU kernels (docs/amd-xdna2-bench.md). The program holds them as Core ML's
// constexpr_blockwise_shift_scale, so that hwnpu-check and the CPU interpreter run it, and the
// same matrices go to out.q4nx in the kernel's 32 x 256 blocks with an index in out.q4nx.json.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/shibukawa/hwnpu/internal/llm"
	_ "github.com/shibukawa/hwnpu/internal/llm/archs"
	"github.com/shibukawa/hwnpu/internal/palette"
)

func main() {
	config := flag.String("config", "", "config.json of the model")
	weights := flag.String("weights", "", "checkpoint: .safetensors (bf16) or .gguf")
	out := flag.String("o", "", "output: an .mlpackage directory (replaced), or an .xml with -backend intel")
	backend := flag.String("backend", "coreml", "runtime to write for: coreml (an .mlpackage for the Neural Engine) or intel (OpenVINO IR for Intel's NPU, with out.parts.json)")
	arch := flag.String("arch", "", "architecture (default: detected from config.json)")
	list := flag.Bool("archs", false, "list the supported architectures and exit")
	seq := flag.Int("seq", 32, "fixed sequence length")
	layers := flag.Int("layers", 0, "keep only the first N layers and output their hidden state (0: whole model)")
	hidden := flag.Bool("hidden", false, "output the final hidden state instead of logits")
	target := flag.String("target", "any", "device whose NPU size limit decides the split: "+targetNames()+", any (the smallest limit of the backend's) or none (never split)")
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
	opts.Seq, opts.Layers, opts.Hidden, opts.Backend = *seq, *layers, *hidden, *backend
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
		opts.Format = a.DefaultFormatFor(model, opts.Backend)
	}
	var blocks *q4nxWriter
	if usesQ4NX(opts) {
		if blocks, err = newQ4NXWriter(strings.TrimSuffix(out, filepath.Ext(out))); err != nil {
			return err
		}
		opts.Q4NX = blocks.add
	}
	_, err = llm.ConvertTiers(a, model, w, out, targetName, nParts, caches, opts, os.Stdout)
	if blocks != nil {
		err = errors.Join(err, blocks.close())
	}
	return err
}

func usesQ4NX(o llm.Options) bool {
	if o.Format == "q4nx" {
		return true
	}
	for _, r := range o.FormatFor {
		if _, f, _ := strings.Cut(r, "="); f == "q4nx" {
			return true
		}
	}
	return false
}

// q4nxWriter writes the q4nx matrices of a conversion as the AMD NPU kernel reads them
// (palette.Q4NX.Blocks) to BASE.q4nx, each at a 64 byte aligned offset, and their index to
// BASE.q4nx.json. A matrix whose rows are not a multiple of 32 or columns of 256 cannot be cut
// into the kernel's blocks; it is listed under "unblocked" and not written. A matrix met again
// (one per cache tier) is written once.
type q4nxWriter struct {
	path  string
	f     *os.File
	off   int64
	index q4nxIndex
	seen  map[string]bool
	err   error
}

type q4nxIndex struct {
	Format    string       `json:"format"`
	Block     [2]int       `json:"block"`
	Group     int          `json:"group"`
	Matrices  []q4nxMatrix `json:"matrices"`
	Unblocked []q4nxMatrix `json:"unblocked,omitempty"`
}

type q4nxMatrix struct {
	Name   string `json:"name"`
	Rows   int    `json:"rows"`
	Cols   int    `json:"cols"`
	Offset int64  `json:"offset,omitempty"`
	Bytes  int    `json:"bytes,omitempty"`
}

func newQ4NXWriter(base string) (*q4nxWriter, error) {
	f, err := os.Create(base + ".q4nx")
	if err != nil {
		return nil, err
	}
	return &q4nxWriter{path: base + ".q4nx", f: f, seen: map[string]bool{},
		index: q4nxIndex{Format: "q4nx", Block: [2]int{palette.Q4NXBlockRows, palette.Q4NXBlockCols}, Group: palette.Q4NXGroup}}, nil
}

func (w *q4nxWriter) add(name string, q palette.Q4NX) {
	if w.err != nil || w.seen[name] {
		return
	}
	w.seen[name] = true
	m := q4nxMatrix{Name: name, Rows: q.Rows, Cols: q.Cols}
	if q.Rows%palette.Q4NXBlockRows != 0 || q.Cols%palette.Q4NXBlockCols != 0 {
		w.index.Unblocked = append(w.index.Unblocked, m)
		return
	}
	if pad := (64 - w.off%64) % 64; pad > 0 {
		if _, w.err = w.f.Write(make([]byte, pad)); w.err != nil {
			return
		}
		w.off += pad
	}
	b := q.Blocks()
	m.Offset, m.Bytes = w.off, len(b)
	if _, w.err = w.f.Write(b); w.err != nil {
		return
	}
	w.off += int64(len(b))
	w.index.Matrices = append(w.index.Matrices, m)
}

func (w *q4nxWriter) close() error {
	err := errors.Join(w.err, w.f.Close())
	if err != nil {
		return err
	}
	j, err := json.MarshalIndent(w.index, "", "  ")
	if err != nil {
		return err
	}
	fmt.Printf("%s: %d q4nx matrices in the AMD kernel's blocks (%.1f MiB), %d not in blocks\n", w.path, len(w.index.Matrices), float64(w.off)/(1<<20), len(w.index.Unblocked))
	return os.WriteFile(w.path+".json", append(j, '\n'), 0o644)
}
