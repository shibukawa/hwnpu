// hwnpu-check compares a conversion with the Hugging Face reference on the CPU, without a Neural
// Engine: it builds the program like hwnpu-convert and runs it with the fp32 interpreter
// (internal/milcpu). It is how a port of a new architecture is checked.
//
//	hwnpu-check -config DIR/config.json -weights DIR/model.safetensors -ref DIR/ref [-format fp16]
//
// The reference is what tools/ref/ref_logits.py writes (REF.ids.bin, REF.hidden.bin,
// REF.logits.bin and REF.json). The check runs, in order:
//
//   - the embeddings looked up on the host against the reference embeddings
//   - every layer on its own, fed the reference hidden state entering it, so that errors do not
//     accumulate and the first wrong layer stands out
//   - what follows the last layer (final norm and output layer) on its own
//   - the whole model from the reference embeddings to the logits
//
// The exit status is 1 if a layer's relative RMS error exceeds -tol. The default format is fp16,
// which checks the structure; -format lut4 measures the quantization loss on top. -package
// instead checks an existing .mlpackage's output (for example hwnpu-convert's lut4 output). Inputs
// other than x, such as an encoder's attention mask, are fed ones: the reference has no padding.
package main

import (
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/llm"
	_ "github.com/shibukawa/hwnpu/internal/llm/archs"
	"github.com/shibukawa/hwnpu/internal/llm/check"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

func main() {
	config := flag.String("config", "", "config.json of the model")
	weights := flag.String("weights", "", "checkpoint: .safetensors (bf16) or .gguf")
	ref := flag.String("ref", "", "reference prefix written by tools/ref/ref_logits.py (REF.ids.bin, ...)")
	arch := flag.String("arch", "", "architecture (default: detected from config.json)")
	layerList := flag.String("layers", "", "comma separated layers to check on their own (default: all)")
	full := flag.Bool("full", true, "also run the whole model from the embeddings to the logits")
	f16 := flag.Bool("f16", false, "round every operation's result to fp16 (storage precision of the Neural Engine)")
	tol := flag.Float64("tol", 0.01, "largest relative RMS error of a layer on its own")
	pkg := flag.String("package", "", "check this .mlpackage's output against the reference logits instead of building")
	var opts llm.Options
	setOpts := llm.BindFlags(flag.CommandLine, &opts, "fp16")
	flag.Parse()
	setOpts()
	var err error
	ok := true
	switch {
	case *pkg != "" && *ref != "":
		err = checkPackage(*pkg, *ref, *f16)
	case *config != "" && *weights != "" && *ref != "":
		ok, err = run(*config, *arch, *weights, *ref, *layerList, *full, *f16, *tol, opts)
	default:
		flag.Usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "hwnpu-check:", err)
		os.Exit(1)
	}
	if !ok {
		os.Exit(1)
	}
}

func run(config, arch, weights, refPrefix, layerList string, full, f16 bool, tol float64, opts llm.Options) (bool, error) {
	a, model, err := llm.LoadConfig(config, arch)
	if err != nil {
		return false, err
	}
	w, err := llm.OpenWeights(weights, a)
	if err != nil {
		return false, err
	}
	defer w.Close()
	ref, err := check.LoadRef(refPrefix)
	if err != nil {
		return false, err
	}
	layers, err := parseLayers(layerList, model.Layers())
	if err != nil {
		return false, err
	}
	fmt.Printf("%s, ", a.Title)
	first, err := checkModel(os.Stdout, model, w, ref, layers, full, f16, tol, opts)
	return first < 0, err
}

// checkModel runs the checks and returns the first layer over tol, or -1.
func checkModel(out io.Writer, model llm.Model, w llm.Weights, ref *check.Ref, layers []int, full, f16 bool, tol float64, opts llm.Options) (int, error) {
	nl, d := model.Layers(), model.Hidden()
	width, logits := model.Output()
	if err := ref.Validate(nl, d, width); err != nil {
		return -1, err
	}
	seq := len(ref.IDs)
	opts.Seq = seq
	fmt.Fprintf(out, "%d layers, hidden %d, output %d", nl, d, width)
	if logits {
		fmt.Fprint(out, " logits")
	}
	fmt.Fprintf(out, "; reference %d tokens", seq)
	if ref.Meta.ModelType != "" {
		fmt.Fprintf(out, " (%s, %s)", ref.Meta.ModelType, ref.Meta.DType)
	}
	fmt.Fprintf(out, "; format %s, fp16 rounding %v\n", opts.Format, f16)

	if missing, unused, mismatched, ok := llm.Inventory(model, w); ok {
		for _, n := range missing {
			fmt.Fprintf(out, "missing tensor: %s\n", n)
		}
		for _, n := range mismatched {
			fmt.Fprintf(out, "shape mismatch: %s\n", n)
		}
		for _, n := range unused {
			fmt.Fprintf(out, "unused tensor: %s\n", n)
		}
		fmt.Fprintf(out, "tensors: %d missing, %d mismatched, %d unused\n", len(missing), len(mismatched), len(unused))
	}

	emb, err := llm.Embed(w, model, ref.IDs)
	if err != nil {
		return -1, err
	}
	st, err := check.Compare(emb, ref.Layer(0, d))
	if err != nil {
		return -1, err
	}
	fmt.Fprintf(out, "embeddings: %v\n", st)

	tmp, err := os.MkdirTemp("", "hwnpu-check")
	if err != nil {
		return -1, err
	}
	defer os.RemoveAll(tmp)
	first := -1
	for _, i := range layers {
		o := opts
		o.Start, o.Layers, o.NoHead = i, 1, true
		start := time.Now()
		got, err := buildRun(model, w, o, ref.Layer(i, d), tmp, f16)
		if err != nil {
			return -1, fmt.Errorf("layer %d: %w", i, err)
		}
		st, err := check.Compare(got, ref.Layer(i+1, d))
		if err != nil {
			return -1, fmt.Errorf("layer %d: %w", i, err)
		}
		mark := ""
		if st.RelRMS > tol || math.IsNaN(st.RelRMS) {
			mark = "  <- over -tol"
			if first < 0 {
				first = i
			}
		}
		fmt.Fprintf(out, "layer %d: %v, %.1fs%s\n", i, st, time.Since(start).Seconds(), mark)
	}

	o := opts
	o.Start = nl
	got, err := buildRun(model, w, o, ref.Layer(nl, d), tmp, f16)
	if err != nil {
		return -1, fmt.Errorf("head: %w", err)
	}
	ls, err := compareOutput(got, ref.Logits, width, logits)
	if err != nil {
		return -1, err
	}
	fmt.Fprintf(out, "after the last layer: %v\n", ls)

	if full {
		start := time.Now()
		got, err := buildRun(model, w, opts, ref.Layer(0, d), tmp, f16)
		if err != nil {
			return -1, fmt.Errorf("whole model: %w", err)
		}
		ls, err := compareOutput(got, ref.Logits, width, logits)
		if err != nil {
			return -1, err
		}
		fmt.Fprintf(out, "whole model: %v, %.1fs\n", ls, time.Since(start).Seconds())
	}
	if first >= 0 {
		fmt.Fprintf(out, "first layer over -tol %g: %d\n", tol, first)
	}
	return first, nil
}

// compareOutput compares the model's output with the reference: as logits, with argmax and KL,
// or as hidden states.
func compareOutput(got, ref []float32, width int, logits bool) (fmt.Stringer, error) {
	if logits {
		return check.CompareLogits(got, ref, width)
	}
	return check.Compare(got, ref)
}

// ones fills the inputs other than x, such as attention masks, with ones (every token valid,
// as in the reference run).
func ones(inputs map[string][]int, x []float32) map[string]milcpu.Tensor {
	feed := map[string]milcpu.Tensor{}
	for name, shape := range inputs {
		if name == "x" {
			feed[name] = milcpu.Tensor{Shape: shape, Data: x}
			continue
		}
		n := 1
		for _, s := range shape {
			n *= s
		}
		v := make([]float32, n)
		for i := range v {
			v[i] = 1
		}
		feed[name] = milcpu.Tensor{Shape: shape, Data: v}
	}
	return feed
}

func parseLayers(list string, n int) ([]int, error) {
	if list == "" {
		all := make([]int, n)
		for i := range all {
			all[i] = i
		}
		return all, nil
	}
	var out []int
	for _, s := range strings.Split(list, ",") {
		i, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil || i < 0 || i >= n {
			return nil, fmt.Errorf("-layers: %q is not a layer of 0..%d", s, n-1)
		}
		out = append(out, i)
	}
	return out, nil
}

// buildRun builds a program into a weight file in dir and runs it on the CPU with input x (and
// ones for any other input).
func buildRun(model llm.Model, w llm.Weights, opts llm.Options, x []float32, dir string, f16 bool) ([]float32, error) {
	path := filepath.Join(dir, "weight.bin")
	blobs, err := mlpackage.NewBlobWriter(path)
	if err != nil {
		return nil, err
	}
	b := mlpackage.NewBuilder("CoreML8", blobs)
	_, out, err := model.Build(b, w, opts)
	if err != nil {
		blobs.Close()
		return nil, err
	}
	prog := b.Finish(&mlpackage.Var{Name: out.Name})
	if err := blobs.Close(); err != nil {
		return nil, err
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	p, err := milcpu.Load(prog, f)
	if err != nil {
		return nil, err
	}
	res, err := p.Run(ones(p.Inputs(), x), milcpu.Options{RoundF16: f16})
	if err != nil {
		return nil, err
	}
	return res[out.Name].Data, nil
}

// checkPackage runs an existing package on the reference embeddings and compares its output
// (logits unless the reference says it holds hidden states).
func checkPackage(dir, refPrefix string, f16 bool) error {
	p, c, err := milcpu.OpenPackage(dir)
	if err != nil {
		return err
	}
	defer c.Close()
	ref, err := check.LoadRef(refPrefix)
	if err != nil {
		return err
	}
	seq := len(ref.IDs)
	if seq == 0 || len(ref.Logits)%seq != 0 {
		return fmt.Errorf("reference has %d tokens and %d outputs", seq, len(ref.Logits))
	}
	width := len(ref.Logits) / seq
	shape, ok := p.Inputs()["x"]
	if !ok || len(shape) != 3 || shape[1] != seq {
		return fmt.Errorf("%s: input x %v does not take the reference's %d tokens", dir, shape, seq)
	}
	d := shape[2]
	start := time.Now()
	out, err := p.Run(ones(p.Inputs(), ref.Layer(0, d)), milcpu.Options{RoundF16: f16})
	if err != nil {
		return err
	}
	ls, err := compareOutput(out["y"].Data, ref.Logits, width, ref.Meta.Output != "hidden")
	if err != nil {
		return err
	}
	fmt.Printf("%s: %v, %.1fs\n", dir, ls, time.Since(start).Seconds())
	return nil
}
