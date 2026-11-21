// Command rotkl measures what a weight format costs a model with and without the residual
// stream rotated (llm.Rotate, docs/design/apple-neural-engine.md): it builds the whole model for a text's tokens,
// runs it on the CPU interpreter (milcpu, float32) and writes the logits; given -ref, the logits
// of another run (fp16, not rotated), it prints the mean KL(ref || this) and the share of
// positions whose top token agrees.
//
//	go run ./tools/quant/rotkl -config DIR/config.json -weights DIR/model.safetensors.index.json \
//	    -tokenizer DIR/tokenizer.json -text FILE -seq 128 -format fp16 -o ref.bin
//	go run ./tools/quant/rotkl ... -format lut4 -rotate 1 -ref ref.bin -o lut4r.bin
//
// -rotate is the architecture's setting (llm.RotateParam, as hwnpu-dev convert takes it).
//
// The format's table fitting is hwnpu-chat's (kmeans-minmax for lutN, 16 rows a table).
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu/internal/llm"
	_ "github.com/shibukawa/hwnpu/internal/llm/archs"
	"github.com/shibukawa/hwnpu/internal/llm/check"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

func main() {
	config := flag.String("config", "", "config.json")
	weights := flag.String("weights", "", "checkpoint")
	tokPath := flag.String("tokenizer", "", "tokenizer.json")
	textPath := flag.String("text", "", "the text whose first -seq tokens are run")
	seq := flag.Int("seq", 128, "tokens")
	seed := flag.Int64("seed", 1, "the rotation's random signs")
	ref := flag.String("ref", "", "logits of the reference run, to compare with")
	out := flag.String("o", "", "write the logits here")
	f16 := flag.Bool("f16", false, "round every operation to fp16 (the Neural Engine's storage)")
	ane := flag.Bool("ane", false, "run on the Neural Engine (Core AI) instead of the CPU interpreter")
	perPart := flag.Int("per-part", 6, "with -ane: layers a program")
	cpuPart := flag.Int("cpu-per-part", 0, "on the CPU: layers a program (0: the whole model at once; a 4B model's float32 takes 16 GB)")
	var opts llm.Options
	set := llm.BindFlags(flag.CommandLine, &opts, "fp16")
	flag.Parse()
	set()
	opts.Backend = llm.BackendCoreML
	if strings.HasPrefix(opts.Format, "lut") && opts.Quant == "" {
		opts.Quant = "kmeans-minmax"
	}
	if err := run(*config, *weights, *tokPath, *textPath, *seq, opts.Params[llm.RotateParam] == "1", *seed, *ref, *out, *f16, *ane, *perPart, *cpuPart, opts); err != nil {
		fmt.Fprintln(os.Stderr, "rotkl:", err)
		os.Exit(1)
	}
}

func run(config, weights, tokPath, textPath string, seq int, rotate bool, seed int64, ref, out string, f16, ane bool, perPart, cpuPart int, opts llm.Options) error {
	a, model, err := llm.LoadConfig(config, "")
	if err != nil {
		return err
	}
	w, err := llm.OpenWeights(weights, a)
	if err != nil {
		return err
	}
	defer w.Close()
	if rotate {
		r, ok := model.(llm.Rotator)
		if !ok {
			return fmt.Errorf("%s cannot be rotated", a.Name)
		}
		rot, err := r.Rotation()
		if err != nil {
			return err
		}
		if w, err = llm.Rotate(w, rot, seed); err != nil {
			return err
		}
	}
	tok, err := tokenizer.Load(tokPath)
	if err != nil {
		return err
	}
	text, err := os.ReadFile(textPath)
	if err != nil {
		return err
	}
	ids := tok.Encode(string(text))
	if len(ids) < seq {
		return fmt.Errorf("the text has %d tokens, under -seq %d", len(ids), seq)
	}
	ids = ids[:seq]
	opts.Seq = seq
	x, err := llm.Embed(w, model, ids)
	if err != nil {
		return err
	}
	dir, err := os.MkdirTemp("", "rotkl")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	t := time.Now()
	var logits []float32
	if ane {
		logits, err = runANE(model, w, opts, x, dir, perPart)
	} else if cpuPart > 0 {
		logits, err = cpuParts(model, w, opts, x, dir, f16, cpuPart)
	} else {
		logits, err = buildRun(model, w, opts, x, dir, f16)
	}
	if err != nil {
		return err
	}
	width, _ := model.Output()
	fmt.Printf("%s rotate=%v ane=%v: %d tokens, built and run in %v\n", opts.Format, rotate, ane, seq, time.Since(t).Round(time.Second))
	if out != "" {
		b := make([]byte, 4*len(logits))
		for i, v := range logits {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
		}
		if err := os.WriteFile(out, b, 0o644); err != nil {
			return err
		}
	}
	if ref != "" {
		b, err := os.ReadFile(ref)
		if err != nil {
			return err
		}
		r := make([]float32, len(b)/4)
		for i := range r {
			r[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		}
		st, err := check.CompareLogits(logits, r, width)
		if err != nil {
			return err
		}
		fmt.Printf("  against %s: %v\n", filepath.Base(ref), st)
	}
	return nil
}

// buildRun is hwnpu-dev check's: the program into a weight file in dir, run on the CPU with x and
// ones for any other input.
func buildRun(model llm.Model, w llm.Weights, opts llm.Options, x []float32, dir string, f16 bool) ([]float32, error) {
	path := filepath.Join(dir, "weight.bin")
	blobs, err := mlpackage.NewBlobWriter(path)
	if err != nil {
		return nil, err
	}
	b := mlpackage.NewBuilder("CoreML8", blobs)
	_, outs, err := model.Build(b, w, opts)
	if err != nil {
		blobs.Close()
		return nil, err
	}
	prog := b.Finish(&mlpackage.Var{Name: outs.Name})
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
	in := map[string]milcpu.Tensor{}
	for name, shape := range p.Inputs() {
		if name == "x" {
			in[name] = milcpu.Tensor{Shape: shape, Data: x}
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
		fmt.Fprintf(os.Stderr, "input %s %v: ones\n", name, shape)
		in[name] = milcpu.Tensor{Shape: shape, Data: v}
	}
	res, err := p.Run(in, milcpu.Options{RoundF16: f16})
	if err != nil {
		return nil, err
	}
	return res[outs.Name].Data, nil
}

// hostHead is the final norm and the output layer of the hidden states h (rows of the hidden
// size) in float32, by the names the model's Rotation gives.
func hostHead(model llm.Model, w llm.Weights, h []float32) ([]float32, error) {
	r, ok := model.(llm.Rotator)
	if !ok {
		return nil, fmt.Errorf("the model has no Rotation for its head's names")
	}
	rot, err := r.Rotation()
	if err != nil {
		return nil, err
	}
	g, _, err := w.Get(rot.HeadNorm)
	if err != nil {
		return nil, err
	}
	head, shape, err := w.Get(rot.Head)
	if err != nil {
		return nil, err
	}
	d, vocab := rot.Dim, shape[0]
	rows := len(h) / d
	out := make([]float32, rows*vocab)
	var wg sync.WaitGroup
	for t := 0; t < rows; t++ {
		row := h[t*d : (t+1)*d]
		var ss float64
		for _, v := range row {
			ss += float64(v) * float64(v)
		}
		inv := float32(1 / math.Sqrt(ss/float64(d)+1e-6))
		n := make([]float32, d)
		for i, v := range row {
			s := g[i]
			if rot.ZeroCentered {
				s = 1 + g[i]
			}
			n[i] = v * inv * s
		}
		wg.Add(1)
		go func(t int, n []float32) {
			defer wg.Done()
			o := out[t*vocab : (t+1)*vocab]
			for v := 0; v < vocab; v++ {
				wr := head[v*d : (v+1)*d]
				var acc float32
				for i := range n {
					acc += wr[i] * n[i]
				}
				o[v] = acc
			}
		}(t, n)
	}
	wg.Wait()
	return out, nil
}

// cpuParts runs the model's layers on the CPU interpreter perPart at a time, the hidden state of
// one part the next one's x, and the final norm and the output layer on the host (hostHead).
func cpuParts(model llm.Model, w llm.Weights, opts llm.Options, x []float32, dir string, f16 bool, perPart int) ([]float32, error) {
	opts.NoHead = true
	h := x
	for start := 0; start < model.Layers(); start += perPart {
		o := opts
		o.Start, o.Layers = start, min(perPart, model.Layers()-start)
		var err error
		if h, err = buildRun(model, w, o, h, dir, f16); err != nil {
			return nil, fmt.Errorf("layers %d..%d: %w", o.Start, o.Start+o.Layers-1, err)
		}
	}
	return hostHead(model, w, h)
}
