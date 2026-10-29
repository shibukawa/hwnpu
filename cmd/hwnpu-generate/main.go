// hwnpu-generate writes text with a causal language model converted by hwnpu-convert, on the
// Neural Engine. The model may be in parts (hwnpu-convert -parts), loaded together as one chain.
//
//	hwnpu-generate -package lfm3.parts.json -config DIR/config.json -weights DIR/model.safetensors -prompt "..."
//
// A model converted with a KV cache (hwnpu-convert -cache) reads the prompt -seq tokens per call
// and then generates one token per call, the keys, values and convolution windows kept in the
// program's input buffers (internal/llm/decode); it stops when the cache is full. A model without
// one runs the whole fixed length program over the tokens so far for every token, and reads the
// logits at the last one: the positions after it hold zeros and do not change the result, because
// the model is causal; when the window is full, the oldest quarter is dropped (after the first
// token). The token embeddings are read from the checkpoint on the host, as the package expects.
package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	_ "github.com/shibukawa/hwnpu/internal/llm/archs"
	"github.com/shibukawa/hwnpu/internal/llm/chain"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

type sampling struct {
	temp, rep float64
	topK      int
	rng       *rand.Rand
}

func main() {
	pkg := flag.String("package", "", "the converted model: an .mlpackage, or the out.parts.json of a model hwnpu-convert split")
	config := flag.String("config", "", "config.json of the model")
	weights := flag.String("weights", "", "checkpoint the token embeddings are read from: .safetensors or .gguf")
	tok := flag.String("tokenizer", "", "tokenizer.json (default: next to config.json)")
	prompt := flag.String("prompt", "", "the user's message")
	system := flag.String("system", "", "system message")
	raw := flag.Bool("raw", false, "continue the prompt as plain text instead of a chat turn")
	maxNew := flag.Int("n", 256, "most tokens to generate")
	temp := flag.Float64("temp", 0, "sampling temperature; 0 picks the most likely token")
	topK := flag.Int("top-k", 50, "sample from the k most likely tokens")
	rep := flag.Float64("rep", 1.1, "repetition penalty on tokens already in the window (1: none)")
	seed := flag.Uint64("seed", 1, "random seed for sampling")
	special := flag.Bool("special", false, "print special tokens such as <think> too")
	cacheDir := flag.String("cache", "", "compile cache (default: hwnpu in the user cache directory)")
	flag.Parse()
	if *pkg == "" || *config == "" || *weights == "" || *prompt == "" {
		flag.Usage()
		os.Exit(2)
	}
	if *tok == "" {
		*tok = filepath.Join(filepath.Dir(*config), "tokenizer.json")
	}
	if *cacheDir == "" {
		if d, err := os.UserCacheDir(); err == nil {
			*cacheDir = filepath.Join(d, "hwnpu")
		}
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	s := sampling{temp: *temp, rep: *rep, topK: *topK, rng: rand.New(rand.NewPCG(*seed, 0))}
	if err := run(ctx, *pkg, *config, *weights, *tok, *system, *prompt, *raw, *special, *maxNew, *cacheDir, s); err != nil {
		fmt.Fprintln(os.Stderr, "hwnpu-generate:", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, pkg, config, weights, tokPath, system, prompt string, raw, special bool, maxNew int, cacheDir string, s sampling) error {
	a, model, err := llm.LoadConfig(config, "")
	if err != nil {
		return err
	}
	if _, logits := model.Output(); !logits {
		return fmt.Errorf("%s is not a causal language model", a.Title)
	}
	tk, err := tokenizer.Load(tokPath)
	if err != nil {
		return err
	}
	w, err := llm.OpenWeights(weights, a)
	if err != nil {
		return err
	}
	defer w.Close()

	opts := []hwnpu.Option{hwnpu.WithBackend("coreml"), hwnpu.WithPlacement(hwnpu.PlacementPrefer)}
	if cacheDir != "" {
		opts = append(opts, hwnpu.WithCacheDir(cacheDir))
	}
	dev, err := hwnpu.Open(ctx, opts...)
	if err != nil {
		return err
	}
	defer dev.Close()
	t0 := time.Now()
	c, err := chain.Open(ctx, dev, pkg)
	if err != nil {
		return err
	}
	defer c.Close()
	for _, p := range c.Parts() {
		where := "NPU"
		if !p.Placement.OnNPU() {
			where = fmt.Sprintf("not all on the NPU (%d on the NPU, %d elsewhere)", p.Placement.NPUOps, len(p.Placement.Ops))
		}
		if p.Layers > 0 {
			fmt.Fprintf(os.Stderr, "%s: layers %d to %d, %.2f GiB, %s\n", filepath.Base(p.Path), p.Start, p.Start+p.Layers-1, float64(p.WeightBytes)/(1<<30), where)
		} else {
			fmt.Fprintf(os.Stderr, "%s: %s\n", filepath.Base(p.Path), where)
		}
	}
	fmt.Fprintf(os.Stderr, "loaded %d parts in %.1fs\n", len(c.Parts()), time.Since(t0).Seconds())

	x, y := c.Input("x"), c.Output()
	if x == nil {
		return fmt.Errorf("%s has no input x", pkg)
	}
	ids, stops, err := promptIDs(tk, system, prompt, raw)
	if err != nil {
		return err
	}
	out := newPrinter(tk, special)
	emb := map[int32][]float32{}
	embed := func(t int32) ([]float32, error) {
		if e, ok := emb[t]; ok {
			return e, nil
		}
		e, err := llm.Embed(w, model, []int32{t})
		emb[t] = e
		return e, err
	}
	if hasCache(c) {
		cm, ok := model.(llm.Cached)
		if !ok {
			return fmt.Errorf("%s has a KV cache, but %s cannot run with one", pkg, a.Title)
		}
		return generateCached(ctx, c.Program(ctx), cm, ids, stops, embed, out, maxNew, s)
	}
	hidden := model.Hidden()
	vocab, _ := model.Output()
	seq := x.Size() / (2 * hidden)
	if y.Size() != 2*seq*vocab {
		return fmt.Errorf("output of %d bytes is not logits (%d, %d) in fp16", y.Size(), seq, vocab)
	}
	if len(ids) >= seq {
		return fmt.Errorf("the prompt is %d tokens; the package takes %d (convert with a larger -seq)", len(ids), seq)
	}
	xb := x.Bytes()
	clear(xb)
	write := func(pos int, t int32) error {
		e, err := embed(t)
		if err != nil {
			return err
		}
		hwnpu.EncodeF16(xb[2*pos*hidden:2*(pos+1)*hidden], e)
		return nil
	}
	for i, t := range ids {
		if err := write(i, t); err != nil {
			return err
		}
	}
	x.SyncToDevice(0, x.Size())

	logits := make([]float32, vocab)
	var gen []int32
	var first, total time.Duration
	slid := false
	for len(gen) < maxNew && ctx.Err() == nil {
		t := time.Now()
		if err := c.Run(ctx); err != nil {
			return err
		}
		n := len(ids)
		off := 2 * (n - 1) * vocab
		y.SyncFromDevice(off, 2*vocab)
		hwnpu.DecodeF16(logits, y.Bytes()[off:off+2*vocab])
		next := s.pick(logits, ids)
		d := time.Since(t)
		if len(gen) == 0 {
			first = d
		} else {
			total += d
		}
		if stops[next] {
			break
		}
		gen = append(gen, next)
		ids = append(ids, next)
		out.add(next)
		if len(ids) == seq {
			// Keep the first token and drop the oldest quarter of the rest.
			drop := max(seq/4, 1)
			ids = append(ids[:1], ids[1+drop:]...)
			clear(xb)
			for i, t := range ids {
				if err := write(i, t); err != nil {
					return err
				}
			}
			x.SyncToDevice(0, x.Size())
			if !slid {
				fmt.Fprintf(os.Stderr, "\n[window of %d tokens full: dropping the oldest]\n", seq)
				slid = true
			}
			continue
		}
		if err := write(len(ids)-1, next); err != nil {
			return err
		}
		x.SyncToDevice(2*(len(ids)-1)*hidden, 2*hidden)
	}
	fmt.Println()
	if len(gen) > 1 {
		ms := float64(total.Microseconds()) / 1000 / float64(len(gen)-1)
		fmt.Fprintf(os.Stderr, "%d tokens; first %.0f ms, then %.1f ms/token (%.1f tokens/s), window %d\n", len(gen), float64(first.Microseconds())/1000, ms, 1000/ms, seq)
	}
	return nil
}

// pick chooses the next token from the logits at the last position.
func (s sampling) pick(logits []float32, context []int32) int32 {
	if s.rep != 1 {
		seen := map[int32]bool{}
		for _, t := range context {
			if seen[t] || int(t) >= len(logits) {
				continue
			}
			seen[t] = true
			if l := logits[t]; l > 0 {
				logits[t] = l / float32(s.rep)
			} else {
				logits[t] = l * float32(s.rep)
			}
		}
	}
	if s.temp <= 0 {
		best := 0
		for i, l := range logits {
			if l > logits[best] {
				best = i
			}
		}
		return int32(best)
	}
	idx := make([]int, len(logits))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return logits[idx[a]] > logits[idx[b]] })
	k := len(idx)
	if s.topK > 0 && s.topK < k {
		k = s.topK
	}
	idx = idx[:k]
	ps := make([]float64, k)
	var sum float64
	for i, j := range idx {
		ps[i] = math.Exp(float64(logits[j]-logits[idx[0]]) / s.temp)
		sum += ps[i]
	}
	r := s.rng.Float64() * sum
	for i, p := range ps {
		if r -= p; r <= 0 {
			return int32(idx[i])
		}
	}
	return int32(idx[k-1])
}

func hasCache(c *chain.Chain) bool {
	for name := range c.Program(context.Background()).In {
		if strings.HasPrefix(name, "kv_") {
			return true
		}
	}
	return false
}

// promptIDs turns the prompt into tokens, in the chat format unless raw, and returns the tokens
// that end the answer.
func promptIDs(tk *tokenizer.Tokenizer, system, prompt string, raw bool) ([]int32, map[int32]bool, error) {
	text := prompt
	if !raw {
		var sb strings.Builder
		sb.WriteString("<|startoftext|>")
		if system != "" {
			sb.WriteString("<|im_start|>system\n" + system + "<|im_end|>\n")
		}
		sb.WriteString("<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n")
		text = sb.String()
	} else if _, ok := tk.ID("<|startoftext|>"); ok {
		text = "<|startoftext|>" + prompt
	}
	stops := map[int32]bool{}
	for _, t := range []string{"<|im_end|>", "<|endoftext|>"} {
		if id, ok := tk.ID(t); ok {
			stops[id] = true
		}
	}
	return tk.Encode(text), stops, nil
}

// printer prints the generated text as it grows, whole characters at a time.
type printer struct {
	tk      *tokenizer.Tokenizer
	special bool
	gen     []int32
	printed int
}

func newPrinter(tk *tokenizer.Tokenizer, special bool) *printer {
	return &printer{tk: tk, special: special}
}

func (p *printer) add(t int32) {
	p.gen = append(p.gen, t)
	if out := p.tk.Decode(p.gen, !p.special); utf8.ValidString(out) && len(out) > p.printed {
		fmt.Print(out[p.printed:])
		p.printed = len(out)
	}
}

// generateCached reads the prompt Q tokens per step and then generates one token per step.
func generateCached(ctx context.Context, prog decode.Program, m llm.Cached, ids []int32, stops map[int32]bool, embed func(int32) ([]float32, error), out *printer, maxNew int, s sampling) error {
	sess, err := decode.New(prog, m)
	if err != nil {
		return err
	}
	if len(ids) >= sess.Cache() {
		return fmt.Errorf("the prompt is %d tokens; the cache holds %d (convert with a larger -cache)", len(ids), sess.Cache())
	}
	step := func(toks []int32) ([]float32, error) {
		var e [][]float32
		for _, t := range toks {
			v, err := embed(t)
			if err != nil {
				return nil, err
			}
			e = append(e, v)
		}
		return sess.Step(e)
	}
	t0 := time.Now()
	var logits []float32
	for i := 0; i < len(ids); i += sess.Q() {
		if logits, err = step(ids[i:min(i+sess.Q(), len(ids))]); err != nil {
			return err
		}
	}
	prefill := time.Since(t0)
	seen := append([]int32(nil), ids...)
	n := 0
	t1 := time.Now()
	for n < maxNew && ctx.Err() == nil {
		next := s.pick(logits, seen)
		if stops[next] {
			break
		}
		out.add(next)
		seen = append(seen, next)
		n++
		if sess.Pos() == sess.Cache() {
			fmt.Fprintf(os.Stderr, "\n[the KV cache of %d tokens is full]\n", sess.Cache())
			break
		}
		if n == maxNew {
			break
		}
		if logits, err = step([]int32{next}); err != nil {
			return err
		}
	}
	fmt.Println()
	gen := time.Since(t1)
	fmt.Fprintf(os.Stderr, "prompt %d tokens in %.0f ms (%d per step); %d tokens in %.1f s, %.1f ms/token (%.1f tokens/s); cache %d of %d\n",
		len(ids), float64(prefill.Microseconds())/1000, sess.Q(), n, gen.Seconds(), float64(gen.Microseconds())/1000/float64(max(n, 1)), float64(n)/gen.Seconds(), sess.Pos(), sess.Cache())
	return nil
}
