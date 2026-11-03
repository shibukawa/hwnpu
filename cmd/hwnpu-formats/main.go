// hwnpu-formats compares the weight formats of a chat model on the NPU: for each format it
// converts the model (as hwnpu-chat does, kept for later runs), loads it, and reads a fixed
// conversation through it a few tokens per call, as generation with drafts runs. It reports how
// often the most likely next token is the reference format's, how far the logits are from the
// reference's, the perplexity of the conversation's answers, and the time per call.
//
//	hwnpu-formats                                   # LFM3-1.7B: int8, int4-g128, int4 against fp16
//	hwnpu-formats -formats int4-g128 -ref int8
//	hwnpu-formats -ref-backend openvino             # the reference on OpenVINO (HWNPU_OV_PYTHON)
//
// The reference runs first and its logits are kept; then each format is loaded alone, so that
// two models are never in memory at once. -json writes the results for a report.
package main

import (
	"cmp"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/signal"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
	_ "github.com/shibukawa/hwnpu/internal/llm/archs"
	"github.com/shibukawa/hwnpu/internal/llm/chain"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// samples are the conversations read: a question and an answer, in Japanese and in English with
// code, as the chat and the agent see them.
var samples = []struct{ user, answer string }{
	{"日本の四季について、短く説明してください。",
		"日本には春、夏、秋、冬の四つの季節があります。春は三月から五月で、桜が咲き、新しい学年や仕事が始まります。" +
			"夏は六月から八月で、梅雨のあとに暑く湿った日が続き、各地で花火大会や祭りが開かれます。" +
			"秋は九月から十一月で、空気が澄み、山の木々が赤や黄色に色づきます。" +
			"冬は十二月から二月で、北の地方や日本海側では雪が多く、温泉やスキーを楽しむ人が増えます。"},
	{"Write a Go function that reverses a slice of ints in place, and explain it briefly.",
		"Here is a function that reverses the slice in place by swapping elements from both ends:\n\n" +
			"```go\nfunc reverse(s []int) {\n\tfor i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {\n\t\ts[i], s[j] = s[j], s[i]\n\t}\n}\n```\n\n" +
			"The two indices start at the first and the last element and move toward the middle. " +
			"Each step swaps the pair, so the loop runs len(s)/2 times and needs no extra memory."},
}

type result struct {
	Format      string  `json:"format"`
	WeightBytes int64   `json:"weight_bytes"`
	LoadSeconds float64 `json:"load_seconds"`
	Cached      bool    `json:"compiled_from_cache"`
	OnNPU       bool    `json:"on_npu"`
	MsPerCall   float64 `json:"ms_per_call"` // median
	Calls       int     `json:"calls"`
	Tokens      int     `json:"tokens"`    // positions compared
	SameNext    int     `json:"same_next"` // most likely next token as the reference's
	MeanRelErr  float64 `json:"mean_rel_err"`
	WorstRelErr float64 `json:"worst_rel_err"`
	Perplexity  float64 `json:"perplexity"` // of the answers' tokens
	Error       string  `json:"error,omitempty"`
}

func main() {
	use := flag.String("use", "LFM3-1.7B", "the chat model: a name that hwnpu-chat -list shows, or a Hugging Face repository OWNER/NAME")
	dir := flag.String("dir", chat.ModelsDir(), "where the models are kept (as hwnpu-chat's -dir); the programs go in FORMATS/ of the model's program directory")
	formats := flag.String("formats", "int8,int4-g128,int4", "weight formats to compare, comma separated")
	ref := flag.String("ref", "fp16", "the reference format (\"none\": no reference, perplexity and time only)")
	backend := flag.String("backend", "", "the NPU (as hwnpu-chat's -backend; default: HWNPU_BACKEND, else the NPU found)")
	refBackend := flag.String("ref-backend", "", "where the reference runs (default: -backend), e.g. openvino to compute it on OpenVINO's CPU")
	cache := flag.Int("cache", 4096, "KV cache of the programs (hwnpu-chat runs short conversations on 4096)")
	seq := flag.Int("seq", 8, "tokens per call (hwnpu-chat and hwnpu-agent generate with 8: a token and 7 drafts)")
	cacheDir := flag.String("cache-dir", "", "compile cache (default: hwnpu in the user cache directory, as hwnpu-chat's -cache)")
	jsonOut := flag.String("json", "", "also write the results to this file")
	flag.Parse()
	if *cacheDir == "" {
		if d, err := os.UserCacheDir(); err == nil {
			*cacheDir = filepath.Join(d, "hwnpu")
		}
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if err := run(ctx, *use, *dir, *formats, *ref, *backend, *refBackend, *cache, *seq, *cacheDir, *jsonOut); err != nil {
		fmt.Fprintln(os.Stderr, "hwnpu-formats:", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, use, root, formatList, refFormat, backend, refBackend string, cache, seq int, cacheDir, jsonOut string) error {
	m, err := chat.Resolve(ctx, use, root, os.Stderr)
	if err != nil {
		return err
	}
	if m.Kind != chat.KindChat {
		return fmt.Errorf("%s is not a chat model", m.Name)
	}
	md := filepath.Join(root, m.Name)
	if err := os.MkdirAll(md, 0o755); err != nil {
		return err
	}
	for _, f := range []string{"config.json", "tokenizer.json", "model.safetensors"} {
		if err := chat.Fetch(ctx, m.Repo, f, filepath.Join(md, f), os.Stderr); err != nil {
			return err
		}
	}
	a, model, err := llm.LoadConfig(filepath.Join(md, "config.json"), "")
	if err != nil {
		return err
	}
	cm, ok := model.(llm.Cached)
	if !ok {
		return fmt.Errorf("%s cannot run with a KV cache", a.Title)
	}
	w, err := llm.OpenWeights(filepath.Join(md, "model.safetensors"), a)
	if err != nil {
		return err
	}
	defer w.Close()
	tok, err := tokenizer.Load(filepath.Join(md, "tokenizer.json"))
	if err != nil {
		return err
	}
	tmpl, err := chat.TemplateFor(a.Name)
	if err != nil {
		return err
	}
	// the conversations' tokens, and from where in each the answer is
	var texts [][]int32
	var answers []int
	total := 0
	for _, s := range samples {
		prompt, _, err := tmpl.Render([]chat.Message{{Role: "user", Content: chat.Content{Text: s.user}}}, nil, chat.Options{NoThink: true})
		if err != nil {
			return err
		}
		p := tok.Encode(prompt)
		ids := append(slices.Clone(p), tok.Encode(s.answer)...)
		texts, answers = append(texts, ids), append(answers, len(p))
		total += len(ids)
	}
	if total > cache {
		return fmt.Errorf("the conversations take %d tokens, over -cache %d", total, cache)
	}

	program, err := llm.ProgramFormat(ctx, backend)
	if err != nil {
		return err
	}
	var refRows [][][]float32
	var results []result
	if refFormat != "none" {
		fmt.Fprintf(os.Stderr, "reference: %s\n", refFormat)
		r, rows := measure(ctx, a, model, cm, w, md, cmp.Or(refBackend, backend), refFormat, cache, seq, cacheDir, texts, answers, nil)
		if r.Error != "" {
			return fmt.Errorf("the reference %s: %s", refFormat, r.Error)
		}
		refRows = rows
		results = append(results, r)
	}
	for _, f := range strings.Split(formatList, ",") {
		f = strings.TrimSpace(f)
		if f == "" || f == refFormat {
			continue
		}
		fmt.Fprintf(os.Stderr, "%s\n", f)
		r, _ := measure(ctx, a, model, cm, w, md, backend, f, cache, seq, cacheDir, texts, answers, refRows)
		results = append(results, r)
		if ctx.Err() != nil {
			break
		}
	}
	report(os.Stdout, m.Name, program, refFormat, cache, seq, results)
	if jsonOut != "" {
		b, _ := json.MarshalIndent(map[string]any{"model": m.Name, "backend": program, "reference": refFormat, "cache": cache, "seq": seq, "results": results}, "", "  ")
		if err := os.WriteFile(jsonOut, append(b, '\n'), 0o644); err != nil {
			return err
		}
	}
	return ctx.Err()
}

// measure converts the model in format (unless it is there), loads it, and reads the texts
// through it. It returns the logits after every token when keep is nil (the reference's), and
// compares them with keep otherwise.
func measure(ctx context.Context, a *llm.Arch, model llm.Model, cm llm.Cached, w llm.Weights, md, backend, format string, cache, seq int, cacheDir string, texts [][]int32, answers []int, keep [][][]float32) (result, [][][]float32) {
	r := result{Format: format}
	fail := func(err error) (result, [][][]float32) {
		r.Error = err.Error()
		fmt.Fprintf(os.Stderr, "  %s: %v\n", format, err)
		return r, nil
	}
	dev, program, err := llm.OpenDevice(ctx, backend, cacheDir)
	if err != nil {
		return fail(err)
	}
	defer dev.Close()
	path, err := convert(a, model, w, md, program, format, cache, seq)
	if err != nil {
		return fail(err)
	}
	t0 := time.Now()
	ch, err := chain.Open(ctx, dev, path)
	if err != nil {
		return fail(err)
	}
	defer ch.Close()
	r.LoadSeconds = time.Since(t0).Seconds()
	r.OnNPU, r.Cached = ch.OnNPU(), true
	for _, p := range ch.Parts() {
		r.WeightBytes += p.WeightBytes
		r.Cached = r.Cached && p.Cached
	}
	fmt.Fprintf(os.Stderr, "  loaded on %s in %.1fs (%.2f GiB of weights)\n", llm.DeviceName(dev), r.LoadSeconds, float64(r.WeightBytes)/(1<<30))
	s, err := decode.New(ch.Program(ctx), cm)
	if err != nil {
		return fail(err)
	}
	if s.Q() < seq {
		return fail(fmt.Errorf("the program takes %d tokens per call, not %d", s.Q(), seq))
	}
	var all [][][]float32
	var times []time.Duration
	var nll float64
	nAnswer := 0
	var sumErr float64
	for ti, ids := range texts {
		s.Reset()
		var rows [][]float32
		for pos := 0; pos < len(ids); {
			if err := ctx.Err(); err != nil {
				return fail(err)
			}
			n := min(seq, len(ids)-pos)
			emb := make([][]float32, n)
			for i := range emb {
				if emb[i], err = llm.TokenRow(w, cm, ids[pos+i]); err != nil {
					return fail(err)
				}
			}
			t := time.Now()
			_, _, err := s.Speculate(emb, func(row func(int) []float32) int {
				for i := range n {
					rows = append(rows, slices.Clone(row(i)))
				}
				return n
			})
			if err != nil {
				return fail(err)
			}
			times = append(times, time.Since(t))
			pos += n
		}
		for i := 0; i+1 < len(ids); i++ {
			row := rows[i]
			if i+1 >= answers[ti] {
				nll -= logSoftmax(row, ids[i+1])
				nAnswer++
			}
			if keep == nil {
				continue
			}
			ref := keep[ti][i]
			if argmax(row) == argmax(ref) {
				r.SameNext++
			}
			e := relErr(row, ref)
			sumErr += e
			r.WorstRelErr = math.Max(r.WorstRelErr, e)
			r.Tokens++
		}
		all = append(all, rows)
	}
	if r.Tokens > 0 {
		r.MeanRelErr = sumErr / float64(r.Tokens)
	}
	r.Perplexity = math.Exp(nll / float64(nAnswer))
	slices.Sort(times)
	r.Calls = len(times)
	r.MsPerCall = float64(times[len(times)/2].Microseconds()) / 1000
	if keep != nil {
		fmt.Fprintf(os.Stderr, "  next token as the reference's at %d of %d, perplexity %.2f, %.1f ms per call\n", r.SameNext, r.Tokens, r.Perplexity, r.MsPerCall)
		return r, nil
	}
	fmt.Fprintf(os.Stderr, "  perplexity %.2f, %.1f ms per call\n", r.Perplexity, r.MsPerCall)
	return r, all
}

// convert writes the program of format unless it is there (NAME.done holds its path, as
// hwnpu-chat keeps its programs) and returns its path.
func convert(a *llm.Arch, model llm.Model, w llm.Weights, md, program, format string, cache, seq int) (string, error) {
	dir := filepath.Join(md, program, "formats")
	if program == llm.BackendCoreML {
		dir = filepath.Join(md, "formats")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	ext := ".mlpackage"
	if program == llm.BackendIntel {
		ext = ".xml"
	}
	name := fmt.Sprintf("%s-%s-c%d-q%d", a.Name, format, cache, seq)
	done := filepath.Join(dir, name+".done")
	if b, err := os.ReadFile(done); err == nil {
		return string(b), nil
	}
	fs := flag.NewFlagSet("", flag.ContinueOnError)
	var opts llm.Options
	set := llm.BindFlags(fs, &opts, "")
	if err := fs.Parse(nil); err != nil {
		return "", err
	}
	set()
	opts.Backend, opts.Format, opts.Seq, opts.Cache = program, format, seq, cache
	if program != llm.BackendIntel && !strings.HasPrefix(format, "int") {
		opts.Quant = "kmeans-minmax" // hwnpu-chat's table fitting
	}
	fmt.Fprintf(os.Stderr, "  converting %s\n", name)
	path, err := llm.Convert(a, model, w, filepath.Join(dir, name+ext), "any", 0, opts, io.Discard)
	if err != nil {
		return "", err
	}
	return path, os.WriteFile(done, []byte(path), 0o644)
}

func report(out io.Writer, model, program, ref string, cache, seq int, results []result) {
	fmt.Fprintf(out, "\n%s for %s, cache %d, %d tokens per call; reference %s\n\n", model, program, cache, seq, ref)
	fmt.Fprintln(out, "| format | weights (GiB) | load (s) | on NPU | ms per call | next token as reference | logits rel. err. (mean / worst) | perplexity |")
	fmt.Fprintln(out, "|---|---|---|---|---|---|---|---|")
	for _, r := range results {
		if r.Error != "" {
			fmt.Fprintf(out, "| %s | | | | | failed: %s | | |\n", r.Format, strings.ReplaceAll(r.Error, "\n", " "))
			continue
		}
		load := fmt.Sprintf("%.1f", r.LoadSeconds)
		if r.Cached {
			load += " (cached)"
		}
		same, errs := "(reference)", ""
		if r.Format != ref {
			same = fmt.Sprintf("%d / %d (%.0f%%)", r.SameNext, r.Tokens, 100*float64(r.SameNext)/float64(max(r.Tokens, 1)))
			errs = fmt.Sprintf("%.3f / %.3f", r.MeanRelErr, r.WorstRelErr)
			if r.Tokens == 0 {
				same = ""
			}
		}
		fmt.Fprintf(out, "| %s | %.2f | %s | %v | %.1f | %s | %s | %.2f |\n", r.Format, float64(r.WeightBytes)/(1<<30), load, r.OnNPU, r.MsPerCall, same, errs, r.Perplexity)
	}
}

func argmax(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}

// logSoftmax is log p(id) under the logits.
func logSoftmax(v []float32, id int32) float64 {
	m := float64(v[argmax(v)])
	var sum float64
	for _, x := range v {
		sum += math.Exp(float64(x) - m)
	}
	return float64(v[id]) - m - math.Log(sum)
}

func relErr(got, ref []float32) float64 {
	var se, sr float64
	for i := range ref {
		d := float64(got[i] - ref[i])
		se += d * d
		sr += float64(ref[i]) * float64(ref[i])
	}
	return math.Sqrt(se / sr)
}
